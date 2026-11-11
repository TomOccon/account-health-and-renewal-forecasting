"""At-risk renewal triage with recommended technical-lead actions.

Priority = dollars at risk x probability of loss. Dollars at risk is the
account's ARR. Probability of loss is 1 minus the renewal rate of the account's
health band, estimated from history (see forecast.band_stats).
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Sequence

from .forecast import BandStats
from .health import FULL_WEIGHTS, band_of, component_scores, health_score
from .synth import Account
from .util import month_index

# component -> (score below which it counts as a weak driver, action text)
ACTIONS = {
    "escalations": (
        50.0,
        "Run a root-cause review of open escalations with engineering. Name one technical owner and send a weekly written status.",
    ),
    "tickets": (
        50.0,
        "Audit ticket themes. Separate product defects from enablement gaps, then agree a fix-or-train plan with the customer admin.",
    ),
    "usage": (
        50.0,
        "Run a usage deep-dive: unused licensed capabilities, broken integrations, stalled rollouts. Follow with an adoption workshop on the top two gaps.",
    ),
    "exec": (
        40.0,
        "Pair with the account owner to secure an executive sponsor meeting. Bring a measured business-outcome review, not a product tour.",
    ),
    "sentiment": (
        50.0,
        "Respond to survey detractor comments within one week and close the loop with the respondent.",
    ),
}
EARLY_ACTION = (
    "First-renewal account with slow time to value: re-baseline the success plan and hold a hands-on session "
    "to the first production use case before the renewal conversation."
)
SAVE_PLAY_DAYS = 90
SLOW_ONBOARDING_DAYS = 60


def recommend(a: Account, comps: Dict[str, float], days_to_renewal: int) -> List[str]:
    weak = sorted((k for k, (floor, _) in ACTIONS.items() if comps[k] < floor), key=lambda k: comps[k])
    acts = [ACTIONS[k][1] for k in weak[:3]]
    if a.renewal_number == 1 and a.days_to_first_value > SLOW_ONBOARDING_DAYS:
        acts.append(EARLY_ACTION)
    if not acts:
        acts.append("No single weak driver. Hold a standard pre-renewal health review and confirm the renewal owner.")
    if days_to_renewal <= SAVE_PLAY_DAYS and weak:
        acts.insert(0, f"Renewal in {days_to_renewal} days: open a save play now and review it weekly.")
    return acts


def at_risk_list(
    accounts: Sequence[Account],
    stats: Dict[str, BandStats],
    as_of: date,
    horizon_months: int = 18,
    weights: Dict[str, float] = FULL_WEIGHTS,
    top: Optional[int] = None,
    min_p_loss: float = 0.0,
) -> List[dict]:
    rows = []
    for a in accounts:
        i = month_index(a.contract_end, as_of)
        if not (0 <= i < horizon_months and a.contract_end >= as_of):
            continue
        score = health_score(a, weights)
        band = band_of(score)
        p_loss = 1 - stats[band].rate
        if p_loss < min_p_loss:
            continue
        comps = component_scores(a)
        days = (a.contract_end - as_of).days
        drivers = [k for k, _ in sorted(comps.items(), key=lambda kv: kv[1]) if k in ACTIONS and comps[k] < ACTIONS[k][0]][:3]
        rows.append(
            {
                "account_id": a.account_id,
                "name": a.name,
                "segment": a.segment,
                "arr": a.arr,
                "renewal_date": a.contract_end.isoformat(),
                "days_to_renewal": days,
                "score": score,
                "band": band,
                "p_loss": round(p_loss, 3),
                "expected_loss": round(a.arr * p_loss, 2),
                "drivers": drivers,
                "actions": recommend(a, comps, days),
            }
        )
    rows.sort(key=lambda r: (-r["expected_loss"], r["days_to_renewal"], r["account_id"]))
    return rows[:top] if top else rows


def early_churn_signal(accounts: Sequence[Account], gap_threshold: float = 0.05) -> dict:
    """Compare churn among first-renewal accounts, split by time to first value.

    Flags early onboarding as a lever when first-renewal churn exceeds later-renewal
    churn by gap_threshold and slow-onboarding first renewals are worse still.
    """
    def churn(grp):
        return (sum(1 for a in grp if not a.renewed) / len(grp)) if grp else float("nan")

    first = [a for a in accounts if a.renewal_number == 1]
    later = [a for a in accounts if a.renewal_number > 1]
    slow = [a for a in first if a.days_to_first_value > SLOW_ONBOARDING_DAYS]
    fast = [a for a in first if a.days_to_first_value <= SLOW_ONBOARDING_DAYS]
    res = {
        "first_n": len(first), "first_churn": churn(first),
        "later_n": len(later), "later_churn": churn(later),
        "slow_n": len(slow), "slow_churn": churn(slow),
        "fast_n": len(fast), "fast_churn": churn(fast),
    }
    res["flag"] = bool(
        first and later and slow and fast
        and res["first_churn"] - res["later_churn"] >= gap_threshold
        and res["slow_churn"] > res["fast_churn"]
    )
    return res


def format_risk(rows: Sequence[dict], n: int = 10) -> str:
    out = []
    for r in rows[:n]:
        out.append(
            f"{r['account_id']} {r['segment']:<10s} ARR {r['arr']:>9,.0f}  renews {r['renewal_date']} "
            f"({r['days_to_renewal']}d)  score {r['score']:.1f} {r['band']}  p_loss {r['p_loss']:.2f}  "
            f"expected loss {r['expected_loss']:,.0f}"
        )
        out.append(f"    drivers: {', '.join(r['drivers']) or 'none'}")
        for act in r["actions"]:
            out.append(f"    - {act}")
    return "\n".join(out)
