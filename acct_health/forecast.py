"""Renewal forecasting out to 18 months, plus NRR and gross dollar renewal.

Steps:
1. renewal_calendar: bucket contract end dates by calendar month.
2. band_stats: from a labeled history cohort, estimate per health band the
   renewal rate (smoothed toward the overall rate) and the average expansion and
   contraction among renewed accounts.
3. monthly_forecast: probability-weighted dollars per month for the book.
4. NRR and GDR, both expected (from the forecast) and realized (from outcomes).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Sequence

from .health import BAND_NAMES, FULL_WEIGHTS, band_of, health_score
from .synth import Account
from .util import add_months, month_index, month_key

HORIZON_MONTHS = 18
PRIOR_STRENGTH = 5.0  # pseudo-accounts pulling a band's rate toward the overall rate


@dataclass(frozen=True)
class BandStats:
    n: int
    rate: float  # smoothed renewal rate
    raw_rate: float
    avg_expansion: float  # mean of max(delta, 0) over renewed accounts
    avg_contraction: float  # mean of max(-delta, 0) over renewed accounts


def renewal_calendar(accounts: Sequence[Account], as_of: date, months: int = HORIZON_MONTHS) -> List[dict]:
    """Monthly count and ARR up for renewal, for every month in the horizon (zeros included)."""
    rows = [{"month": month_key(add_months(as_of.replace(day=1), m)), "accounts": 0, "arr_up": 0.0} for m in range(months)]
    for a in accounts:
        i = month_index(a.contract_end, as_of)
        if 0 <= i < months and a.contract_end >= as_of:
            rows[i]["accounts"] += 1
            rows[i]["arr_up"] += a.arr
    return rows


def band_stats(history: Sequence[Account], weights: Dict[str, float] = FULL_WEIGHTS, prior_strength: float = PRIOR_STRENGTH) -> Dict[str, BandStats]:
    if not history:
        raise ValueError("history is empty")
    overall = sum(a.renewed for a in history) / len(history)
    groups: Dict[str, List[Account]] = {b: [] for b in BAND_NAMES}
    for a in history:
        groups[band_of(health_score(a, weights))].append(a)
    out: Dict[str, BandStats] = {}
    for b, grp in groups.items():
        n = len(grp)
        k = sum(a.renewed for a in grp)
        renewed = [a for a in grp if a.renewed]
        out[b] = BandStats(
            n=n,
            rate=(k + prior_strength * overall) / (n + prior_strength),
            raw_rate=(k / n) if n else overall,
            avg_expansion=(sum(max(a.renewal_delta_pct, 0.0) for a in renewed) / len(renewed)) if renewed else 0.0,
            avg_contraction=(sum(max(-a.renewal_delta_pct, 0.0) for a in renewed) / len(renewed)) if renewed else 0.0,
        )
    return out


def monthly_forecast(
    accounts: Sequence[Account],
    stats: Dict[str, BandStats],
    as_of: date,
    months: int = HORIZON_MONTHS,
    weights: Dict[str, float] = FULL_WEIGHTS,
) -> List[dict]:
    rows = [
        {
            "month": r["month"], "accounts": r["accounts"], "arr_up": r["arr_up"],
            "expected_gross": 0.0, "expected_net": 0.0, "expected_churn": 0.0,
        }
        for r in renewal_calendar([], as_of, months)
    ]
    for a in accounts:
        i = month_index(a.contract_end, as_of)
        if not (0 <= i < months and a.contract_end >= as_of):
            continue
        s = stats[band_of(health_score(a, weights))]
        r = rows[i]
        r["accounts"] += 1
        r["arr_up"] += a.arr
        r["expected_gross"] += a.arr * s.rate * (1 - s.avg_contraction)
        r["expected_net"] += a.arr * s.rate * (1 + s.avg_expansion - s.avg_contraction)
        r["expected_churn"] += a.arr * (1 - s.rate)
    return rows


def rollup_quarters(rows: Sequence[dict]) -> List[dict]:
    """Group consecutive 3-month blocks (Q+1, Q+2, ...) starting at the first month."""
    out = []
    for q in range(0, len(rows), 3):
        grp = rows[q:q + 3]
        agg = {"quarter": f"Q+{q // 3 + 1}", "months": f"{grp[0]['month']}..{grp[-1]['month']}"}
        for k in ("accounts", "arr_up", "expected_gross", "expected_net", "expected_churn"):
            agg[k] = sum(r[k] for r in grp)
        out.append(agg)
    return out


def gross_dollar_retention(start_arr: float, contraction: float, churn: float) -> float:
    """GDR = (start - contraction - churn) / start. Expansion excluded; bounded to [0, 1]."""
    if start_arr <= 0:
        raise ValueError("start_arr must be positive")
    return max(0.0, min(1.0, (start_arr - contraction - churn) / start_arr))


def net_revenue_retention(start_arr: float, expansion: float, contraction: float, churn: float) -> float:
    """NRR = (start + expansion - contraction - churn) / start. Can exceed 1."""
    if start_arr <= 0:
        raise ValueError("start_arr must be positive")
    return max(0.0, (start_arr + expansion - contraction - churn) / start_arr)


def realized_retention(accounts: Sequence[Account]) -> dict:
    """NRR and GDR from the realized outcomes of the accounts passed in."""
    start = sum(a.arr for a in accounts)
    churn = sum(a.arr for a in accounts if not a.renewed)
    exp = sum(a.arr * max(a.renewal_delta_pct, 0.0) for a in accounts if a.renewed)
    con = sum(a.arr * max(-a.renewal_delta_pct, 0.0) for a in accounts if a.renewed)
    return {
        "start_arr": start, "expansion": exp, "contraction": con, "churn": churn,
        "nrr": net_revenue_retention(start, exp, con, churn),
        "gdr": gross_dollar_retention(start, con, churn),
    }


def expected_retention(rows: Sequence[dict]) -> dict:
    """NRR and GDR implied by a forecast: expected dollars kept over dollars up for renewal."""
    start = sum(r["arr_up"] for r in rows)
    if start <= 0:
        return {"start_arr": 0.0, "nrr": float("nan"), "gdr": float("nan")}
    return {
        "start_arr": start,
        "nrr": sum(r["expected_net"] for r in rows) / start,
        "gdr": sum(r["expected_gross"] for r in rows) / start,
    }


def realized_monthly(accounts: Sequence[Account], as_of: date, months: int = HORIZON_MONTHS) -> List[dict]:
    """Actual renewed ARR per month (with expansion and contraction), for forecast checks."""
    rows = [{"month": r["month"], "actual_net": 0.0} for r in renewal_calendar([], as_of, months)]
    for a in accounts:
        i = month_index(a.contract_end, as_of)
        if 0 <= i < months and a.contract_end >= as_of and a.renewed:
            rows[i]["actual_net"] += a.arr * (1 + a.renewal_delta_pct)
    return rows


def forecast_error(rows: Sequence[dict], actual: Sequence[dict]) -> dict:
    """Total forecast error and mean absolute error by month, in dollars and percent of actual."""
    fc = sum(r["expected_net"] for r in rows)
    ac = sum(r["actual_net"] for r in actual)
    mae = sum(abs(r["expected_net"] - x["actual_net"]) for r, x in zip(rows, actual)) / len(rows)
    return {
        "forecast_total": fc, "actual_total": ac,
        "total_error_pct": (fc - ac) / ac if ac else float("nan"),
        "monthly_mae": mae,
    }


def format_forecast(rows: Sequence[dict]) -> str:
    lines = [f"{'month':8s} {'accts':>5s} {'ARR up':>12s} {'exp. gross':>12s} {'exp. net':>12s} {'exp. churn':>12s}"]
    for r in rows:
        lines.append(
            f"{r['month']:8s} {r['accounts']:5d} {r['arr_up']:12,.0f} {r['expected_gross']:12,.0f} "
            f"{r['expected_net']:12,.0f} {r['expected_churn']:12,.0f}"
        )
    return "\n".join(lines)
