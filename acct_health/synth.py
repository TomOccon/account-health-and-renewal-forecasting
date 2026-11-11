"""Seeded synthetic account generator. Fully fictional; no real companies or people.

Latent model (per account):
  h  engagement latent, drives usage, sentiment, executive engagement
  g  friction latent, drives tickets and escalations, only weakly visible in surveys
Renewal outcome depends on both h and g. A health score without ticket and
escalation inputs sees g only through noisy survey scores, so adding those
inputs should help. The backtest in health.py measures how much.
First-renewal accounts (renewal_number == 1) also churn more when time to first
value was slow, which is the early churn signal used in risk.py.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional

from .util import add_months

DEFAULT_AS_OF = date(2026, 1, 1)

# name, share, median ARR, lognormal sigma, logit shift
SEGMENTS = (
    ("SMB", 0.50, 18_000, 0.50, 0.0),
    ("Mid-Market", 0.35, 75_000, 0.50, 0.1),
    ("Enterprise", 0.15, 300_000, 0.55, 0.3),
)


@dataclass(frozen=True)
class Account:
    account_id: str
    name: str
    segment: str
    arr: float
    contract_end: date
    term_months: int
    renewal_number: int
    usage_index: float  # 0..1, 90-day usage vs expected
    usage_trend: float  # -1..1, change vs prior 90 days
    tickets_90d: int
    escalations_90d: int
    nps: Optional[int]  # 0..10 survey, None if no response
    csat: Optional[float]  # 1..5, None if no response
    exec_engagement: float  # 0..1, executive sponsor touches in 90 days
    days_to_first_value: int
    renewed: bool  # realized outcome of the upcoming renewal (simulated)
    renewal_delta_pct: float  # expansion (+) or contraction (-) if renewed, else 0.0


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _poisson(rng: random.Random, lam: float) -> int:
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def generate_accounts(n: int = 300, seed: int = 42, as_of: date = DEFAULT_AS_OF) -> List[Account]:
    """Generate n accounts whose contracts end within 18 months after as_of."""
    rng = random.Random(seed)
    horizon_days = (add_months(as_of, 18) - as_of).days
    shares = [s[1] for s in SEGMENTS]
    out: List[Account] = []
    for i in range(1, n + 1):
        seg, _, median, sigma, shift = rng.choices(SEGMENTS, weights=shares)[0]
        arr = round(median * math.exp(rng.gauss(0, sigma)), -2)
        h = rng.gauss(0, 1)
        g = -0.3 * h + math.sqrt(1 - 0.09) * rng.gauss(0, 1)

        usage = _clip(0.55 + 0.18 * h + rng.gauss(0, 0.12), 0.0, 1.0)
        trend = _clip(0.25 * h + rng.gauss(0, 0.3), -1.0, 1.0)
        tickets = _poisson(rng, math.exp(1.9 + 0.45 * g))
        esc = _poisson(rng, math.exp(-0.9 + 0.7 * g))
        nps = None if rng.random() < 0.25 else int(_clip(round(6.5 + 1.3 * h - 0.4 * g + rng.gauss(0, 1.5)), 0, 10))
        csat = None if rng.random() < 0.35 else round(_clip(3.6 + 0.5 * h - 0.15 * g + rng.gauss(0, 0.6), 1, 5), 1)
        execs = _clip(0.5 + 0.2 * h + rng.gauss(0, 0.22), 0.0, 1.0)
        dtfv = int(_clip(40 + 14 * g + rng.gauss(0, 15), 7, 150))

        term = rng.choices([12, 24, 36], weights=[0.6, 0.25, 0.15])[0]
        renewal_number = rng.choices([1, 2, 3], weights=[0.4, 0.35, 0.25])[0]
        end = as_of + timedelta(days=rng.randint(0, horizon_days - 1))

        logit = 1.6 + shift + 0.6 * h - 0.9 * g + rng.gauss(0, 0.5)
        if renewal_number == 1:
            logit -= 0.5 + 0.025 * (dtfv - 45)
        renewed = rng.random() < 1 / (1 + math.exp(-logit))
        delta = round(_clip(0.03 + 0.07 * h + rng.gauss(0, 0.08), -0.3, 0.5), 3) if renewed else 0.0

        out.append(
            Account(
                account_id=f"ACC-{i:04d}",
                name=f"Customer {i:04d}",
                segment=seg,
                arr=arr,
                contract_end=end,
                term_months=term,
                renewal_number=renewal_number,
                usage_index=round(usage, 3),
                usage_trend=round(trend, 3),
                tickets_90d=tickets,
                escalations_90d=esc,
                nps=nps,
                csat=csat,
                exec_engagement=round(execs, 3),
                days_to_first_value=dtfv,
                renewed=renewed,
                renewal_delta_pct=delta,
            )
        )
    return out
