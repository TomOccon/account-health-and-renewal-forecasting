"""Composite account health score and a backtest of the tickets + escalations inputs.

Each component is scaled 0..100 (higher is healthier). The composite is a
weighted mean over the components present in the weight dict, renormalized, so
dropping a component is just omitting its key. No model is fit: the weights are
a hand-set prior, which keeps the backtest honest about what the inputs add.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from .synth import Account

BASE_WEIGHTS: Dict[str, float] = {"usage": 0.40, "sentiment": 0.30, "exec": 0.30}
PLUS_TICKETS_WEIGHTS: Dict[str, float] = {"usage": 0.35, "sentiment": 0.25, "exec": 0.25, "tickets": 0.15}
FULL_WEIGHTS: Dict[str, float] = {
    "usage": 0.30, "sentiment": 0.20, "exec": 0.20, "tickets": 0.15, "escalations": 0.15,
}
VARIANTS: Dict[str, Dict[str, float]] = {
    "base": BASE_WEIGHTS,
    "base+tickets": PLUS_TICKETS_WEIGHTS,
    "full (+tickets +escalations)": FULL_WEIGHTS,
}

TICKET_CAP = 15  # tickets in 90 days that score 0
ESCALATION_CAP = 3  # escalations in 90 days that score 0

# (name, minimum score)
BANDS: Tuple[Tuple[str, float], ...] = (("healthy", 70.0), ("stable", 55.0), ("watch", 40.0), ("critical", 0.0))
BAND_NAMES = tuple(b[0] for b in BANDS)


def component_scores(a: Account) -> Dict[str, float]:
    usage = 0.7 * a.usage_index + 0.3 * (0.5 + a.usage_trend / 2)
    parts = []
    if a.nps is not None:
        parts.append(a.nps / 10.0)
    if a.csat is not None:
        parts.append((a.csat - 1) / 4.0)
    sentiment = sum(parts) / len(parts) if parts else 0.5  # neutral when no survey
    return {
        "usage": 100 * usage,
        "sentiment": 100 * sentiment,
        "exec": 100 * a.exec_engagement,
        "tickets": 100 * max(0.0, 1 - a.tickets_90d / TICKET_CAP),
        "escalations": 100 * max(0.0, 1 - a.escalations_90d / ESCALATION_CAP),
    }


def health_score(a: Account, weights: Dict[str, float] = FULL_WEIGHTS) -> float:
    comps = component_scores(a)
    total = sum(weights.values())
    return round(sum(comps[k] * w for k, w in weights.items()) / total, 1)


def band_of(score: float) -> str:
    for name, floor in BANDS:
        if score >= floor:
            return name
    return BANDS[-1][0]


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve via the rank-sum identity; ties get average ranks.

    labels True = positive (renewed). Equals P(score_pos > score_neg) + 0.5 P(tie).
    """
    n = len(scores)
    pos = sum(1 for y in labels if y)
    neg = n - pos
    if pos == 0 or neg == 0:
        raise ValueError("auc needs both classes")
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    rank_sum = sum(ranks[k] for k in range(n) if labels[k])
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def best_accuracy(scores: Sequence[float], labels: Sequence[bool]) -> Tuple[float, float]:
    """Highest accuracy over thresholds (predict renewed if score >= t). In-sample."""
    n = len(scores)
    best_t, best_acc = min(scores), -1.0
    for t in sorted(set(scores)):
        acc = sum((s >= t) == y for s, y in zip(scores, labels)) / n
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_acc, best_t


def backtest(
    accounts: Sequence[Account],
    variants: Dict[str, Dict[str, float]] = VARIANTS,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Compare renewal-prediction AUC and accuracy across weight variants.

    The first variant is the baseline; the last is the one under test. The 95%
    interval on the AUC difference comes from a seeded bootstrap over accounts,
    scoring both variants on the same resample.
    """
    labels = [a.renewed for a in accounts]
    names = list(variants)
    scored = {k: [health_score(a, w) for a in accounts] for k, w in variants.items()}
    result = {"n": len(accounts), "renewal_rate": sum(labels) / len(labels), "variants": {}}
    for k in names:
        acc, thr = best_accuracy(scored[k], labels)
        result["variants"][k] = {"auc": auc(scored[k], labels), "accuracy": acc, "threshold": thr}

    base, full = names[0], names[-1]
    result["baseline"], result["candidate"] = base, full
    result["auc_delta"] = result["variants"][full]["auc"] - result["variants"][base]["auc"]
    result["accuracy_delta"] = result["variants"][full]["accuracy"] - result["variants"][base]["accuracy"]

    rng = random.Random(seed)
    n = len(accounts)
    deltas: List[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        y = [labels[i] for i in idx]
        if all(y) or not any(y):
            continue
        deltas.append(auc([scored[full][i] for i in idx], y) - auc([scored[base][i] for i in idx], y))
    deltas.sort()
    if deltas:
        result["auc_delta_ci95"] = (deltas[int(0.025 * len(deltas))], deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))])
    else:
        result["auc_delta_ci95"] = (float("nan"), float("nan"))
    return result


def backtest_seeds(seeds: Sequence[int], n: int = 300) -> dict:
    """Repeat the backtest on independent synthetic books. Shows how often the inputs help at this sample size."""
    from .synth import generate_accounts

    deltas = [backtest(generate_accounts(n, s), n_boot=0)["auc_delta"] for s in seeds]
    return {
        "n": n,
        "seeds": len(deltas),
        "mean_auc_delta": sum(deltas) / len(deltas),
        "share_positive": sum(d > 0 for d in deltas) / len(deltas),
    }


def format_backtest(res: dict) -> str:
    lines = [f"Backtest on {res['n']} accounts (renewal rate {res['renewal_rate']:.1%})", ""]
    lines.append(f"{'variant':32s} {'AUC':>6s} {'accuracy':>9s}")
    for k, v in res["variants"].items():
        lines.append(f"{k:32s} {v['auc']:6.3f} {v['accuracy']:9.1%}")
    lo, hi = res["auc_delta_ci95"]
    lines += [
        "",
        f"AUC change, {res['baseline']} -> full: {res['auc_delta']:+.3f} (95% bootstrap interval {lo:+.3f} to {hi:+.3f})",
        f"Accuracy change (best threshold, in-sample): {res['accuracy_delta']:+.1%}",
    ]
    return "\n".join(lines)
