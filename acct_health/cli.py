"""Command line: python -m acct_health [backtest|forecast|risk|all]."""

from __future__ import annotations

import argparse
import sys

from .forecast import (
    band_stats, expected_retention, forecast_error, format_forecast, monthly_forecast,
    realized_monthly, realized_retention, rollup_quarters,
)
from .health import backtest, backtest_seeds, format_backtest
from .risk import at_risk_list, early_churn_signal, format_risk
from .synth import DEFAULT_AS_OF, generate_accounts

HISTORY_N = 900
HISTORY_SEED_OFFSET = 1000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="acct_health")
    ap.add_argument("command", nargs="?", default="all", choices=["backtest", "forecast", "risk", "all"])
    ap.add_argument("--n", type=int, default=300, help="accounts in the current book")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top", type=int, default=5, help="rows in the at-risk list")
    args = ap.parse_args(argv)

    book = generate_accounts(args.n, args.seed)
    history = generate_accounts(HISTORY_N, args.seed + HISTORY_SEED_OFFSET)  # separate labeled cohort
    stats = band_stats(history)
    as_of = DEFAULT_AS_OF

    if args.command in ("backtest", "all"):
        print("Labeled history cohort:")
        print(format_backtest(backtest(history)))
        print()
        bs = backtest_seeds(range(args.seed, args.seed + 20), n=args.n)
        print(
            f"Repeated on {bs['seeds']} independent books of {bs['n']} accounts: "
            f"mean AUC change {bs['mean_auc_delta']:+.3f}, positive in {bs['share_positive']:.0%} of books"
        )
        print()
    if args.command in ("forecast", "all"):
        rows = monthly_forecast(book, stats, as_of)
        print("18-month renewal forecast (probability weighted)")
        print(format_forecast(rows))
        print()
        for q in rollup_quarters(rows):
            print(f"{q['quarter']} {q['months']}: ARR up {q['arr_up']:,.0f}, expected net {q['expected_net']:,.0f}")
        exp = expected_retention(rows)
        act = realized_retention(book)
        err = forecast_error(rows, realized_monthly(book, as_of))
        print()
        print(f"Expected NRR {exp['nrr']:.1%}, GDR {exp['gdr']:.1%}")
        print(f"Realized  NRR {act['nrr']:.1%}, GDR {act['gdr']:.1%} (synthetic outcomes)")
        print(f"Forecast total error {err['total_error_pct']:+.1%}, monthly MAE {err['monthly_mae']:,.0f}")
        print()
    if args.command in ("risk", "all"):
        print(f"Top {args.top} at-risk renewals (expected loss = ARR x p_loss)")
        print(format_risk(at_risk_list(book, stats, as_of), args.top))
        sig = early_churn_signal(book)
        print()
        print(
            f"Early churn: first-renewal {sig['first_churn']:.1%} vs later {sig['later_churn']:.1%}; "
            f"slow onboarding {sig['slow_churn']:.1%} vs fast {sig['fast_churn']:.1%}; flag={sig['flag']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
