import math
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from acct_health.forecast import (
    band_stats,
    expected_retention,
    forecast_error,
    gross_dollar_retention,
    monthly_forecast,
    net_revenue_retention,
    realized_monthly,
    realized_retention,
    renewal_calendar,
    rollup_quarters,
)
from acct_health.health import (
    BAND_NAMES,
    BASE_WEIGHTS,
    FULL_WEIGHTS,
    auc,
    backtest,
    backtest_seeds,
    band_of,
    best_accuracy,
    component_scores,
    health_score,
)
from acct_health.risk import at_risk_list, early_churn_signal, recommend
from acct_health.synth import DEFAULT_AS_OF, generate_accounts
from acct_health.util import add_months, month_index

ROOT = Path(__file__).resolve().parent.parent
AS_OF = DEFAULT_AS_OF


@pytest.fixture(scope="module")
def book():
    return generate_accounts(300, 42)


@pytest.fixture(scope="module")
def history():
    return generate_accounts(900, 1042)


@pytest.fixture(scope="module")
def stats(history):
    return band_stats(history)


# synthetic data

def test_generator_reproducible():
    assert generate_accounts(50, 3) == generate_accounts(50, 3)
    assert generate_accounts(50, 3) != generate_accounts(50, 4)


def test_generator_shape(book):
    assert len(book) == 300
    assert len({a.account_id for a in book}) == 300
    horizon = add_months(AS_OF, 18)
    assert all(AS_OF <= a.contract_end < horizon for a in book)
    assert all(a.name.startswith("Customer ") for a in book)
    assert {a.segment for a in book} == {"SMB", "Mid-Market", "Enterprise"}


# health

def test_auc_known_values():
    assert auc([1, 2, 3, 4], [False, False, True, True]) == 1.0
    assert auc([4, 3, 2, 1], [False, False, True, True]) == 0.0
    assert auc([1, 1, 1, 1], [False, True, False, True]) == 0.5
    # one inversion out of four pos/neg pairs
    assert auc([1, 3, 2, 4], [False, False, True, True]) == 0.75


def test_auc_matches_bruteforce():
    import random

    rng = random.Random(1)
    s = [rng.randint(0, 5) for _ in range(60)]
    y = [rng.random() < 0.4 for _ in range(60)]
    pos = [a for a, l in zip(s, y) if l]
    neg = [a for a, l in zip(s, y) if not l]
    brute = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
    assert auc(s, y) == pytest.approx(brute)


def test_auc_needs_both_classes():
    with pytest.raises(ValueError):
        auc([1, 2], [True, True])


def test_best_accuracy():
    acc, thr = best_accuracy([1, 2, 3, 4], [False, False, True, True])
    assert acc == 1.0 and thr == 3


def test_score_bounds_and_bands(book):
    for a in book:
        s = health_score(a)
        assert 0 <= s <= 100
        assert band_of(s) in BAND_NAMES
    assert band_of(100) == "healthy" and band_of(0) == "critical"


def test_score_monotonic_in_tickets_and_escalations(book):
    a = book[0]
    calm = replace(a, tickets_90d=0, escalations_90d=0)
    hot = replace(a, tickets_90d=20, escalations_90d=4)
    assert health_score(calm) > health_score(hot)
    # base weights ignore both inputs
    assert health_score(calm, BASE_WEIGHTS) == health_score(hot, BASE_WEIGHTS)


def test_missing_survey_is_neutral(book):
    a = replace(book[0], nps=None, csat=None)
    assert component_scores(a)["sentiment"] == 50.0


def test_adding_tickets_and_escalations_improves_auc(history):
    res = backtest(history, n_boot=100)
    assert res["auc_delta"] > 0
    assert res["variants"]["full (+tickets +escalations)"]["auc"] > res["variants"]["base"]["auc"]
    lo, hi = res["auc_delta_ci95"]
    assert lo > 0 and hi >= res["auc_delta"] - 0.05
    names = list(res["variants"])
    aucs = [res["variants"][n]["auc"] for n in names]
    assert aucs == sorted(aucs)  # each added input helps


def test_backtest_across_books():
    r = backtest_seeds(range(100, 110), n=300)
    assert r["mean_auc_delta"] > 0 and r["share_positive"] >= 0.8


# forecast

def test_add_months_clamps():
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert add_months(date(2026, 11, 15), 3) == date(2027, 2, 15)
    assert month_index(date(2027, 6, 30), date(2026, 1, 1)) == 17


def test_calendar_covers_18_months(book):
    cal = renewal_calendar(book, AS_OF)
    assert len(cal) == 18
    assert cal[0]["month"] == "2026-01" and cal[-1]["month"] == "2027-06"
    assert sum(r["accounts"] for r in cal) == 300
    assert sum(r["arr_up"] for r in cal) == pytest.approx(sum(a.arr for a in book))


def test_calendar_excludes_outside_horizon(book):
    far = replace(book[0], contract_end=date(2028, 1, 1))
    past = replace(book[0], contract_end=date(2025, 12, 31))
    assert sum(r["accounts"] for r in renewal_calendar([far, past], AS_OF)) == 0


def test_band_rates_ordered(stats):
    rates = [stats[b].rate for b in BAND_NAMES]
    assert rates == sorted(rates, reverse=True)
    assert all(0 < r < 1 for r in rates)


def test_forecast_conserves_dollars(book, stats):
    rows = monthly_forecast(book, stats, AS_OF)
    assert len(rows) == 18
    for r in rows:
        # churn plus retained (before contraction) equals ARR up
        assert r["expected_churn"] <= r["arr_up"] + 1e-6
        assert r["expected_gross"] <= r["arr_up"] - r["expected_churn"] + 1e-6
    assert sum(r["arr_up"] for r in rows) == pytest.approx(sum(a.arr for a in book))


def test_forecast_close_to_realized(book, stats):
    rows = monthly_forecast(book, stats, AS_OF)
    err = forecast_error(rows, realized_monthly(book, AS_OF))
    assert abs(err["total_error_pct"]) < 0.12


def test_quarter_rollup(book, stats):
    rows = monthly_forecast(book, stats, AS_OF)
    q = rollup_quarters(rows)
    assert len(q) == 6
    assert sum(x["expected_net"] for x in q) == pytest.approx(sum(r["expected_net"] for r in rows))


def test_nrr_gdr_formulas():
    assert net_revenue_retention(100, 20, 5, 10) == pytest.approx(1.05)
    assert gross_dollar_retention(100, 5, 10) == pytest.approx(0.85)
    assert gross_dollar_retention(100, 60, 60) == 0.0
    with pytest.raises(ValueError):
        net_revenue_retention(0, 0, 0, 0)


def test_gdr_never_exceeds_nrr_or_one(book, stats):
    r = realized_retention(book)
    assert r["gdr"] <= 1.0 and r["gdr"] <= r["nrr"]
    e = expected_retention(monthly_forecast(book, stats, AS_OF))
    assert e["gdr"] <= e["nrr"] and e["gdr"] <= 1.0
    assert abs(e["nrr"] - r["nrr"]) < 0.08


# risk

def test_risk_ordered_by_expected_loss(book, stats):
    rows = at_risk_list(book, stats, AS_OF)
    losses = [r["expected_loss"] for r in rows]
    assert losses == sorted(losses, reverse=True)
    top = rows[0]
    assert top["expected_loss"] == pytest.approx(top["arr"] * top["p_loss"], rel=1e-3)
    assert all(r["actions"] for r in rows)


def test_risk_top_and_threshold(book, stats):
    assert len(at_risk_list(book, stats, AS_OF, top=7)) == 7
    hi = at_risk_list(book, stats, AS_OF, min_p_loss=0.4)
    assert all(r["p_loss"] >= 0.4 for r in hi)


def test_actions_follow_weak_drivers(book):
    a = replace(book[0], escalations_90d=5, tickets_90d=20, renewal_number=2)
    comps = component_scores(a)
    acts = recommend(a, comps, 200)
    assert any("escalations" in x for x in acts)
    assert any("ticket" in x.lower() for x in acts)


def test_save_play_when_renewal_close(book):
    a = replace(book[0], escalations_90d=5)
    acts = recommend(a, component_scores(a), 45)
    assert acts[0].startswith("Renewal in 45 days")
    assert not recommend(a, component_scores(a), 200)[0].startswith("Renewal in")


def test_early_onboarding_action(book):
    a = replace(book[0], renewal_number=1, days_to_first_value=100)
    assert any("First-renewal" in x for x in recommend(a, component_scores(a), 200))


def test_early_churn_signal_flags(book):
    sig = early_churn_signal(generate_accounts(900, 5))
    assert sig["flag"] and sig["slow_churn"] > sig["fast_churn"]
    assert math.isnan(early_churn_signal([])["first_churn"])
    assert early_churn_signal([])["flag"] is False


# cli

def test_cli_runs():
    p = subprocess.run([sys.executable, "-m", "acct_health", "all", "--top", "2"], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    for text in ("Backtest on", "18-month renewal forecast", "Expected NRR", "at-risk renewals", "Early churn"):
        assert text in p.stdout
