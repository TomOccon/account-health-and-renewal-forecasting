# account-health-and-renewal-forecasting

A composite account health score, an 18-month probability-weighted renewal forecast, and an at-risk renewal list with recommended technical-lead actions. Everything runs on seeded synthetic data (300 fictional accounts, no real companies or people). Python standard library only; tests use pytest.

## Quickstart

```
python -m acct_health            # backtest, 18-month forecast, NRR/GDR, top at-risk list
python -m acct_health backtest   # just the tickets + escalations backtest
python -m acct_health risk --top 10
pytest -q
```

Options: `--n` (book size, default 300), `--seed` (default 42), `--top` (rows in the risk list). Same arguments give identical output.

## Method

### 1. Health score (`health.py`)

Five components, each scaled 0 to 100 (higher is healthier):

| Component | Input | Scaling |
|---|---|---|
| usage | 90-day usage vs expected, plus trend vs prior 90 days | `0.7 * usage + 0.3 * (0.5 + trend / 2)` |
| sentiment | NPS (0 to 10) and CSAT (1 to 5), whichever exist | mean of the rescaled values; 50 if no survey |
| exec | executive sponsor touches in 90 days (0 to 1) | linear |
| tickets | support tickets in 90 days | `max(0, 1 - tickets / 15)` |
| escalations | escalations in 90 days | `max(0, 1 - escalations / 3)` |

Composite = weighted mean over the components in the weight dict, renormalized. Weights are a hand-set prior; nothing is fit, so the backtest measures what the inputs add and not how well a model memorized the data.

| Variant | usage | sentiment | exec | tickets | escalations |
|---|---:|---:|---:|---:|---:|
| base | 0.40 | 0.30 | 0.30 | | |
| base + tickets | 0.35 | 0.25 | 0.25 | 0.15 | |
| full | 0.30 | 0.20 | 0.20 | 0.15 | 0.15 |

Bands: healthy >= 70, stable >= 55, watch >= 40, critical below 40.

### 2. Backtest: do tickets and escalations earn their place?

`backtest()` scores every account under each variant and compares how well the score separates renewed from churned accounts.

- **AUC** is computed from the rank-sum identity, with average ranks for ties, so no sklearn: `AUC = (R_pos - n_pos (n_pos + 1) / 2) / (n_pos * n_neg)`. It equals the probability that a random renewed account outscores a random churned one.
- **Accuracy** at the best threshold. This is in-sample, so read it as a rough guide only.
- **Interval** on the AUC difference from a seeded bootstrap that scores both variants on the same resample.

Result on the 900-account labeled history cohort (synthetic, seed 1042):

```
variant                             AUC  accuracy
base                              0.639     75.8%
base+tickets                      0.670     76.2%
full (+tickets +escalations)      0.680     76.4%
AUC change: +0.041 (95% bootstrap interval +0.024 to +0.062)
```

At 300 accounts one book is noisy; a single draw can have an interval that crosses zero. Over 20 independent 300-account books the mean AUC change is +0.048, positive in all 20. The generator is built so ticket and escalation volume carries signal the surveys miss, so this demonstrates the measurement method. It is not evidence about any real customer base. On real data, run the same function on your own labeled renewals.

### 3. Forecast (`forecast.py`)

1. **Calendar.** Bucket each contract end date into a calendar month for 18 months from the as-of date. Empty months are kept.
2. **Band rates.** From a separate labeled history cohort, per band: renewal rate `p = (k + 5 * overall) / (n + 5)` (shrunk toward the overall rate so thin bands do not overfit), plus average expansion `e` and contraction `c` among renewed accounts.
3. **Probability-weighted dollars** per account, with `ARR` and its band's `p`, `e`, `c`:
   - expected churn = `ARR * (1 - p)`
   - expected gross renewed = `ARR * p * (1 - c)`
   - expected net renewed = `ARR * p * (1 + e - c)`
4. **Retention metrics**:
   - `GDR = (start - contraction - churn) / start` (expansion excluded, bounded to 0 to 1)
   - `NRR = (start + expansion - contraction - churn) / start` (can exceed 1)
   - Expected NRR and GDR come from the forecast; realized ones come from outcomes.

Extending from two quarters to 18 months is mostly the calendar and the rates. The cost is that health scores are measured today and applied to renewals up to 18 months away, so far-out months deserve wider error bars. Rescore monthly and re-forecast; do not treat month 15 as a commitment.

Sample result (seed 42, synthetic): forecast total within 3% of realized net renewed dollars, expected NRR 77.3% vs realized 75.4%.

### 4. At-risk triage (`risk.py`)

Priority = dollars at risk x probability of loss, where dollars at risk is ARR and probability of loss is `1 - p` for the account's band. Probability is band-level, so inside a band the order is driven by ARR, then days to renewal.

Each row lists the weakest drivers (components under a floor) and maps them to a technical-lead action: escalation root-cause review with a named owner, ticket-theme audit (defect vs enablement gap), usage deep-dive and adoption workshop, executive sponsor meeting with a measured outcome review, survey detractor follow-up. Renewals inside 90 days with a weak driver get a save-play action first.

## Decisions this drives

- **Save plays.** Work the list top-down: the largest expected loss with the nearest renewal date gets a technical lead first. The 90-day rule pulls near-term renewals forward.
- **Early onboarding.** `early_churn_signal()` compares first-renewal churn to later-renewal churn and slow vs fast time to first value. When first-renewal accounts churn at least 5 points more than later ones and slow onboarding is worse, the fix is upstream: put technical effort into the first 60 days, not into save plays at month 11. Sample result: slow-onboarding first renewals churn at roughly three times the rate of fast ones.
- **Instrumentation.** If the backtest shows tickets and escalations add AUC, they stay in the score. If they do not on your data, drop them and save the pipeline work.
- **Capacity.** The monthly calendar shows which months stack renewals so technical leads can be scheduled ahead.

## Layout

```
acct_health/   synth (generator), health, forecast, risk, cli
tests/         pytest suite
```

## Limits

- Synthetic data. Effect sizes are set by the generator.
- Weights, floors, and caps are priors to tune against your history.
- Band-level probabilities do not separate accounts inside a band.
- Backtest accuracy is in-sample; AUC does not depend on a threshold.
