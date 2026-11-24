# ercot-sced-replica

[![ci](https://github.com/fig-tensor/ercot-sced-replica/actions/workflows/ci.yml/badge.svg)](https://github.com/fig-tensor/ercot-sced-replica/actions/workflows/ci.yml) [![release](https://img.shields.io/github/v/release/fig-tensor/ercot-sced-replica)](https://github.com/fig-tensor/ercot-sced-replica/releases) [![license](https://img.shields.io/github/license/fig-tensor/ercot-sced-replica)](LICENSE)

An open, validated nodal model of ERCOT's real-time market, built from public data. When no transmission constraint binds, the median error against ERCOT's published price is one cent (80% of runs within $0.85). With transmission constraints estimated from public prices, the congested-run median error drops from between -$2 and -$6 to within $0.50 on ordinary days. A day-ahead model built on ERCOT's public forecasts predicts which constraints will bind in each hour (AUC 0.88; 19% lower Brier score than the recent binding rate).

![ERCOT system lambda vs replica with and without the network, 2026-09-15](docs/lambda_network_2026-09-15.png)

## Status

Clearing of every SCED run in a day from real 60-day data, offer curves solved as a convex QP (HiGHS), with transmission constraints from estimated shift factors. A scheduled daily run validates the newest disclosed day with and without the network; running record in `results/daily.csv`. A day-ahead forecasting layer predicts binding constraints from ERCOT's public forecasts. ERCOT's unplanned generation outage reports are collected daily for a future outage feature.

## Run it

Run the tests:

```bash
uv sync
uv run pytest
```

Re-clear one day (needs ERCOT Public API credentials in `.env`: `ERCOT_API_USERNAME`, `ERCOT_API_PASSWORD`, `ERCOT_PUBLIC_API_SUBSCRIPTION_KEY`):

```bash
uv run --env-file .env scedrep YYYY-MM-DD
```

Build the forecasting dataset and run the walk-forward test (save as a script and run it with `uv run --env-file .env python <script>`; downloads are cached in `data/`):

```python
from gridstatus.ercot_api.ercot_api import ErcotAPI

from scedrep.forecast import dataset, score, walk_forward

X, Y = dataset(ErcotAPI(), "2026-06-01", "2026-11-20")
R = walk_forward(X, Y, "2026-08-01", "2026-11-20")
print(score(R))
```

## Modeling choices

- Offer curves - SCED2 (mitigated) curves, linear between points, so cost is quadratic and the problem is a convex QP. Checked against ERCOT: hand-interpolating the marginal unit matched published λ to the cent.
- Dispatchable set - online units only (status ON*) with HDL - LDL > 0.1 MW. Offline units (OFF, OUT, EMR) still carry live HDL/LDL in the data; leaving them in cost $13.90 on the first interval.
- Load and fixed injections (fixed gens, ESRs) - taken at SCED base points.
- Validation target - UncappedSystemLambda. Where ERCOT's uncapped value is missing (older days), CappedSystemLambda is used instead; if the live endpoint has no data, the archive's SystemLambda (no capped/uncapped split). The two matched on every 2026-09-08 run.
- Numerical cleaning - segments under 0.1 MW dropped, slopes outside [0.001, 1000] flattened at average price. HiGHS tolerances: primal 1e-4 MW (scale-appropriate for a ~60 GW balance), dual 1e-6 $/MWh. 20 s time limit per SCED run.

## Results without the network

Error = model λ - ERCOT system lambda. 2026-08-10 to 2026-09-19 (40 days with per-run results, 11,629 SCED runs, 98.4% solved):

- Uncongested runs (3,350) - median error -$0.01/MWh; middle 80% of runs within -$0.83 to +$0.06. The replica reproduces SCED when no transmission constraint binds.
- Congested runs (8,090, 71% of all runs) - median -$4.23; middle 80% within -$32.85 to -$0.06. Copper plate misses congestion; the model runs low (without network constraints; see Congestion).
- Worst uncongested day - 2026-08-27, median -$26.90: a whole-day miss without congestion points at scarcity / ancillary-service pricing (next: compare against ancillary-service clearing prices, MCPC).
- 2026-09-08 by hour - essentially exact 0:00-8:00; -$6 to -$11 through the solar hours (9-17); -$30 to -$53 at the evening net peak (18-21), with ERCOT at $459 at 21:15.
- Daily summaries in `results/daily.csv`; per-run detail in `results/runs/`.

## Congestion

Earlier versions left out transmission limits. When a line is full, cheap power can't reach load, so ERCOT dispatches more expensive units and prices rise. A model without the network misses this and comes out too low on congested runs.

ERCOT posts shift factors (how much each unit's output loads each line) only to the MIS Secure Area, which requires market-participant credentials, so this project estimates them from public data. In each SCED run, a node's price is LMP = λ - Σ SF·μ, summed over the binding constraints. Regressing node prices on ERCOT's published shadow prices μ over the previous 14 days recovers the shift factors. The replica then adds one constraint per binding line, anchored at ERCOT's flow, with ERCOT's shadow-price cap as the penalty for overloading it.

Median λ error on congested runs, without and with the network:

| Date | Without | With |
|---|---|---|
| 2026-09-15 | -$6.01 | -$0.02 |
| 2026-09-16 | -$19.62 | -$18.59 |
| 2026-09-17 | -$10.32 | -$4.41 |
| 2026-09-18 | -$2.18 | -$0.49 |
| 2026-09-19 | -$2.70 | -$0.15 |
| 2026-09-20 | -$4.36 | -$0.09 |
| 2026-09-21 | -$2.58 | -$0.23 |

On ordinary days the bias is gone. 2026-09-16 is the exception: most of its congestion was on lines that hadn't bound in the prior two weeks, so there were no estimates for them. The regression test predicted this before the replica ran. Out of sample, it explained 97% of the congestion pattern on the median day, and only 16% on 2026-09-16.

What this does and doesn't show:
- The replica is told which constraints bind and their flows, the same way it's told the load. It reproduces ERCOT's clearing given the network state ERCOT saw. It does not predict congestion.
- A constraint that didn't bind in the 14-day training window has no estimate. On an average day, about a quarter of the shadow price falls on such constraints.
- The same line under a different contingency is a separate constraint with its own shift factors, so coverage grows slowly with more history.
- Resources are matched to nodes using ERCOT's latest public mapping file (NP4-160), not the one in force on the day being modeled. 99.9% of resources match.
- The QP uses every estimated shift factor. For the few runs where the solver stalls, it retries with shift factors below 0.01 set to zero. Cutting them everywhere roughly doubled the error: small shift factors carry real signal in aggregate.

## Day-ahead congestion forecast

Predicts, the day before, which transmission constraints will bind in each hour. Inputs are ERCOT's public day-ahead forecasts (wind and solar by region, load by weather zone, generation outages by zone), using only what was published by 9 AM the previous day.

Walk-forward test over 16 weeks (2026-08-01 to 2026-11-20), retrained weekly on the prior 8 weeks, one model per frequently binding constraint:

| Method | Brier | Skill vs recent rate | AUC |
|---|---|---|---|
| Recent 14-day binding rate | 0.0648 | 0 | 0.54 |
| Same hour two days earlier | 0.0922 | -0.42 | 0.60 |
| Logistic regression (L2, C = 0.01) | 0.0524 | +0.19 | 0.88 |

Skill is the reduction in Brier score (mean squared error of the predicted probabilities) against the recent binding rate. AUC is the chance a binding hour is ranked above a non-binding one.

The drivers it finds make physical sense. A sparse (lasso) version of the model picks Coastal wind as the main driver for two Rio Grande Valley constraints (NELRIO and WESLCO_HIDLBRG_1), and solar for daytime constraints.

The model's learned wind and solar effects agree in sign with the network's estimated shift factors for 88% of constraints with a clear sensitivity (rank correlation about +0.55, across 74 constraints), although the model never sees the shift factors.

Adding predicted line loading from the estimated shift factors (wind and solar forecasts weighted by each constraint's shift factors) does not help on frequent constraints: it is a linear combination of inputs the model already has, and the model learns that direction from the binding history.

On rare constraints (binding in 1-5% of hours), a 3-feature model using the predicted loading matched or beat the full 37-feature model (+0.02 skill; 95% bootstrap range -0.001 to +0.045). Not established yet.

Congestion spreads (LMP - λ), day ahead: the dollar size of real-time spreads is not forecastable from day-ahead public information beyond a few percent; even ERCOT's day-ahead market does no better than predicting zero on absolute error at the hubs. Whether a spread event happens is forecastable. For an hourly spread below -$10 (a discount) or above +$10 (a premium), walk-forward over 10 weeks (2026-09-12 to 2026-11-20):

| Locations | Event | Skill vs recent rate | Model AUC | Day-ahead market AUC |
|---|---|---|---|---|
| Hubs and load zones | Discount | +0.10 | 0.72 | 0.62 |
| Hubs and load zones | Premium | +0.15 | 0.74 | 0.79 |
| Resource nodes | Discount | +0.14 | 0.73 | 0.69 |
| Resource nodes | Premium | +0.15 | 0.71 | 0.80 |

The 9 AM model ranks discount hours better than ERCOT's day-ahead prices, which clear hours later; the day-ahead market is better on premiums. The day-ahead price is scored only as a ranking (it is not a probability). The hub comparison rests on 15 locations and has not been tested for significance.

Limits:
- The penalty C was chosen by comparing values on the test weeks; skill is flat for C between 0.003 and 0.03, so the effect is small.
- Transmission outages are not among the inputs: ERCOT's transmission outage report is not public. One frequent constraint (6437__F) is not predicted better than its recent rate, and outages are the likely reason. ERCOT's unit-level unplanned generation outage reports are public but kept online for only 31 days, so they are now saved daily to `results/outages/` to build history for an outage feature.
- Constraints change with the season; summer's frequent constraints mostly stopped binding in the fall, which is why the model is retrained weekly.

## Known gaps

- Energy only. Post-RTC+B SCED co-optimizes AS; holding SCED's AS awards fixed moved λ by $0.01 on the first interval.
- Transmission - covers only constraints that bound in the prior 14 days, and is given ERCOT's binding set and flows (see Congestion).
- No unit commitment - offline and quick-start units are never started.
- Runs that hit the 20 s limit are recorded as failed (ok = False), not solved.
- Daylight-saving: the fall-back repeated hour is matched using ERCOT's RepeatedHourFlag; spring-forward unverified until the daily run reaches mid-March.

## Notes

Degenerate λ at supply-curve steps - when load lands exactly on a step, any λ between the adjacent offers clears. The original LP returned the upper end (50 of [20, 50]); the segment QP returns the lower (20). Solver/formulation artifact, not a market rule.

## Daily run

A scheduled GitHub workflow re-clears the newest disclosed day (about 61 days back), with and without the network, and appends a summary row to `results/daily.csv`. It also saves ERCOT's newest unplanned generation outage report (published that morning) to `results/outages/`.

## Teaching

The repo is also intended as a worked example for teaching optimization and applied ML on real market data. Built so far - LP and QP, from a two-unit toy dispatch with hand-checkable duals, through offer-curve segments as a convex QP, to clearing real ERCOT intervals; then least-squares recovery of shift factors from public prices, with out-of-sample tests and a tuned rolling window; then a day-ahead classifier for binding constraints, with leakage-safe features, a walk-forward test against honest baselines, a regularization sweep, a lasso for interpretation and a paired bootstrap. The known-answer tests are the worked examples. Planned - supervised learning on the validation results (regressing λ error on congestion, load, renewables and hour, with a date-based train/test split).

## License

MIT. See [LICENSE](LICENSE).
