# ercot-sced-replica

[![ci](https://github.com/fig-tensor/ercot-sced-replica/actions/workflows/ci.yml/badge.svg)](https://github.com/fig-tensor/ercot-sced-replica/actions/workflows/ci.yml) [![release](https://img.shields.io/github/v/release/fig-tensor/ercot-sced-replica)](https://github.com/fig-tensor/ercot-sced-replica/releases) [![license](https://img.shields.io/github/license/fig-tensor/ercot-sced-replica)](LICENSE)

An open, validated nodal model of ERCOT's real-time market, built from public data. When no transmission constraint binds, the median error against ERCOT's published price is one cent (80% of runs within $0.85). With transmission constraints estimated from public prices, the congested-run median error drops from between -$2 and -$6 to within $0.50 on ordinary days. With ancillary services co-optimized the way ERCOT's real-time market has cleared them since December 2025, the congested-run median absolute error falls further on 60 of 61 days tested (July 27 to September 25, 2026; monthly medians $0.23-0.32 down to $0.14-0.18), and ancillary-service prices are reproduced to within half a cent (median; 1-4 cents at the 90th percentile, 12 cents for Regulation Down). A day-ahead model built on ERCOT's public forecasts predicts which constraints will bind in each hour (AUC 0.88; 19% lower Brier score than the recent binding rate).

![ERCOT system lambda vs replica with and without the network, 2026-09-15](docs/lambda_network_2026-09-15.png)

## Status

Clearing of every SCED run in a day from real 60-day data, offer curves solved as a convex QP (PIQP, interior point, with a per-run fallback to other solvers), with transmission constraints from estimated shift factors. Batteries (ESRs) clear as dispatchable resources, within state-of-charge limits. Ancillary services (RegUp, RegDn, RRS, ECRS, Non-Spin) can be co-optimized with energy, as in ERCOT's market since RTC+B; this is validated on 61 days (2026-07-27 to 09-25, 17,727 SCED runs). A scheduled daily run validates the newest disclosed day with and without the network, and with ancillary services; running record in `results/daily.csv`. A day-ahead forecasting layer predicts binding constraints from ERCOT's public forecasts. ERCOT's unplanned generation outage reports are collected daily for a future outage feature.

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

Add `--as` to also clear it with ancillary services co-optimized (results in `results/runs_as/`), and give a second date to run a range. Only what is missing is solved: a model whose per-run file for that day exists is not run again (older days predate `runs_net/` and `runs_both/`, so a range fills those in):

```bash
uv run --env-file .env scedrep YYYY-MM-DD --as
uv run --env-file .env scedrep 2026-07-27 2026-09-25 --as
```

Summarise the ancillary-service results already saved in `results/` (no downloads): reserve-price errors by service, and λ errors by month against the energy-only model:

```bash
uv run python scripts/as_report.py 2026-07-27 2026-09-25
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
- Load and fixed injections (fixed gens) - taken at SCED base points. Batteries are dispatched by the model (see Batteries).
- Validation target - UncappedSystemLambda. Where ERCOT's uncapped value is missing (older days), CappedSystemLambda is used instead; if the live endpoint has no data, the archive's SystemLambda (no capped/uncapped split). The two matched on every 2026-09-08 run.
- Numerical cleaning - segments under 0.1 MW dropped, slopes outside [0.001, 1000] flattened at average price.
- Solver - PIQP (interior point), with the model written as sparse matrices. If a run fails, it is retried with stricter PIQP settings, then with HiGHS (active set; tolerances primal 1e-4 MW, dual 1e-6 $/MWh, 20 s limit); each run records which solver produced it. See Solver.

## Results without the network

Error = model λ - ERCOT system lambda. 2026-08-10 to 2026-09-19 (40 days with per-run results, 11,629 SCED runs, 98.4% solved):

- Uncongested runs (3,350) - median error -$0.01/MWh; middle 80% of runs within -$0.83 to +$0.06. The replica reproduces SCED when no transmission constraint binds.
- Congested runs (8,090, 71% of all runs) - median -$4.23; middle 80% within -$32.85 to -$0.06. Copper plate misses congestion; the model runs low (without network constraints; see Congestion).
- Worst uncongested day - 2026-08-27, median -$26.90: a whole-day miss without congestion was reserve scarcity: with ancillary services co-optimized, that day's median uncongested error is $0.08 (see Ancillary services).
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
- The QP uses every estimated shift factor. Cutting shift factors below 0.01 everywhere roughly doubled the error: small shift factors carry real signal in aggregate. With HiGHS, runs that stalled were retried with that cut; with PIQP the stalls are gone (on 2026-09-15 to 09-21, the 53 runs where HiGHS needed the cut solved with every shift factor, and PIQP was closer to ERCOT on 41 of them). The retry remains as a last resort.

## Batteries

Batteries are now a large share of evening supply in ERCOT, and they also set prices overnight when they charge. Since RTC+B, ERCOT publishes each battery as one offer curve covering both directions: charging (negative MW, down to LDL) and discharging (positive MW, up to HDL). The replica clears batteries on those curves like any other resource, so their output responds to price instead of being fixed at ERCOT's answer.

State of charge needed one fix. The published LDL/HDL ignore how full a battery is: on 2026-09-21 at 04:45, four full batteries still showed full charging ability, and the model charged them about 280 MW while ERCOT charged them about 0. The replica caps each battery by its stored energy over one 5-minute interval:

- charging limit: max(LDL, -(Max SOC - SOC) × 12)
- discharging limit: min(HDL, (SOC - Min SOC) × 12)

Checked against every battery in every run that day (87,360 battery-runs), ERCOT's base points fall outside these limits in 0.6% (charging) and 0.02% (discharging) of cases, mostly sub-1 MW trickle charging on full batteries.

Median λ error on congested runs:

| Date | Neither | Network | Batteries | Both |
|---|---|---|---|---|
| 2026-09-15 | -$6.01 | -$0.02 | -$4.35 | -$0.10 |
| 2026-09-16 | -$19.62 | -$18.59 | -$7.24 | -$7.11 |
| 2026-09-17 | -$10.32 | -$4.41 | -$4.49 | -$0.55 |
| 2026-09-18 | -$2.18 | -$0.49 | -$0.66 | -$0.24 |
| 2026-09-19 | -$2.70 | -$0.15 | -$2.05 | -$0.13 |
| 2026-09-20 | -$4.36 | -$0.09 | -$1.89 | -$0.14 |
| 2026-09-21 | -$2.58 | -$0.23 | -$1.67 | -$0.09 |

A median near zero can hide errors that cancel: on 2026-09-15 and 2026-09-20, Network sits closer to zero than Both, but its median absolute error is higher (0.29 vs 0.20, 1.15 vs 0.37).

The two fixes do different jobs. The network fixes congested prices; batteries fix the price level when nothing binds (median uncongested error on 2026-09-20: $6.29 without batteries, $0.40 with). They also interact: on 2026-09-17, the median evening error (hours 18-21) was -$146 with neither, -$12 with the network, -$119 with batteries, and -$2.21 with both. Batteries discharging behind a binding line only show up in λ when the model can see that line.

## Ancillary services

Since RTC+B (December 2025), ERCOT's real-time market clears energy and five ancillary services together: Regulation Up and Down, Responsive Reserve (RRS), Contingency Reserve (ECRS) and Non-Spinning Reserve. Each service has a price (MCPC) set by its own supply and demand, and holding capacity in reserve competes with producing energy, so reserve prices feed into λ. An energy-only model cannot produce that.

The replica co-optimizes them from ERCOT's public data:
- Offers - each resource offers up to six blocks; a block has one MW quantity shared by every service it prices (verified on 2026-09-21: ERCOT's awards fit this reading on every unit-run). RRS sub-types keep their own prices: primary frequency response (PFR) and fast frequency response (FFR); under-frequency relay (UFR) comes from Load Resources. ERCOT's proxy offers for resources that did not offer are already in the published data.
- Demand - ERCOT's hourly demand curve for each service, linear between points. ERCOT buys the full planned quantity of RegUp, RegDn, RRS and ECRS, with the price set by supply; Non-Spin clears on the sloped part of its curve.
- Fixed - resources the replica does not dispatch (Load Resources, offline units) keep ERCOT's awards.

**Rules recovered from ERCOT's own awards.** ERCOT's base points and awards must satisfy every rule ERCOT enforces, so each candidate rule was scored against them over 2026-09-15 to 09-21 (2.4 million unit-runs): a real rule is never broken and binds sometimes.
- Capacity - output plus up-reserves within the larger of the high sustained limit (HSL) and the high dispatch limit (HDL). ERCOT's solution never exceeds it; HSL alone was exceeded 19 times, all by under 1 MW, on units whose HDL sat just above HSL.
- Award caps - each award within the unit's telemetered capability for that service times its deployment time: 5 minutes for regulation, 10 for ECRS, 30 for Non-Spin. Never broken by any of ERCOT's ~650,000 awards, and binding tens of thousands of times. The same rule with the disclosed ramp rate in place of the capability was broken about 4,600 times.
- RRS from PFR - at least 1,377 MW system-wide, at most 157 MW per resource (ERCOT's 2026 AS methodology). Both bind: ERCOT's total PFR sits at exactly 1,377 MW in nearly every run, and its largest single PFR award is exactly 157.0 MW every day. The minimum is imposed as a cap on FFR (FFR at most the room left after the required PFR), which is the same thing when the RRS curve is bought in full and keeps the solver well-conditioned (see Solver).
- Batteries hold enough stored energy to sustain each award (ERCOT's qualification durations: 0.5 h for regulation and RRS, 1 h for ECRS, 4 h for Non-Spin), and regulation awards respect a 5/7 ramp factor from ERCOT's training material.

**Prices.** ERCOT's awards obey the PFR minimum, but its RRS price does not reflect it: on 2026-09-15 and 09-18, RRS prices from a solve without the PFR rows matched ERCOT's (median error $0.004, no run off by more than $1), while the same solve with them was off by more than $1 in over 200 runs. So quantities come from the full model and prices from a second solve without the PFR rows. Why ERCOT's price behaves this way is not established.

Median absolute λ error on congested runs and median evening (18-21) error, energy only vs with ancillary services (network and batteries in both), 2026-07-27 to 09-25:

| Period | Days | Congested: energy only | With AS | Evening: energy only | With AS |
|---|---|---|---|---|---|
| Jul 27-31 | 5 | $0.32 | $0.18 | -$0.50 | $0.00 |
| August | 31 | $0.23 | $0.14 | -$0.42 | $0.00 |
| Sep 1-25 | 25 | $0.24 | $0.15 | -$0.50 | -$0.02 |

The congested error is lower on 60 of the 61 days (unchanged on the other). The clearest cases are days the energy-only model missed without congestion, where the missing piece was reserve scarcity: 2026-08-27, median uncongested error $13.08 -> $0.08; 2026-08-09, $0.84 -> $0.03. Evening misses close too: -$31.70 -> $0.08 on 2026-08-23, -$9.94 -> -$0.26 on 2026-09-02. Three evenings stay off by about the same with or without reserves (2026-08-17 -$15, 09-16 -$266, 09-22 -$53): energy-side errors (09-16 is the transmission coverage gap; see Congestion). 17,688 of 17,727 runs solved.

Ancillary-service prices, 61 days:

| Service | Median abs. error | 90th percentile | Awards on a different unit than ERCOT's |
|---|---|---|---|
| RegUp | $0.001 | $0.04 | 4% |
| RegDn | $0.000 | $0.12 | 8% |
| RRS | $0.004 | $0.01 | 26% |
| ECRS | $0.003 | $0.01 | 14% |
| Non-Spin | $0.000 | $0.04 | 4% |

($/MW-h. Each month on its own gives the same picture. RRS and ECRS awards often land on other units at the right price: many offers carry identical prices.)

Limits:
- Load Resources and offline units are held at ERCOT's awards (about 27% of Non-Spin supply, 42% of RRS).
- Battery durations are ERCOT's qualification values; ERCOT's own PFR awards occasionally exceed the 0.5 h RRS duration slightly.
- The FFR maximum (450 MW) and the Load Resource share of RRS (60%) are not modeled: neither came close to binding in the week tested.
- The PFR minimum is on file for 2026 only.
- Scarcity evenings: on 2026-08-26 (about 21:00-22:50, reserve prices $150-770), the replica's reserve prices ran above ERCOT's (8-32% in the runs examined). The cause is on the energy side, not in the reserve rules. ERCOT's awards satisfy every rule the replica enforces, and with them held fixed the replica reproduces ERCOT's Non-Spin price within 0.5%. Left free, it holds 50-180 MW less Non-Spin, because carrying ERCOT's reserves would raise its λ $75-140 above ERCOT's. The same happens without the network, so the network is not the cause. Non-Spin clears on the sloped part of its curve, so that quantity gap sets the price gap. Why the replica's energy supply is steeper than ERCOT's at the margin is not established.
- In 13 of the 17,688 solved runs, PFR on offer could not fill the 1,377 MW minimum; the run then buys less RRS and reports the shortfall. 12 of them are that same evening (21:55-22:50). ERCOT was short too (1,358 MW at 22:00), and, like the replica, it did not make up the difference with FFR.

## Solver

With ancillary services, a run has about 17,000 variables (thousands of 1-MW demand-curve segments). HiGHS's QP method (active set, which changes one constraint per iteration) slowed from 0.2 s at 200 segments to 8 s at 6,600 on a synthetic test, and on real runs stopped at its time limit or with a false "non-convex" status on problems that are convex and feasible. An interior-point method takes Newton steps on the whole system at once, so its iteration count barely depends on size. Of the open-source QP solvers tested (PIQP, Clarabel, OSQP, SCS, QPALM, ProxQP, HiGHS), PIQP and Clarabel matched HiGHS's answers exactly with default settings; PIQP was fastest and is the default.

Regression, energy only, 2026-09-15 to 09-21: on the 2,019 runs both solvers solved, they never differed by more than $0.01 on the same model. PIQP solved 13 runs HiGHS could not (2,032 against 2,019), every validated median was equal or better, and runs took about 0.02 s instead of 1-2 s.

Telemetry is not always consistent: a unit's sustained limit can sit below its dispatch floor (HSL < LDL, e.g. during a derate). Using the larger of HSL and HDL in the capacity limit keeps every such run feasible.

Interior-point methods need room inside the constraints to move. A first version imposed the PFR minimum as a floor with a penalised shortfall ($10,000/MW). On 20 runs PIQP stalled: in some, the floor coincided with the end of the RRS demand curve, or sat 0.01 MW beyond it, which leaves no interior; in others the large penalty coefficient alone was enough. Stating the minimum as a cap on FFR, with FFR offers dropped when there is no room for them, solved them all, with no penalty. The HiGHS fallback fails on most runs that stall PIQP (it hits its time limit), so it rarely rescues one.

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

- Ancillary services - Load Resources and offline units keep ERCOT's awards; battery durations are not calibrated; ERCOT's own pricing run (reliability deployment price adders) is not modeled (see Ancillary services).
- Transmission - covers only constraints that bound in the prior 14 days, and is given ERCOT's binding set and flows (see Congestion).
- No unit commitment - offline and quick-start units are never started.
- A run is recorded as failed (ok = False) only if PIQP, PIQP with stricter settings and HiGHS all fail.
- 39 of 17,727 runs (0.2%) in the ancillary-service validation do not solve (PIQP stalls; the HiGHS fallback times out).
- Batteries - charging efficiency is taken as 1. Where ERCOT charged past the state-of-charge limit, it did so by about 14% (median), consistent with efficiency near 0.88, but that sample only includes the overshoots, so it isn't a fitted value.
- Daylight-saving: the fall-back repeated hour is matched using ERCOT's RepeatedHourFlag; spring-forward unverified until the daily run reaches mid-March.

## Notes

Degenerate λ at supply-curve steps - when load lands exactly on a step, any λ between the adjacent offers clears. Which one a solver returns depends on the algorithm: active-set and simplex methods stop at an end of the interval, interior-point methods near its middle (on a textbook case with valid λ in [20, 50], solvers returned 20, 30, 35, 43 and 50). It is not a market rule. On real runs it did not matter: in the solver regression, the same model never gave λ more than $0.01 apart.

## Daily run

A scheduled GitHub workflow re-clears the newest disclosed day (about 61 days back), with and without the network, and with ancillary services co-optimized (`scedrep DATE --as`), and writes one summary row per day to `results/daily.csv`, kept in date order. The AS columns (`*_as`, `med_abs_err_mcpc_*`) compare λ from the floor-free solve and each service's price with ERCOT's; per-run detail is in `results/runs_as/`. It also saves ERCOT's newest unplanned generation outage report (published that morning) to `results/outages/`. From 2026-11-24 the daily run uses PIQP; the network series changes only on runs where HiGHS used to need the reduced shift-factor retry.

## Teaching

The repo is also intended as a worked example for teaching optimization and applied ML on real market data. Built so far - LP and QP, from a two-unit toy dispatch with hand-checkable duals, through offer-curve segments as a convex QP, to clearing real ERCOT intervals; then least-squares recovery of shift factors from public prices, with out-of-sample tests and a tuned rolling window; then co-optimized energy and reserves, with a comparison of active-set and interior-point QP methods on the same model, and market rules recovered by scoring candidates against the operator's own solution; then a day-ahead classifier for binding constraints, with leakage-safe features, a walk-forward test against honest baselines, a regularization sweep, a lasso for interpretation and a paired bootstrap. The known-answer tests are the worked examples. Planned - supervised learning on the validation results (regressing λ error on congestion, load, renewables and hour, with a date-based train/test split).

## License

MIT. See [LICENSE](LICENSE).
