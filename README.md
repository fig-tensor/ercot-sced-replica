# ercot-sced-replica

[![ci](https://github.com/fig-tensor/ercot-sced-replica/actions/workflows/ci.yml/badge.svg)](https://github.com/fig-tensor/ercot-sced-replica/actions/workflows/ci.yml) [![release](https://img.shields.io/github/v/release/fig-tensor/ercot-sced-replica)](https://github.com/fig-tensor/ercot-sced-replica/releases) [![license](https://img.shields.io/github/license/fig-tensor/ercot-sced-replica)](LICENSE)

An open, validated nodal model of ERCOT's real-time market, built from public data. When no transmission constraint binds, the median error against ERCOT's published price is one cent (80% of runs within $0.85). With transmission constraints estimated from public prices, the congested-run median error drops from -$2 to -$6 to within $0.50 on ordinary days.

![ERCOT system lambda vs replica with and without the network, 2026-09-15](docs/lambda_network_2026-09-15.png)


Status: clearing of every SCED run in a day from real 60-day data, offer curves solved as a convex QP (HiGHS), with transmission constraints from estimated shift factors. A scheduled daily run validates the newest disclosed day with and without the network; running record in results/daily.csv.

Run the tests: uv sync, then uv run pytest
Run a day: uv run --env-file .env scedrep YYYY-MM-DD (needs ERCOT Public API credentials in .env: ERCOT_API_USERNAME, ERCOT_API_PASSWORD, ERCOT_PUBLIC_API_SUBSCRIPTION_KEY)

Modeling choices:
- Offer curves - SCED2 (mitigated) curves, linear between points, so cost is quadratic and the problem is a convex QP. Checked against ERCOT: hand-interpolating the marginal unit matched published λ to the cent.
- Dispatchable set - online units only (status ON*) with HDL - LDL > 0.1 MW. Offline units (OFF, OUT, EMR) still carry live HDL/LDL in the data; leaving them in cost $13.90 on the first interval.
- Load and fixed injections (fixed gens, ESRs) - taken at SCED base points.
- Validation target - UncappedSystemLambda. Where ERCOT's uncapped value is missing (older days), CappedSystemLambda is used instead; if the live endpoint has no data, the archive's SystemLambda (no capped/uncapped split). The two matched on every 2026-09-08 run.
- Numerical cleaning - segments under 0.1 MW dropped, slopes outside [0.001, 1000] flattened at average price. HiGHS tolerances: primal 1e-4 MW (scale-appropriate for a ~60 GW balance), dual 1e-6 $/MWh. 20 s time limit per SCED run.

Results - error = model λ - ERCOT system lambda. 2026-08-10 to 2026-09-19 (40 days with per-run results, 11,629 SCED runs, 98.4% solved):
- Uncongested runs (3,350) - median error -$0.01/MWh; middle 80% of runs within -$0.83 to +$0.06. The replica reproduces SCED when no transmission constraint binds.
- Congested runs (8,090, 71% of all runs) - median -$4.23; middle 80% within -$32.85 to -$0.06. Copper plate misses congestion; the model runs low (without network constraints; see Congestion).
- Worst uncongested day - 2026-08-27, median -$26.90: a whole-day miss without congestion points at scarcity / AS pricing (next: MCPC axis).
- 2026-09-08 by hour - essentially exact 0:00-8:00; -$6 to -$11 through the solar hours (9-17); -$30 to -$53 at the evening net peak (18-21), with ERCOT at $459 at 21:15.
- Daily summaries in results/daily.csv; per-run detail in results/runs/.


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

The daily run now solves each day both ways and records both in `results/daily.csv`.

What this does and doesn't show:
- The replica is told which constraints bind and their flows, the same way it's told the load. It reproduces ERCOT's clearing given the network state ERCOT saw. It does not predict congestion.
- A constraint that didn't bind in the 14-day training window has no estimate. On an average day, about a quarter of the shadow price falls on such constraints.
- The same line under a different contingency is a separate constraint with its own shift factors, so coverage grows slowly with more history.
- Resources are matched to nodes using ERCOT's latest public mapping file (NP4-160), not the one in force on the day being modeled. 99.9% of resources match.
- The QP uses every estimated shift factor. For the few runs where the solver stalls, it retries with shift factors below 0.01 set to zero. Cutting them everywhere roughly doubled the error: small shift factors carry real signal in aggregate.

## Day-ahead congestion forecast (in progress)

Predicts, the day before, which transmission constraints will bind in each hour. Inputs are ERCOT's public day-ahead forecasts (wind and solar by region, load by weather zone, generation outages by zone), using only what was published by 9 AM the previous day.

Walk-forward test over 16 weeks (2026-08-01 to 2026-11-20), retrained weekly on the prior 8 weeks, one model per frequently binding constraint:

| Method | Brier | Skill vs recent rate | AUC |
|---|---|---|---|
| Recent 14-day binding rate | 0.0648 | 0 | 0.54 |
| Same hour two days earlier | 0.0922 | -0.42 | 0.60 |
| Logistic regression (L2, C = 0.01) | 0.0524 | +0.19 | 0.88 |

Skill is the reduction in Brier score (mean squared error of the predicted probabilities) against the recent binding rate. AUC is the chance a binding hour is ranked above a non-binding one.

The drivers it finds make physical sense. A sparse (lasso) version of the model picks Coastal wind as the main driver for two Rio Grande Valley constraints (NELRIO and WESLCO_HIDLBRG_1), and solar for daytime constraints.

Limits:
- The penalty C was chosen by comparing values on the test weeks; skill is flat for C between 0.003 and 0.03, so the effect is small.
- Transmission outages are not among the inputs. One frequent constraint (6437__F) is not predicted better than its recent rate.
- Constraints change with the season; summer's frequent constraints mostly stopped binding in the fall, which is why the model is retrained weekly.

Adding predicted line loading from the estimated shift factors (wind and solar forecasts weighted by each constraint's shift factors) does not help on frequent constraints: it is a linear combination of inputs the model already has, and the model learns that direction from the binding history.

On rare constraints (binding in 1-5% of hours), a 3-feature model using the predicted loading matched or beat the full 37-feature model (+0.02 skill; 95% bootstrap range -0.001 to +0.045). Not established yet.

Known gaps:
- Energy only. Post-RTC+B SCED co-optimizes AS; holding SCED's AS awards fixed moved λ by $0.01 on the first interval.
- Transmission - covers only constraints that bound in the prior 14 days, and is given ERCOT's binding set and flows (see Congestion).
- No unit commitment - offline and quick-start units are never started.
- Runs that hit the 20 s limit are recorded as failed (ok = False), not solved.
- Daylight-saving: the fall-back repeated hour is matched using ERCOT's RepeatedHourFlag; spring-forward unverified until the daily run reaches mid-March.

Notes: Degenerate λ at supply-curve steps - when load lands exactly on a step, any λ between the adjacent offers clears. The original LP returned the upper end (50 of [20, 50]); the segment QP returns the lower (20). Solver/formulation artifact, not a market rule.

Daily run: a scheduled GitHub workflow re-clears the newest disclosed day (about 61 days back), with and without the network, and appends a summary row to results/daily.csv.

Teaching: the repo is also intended as a worked example for teaching optimization and applied ML on real market data. Built so far - LP and QP, from a two-unit toy dispatch with hand-checkable duals, through offer-curve segments as a convex QP, to clearing real ERCOT intervals; then least-squares recovery of shift factors from public prices, with out-of-sample tests and a tuned rolling window. The known-answer tests are the worked examples. Planned - supervised learning on the validation results (regressing λ error on congestion, load, renewables and hour, with a date-based train/test split).
