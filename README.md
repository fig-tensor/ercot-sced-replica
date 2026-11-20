# ercot-sced-replica

A replica of ERCOT's real-time market clearing (SCED), being rebuilt from public 60-day disclosure data and validated against published prices.

Status: energy-only clearing of every SCED run in a day from real 60-day data, offer curves solved as a convex QP (HiGHS). Validated on 2026-09-08 (290 runs) - see Results.

Run the tests: uv sync, then uv run pytest

Modeling choices:
- Offer curves - SCED2 (mitigated) curves, linear between points, so cost is quadratic and the problem is a convex QP. Checked against ERCOT: hand-interpolating the marginal unit matched published λ to the cent.
- Dispatchable set - online units only (status ON*) with HDL - LDL > 0.1 MW. Offline units (OFF, OUT, EMR) still carry live HDL/LDL in the data; leaving them in cost $13.90 on the first interval.
- Load and fixed injections (fixed gens, ESRs) - taken at SCED base points.
- Validation target - UncappedSystemLambda.
- Numerical cleaning - segments under 0.1 MW dropped, slopes outside [0.001, 1000] flattened at average price, HiGHS feasibility tolerances 1e-6.

Known gaps:
- Energy only. Post-RTC+B SCED co-optimizes AS; holding SCED's AS awards fixed moved λ by $0.01 on the first interval.
- Copper plate - no transmission constraints.
- No unit commitment - offline and quick-start units are never started.

Notes: Degenerate λ at supply-curve steps - when load lands exactly on a step, any λ between the adjacent offers clears. The original LP returned the upper end (50 of [20, 50]); the segment QP returns the lower (20). Solver/formulation artifact, not a market rule.

Results - 2026-09-08, 290 SCED runs, error = model λ - ERCOT UncappedSystemLambda:
- Uncongested runs (116) - median error -$0.04/MWh, mean -$0.77. The replica reproduces SCED when no transmission constraint binds.
- Congested runs (174) - median -$8.71, mean -$17.52. Copper plate misses congestion; the model runs low.
- By hour - essentially exact 0:00-8:00; -$6 to -$11 through the solar hours (9-17); -$30 to -$53 at the evening net peak (18-21), with ERCOT at $459 at 21:15.
- Congested bucket still mixes congestion with evening AS scarcity; separating them (MCPC axis) is next.
