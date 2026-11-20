# ercot-sced-replica

A replica of ERCOT's real-time market clearing (SCED), being rebuilt from public 60-day disclosure data and validated against published prices.

Status: energy-only clearing of single SCED intervals from real 60-day data, offer curves solved as a convex QP (HiGHS). First result - 2026-09-08 00:00:20, uncongested: model λ $33.06 vs ERCOT $32.72 (+1.0%). Full-day validation is next.

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
