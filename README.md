A replica of ERCOT’s real-time market clearing (SCED), being rebuilt from public 60-day disclosure data and validated against published prices.

Status: single-interval economic dispatch LP with known-answer tests. ERCOT data ingestion is next.

Notes:  Solver artifact behavior at the boundary - degenerate lambdas on the margin. With coal (100 MW, $20), gas (100 MW, $50) and load at exactly 100 MW, any lambda in [20, 50] clears; HiGHS returns 50, the upper end.
    
