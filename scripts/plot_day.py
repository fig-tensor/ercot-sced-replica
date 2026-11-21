import sys

import matplotlib.pyplot as plt
import pandas as pd

date = sys.argv[1]
d = pd.read_csv(f"results/runs/{date}.csv", parse_dates=["SCED Timestamp"])
t = d["SCED Timestamp"].dt.tz_convert("US/Central")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, d["UncappedSystemLambda"], label="ERCOT published λ", color="black", linewidth=1.5)
ax.plot(t, d["model_lambda"], label="Replica λ", color="tab:blue", linewidth=1, alpha=0.8)
cong = d["congested"].astype(bool)
ax.scatter(t[cong], d.loc[cong, "model_lambda"], s=6, color="tab:red", label="Congested run", zorder=3)
ax.set_ylabel("System lambda ($/MWh)")
ax.set_title(f"ERCOT SCED system lambda vs replica, {date}")
ax.legend(loc="upper left")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"docs/lambda_{date}.png", dpi=150)