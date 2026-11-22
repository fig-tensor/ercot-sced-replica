"""Plot one day's lambda: ERCOT vs the replica without and with network constraints."""

import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

date = sys.argv[1]

# Per-run results: the no-network run, and the network run (daily output or the eval script's copy)
base = pd.read_csv(f"results/runs/{date}.csv")
net_path = f"results/runs_net/{date}.csv"
if not os.path.exists(net_path):
    net_path = f"data/net_{date}.csv"
net = pd.read_csv(net_path)
for df in (base, net):
    # Plot in Texas local time (timestamps carry their UTC offset)
    df["t"] = pd.to_datetime(df["SCED Timestamp"], utc=True).dt.tz_convert("US/Central")

# Colors: ERCOT in ink (the reference), models in the first two categorical slots
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
WITHOUT, WITH = "#eb6834", "#2a78d6"

fig, (top, bot) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[3, 2])
fig.patch.set_facecolor("#fcfcfb")

# Top: lambda through the day
top.plot(base["t"], base["model_lambda"], color=WITHOUT, lw=2, label="Replica, no network")
top.plot(net["t"], net["model_lambda"], color=WITH, lw=2, label="Replica, with network")
top.plot(base["t"], base["UncappedSystemLambda"], color=INK, lw=1.2, label="ERCOT")  # drawn last, on top
top.set_ylabel("System lambda ($/MWh)", color=MUTED)
top.legend(frameon=False, loc="upper left")
top.set_title(f"ERCOT system lambda vs replica, {date}", loc="left", color=INK)

# Bottom: error (model - ERCOT), congested runs only, so the network's effect is isolated
for df, color, label in ((base, WITHOUT, "no network"), (net, WITH, "with network")):
    c = df[df["congested"] & df["ok"]]
    bot.scatter(c["t"], c["err"], s=10, color=color, label=label)
bot.axhline(0, color=MUTED, lw=1)
bot.set_ylabel("Model - ERCOT ($/MWh)", color=MUTED)
bot.set_title("Error on congested runs (colors as above)", loc="left", color=MUTED, fontsize=10)

# Recessive axes and grid
for ax in (top, bot):
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color=GRID, lw=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED)
bot.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="US/Central"))
bot.set_xlabel("Hour (Central time)", color=MUTED)

os.makedirs("docs", exist_ok=True)
fig.tight_layout()
fig.savefig(f"docs/lambda_network_{date}.png", dpi=150, facecolor=fig.get_facecolor())
print(f"saved docs/lambda_network_{date}.png")