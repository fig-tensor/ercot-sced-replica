import gridstatus

d = gridstatus.Ercot().get_60_day_sced_disclosure("2026-09-08", process=True)
print({k: v.shape for k, v in d.items()})
