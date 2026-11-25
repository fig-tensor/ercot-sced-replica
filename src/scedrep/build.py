import pandas as pd

DT = 5 / 60  # SCED interval length in hours


def build_interval(gen, ts, esr=None):
    """Model inputs for one SCED run: dispatchable units and the load they must meet.

    gen: 60-day SCED generator table. esr: optional 60-day SCED battery (ESR) table; if given,
    online batteries are dispatchable too, from LDL (charging, negative MW) to HDL (discharging).
    """
    rows = gen[gen["SCED Timestamp"] == ts]
    if esr is not None:
        b = esr[esr["SCED Timestamp"] == ts].copy()
        # ERCOT's LDL/HDL ignore state of charge, so cap them by stored energy:
        # MW limit = MWh of room / interval length in hours.
        b["LDL"] = b["LDL"].clip(lower=-(b["Max SOC"] - b["SOC"]) / DT)   # can't charge past Max SOC
        b["HDL"] = b["HDL"].clip(upper=(b["SOC"] - b["Min SOC"]) / DT)    # can't discharge below Min SOC
        rows = pd.concat([rows, b])
    # keep only online units in dispatchable set
    rows = rows[rows["Telemetered Resource Status"].astype(str).str.startswith("ON")]
    # keep only dispatchable rows: HDL > LDL only if its range is meaningful
    rows = rows[rows["HDL"] - rows["LDL"] > 0.1]
    # Limits the AS constraints need: HSL/LSL (capacity rows), ramp rates in MW/min (ramp-minute rows) and,
    # for batteries, state of charge. A column absent from the table (generators have no SOC; small test
    # frames lack HSL) becomes NaN.
    extra = {"hsl": "HSL", "lsl": "LSL", "rru": "Ramp Rate Up", "rrd": "Ramp Rate Down",
             "soc": "SOC", "soc_min": "Min SOC", "soc_max": "Max SOC"}
    for col in extra.values():
        if col not in rows.columns:
            rows = rows.assign(**{col: float("nan")})
    # units = {name: {"curve": SCED2 curve, "ldl": LDL, "hdl": HDL, "bp": base point, + extra}} for those rows
    units = {
        name: {"curve": c, "ldl": lo, "hdl": hi, "bp": bp}
        for name, c, lo, hi, bp in zip(
            rows["Resource Name"], rows["SCED2 Offer Curve"], rows["LDL"], rows["HDL"], rows["Base Point"], strict=True
        )
    }
    for key, col in extra.items():
        for name, v in zip(rows["Resource Name"], rows[col], strict=True):
            units[name][key] = float(v)
    # load = sum of Base Point over those rows
    load = rows["Base Point"].sum()
    # return units, load
    return units, load
