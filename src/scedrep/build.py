def build_interval(gen, ts):
    # keep only rows for this SCED timestamp
    rows = gen[gen["SCED Timestamp"] == ts]
    # keep only online units in dispatchable set
    rows = rows[rows["Telemetered Resource Status"].astype(str).str.startswith("ON")]
    # keep only dispatchable rows: HDL > LDL only if its range is meaningful
    rows = rows[rows["HDL"] - rows["LDL"] > 0.1]
    # units = {name: {"curve": SCED2 curve, "ldl": LDL, "hdl": HDL}} for those rows
    units = {
        name: {"curve": c, "ldl": lo, "hdl": hi}
        for name, c, lo, hi in zip(
            rows["Resource Name"], rows["SCED2 Offer Curve"], rows["LDL"], rows["HDL"]
        )
    }
    # load = sum of Base Point over those rows
    load = rows["Base Point"].sum()
    # return units, load
    return units, load
