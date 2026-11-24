import pandas as pd


def build_interval(gen, ts, esr=None):
    """Model inputs for one SCED run: dispatchable units and the load they must meet.

    gen: 60-day SCED generator table. esr: optional 60-day SCED battery (ESR) table; if given,
    online batteries are dispatchable too, from LDL (charging, negative MW) to HDL (discharging).
    """
    rows = gen[gen["SCED Timestamp"] == ts]
    if esr is not None:
        rows = pd.concat([rows, esr[esr["SCED Timestamp"] == ts]])
    # keep only online units in dispatchable set
    rows = rows[rows["Telemetered Resource Status"].astype(str).str.startswith("ON")]
    # keep only dispatchable rows: HDL > LDL only if its range is meaningful
    rows = rows[rows["HDL"] - rows["LDL"] > 0.1]
    # units = {name: {"curve": SCED2 curve, "ldl": LDL, "hdl": HDL}} for those rows
    units = {
        name: {"curve": c, "ldl": lo, "hdl": hi, "bp": bp}
        for name, c, lo, hi, bp in zip(
            rows["Resource Name"], rows["SCED2 Offer Curve"], rows["LDL"], rows["HDL"], rows["Base Point"], strict=True
        )
    }
    # load = sum of Base Point over those rows
    load = rows["Base Point"].sum()
    # return units, load
    return units, load
