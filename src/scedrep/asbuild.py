"""Ancillary-service (AS) inputs for one SCED run: offer blocks, fixed supply, demand curves.

Data in, model inputs out; no optimization here. Products k: REGUP, REGDN, RRS, ECRS, NSPIN. RRS is offered
and awarded by sub-type (PFR, UFR, FFR); blocks keep each sub-type's own price.
"""

from itertools import pairwise

import pandas as pd

PRODUCTS = ["REGUP", "REGDN", "RRS", "ECRS", "NSPIN"]
UP = ["REGUP", "RRS", "ECRS", "NSPIN"]  # products offered in ONRES rows; REGDN has its own rows
SUB = ["REGUP", "REGDN", "RRSPFR", "RRSUFR", "RRSFFR", "NSPIN", "ECRS"]  # price columns as offered
RRS_SUB = ["RRSPFR", "RRSUFR", "RRSFFR"]
TZ = "US/Central"

# RRS from Primary Frequency Response: system minimum (MW) by year, from ERCOT's AS methodology for that year
# (2026: market notice M-A121925-01), and the most one resource may carry (MW). In ERCOT's own awards,
# Sept 15-21 2026, total PFR sits at exactly 1,377 MW in nearly every run and the largest award at 157.0 MW.
PFR_MIN = {2026: 1377.0}
PFR_UNIT_MAX = 157.0

# ERCOT renamed the price-column suffixes in late March 2026; map old names to new ones.
OLD_TO_NEW = {"_URS": "_REGUP", "_DRS": "_REGDN", "_NS": "_NSPIN",
              "_RRSPF": "_RRSPFR", "_RRSUF": "_RRSUFR", "_RRSFF": "_RRSFFR"}

# Award columns per product: gridstatus's processed names first, then ERCOT's raw names.
AWARD_COLS = {
    "REGUP": ["AS Awards RegUp", "AS Awards REGUP"],
    "REGDN": ["AS Awards RegDown", "AS Awards REGDN"],
    "ECRS": ["AS Awards ECRS"],
    "NSPIN": ["AS Awards NonSpin", "AS Awards NSPIN"],
    "RRS": ["AS Awards RRSPFR", "AS Awards RRSUFR", "AS Awards RRSFFR"],  # summed
}


# ---------- 1. offer blocks ----------

def _rename_suffixes(df):
    """Old price suffixes -> new (_URS -> _REGUP, ...); new-style files pass through unchanged."""
    def new_name(col):
        for old, new in OLD_TO_NEW.items():
            if col.startswith("PRICE") and col.endswith(old):
                return col[: -len(old)] + new
        return col
    return df.rename(columns=new_name)


def as_offer_blocks(offers):
    """Raw AS offer table -> one row per (run, unit, offer type, block b, product k) that is offered.

    Each block b has one MW quantity Q_b shared by every product it prices. Kept: blocks with
    Q_b > 0 and a price for k; up-products from ONRES rows, REGDN from REGDN rows. RRS sub-types each keep
    their own row and price (product RRS, sub RRSPFR / RRSUFR / RRSFFR); for other products sub = product.
    Columns: SCED Timestamp, Resource Name, Offer Type, block, mw, product, sub, price.
    """
    if "Offer Type" not in offers.columns:
        raise ValueError("no 'Offer Type' column: files before operating day 2026-06-25 are not supported")
    df = _rename_suffixes(offers)
    price_cols = [c for c in df.columns if c.startswith("PRICE")]
    # "Not offered" must be coded as missing. If no price is missing but zeros exist, a $0 offer
    # can't be told apart from no offer, so stop rather than guess.
    p = df[price_cols]
    if not p.isna().any().any() and (p == 0).any().any():
        raise ValueError("'not offered' appears coded as 0, not missing: cannot read prices")

    frames = []
    for b in range(1, 7):
        qcol = f"QUANTITY_MW{b}"
        if qcol not in df.columns:
            continue
        for s in SUB:
            pcol = f"PRICE{b}_{s}"
            if pcol not in df.columns:
                continue
            keep = df[pcol].notna() & (df[qcol] > 0)  # priced and with MW
            sel = df.loc[keep, ["SCED Timestamp", "Resource Name", "Offer Type", qcol, pcol]]
            frames.append(sel.set_axis(["SCED Timestamp", "Resource Name", "Offer Type", "mw", "price"], axis=1)
                          .assign(block=b, sub=s))
    cols = ["SCED Timestamp", "Resource Name", "Offer Type", "block", "mw", "product", "sub", "price"]
    if not frames:
        return pd.DataFrame(columns=cols)
    x = pd.concat(frames, ignore_index=True)
    x["Resource Name"] = x["Resource Name"].astype(str)
    x["Offer Type"] = x["Offer Type"].astype(str)
    x["product"] = x["sub"].where(~x["sub"].isin(RRS_SUB), "RRS")  # sub-types belong to RRS

    # Keep products from the row type that carries them
    up = (x["Offer Type"] == "ONRES") & x["product"].isin(UP)
    dn = (x["Offer Type"] == "REGDN") & (x["product"] == "REGDN")
    x = x[up | dn]
    return x[cols].reset_index(drop=True)


# ---------- 2. awards ----------

def as_awards(*tables):
    """ERCOT AS awards per (run, resource) from any number of resource tables (gen, ESR, load).

    Columns: SCED Timestamp, Resource Name, REGUP, REGDN, RRS (PFR+UFR+FFR), ECRS, NSPIN, and the RRS
    sub-types RRSPFR, RRSUFR, RRSFFR (MW; a missing sub-type column counts as 0).
    """
    parts = []
    for t in tables:
        out = t[["SCED Timestamp"]].copy()
        out["Resource Name"] = t["Resource Name"].astype(str)
        for k, names in AWARD_COLS.items():
            present = [c for c in names if c in t.columns]
            if not present:
                raise ValueError(f"no award column for {k} (looked for {names})")
            if k == "RRS":
                out[k] = t[present].fillna(0).sum(axis=1)   # sub-types add up to RRS
            else:
                out[k] = t[present[0]].fillna(0)            # processed name if present, else raw
        for s in RRS_SUB:
            c = f"AS Awards {s}"
            out[s] = t[c].fillna(0) if c in t.columns else 0.0
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


# ---------- 3. fixed supply ----------

def fixed_supply(awards, ts, units):
    """F_k = sum of ERCOT awards at run ts over resources the model does not dispatch (not in units).

    Covers Load Resources, offline units and units with a pinned range. With F_k fixed, the AS balance
    sum_{u in units} a_uk + F_k = q_k holds exactly at ERCOT's own solution. Also F for RRSPFR (the PFR
    floor row needs it).
    """
    if "RRSPFR" not in awards.columns:
        raise ValueError("awards table has no RRSPFR column: rebuild it with as_awards "
                         "(tables made before RRS sub-types were split lack it)")
    a = awards[awards["SCED Timestamp"] == ts]
    rest = a[~a["Resource Name"].isin(set(units))]
    return {k: float(rest[k].sum()) for k in PRODUCTS + ["RRSPFR"]}


def pfr_limits(ts):
    """PFR floor and per-resource cap for run ts: {"min": MW, "unit_max": MW}. Raises for a year with no
    known floor rather than guess (the floor changes every year)."""
    year = pd.Timestamp(ts).tz_convert(TZ).year
    if year not in PFR_MIN:
        raise ValueError(f"no RRS-PFR minimum on file for {year}: add it to PFR_MIN from that year's AS methodology")
    return {"min": PFR_MIN[year], "unit_max": PFR_UNIT_MAX}


# ---------- 4. demand curves ----------

def asdc_segments(curve):
    """Demand-curve points [(q_i, P_i), ...] -> segments [(P_i, s_i, w_i), ...].

    Segment i spans w_i = q_{i+1} - q_i MW, starting at price P_i with slope s_i = (P_{i+1} - P_i) / w_i.
    Filling y MW of it is worth P_i*y + s_i*y^2/2. Zero-width steps are skipped. Prices must not rise
    with MW (s_i <= 0), so the value is concave and the clearing problem stays convex.
    """
    segs = []
    for (q0, p0), (q1, p1) in pairwise(curve):
        w = q1 - q0
        if w <= 0:
            continue  # vertical step: no MW to fill
        s = (p1 - p0) / w
        if s > 1e-9:
            raise ValueError(f"demand curve price rises between {q0} and {q1} MW: not concave")
        segs.append((float(p0), float(s), float(w)))
    return segs


def hour_key(ts):
    """SCED timestamp -> (HourEnding 'HH:00', RepeatedHourFlag 'Y'/'N') in Central time.

    'Y' only on the fall-back day, for the second (standard-time) occurrence of the repeated hour.
    """
    local = pd.Timestamp(ts).tz_convert(TZ)
    he = f"{local.hour + 1:02d}:00"
    wall = local.tz_localize(None)
    # A wall time is ambiguous if reading it as daylight time and as standard time gives two instants
    ambiguous = wall.tz_localize(TZ, ambiguous=True) != wall.tz_localize(TZ, ambiguous=False)
    # Repeated hour = the second (standard-time, dst() == 0) occurrence of an ambiguous wall time
    repeated = "Y" if ambiguous and local.dst() == pd.Timedelta(0) else "N"
    return he, repeated


def asdc_for_run(asdc, ts):
    """Segments of each product's demand curve for the hour containing run ts: {product: segments}."""
    he, rep = hour_key(ts)
    rows = asdc[(asdc["HourEnding"].astype(str) == he) & (asdc["RepeatedHourFlag"].astype(str) == rep)]
    out = {r.ASType: asdc_segments(r.curve) for r in rows.itertuples()}
    missing = set(PRODUCTS) - set(out)
    if missing:
        raise ValueError(f"no demand curve for {sorted(missing)} at {he} (repeated={rep})")
    return out


# ---------- 5. one run ----------

def build_as_interval(blocks, awards, asdc, ts, units):
    """AS inputs for one SCED run.

    blocks: output of as_offer_blocks; awards: output of as_awards; asdc: demand-curve table
    (one row per HourEnding, RepeatedHourFlag, ASType with a 'curve'); units: the names the energy model
    dispatches (keys of build_interval's units). Returns {"blocks", "fixed", "demand", "pfr"}.
    """
    names = set(units)
    b = blocks[(blocks["SCED Timestamp"] == ts) & blocks["Resource Name"].isin(names)].reset_index(drop=True)
    return {
        "blocks": b,                                  # one row per (unit, offer type, block, product)
        "fixed": fixed_supply(awards, ts, names),     # F_k
        "demand": asdc_for_run(asdc, ts),             # segments per product
        "pfr": pfr_limits(ts),                        # RRS-PFR floor and per-resource cap
    }


def block_capacity(blocks):
    """Shared MW per block: one row per (unit, offer type, block) with Q_b (helper for the model)."""
    return blocks.groupby(["Resource Name", "Offer Type", "block"], as_index=False, observed=True)["mw"].first()