import pandas as pd
import pytest

from scedrep.forecast import snapshot


def ts(s):
    return pd.Timestamp(s, tz="US/Central")


def test_snapshot_uses_last_report_before_cutoff():
    # Three reports, each forecasting hours 0 and 1 of Sept 15 (plus one Sept 14 row)
    pubs = ["2026-09-13 08:00", "2026-09-14 08:55", "2026-09-14 10:00"]
    rows = []
    for k, p in enumerate(pubs):
        for h in ["2026-09-14 23:00", "2026-09-15 00:00", "2026-09-15 01:00"]:
            rows.append({"Publish Time": ts(p), "Interval Start": ts(h), "STWPF WEST": 100 * (k + 1)})
    df = pd.DataFrame(rows)

    out = snapshot(df, "2026-09-15")

    # The 08:55 report on Sept 14 is the last one before 09:00; the 10:00 one is too late
    assert (out["Publish Time"] == ts("2026-09-14 08:55")).all()
    assert out["STWPF WEST"].tolist() == [200, 200]

    # Only the target day's hours, in order
    assert out["Interval Start"].tolist() == [ts("2026-09-15 00:00"), ts("2026-09-15 01:00")]


def test_snapshot_without_a_report_before_cutoff():
    # The only report is published after the cutoff, so there is nothing legal to use
    df = pd.DataFrame({
        "Publish Time": [ts("2026-09-14 10:00")],
        "Interval Start": [ts("2026-09-15 00:00")],
        "STWPF WEST": [1.0],
    })
    with pytest.raises(ValueError):
        snapshot(df, "2026-09-15")