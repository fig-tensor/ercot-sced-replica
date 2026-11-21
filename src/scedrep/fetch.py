import pandas as pd


def get_lambda(api, date):
    """ERCOT system lambda for every SCED run on date (live API, archive fallback)."""
    # Live endpoint: every SCED run from date to date + 1 day
    nxt = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    lam = api.hit_ercot_api(
        "/np6-322-cd/sced_system_lambda",
        SCEDTimestampFrom=f"{date}T00:00:00",
        SCEDTimestampTo=f"{nxt}T00:00:00",
    )

    # Fall back to archive data when the live endpoint returns nothing
    if len(lam) == 0:
        tz = "US/Central"
        lam = api.get_historical_data(
            "/np6-322-cd/sced_system_lambda",
            pd.Timestamp(date, tz=tz),
            pd.Timestamp(f"{nxt} 01:00", tz=tz),
        ).rename(
            columns={
                "SCEDTimeStamp": "SCEDTimestamp",
                "SystemLambda": "UncappedSystemLambda",
            }
        )

    # Use capped lambda where ERCOT's uncapped value is missing (older days)
    if "CappedSystemLambda" in lam.columns:
        lam["UncappedSystemLambda"] = pd.to_numeric(
            lam["UncappedSystemLambda"], errors="coerce"
        ).fillna(lam["CappedSystemLambda"])

    return lam