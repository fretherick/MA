from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal
from pathlib import Path


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)

# df = pd.read_parquet(Path("data") / "aapl.pq")
# df.loc["2025-02-14 09:25:00": "2025-02-14 17:30:00"]



def get_min_max_timestamp_per_day(df: pd.DataFrame) -> pd.DataFrame:
    """Get the min and max timestamp per day from a DataFrame with a datetime index. The index is called 'timestamp'."""
    df.index = pd.to_datetime(df.index)
    df_reset = df.reset_index()
    return df_reset.groupby(df_reset['timestamp'].dt.date).agg({'timestamp': ['min', 'max']})


def convert_utc_to_ny(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the timestamp index of a `df` from UTC to New York time."""
    df.index = pd.to_datetime(df.index)
    zi_utc = ZoneInfo('UTC')
    zi_ny = ZoneInfo('America/New_York')
    df.index = df.index.tz_localize(zi_utc).tz_convert(zi_ny)
    return df


def filter_market_hours(df: pd.DataFrame, mkt_cal: mcal.MarketCalendar) -> pd.DataFrame:
    """Filter the DataFrame to keep only data inside market hours."""
    df.index = pd.to_datetime(df.index)
    schedule = mkt_cal.schedule(start_date=df.index.min(), end_date=df.index.max())
    filtered_df = pd.DataFrame()  # empty df
    for date, row in schedule.iterrows():
        open_time = row['market_open']
        close_time = row['market_close']
        # Filter for the current day within market hours
        daily_data = df[(df.index >= open_time) & (df.index <= close_time)]
        filtered_df = pd.concat([filtered_df, daily_data])
    return filtered_df


def resample_min_candles(df: pd.DataFrame, n_minutes: int) -> pd.DataFrame:
    """Resample and aggregate candles every n minutes."""
    df.index = pd.to_datetime(df.index)
    resampled_df = df.resample(f'{n_minutes}min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    # Drop any rows with NaN values (e.g., if there are incomplete resampling periods)
    resampled_df.dropna(inplace=True)
    return resampled_df


def resample_daily_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Resample and aggregate candles per day."""
    df.index = pd.to_datetime(df.index)
    resampled_df = df.resample(f'1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return resampled_df


def exclude_days_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude days with NaN values present in any column/row of the DataFrame."""
    df.index = pd.to_datetime(df.index)
    # Group by date and filter out groups with NaN values
    filtered_df = df.groupby(df.index.date).filter(lambda x: x.notna().all().all())
    return filtered_df


def remove_tz_from_index(df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Remove timezone information from the DataFrame index."""
    df.index = df.index.tz_localize(None)
    return df


if __name__ == "__main__":
    symbol = "aapl"
    path_to_file = Path("./data") / f"{symbol}.pq"
    df = pd.read_parquet(path_to_file)
    df.index = pd.to_datetime(df.index)
    df_ny = convert_utc_to_ny(df)  # Convert UTC to NY time
    nyse_cal = mcal.get_calendar('NYSE')

    # Filter to market hours
    filtered_df = filter_market_hours(df_ny, nyse_cal)
    # min_max_df = get_min_max_timestamp_per_day(filtered_df)
    filtered_df.to_parquet(Path("data") / f"{symbol}_ny.pq")

    # Resample to 5-minute candles
    resample_candles_df = resample_min_candles(filtered_df, n_minutes=5)
    resample_candles_df.to_parquet(Path("data") / f"{symbol}_5min.pq")

    # Resample to daily candles
    resample_candles_df = resample_daily_candles(filtered_df)
    resample_candles_df.to_parquet(Path("data") / f"{symbol}_daily.pq")

