from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import pytz


def datetime_to_utc_timestamp_ms(dt: datetime) -> int:
    """
    Convert a timezone-aware datetime to a UTC timestamp in milliseconds.

    Args:
        dt: A timezone-aware datetime object.

    Returns:
        UTC timestamp in milliseconds.

    Raises:
        ValueError: If the datetime object is not timezone-aware.
    """
    if dt.tzinfo is None:
        raise ValueError("Datetime object must be timezone-aware")
    dt_utc = dt.astimezone(pytz.UTC)
    return int(dt_utc.timestamp() * 1000)


def json_to_dataframe(json_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of JSON objects to a pandas DataFrame.

    Args:
        json_data: List of JSON objects.

    Returns:
        A pandas DataFrame containing the data.

    Raises:
        ValueError: If input JSON data is empty or conversion results in empty DataFrame.
    """
    if not json_data:
        raise ValueError("Input JSON data is empty")
    
    df = pd.DataFrame(json_data)
    
    if df.empty:
        raise ValueError("Converted DataFrame is empty")
    
    return df


def filter_instruments(
    instruments: List[Dict[str, Any]], 
    strike: Optional[float] = None, 
    expiration_timestamp: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Filter a list of instrument dictionaries based on strike price and expiration timestamp.

    Args:
        instruments: List of instrument dictionaries.
        strike: Strike price to filter by (optional).
        expiration_timestamp: Expiration timestamp to filter by (optional).

    Returns:
        Filtered list of instrument dictionaries matching the criteria.
    """
    filtered = instruments
    if strike is not None:
        filtered = [inst for inst in filtered if inst.get('strike') == strike]
    if expiration_timestamp is not None:
        filtered = [inst for inst in filtered if inst.get('expiration_timestamp') == expiration_timestamp]
    
    return filtered
