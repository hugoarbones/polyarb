from typing import Optional
import pandas as pd
import requests
import time
from tqdm import tqdm
from utils import filter_instruments, json_to_dataframe
import config

def flatten_instrument(instrument: dict) -> dict:
    """
    Flattens the 'stats' list in the instrument dict into top-level keys.

    Args:
        instrument: The instrument dictionary containing a 'stats' key.

    Returns:
        The instrument dictionary with 'stats' flattened into top-level keys.
    """
    stats = instrument.get('stats', [])
    for key, value in stats.items():
        if key not in instrument:
            instrument[key] = value
        
    return instrument

def get_ticker(instrument_name: str) -> dict:
    """
    Fetch the ticker for a given instrument name from the API.

    Args:
        instrument_name: The name of the instrument to fetch the ticker for.

    Returns:
        A dictionary containing the ticker information.

    Raises:
        requests.HTTPError: If the API request fails.
        ValueError: If the response format is invalid.
    """
    url = f"{config.BASE_URL}/public/ticker?instrument_name={instrument_name}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if 'result' in data and data['result']:
        return flatten_instrument(data['result'])
    else:
        raise ValueError("Invalid response format or no result found.")
    

def get_btc_option_instruments(
    strike: Optional[float] = None, 
    expiration_timestamp: Optional[int] = None
) -> pd.DataFrame:
    """
    Fetch BTC option instruments from the API and filter them by strike and expiration timestamp.

    Args:
        base_url: The base URL of the API.
        strike: Strike price to filter by (optional).
        expiration_timestamp: Expiration timestamp to filter by (optional).

    Returns:
        Filtered pandas DataFrame of BTC option instruments.

    Raises:
        requests.HTTPError: If the API request fails.
        ValueError: If no instruments are found or response format is invalid.
        TypeError: If strike or expiration_timestamp parameters are of invalid types.
    """
    if strike is not None and not isinstance(strike, (int, float)):
        raise TypeError("strike must be a number")
    if expiration_timestamp is not None and not isinstance(expiration_timestamp, int):
        raise TypeError("expiration_timestamp must be an integer")

    url = f"{config.BASE_URL}/public/get_instruments?currency=BTC&kind=option"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if 'result' in data and data['result']:
        instruments = data['result']
        filtered_instruments = filter_instruments(instruments, strike=strike, expiration_timestamp=expiration_timestamp)
        for instrument in tqdm(filtered_instruments):
            extra_data = get_ticker(instrument['instrument_name'])
            instrument.update(extra_data)
            time.sleep(config.REQUEST_DELAY)
        return json_to_dataframe(filtered_instruments)
    else:
        raise ValueError("No instruments found or invalid response format.")