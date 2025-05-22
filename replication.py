from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


def replicate_binary_option(
    vanilla_option_data: pd.DataFrame,
    option_type: str,  # 'call' or 'put' for the binary option
    strike: float, 
    expiry_timestamp: int, 
    replication_precision: float = 500
) -> Dict[str, Any]:
    """
    Replicate a binary option payoff using two vanilla options.

    Args:
        vanilla_option_data: DataFrame containing option data including 'option_type', 'strike', 'expiration_timestamp', 'instrument_name'.
        option_type: 'call' or 'put' (binary option type).
        strike: Target strike price for the binary option.
        expiry_timestamp: Expiration timestamp to filter options.
        replication_precision: Minimum difference between strikes used in replication.

    Returns:
        Dictionary with details of the two vanilla options and their weights for replication.

    Raises:
        ValueError: If no suitable strikes are found or option_type is invalid.
    """
    df_expiry = vanilla_option_data[vanilla_option_data['expiration_timestamp'] == expiry_timestamp]
    if option_type == 'call':
        df_calls = df_expiry[df_expiry['option_type'] == 'call']
        strikes = np.sort(df_calls['strike'].unique())
        
        strike1_idx = np.searchsorted(strikes, strike, side='left')
        strike2_idx = np.searchsorted(strikes, strike + replication_precision, side='left')

        if strike1_idx >= len(strikes) or strike2_idx >= len(strikes):
            raise ValueError("No suitable strikes found for replication.")

        K1 = strikes[strike1_idx]
        K2 = strikes[strike2_idx]

        opt1 = df_calls[df_calls['strike'] == K1].iloc[0]
        opt2 = df_calls[df_calls['strike'] == K2].iloc[0]

        weight1 = 1 / (K2 - K1)
        weight2 = -1 / (K2 - K1)

        return {
            'option_1': {'instrument': opt1['instrument_name'], 'weight': weight1, 'cost': opt1['last_price']},
            'option_2': {'instrument': opt2['instrument_name'], 'weight': weight2, 'cost': opt2['last_price']},
            'note': f"Binary call approx by (C({K1}) - C({K2})) / ({K2} - {K1})",
            "total_cost": (weight1 * opt1['last_price'] + weight2 * opt2['last_price']),
        }

    elif option_type == 'put':
        df_puts = df_expiry[df_expiry['option_type'] == 'put']
        strikes = np.sort(df_puts['strike'].unique())

        strike2_idx = np.searchsorted(strikes, strike, side='right') - 1
        strike1_idx = np.searchsorted(strikes, strike - replication_precision, side='right') - 1

        if strike1_idx < 0 or strike2_idx < 0:
            raise ValueError("No suitable strikes found for replication.")

        K1 = strikes[strike1_idx]
        K2 = strikes[strike2_idx]

        opt1 = df_puts[df_puts['strike'] == K1].iloc[0]
        opt2 = df_puts[df_puts['strike'] == K2].iloc[0]

        weight1 = 1 / (K2 - K1)
        weight2 = -1 / (K2 - K1)

        return {
            'option_1': {'instrument': opt1['instrument_name'], 'weight': weight1, 'cost': opt1['last_price']},
            'option_2': {'instrument': opt2['instrument_name'], 'weight': weight2, 'cost': opt2['last_price']},
            'note': f"Binary put approx by (P({K1}) - P({K2})) / ({K2} - {K1})",
            "total_cost": weight1 * opt1['last_price'] + weight2 * opt2['last_price']
        }

    else:
        raise ValueError("Option type must be 'call' or 'put'.")


def plot_binary_replication_payoff(
    strike: float,
    option_type: str,
    replication: Dict[str, Any],
    price_range: Optional[np.ndarray] = None,
) -> None:
    """
    Plot the ideal binary option payoff vs. the replicated payoff using two vanilla options.

    Args:
        strike: Strike price of the binary option.
        option_type: 'call' or 'put'.
        replication: Dictionary with replication info from replicate_binary_option().
        price_range: Array of underlying prices to plot over. If None, generates automatically.
    """
    if price_range is None:
        price_range = np.linspace(strike * 0.5, strike * 1.5, 500)

    # Parse strikes from replication note using regex
    note = replication.get('note', '')
    match = re.findall(r"\((\d+\.?\d*)\)", note)
    if len(match) >= 2:
        K1, K2 = map(float, match[:2])
    else:
        # Fallback: estimate strikes using weights
        w1 = replication['option_1']['weight']
        diff = 1 / w1
        K1 = strike
        K2 = strike + diff

    def call_payoff(S, K):
        return np.maximum(S - K, 0)

    def put_payoff(S, K):
        return np.maximum(K - S, 0)

    if option_type == 'call':
        payoff_opt1 = call_payoff(price_range, K1)
        payoff_opt2 = call_payoff(price_range, K2)
    elif option_type == 'put':
        payoff_opt1 = put_payoff(price_range, K1)
        payoff_opt2 = put_payoff(price_range, K2)
    else:
        raise ValueError("Option type must be 'call' or 'put'.")

    w1 = replication['option_1']['weight']
    w2 = replication['option_2']['weight']

    replicated_payoff = w1 * payoff_opt1 + w2 * payoff_opt2

    if option_type == 'call':
        ideal_payoff = (price_range >= strike).astype(float)
    else:
        ideal_payoff = (price_range <= strike).astype(float)

    plt.figure(figsize=(10, 6))
    plt.plot(price_range, ideal_payoff, label='Ideal Binary Payoff', linestyle='--', color='black')
    plt.plot(price_range, replicated_payoff, label='Replicated Payoff', color='blue')
    plt.axvline(strike, color='red', linestyle=':', label=f'Strike {strike}')
    plt.title(f"Binary Option Payoff vs Replication ({option_type.capitalize()})")
    plt.xlabel("Underlying Price at Expiry")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.show()
