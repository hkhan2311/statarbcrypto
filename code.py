import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
from ta_functions import calc_adx, calc_macd, calc_obv, calc_cci, calc_cmo


# Load daily data from CoinGecko API
def load_crypto_data(symbol: str, days_back: int = 180) -> pd.DataFrame:
    cg = CoinGeckoAPI()
    id_map = {"BTC": "bitcoin", "ETH": "ethereum"}
    coin_id = id_map.get(symbol.upper())

    if not coin_id:
        raise ValueError(f"Unsupported symbol: {symbol}")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    from_ts = int(start_date.timestamp())
    to_ts = int(end_date.timestamp())

    print(f"Fetching {symbol} from {start_date.date()} to {end_date.date()}...")

    try:
        market_data = cg.get_coin_market_chart_range_by_id(
            id=coin_id,
            vs_currency='usd',
            from_timestamp=from_ts,
            to_timestamp=to_ts
        )
    except Exception as e:
        raise RuntimeError(f"Error fetching data for {symbol}: {e}")

    prices = pd.DataFrame(market_data['prices'], columns=['timestamp', 'price'])
    volumes = pd.DataFrame(market_data['total_volumes'], columns=['timestamp', 'volume'])

    df = prices.copy()
    df['volume'] = volumes['volume']
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('date').resample('1D').ffill()

    df['open'] = df['price']
    df['high'] = df['price']
    df['low'] = df['price']
    df['close'] = df['price']
    df = df[['open', 'high', 'low', 'close', 'volume']]

    return df
#
# Compute technical indicators: ADX, MACD, OBV, CCI, CMO
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    ind_df = pd.DataFrame(index=df.index)
    ind_df = ind_df.join(calc_adx(df, [{'period': 14}]))
    ind_df = ind_df.join(calc_macd(df, [{'fast_period': 12, 'slow_period': 26, 'signal_period': 9}]))
    ind_df = ind_df.join(calc_obv(df, [{'smooth_period': 0}]))
    ind_df = ind_df.join(calc_cci(df, [{'period': 20}]))
    ind_df = ind_df.join(calc_cmo(df, [{'period': 14}]))
    return ind_df.dropna()


# Conduct PCA to unify indicators into one PC: "Trend Strength and Momentum Confirmation"

def create_trend_strength_component(ind_df: pd.DataFrame) -> pd.Series:
    pca = PCA(n_components=1)
    component = pca.fit_transform(ind_df)
    return pd.Series(component.flatten(), index=ind_df.index, name="Trend Strength and Momentum Confirmation")

# Long-short strategy 
def long_short_strategy(btc_df: pd.DataFrame, eth_df: pd.DataFrame) -> pd.DataFrame:
    btc_ind = compute_indicators(btc_df)
    eth_ind = compute_indicators(eth_df)

    btc_pca = create_trend_strength_component(btc_ind)
    eth_pca = create_trend_strength_component(eth_ind)

    combined = pd.DataFrame({
        'BTC': btc_pca,
        'ETH': eth_pca
    }).dropna()

    combined['Signal'] = combined['BTC'] - combined['ETH']
    combined['Position'] = np.where(combined['Signal'] < 0, 1, -1)

    btc_returns = np.log(btc_df['close']).diff()
    eth_returns = np.log(eth_df['close']).diff()

    combined['Daily Return'] = combined['Position'].shift(1) * (btc_returns - eth_returns)
    combined['Cumulative Return'] = combined['Daily Return'].cumsum()

    # Also calculate simple BTC and ETH buy & hold returns
    combined['BTC Buy&Hold'] = btc_returns.cumsum()
    combined['ETH Buy&Hold'] = eth_returns.cumsum()

    return combined

# Performance Metrics
def calculate_metrics(results: pd.DataFrame) -> dict:
    daily_returns = results['Daily Return'].dropna()
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else np.nan
    cumulative_return = results['Cumulative Return'].iloc[-1]
    cumulative_pct = (np.exp(cumulative_return) - 1) * 100

    cum_return = results['Cumulative Return'].dropna()
    running_max = cum_return.cummax()
    drawdown = cum_return - running_max
    max_dd = drawdown.min()

    return {
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Cumulative Return (%)': cumulative_pct
    }

# Plotting + Summary
def plot_results(results: pd.DataFrame):
    print("\n Indicators used:")
    print(" - ADX (trend strength)")
    print(" - MACD (momentum crossover)")
    print(" - OBV (volume-driven price pressure)")
    print(" - CCI (commodity channel index)")
    print(" - CMO (momentum oscillator)")

    print("\n Strategy Engine:")
    print(" - PCA is applied to these five indicators")
    print(" - Resulting in a single feature: 'Trend Strength and Momentum Confirmation'")
    print(" - Trade Logic: Long BTC / Short ETH if BTC component < ETH component")

    print("\n Strategy Summary Statistics:")
    print(results[['Daily Return', 'Position']].describe())

    metrics = calculate_metrics(results)
    print("\n Performance Metrics:")
    for key, value in metrics.items():
        print(f" - {key}: {value:.4f}" if isinstance(value, float) else f" - {key}: {value}")

    # Prepare cumulative returns separately for long and short signals
    long_returns = results['Daily Return'] * (results['Position'].shift(1) == 1)
    short_returns = results['Daily Return'] * (results['Position'].shift(1) == -1)

    long_cumulative = long_returns.cumsum() * 100  # Convert to percentage
    short_cumulative = short_returns.cumsum() * 100
    combined_cumulative = results['Cumulative Return'] * 100

    # Create figure with three subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Long side plot
    axs[0].plot(results.index, long_cumulative, color='green', label="Long Side Returns (%)")
    axs[0].set_title("Long Side Performance")
    axs[0].legend()
    axs[0].grid(True)

    # Short side plot
    axs[1].plot(results.index, short_cumulative, color='red', label="Short Side Returns (%)")
    axs[1].set_title("Short Side Performance")
    axs[1].legend()
    axs[1].grid(True)

    # Combined strategy plot
    axs[2].plot(results.index, combined_cumulative, color='blue', label="Long-Short Combined Returns (%)")
    axs[2].set_title("Combined Strategy Performance")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Plot Buy & Hold vs Strategy separately
    plt.figure(figsize=(14, 6))
    plt.plot(results.index, results['Cumulative Return'], label="Long-Short Strategy", color='blue')
    plt.plot(results.index, results['BTC Buy&Hold'], label="BTC Buy & Hold", color='orange')
    plt.plot(results.index, results['ETH Buy&Hold'], label="ETH Buy & Hold", color='green')
    plt.xlabel("Date")
    plt.ylabel("Cumulative Log Return")
    plt.title("Performance Comparison: Long-Short Strategy vs Buy & Hold BTC/ETH")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# Run Script
if __name__ == "__main__":
    try:
        btc_df = load_crypto_data("BTC", days_back=180)
        eth_df = load_crypto_data("ETH", days_back=180)

        result = long_short_strategy(btc_df, eth_df)
        plot_results(result)

    except Exception as e:
        print(f"\n ERROR: {e}")
