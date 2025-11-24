import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from data_loader import result
import talib
import numpy as np
import seaborn as sns

# Set seaborn style
sns.set_style("darkgrid")  # Options: "darkgrid", "whitegrid", "dark", "white", "ticks"
sns.set_palette("husl")  # Beautiful color palette

# 1. Import tickers from data_loader result
tickers_from_cols = result.columns.get_level_values(1).unique().tolist()
print(f"From columns: {tickers_from_cols}")


def smas_indicator(plot=False):
    smas = {}

    for ticker in tickers_from_cols:
        smas[ticker] = {
            'SMA_20': result['Close'][ticker].rolling(window=20).mean(),
            'SMA_50': result['Close'][ticker].rolling(window=50).mean(),
            'SMA_200': result['Close'][ticker].rolling(window=200).mean()
        }

    # Then plot

        if plot:
            plt.plot(result['Close'][ticker], label='Close Price')
            plt.plot(smas[ticker]['SMA_20'], label='20-Day SMA', linestyle='--')
            plt.plot(smas[ticker]['SMA_50'], label='50-Day SMA', linestyle='-.')
            plt.plot(smas[ticker]['SMA_200'], label='200-Day SMA', linestyle=':')
            plt.title(f'{ticker} Stock Price with SMAs')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True)
            plt.show()

    return smas

# for ticker in tickers_from_cols:
#     # Calculate SMAs
#     sma_20 = result['Close'][ticker].rolling(window=20).mean()
#     sma_50 = result['Close'][ticker].rolling(window=50).mean()
#     sma_200 = result['Close'][ticker].rolling(window=200).mean()
#
#     # Create individual figure for this ticker
#     plt.figure(figsize=(12, 6))
#     plt.plot(result['Close'][ticker], label='Close Price', alpha=0.7)
#     plt.plot(sma_20, label='20-Day SMA', linestyle='--')
#     plt.plot(sma_50, label='50-Day SMA', linestyle='-.')
#     plt.plot(sma_200, label='200-Day SMA', linestyle=':')
#     plt.title(f'{ticker} Stock Price with SMAs')
#     plt.xlabel('Date')
#     plt.ylabel('Price ($)')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()


def ema_indicator(plot=False):
    ema = {}
    for ticker in tickers_from_cols:
        ema[ticker] = {
            'EMA_12': result['Close'][ticker].ewm(span=12, adjust=False).mean(),
            'EMA_26': result['Close'][ticker].ewm(span=26, adjust=False).mean(),
        }

        if plot:
            plt.figure(figsize=(12, 8))

            plt.plot(result['Close'][ticker], label=f'{ticker} Close')
            plt.plot(ema[ticker]['EMA_12'], label=f'{ticker} EMA 12', linestyle='-.')
            plt.plot(ema[ticker]['EMA_26'], label=f'{ticker} EMA 26', linestyle='--')

            plt.title('12 and 26-day EMAs')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.show()

    return ema


def rsi_indicator(plot=False):
    rsi = {}
    for ticker in tickers_from_cols:
        # Get the Close price from the result DataFrame
        close_prices = result['Close'][ticker].dropna()

        try:
            rsi[ticker] = talib.RSI(close_prices)
        except:
            # Fallback RSI calculation
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi[ticker] = 100 - (100 / (1 + rs))

        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(result['Close'][ticker], label=f'{ticker} Close')
            plt.plot(rsi[ticker], label=f'{ticker} RSI', linestyle='--')
            plt.axhline(70, linestyle='--', color='red', label='Overbought (70)')
            plt.axhline(30, linestyle='--', color='green', label='Oversold (30)')

            plt.title('Relative Strength Index (RSI)')
            plt.xlabel('Date')
            plt.ylabel('RSI')
            plt.legend()
            plt.grid(True)
            plt.show()

    return rsi


def mcad_indicator(plot=False):
    """
calculates the MACD of stocks
    """
    macd = {}
    for ticker in tickers_from_cols:
        close_prices = result['Close'][ticker]

        # Calculate MACD using TA-Lib
        # Returns: (MACD line, Signal line, Histogram)

        try:
            macd_line, signal_line, histogram = talib.MACD(close_prices)
            macd[ticker] = {
                'MACD': macd_line,
                'Signal': signal_line,
                'Histogram': histogram
            }
        except:
            # Fallback MACD calculation
            ema = ema_indicator()
            ema_12 = ema[ticker]['EMA_12']
            ema_26 = ema[ticker]['EMA_26']
            macd_ema = ema_12 - ema_26
            signal = macd_ema.ewm(span=9, adjust=False).mean()
            macd[ticker]['macd'] = macd
            macd[ticker]['macd_signal'] = signal
            macd[ticker]['macd_histogram'] = macd_ema - signal

        macd_line = macd[ticker]['MACD']
        signal_line_values = macd[ticker]['Signal']
        histogram = macd[ticker]['Histogram']

        # Drop NaN values before plotting and detecting crossovers
        valid_data = ~(macd_line.isna() | signal_line_values.isna())
        macd_clean = macd_line[valid_data]
        signal_clean = signal_line_values[valid_data]
        histogram_clean = histogram[valid_data]

        if plot:
            plt.plot(macd_clean, label='MACD Line', linestyle='--')
            plt.plot(signal_clean, label='Signal', linestyle='-.')
            # Plot histogram as bars
            colors = ['green' if val >= 0 else 'red' for val in histogram_clean]
            plt.bar(histogram_clean.index, histogram_clean, color=colors, alpha=0.3, label='Histogram')
            # Add zero line
            plt.axhline(color='black', linewidth=0.5, linestyle='--', alpha=0.5)
            # Detect crossovers (on clean data only)
            macd_above_signal = macd_clean > signal_clean
            # Shift and compare - this finds where the state changes
            bullish_cross = macd_above_signal & (~macd_above_signal.shift(1).fillna(False))
            bearish_cross = (~macd_above_signal) & (macd_above_signal.shift(1).fillna(False))

            # Count crossovers
            num_bullish = bullish_cross.sum()
            num_bearish = bearish_cross.sum()

            print(f"{ticker}: {num_bullish} bullish crossovers, {num_bearish} bearish crossovers")

            if num_bullish > 0:
                plt.scatter(macd_clean[bullish_cross].index, macd_clean[bullish_cross], color='green', marker='^',
                            s=100,
                            label=f'Bullish Cross {num_bullish}', zorder=5, edgecolors='darkgreen', linewidths=1.5)

            if num_bearish > 0:
                plt.scatter(macd_clean[bearish_cross].index, macd_clean[bearish_cross], color='red', marker='v', s=100,
                            label=f'Bearish Cross {num_bearish}', zorder=5, edgecolors='darkred', linewidths=1.5)

            plt.title(f'{ticker} - MACD (12, 26, 9) with crossovers')
            plt.xlabel('Date')
            plt.ylabel('MACD Value')
            plt.legend()
            plt.grid(True)
            plt.show()

    return macd


def stochastic_oscillator_talib(plot=True):
    """
    Calculate Stochastic Oscillator using TA-Lib.

    Returns:
        dict: Dictionary with ticker as key and stochastic data as values
    """
    stochastic = {}

    for ticker in tickers_from_cols:

        stochastic[ticker] = {}

        # Get price data
        high_prices = result['High'][ticker]
        low_prices = result['Low'][ticker]
        close_prices = result['Close'][ticker]

        try:

            slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices,
                                       fastk_period=14, slowk_matype=0, slowd_matype=0)
            stochastic[ticker]['%K'] = slowk
            stochastic[ticker]['%D'] = slowd
        except Exception as e:
            # Manual stochastic calculation
            period = 14
            stoch_k = pd.Series(index=result.index, dtype=float)
            for i in range(period - 1, len(result)):
                recent_high = result['High'][ticker].iloc[i - period + 1:i + 1].max()
                recent_low = result['Low'][ticker].iloc[i - period + 1:i + 1].min()
                if recent_high != recent_low:
                    stoch_k.iloc[i] = ((result['Close'][ticker].iloc[i] - recent_low) /
                                       (recent_high - recent_low)) * 100
            stochastic[ticker]['%K'] = stoch_k
            stochastic[ticker]['%D'] = stoch_k.rolling(window=3).mean()

        # Check current status
        current_k = stochastic[ticker]['%K'].dropna().iloc[-1] if not stochastic[ticker]['%K'].dropna().empty else None
        current_d = stochastic[ticker]['%D'].iloc[-1] if not stochastic[ticker]['%D'].dropna().empty else None

        if current_k is not None:
            print(f"\n{ticker} Current Stochastic:")
            print(f"  %K: {current_k:.2f}")
            print(f"  %D: {current_d:.2f}")

            if current_k >= 80:
                print(f"  OVERBOUGHT (above 80)")
            elif current_k <= 20:
                print(f"  OVERSOLD (below 20)")
            else:
                print(f"  ✓ Neutral zone")
        if plot:
            plt.figure(figsize=(12, 4 * len(tickers_from_cols)))

            slow_k = stochastic[ticker]['%K']
            slow_d = stochastic[ticker]['%D']

            # Clean data
            valid_data = ~(slow_k.isna() | slow_d.isna())
            slowk_clean = slow_k[valid_data]
            slowd_clean = slow_d[valid_data]

            # Plot
            plt.plot(slowk_clean.index, slowk_clean,
                     label='%K', linewidth=1.5, color='blue')
            plt.plot(slowd_clean.index, slowd_clean,
                     label='%D', linewidth=1.5, color='red')

            # Overbought/Oversold lines
            plt.axhline(80, color='red', linestyle='--',
                        linewidth=1, alpha=0.7)
            plt.axhline(20, color='green', linestyle='--',
                        linewidth=1, alpha=0.7)

            # Fill zones
            plt.fill_between(slowk_clean.index, 80, 100,
                             alpha=0.1, color='red')
            plt.fill_between(slowk_clean.index, 0, 20,
                             alpha=0.1, color='green')

            # Detect crossovers
            k_above_d = slowk_clean > slowd_clean
            bullish_cross = k_above_d & (~k_above_d.shift(1).fillna(False))
            bearish_cross = (~k_above_d) & (k_above_d.shift(1).fillna(False))

            # Plot crossovers
            if bullish_cross.sum() > 0:
                plt.scatter(slowk_clean[bullish_cross].index,
                            slowk_clean[bullish_cross],
                            color='green', marker='^', s=100,
                            label='Bullish Cross', zorder=5)

            if bearish_cross.sum() > 0:
                plt.scatter(slowk_clean[bearish_cross].index,
                            slowk_clean[bearish_cross],
                            color='red', marker='v', s=100,
                            label='Bearish Cross', zorder=5)

            plt.title(f'{ticker} - Stochastic Oscillator (14, 3, 3)',
                      fontsize=12, fontweight='bold')
            plt.xlabel('Date', fontsize=10)
            plt.ylabel('Stochastic Value', fontsize=10)
            plt.ylim(0, 100)
            plt.legend(loc='best', fontsize=9)
            plt.grid(True, alpha=0.3, linestyle=':')

            plt.show()

    return stochastic


def calculate_bollinger_bands(window=20, num_std_dev=2, plot=True):
    """
    Calculates Bollinger Bands for a given price series.

    Args:
        result (pd.Series): The price series (e.g., 'Close' prices).
        window (int): The lookback period for the Simple Moving Average (SMA) and Standard Deviation.
        num_std_dev (int): The number of standard deviations for the upper and lower bands.
        :param window:
        :param num_std_dev:
        :param plot:

    Returns:
        pd.DataFrame: A DataFrame with 'SMA', 'Upper Band', and 'Lower Band' columns.

    """
    bollinger_bands = {}
    for ticker in tickers_from_cols:
        sma = result['Close'][ticker].rolling(window=window).mean()
        std_dev = result['Close'][ticker].rolling(window=window).std()

        upper_band = sma + (std_dev * num_std_dev)
        lower_band = sma - (std_dev * num_std_dev)

        bollinger_bands[ticker] = {
            'SMA': sma,
            'Upper Band': upper_band,
            'Lower Band': lower_band
        }

        # Plotting
        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(result['Close'][ticker], label='Close Price')
            plt.plot(bollinger_bands[ticker]['SMA'], label='SMA', linestyle='--')
            plt.plot(bollinger_bands[ticker]['Upper Band'], label='Upper Band', color='red')
            plt.plot(bollinger_bands[ticker]['Lower Band'], label='Lower Band', color='green')
            plt.title('Bollinger Bands')
            plt.xlabel('Data Point')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.show()

    return bollinger_bands


def calculate_atr(period=14, plot=False):
    """
    Calculates the Average True Range (ATR) for multiple tickers in the 'result' DataFrame.

    Args:
        period (int): The number of periods to use for the ATR calculation.
        plot (bool): Whether to plot the ATR values.

    Returns:
        dict: A dictionary containing the ATR values for each ticker.
              Format: {ticker: {date: atr_value, ...}, ...}

    """
    atr_tickers = {}
    # Calculate True Range (TR) components
    for ticker in tickers_from_cols:
        high_low = result['High'][ticker] - result['Low'][ticker]
        high_prev_close = abs(result['High'][ticker] - result['Close'][ticker].shift(1))
        low_prev_close = abs(result['Low'][ticker] - result['Close'][ticker].shift(1))

        # True Range is the maximum of the three
        true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)

        # Calculate ATR using Wilder's smoothing method
        atr = pd.Series(np.nan, index=result.index)
        atr.iloc[period - 1] = true_range.iloc[:period].mean()  # Initial ATR

        for i in range(period, len(result)):
            atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + true_range.iloc[i]) / period

        atr_tickers[ticker] = atr.to_dict()

        if plot:
            fig, ax1 = plt.subplots(figsize=(14, 7))

            # Primary y-axis: Close Price
            color1 = 'tab:blue'
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Close Price ($)', color=color1, fontsize=12)
            ax1.plot(result.index, result['Close'][ticker], color=color1, linewidth=2, label=f'{ticker} Close Price')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, alpha=0.3)

            # Secondary y-axis: ATR
            ax2 = ax1.twinx()
            color2 = 'tab:orange'
            ax2.set_ylabel('ATR Value ($)', color=color2, fontsize=12)
            atr_clean = atr.dropna()
            ax2.plot(atr_clean.index, atr_clean.values, color=color2, linewidth=2, label=f'{ticker} ATR', alpha=0.8)
            ax2.tick_params(axis='y', labelcolor=color2)

            # Title and legends
            plt.title(f'{ticker} - Close Price and Average True Range (ATR)', fontsize=14, fontweight='bold')

            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            plt.tight_layout()
            plt.show()
        # Convert to dictionary
    return atr_tickers


def calculate_obv(plot=False):
    """
    Calculates the On-Balance Volume (OBV) for multiple tickers in the 'result' DataFrame.

    OBV Logic:
    - If today's close > yesterday's close: OBV = previous OBV + today's volume
    - If today's close < yesterday's close: OBV = previous OBV - today's volume
    - If today's close = yesterday's close: OBV = previous OBV

    Args:
        plot (bool): Whether to plot the OBV values with price.

    Returns:
        dict: A dictionary containing the OBV values for each ticker.
              Format: {ticker: {date: obv_value, ...}, ...}
    """
    obv_all_tickers = {}

    # Calculate OBV for each ticker
    for ticker in tickers_from_cols:
        # Get close prices and volume for this ticker
        close_prices = result['Close'][ticker]
        volumes = result['Volume'][ticker]

        # Initialize OBV series
        obv_series = pd.Series(index=result.index, dtype=float)
        obv_series.iloc[0] = 0  # Start at 0

        # Calculate OBV using vectorized operations (more efficient)
        # Method 1: Using numpy where for vectorized calculation
        price_diff = close_prices.diff()  # Today's close - yesterday's close

        # Create the directional volume array
        # +volume when price up, -volume when price down, 0 when unchanged
        directional_volume = np.where(price_diff > 0, volumes,
                                      np.where(price_diff < 0, -volumes, 0))

        # Calculate cumulative sum (this is the OBV)
        obv_series = pd.Series(directional_volume, index=result.index).cumsum()

        # Store in dictionary (drop NaN values)
        obv_all_tickers[ticker] = obv_series.dropna().to_dict()

        # Print summary statistics
        print(f"{ticker} OBV range: {obv_series.min():,.0f} to {obv_series.max():,.0f}")
        print(f"{ticker} Latest OBV: {obv_series.iloc[-1]:,.0f}")
        print(
            f"{ticker} OBV trend: {'Rising' if obv_series.iloc[-1] > obv_series.iloc[-20] else 'Falling'} (last 20 days)")
        print()

        # Plot if requested
        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

            # Top subplot: Close Price
            color1 = 'tab:blue'
            ax1.plot(result.index, close_prices, color=color1, linewidth=2, label=f'{ticker} Close Price')
            ax1.set_ylabel('Close Price ($)', color=color1, fontsize=12, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left', fontsize=10)
            ax1.set_title(f'{ticker} - Price and On-Balance Volume (OBV)', fontsize=14, fontweight='bold')

            # Bottom subplot: OBV
            color2 = 'tab:green'
            ax2.plot(result.index, obv_series, color=color2, linewidth=2, label=f'{ticker} OBV')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('OBV (Cumulative Volume)', color=color2, fontsize=12, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor=color2)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper left', fontsize=10)

            # Add zero line to OBV
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)

            # Format y-axis to show numbers with commas
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

            plt.tight_layout()
            plt.show()

    return obv_all_tickers


# Alternative: OBV with Moving Average for better signals
# def calculate_obv_with_ma(ma_period=20, plot=False):
#     """
#     Calculates OBV with a moving average overlay for signal detection.
#
#     Args:
#         ma_period (int): Period for OBV moving average
#         plot (bool): Whether to plot the results
#
#     Returns:
#         dict: Contains OBV, OBV MA, and crossover signals
#     """
#     obv_all_tickers = {}
#
#     for ticker in tickers_from_cols:
#         close_prices = result['Close'][ticker]
#         volumes = result['Volume'][ticker]
#
#         # Calculate OBV
#         price_diff = close_prices.diff()
#         directional_volume = np.where(price_diff > 0, volumes,
#                                       np.where(price_diff < 0, -volumes, 0))
#         obv_series = pd.Series(directional_volume, index=result.index).cumsum()
#
#         # Calculate OBV Moving Average
#         obv_ma = obv_series.rolling(window=ma_period).mean()
#
#         # Detect crossovers
#         obv_above_ma = obv_series > obv_ma
#         bullish_cross = obv_above_ma & ~obv_above_ma.shift(1)  # OBV crosses above MA
#         bearish_cross = ~obv_above_ma & obv_above_ma.shift(1)  # OBV crosses below MA
#
#         obv_all_tickers[ticker] = {
#             'obv': obv_series.to_dict(),
#             'obv_ma': obv_ma.dropna().to_dict(),
#             'bullish_signals': result.index[bullish_cross].tolist(),
#             'bearish_signals': result.index[bearish_cross].tolist()
#         }
#
#         if plot:
#             fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
#
#             # Top: Price with signals
#             ax1.plot(result.index, close_prices, linewidth=2, label=f'{ticker} Close', color='blue')
#
#             # Mark bullish and bearish signals on price chart
#             if bullish_cross.any():
#                 ax1.scatter(result.index[bullish_cross], close_prices[bullish_cross],
#                             color='green', s=100, marker='^', label='Bullish OBV Signal', zorder=5)
#             if bearish_cross.any():
#                 ax1.scatter(result.index[bearish_cross], close_prices[bearish_cross],
#                             color='red', s=100, marker='v', label='Bearish OBV Signal', zorder=5)
#
#             ax1.set_ylabel('Close Price ($)', fontsize=12, fontweight='bold')
#             ax1.legend(loc='upper left')
#             ax1.grid(True, alpha=0.3)
#             ax1.set_title(f'{ticker} - Price with OBV Signals', fontsize=14, fontweight='bold')
#
#             # Bottom: OBV with MA
#             ax2.plot(result.index, obv_series, linewidth=2, label='OBV', color='green')
#             ax2.plot(result.index, obv_ma, linewidth=2, linestyle='--', label=f'OBV MA({ma_period})', color='orange')
#             ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
#             ax2.set_xlabel('Date', fontsize=12)
#             ax2.set_ylabel('OBV Value', fontsize=12, fontweight='bold')
#             ax2.legend(loc='upper left')
#             ax2.grid(True, alpha=0.3)
#             ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
#
#             plt.tight_layout()
#             plt.show()
#
#     return obv_all_tickers


def calculate_vwap(plot=True, style='darkgrid'):
    """
    Calculates the Volume Weighted Average Price (VWAP) for multiple tickers.

    VWAP = Cumulative(Typical Price × Volume) / Cumulative(Volume)
    Typical Price = (High + Low + Close) / 3

    Args:
        plot (bool): Whether to plot the VWAP with price and crossovers.
        style (str): Seaborn style - 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'

    Returns:
        dict: A dictionary containing VWAP values and crossover information for each ticker.
    """
    # Set seaborn style
    sns.set_style(style)
    sns.set_context("notebook", font_scale=1.1)  # Makes fonts slightly larger

    vwap_all_tickers = {}

    for ticker in tickers_from_cols:
        # Calculate the Typical Price (TP)
        typical_price = (result['High'][ticker] + result['Low'][ticker] + result['Close'][ticker]) / 3

        # Calculate Price * Volume (PV)
        pv = typical_price * result['Volume'][ticker]

        # Calculate Cumulative Price * Volume
        cumulative_pv = pv.cumsum()

        # Calculate Cumulative Volume
        cumulative_volume = result['Volume'][ticker].cumsum()

        # Calculate VWAP
        vwap_series = cumulative_pv / cumulative_volume

        # Get close prices
        close_prices = result['Close'][ticker]

        # Detect crossovers
        price_above_vwap = close_prices > vwap_series
        bullish_cross = price_above_vwap & ~price_above_vwap.shift(1).fillna(False)
        bearish_cross = ~price_above_vwap & price_above_vwap.shift(1).fillna(False)

        # Store results
        vwap_all_tickers[ticker] = {
            'vwap': vwap_series.to_dict(),
            'bullish_crosses': result.index[bullish_cross].tolist(),
            'bearish_crosses': result.index[bearish_cross].tolist()
        }

        # Check for crossover in the most recent bar
        if len(result) >= 2:
            last_close = close_prices.iloc[-1]
            second_last_close = close_prices.iloc[-2]
            last_vwap = vwap_series.iloc[-1]
            second_last_vwap = vwap_series.iloc[-2]

            print(f"\n{ticker} - Recent Crossover Analysis:")
            if second_last_close > second_last_vwap and last_close < last_vwap:
                print(f"  ⚠️  Price Crossed BELOW VWAP (Bearish)")
                print(f"  Last Close: ${last_close:.2f} | VWAP: ${last_vwap:.2f}")
            elif second_last_close < second_last_vwap and last_close > last_vwap:
                print(f"  🚀 Price Crossed ABOVE VWAP (Bullish)")
                print(f"  Last Close: ${last_close:.2f} | VWAP: ${last_vwap:.2f}")
            else:
                position = "above" if last_close > last_vwap else "below"
                print(f"  No recent crossover - Price is {position} VWAP")
                print(f"  Last Close: ${last_close:.2f} | VWAP: ${last_vwap:.2f}")

        # Plotting with Seaborn
        if plot:
            fig, ax = plt.subplots(figsize=(16, 8))

            # Create a DataFrame for easier seaborn plotting
            plot_data = pd.DataFrame({
                'Date': result.index,
                'Close': close_prices.values,
                'VWAP': vwap_series.values
            })

            # Plot Close Price with seaborn (smoother lines)
            sns.lineplot(data=plot_data, x='Date', y='Close',
                         label=f'{ticker} Close Price',
                         linewidth=2.5, color='#2E86AB', ax=ax)

            # Plot VWAP with seaborn
            sns.lineplot(data=plot_data, x='Date', y='VWAP',
                         label='VWAP',
                         linewidth=2.5, linestyle='--', color='#F77F00', ax=ax)

            # Optional: Add a filled area between price and VWAP
            ax.fill_between(result.index, close_prices, vwap_series,
                            where=(close_prices > vwap_series),
                            alpha=0.2, color='green', label='Price > VWAP')
            ax.fill_between(result.index, close_prices, vwap_series,
                            where=(close_prices <= vwap_series),
                            alpha=0.2, color='red', label='Price < VWAP')

            # Plot bullish crossovers
            if bullish_cross.sum() > 0:
                ax.scatter(result.index[bullish_cross],
                           close_prices[bullish_cross],
                           color='#06D6A0', marker='^', s=200,
                           label='Bullish Cross', zorder=5,
                           edgecolors='white', linewidths=2)

            # Plot bearish crossovers
            if bearish_cross.sum() > 0:
                ax.scatter(result.index[bearish_cross],
                           close_prices[bearish_cross],
                           color='#EF476F', marker='v', s=200,
                           label='Bearish Cross', zorder=5,
                           edgecolors='white', linewidths=2)

            # Styling
            ax.set_title(f'{ticker} - VWAP Analysis',
                         fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Date', fontsize=13, fontweight='bold')
            ax.set_ylabel('Price ($)', fontsize=13, fontweight='bold')
            ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')

            plt.tight_layout()
            plt.show()

            # Print summary
            print(f"\n{ticker} Summary:")
            print(f"  Total Bullish Crosses: {bullish_cross.sum()}")
            print(f"  Total Bearish Crosses: {bearish_cross.sum()}")
            print(f"  Current VWAP: ${vwap_series.iloc[-1]:.2f}")
            print(f"  Current Close: ${close_prices.iloc[-1]:.2f}")

    return vwap_all_tickers
