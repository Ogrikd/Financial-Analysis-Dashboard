import yfinance as yf
import pandas as pd
from typing import Optional, Union

tickers_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']


def data_loader(ticker: Union[str, list[str]], start_date: str = '2020-01-01',
                end_date: str = '2025-01-01') -> Optional[pd.DataFrame]:
    """
    Download stock data from Yahoo Finance with error handling.

    Args:
        ticker: Stock ticker symbol or list of tickers (['AAPL', 'MSFT'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format

    Returns:
        DataFrame with stock data, or None if download fails
    """

    try:
        print(f"Downloading data for {ticker}...")

        # Download the data
        data = yf.download(ticker, start=start_date, end=end_date,
                           progress=False)  # Suppress progress bar

        # Check if data is empty
        if data.empty:
            print(f"Warning: No data retrieved for {ticker}")
            return None

        # Check for missing values
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: Found {missing_count} missing values, dropping rows...")

        data_cleaned = data.dropna()

        # Verify we still have data after cleaning
        if data_cleaned.empty:
            print(f"Error: No data remaining after cleaning for {ticker}")
            return None

        # Store ticker info as an attribute
        data_cleaned.attrs['tickers'] = ticker

        print(f"Successfully loaded {len(data_cleaned)} rows for {ticker}")
        data_cleaned.to_csv("Stocks Requested.csv")

        return data_cleaned

    except Exception as e:
        print(f"Error downloading data for {ticker}: {str(e)}")
        return None


result = data_loader(tickers_list)
