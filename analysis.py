import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import yfinance as yf
from scipy import stats

"""
Financial Metrics and Risk Analysis Module

This module calculates comprehensive financial metrics including:
- Returns (daily, weekly, monthly, cumulative, annualized)
- Risk metrics (volatility, beta, Sharpe ratio, max drawdown, VaR)
- Correlation and covariance analysis
"""


class FinancialMetrics:
    """
    Calculate comprehensive financial metrics for stock analysis
    """

    def __init__(self, price_data: pd.DataFrame, risk_free_rate: float = 0.04):
        """
        Initialize with price data

        Args:
            price_data: DataFrame with stock prices (columns = tickers, index = dates)
            risk_free_rate: Annual risk-free rate (default 4% = 0.04)
        """
        self.prices = price_data
        self.risk_free_rate = risk_free_rate
        self.daily_returns = None
        self.calculate_daily_returns()

    def calculate_daily_returns(self):
        """Calculate daily returns for all stocks"""
        self.daily_returns = self.prices.pct_change().dropna()
        return self.daily_returns

    # ==================== RETURNS CALCULATIONS ====================

    def get_weekly_returns(self) -> pd.DataFrame:
        """Calculate weekly returns"""
        weekly_prices = self.prices.resample('W').last()
        weekly_returns = weekly_prices.pct_change().dropna()
        return weekly_returns

    def get_monthly_returns(self) -> pd.DataFrame:
        """Calculate monthly returns"""
        monthly_prices = self.prices.resample('M').last()
        monthly_returns = monthly_prices.pct_change().dropna()
        return monthly_returns

    def get_cumulative_returns(self) -> pd.DataFrame:
        """Calculate cumulative returns"""
        cumulative_returns = (1 + self.daily_returns).cumprod() - 1
        return cumulative_returns

    def get_annualized_returns(self) -> pd.Series:
        """
        Calculate annualized returns

        Returns:
            Series with annualized returns for each stock
        """
        total_days = len(self.daily_returns)
        cumulative_return = (1 + self.daily_returns).prod() - 1
        years = total_days / 252  # 252 trading days per year
        annualized_return = (1 + cumulative_return) ** (1 / years) - 1
        return annualized_return

    def get_total_return(self) -> pd.Series:
        """Calculate total return over the entire period"""
        total_return = (self.prices.iloc[-1] / self.prices.iloc[0]) - 1
        return total_return

    # ==================== RISK METRICS ====================

    def get_volatility(self, annualized: bool = True) -> pd.Series:
        """
        Calculate volatility (standard deviation of returns)

        Args:
            annualized: If True, annualize the volatility

        Returns:
            Series with volatility for each stock
        """
        volatility = self.daily_returns.std()

        if annualized:
            volatility = volatility * np.sqrt(252)  # Annualize

        return volatility

    def get_beta(self, market_ticker: str = 'SPY') -> pd.Series:
        """
        Calculate beta relative to market (SPY)

        Args:
            market_ticker: Market benchmark ticker (default SPY)

        Returns:
            Series with beta for each stock
        """
        # Download market data if not in price data
        if market_ticker not in self.prices.columns:
            market_data = yf.download(market_ticker,
                                      start=self.prices.index[0],
                                      end=self.prices.index[-1],
                                      progress=False)['Adj Close']
            market_returns = market_data.pct_change().dropna()
        else:
            market_returns = self.daily_returns[market_ticker]

        betas = {}
        for ticker in self.daily_returns.columns:
            if ticker == market_ticker:
                betas[ticker] = 1.0
            else:
                # Calculate beta using covariance
                covariance = self.daily_returns[ticker].cov(market_returns)
                market_variance = market_returns.var()
                betas[ticker] = covariance / market_variance

        return pd.Series(betas)

    def get_sharpe_ratio(self, annualized: bool = True) -> pd.Series:
        """
        Calculate Sharpe Ratio

        Sharpe Ratio = (Return - Risk Free Rate) / Volatility

        Args:
            annualized: If True, use annualized values

        Returns:
            Series with Sharpe ratio for each stock
        """
        if annualized:
            returns = self.get_annualized_returns()
            volatility = self.get_volatility(annualized=True)
            sharpe = (returns - self.risk_free_rate) / volatility
        else:
            daily_rf = self.risk_free_rate / 252
            excess_returns = self.daily_returns.mean() - daily_rf
            volatility = self.daily_returns.std()
            sharpe = excess_returns / volatility
            sharpe = sharpe * np.sqrt(252)  # Annualize

        return sharpe

    def get_sortino_ratio(self, annualized: bool = True) -> pd.Series:
        """
        Calculate Sortino Ratio (like Sharpe but only considers downside volatility)

        Returns:
            Series with Sortino ratio for each stock
        """
        if annualized:
            returns = self.get_annualized_returns()
            # Downside deviation
            downside_returns = self.daily_returns[self.daily_returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino = (returns - self.risk_free_rate) / downside_std
        else:
            daily_rf = self.risk_free_rate / 252
            excess_returns = self.daily_returns.mean() - daily_rf
            downside_returns = self.daily_returns[self.daily_returns < 0]
            downside_std = downside_returns.std()
            sortino = excess_returns / downside_std * np.sqrt(252)

        return sortino

    def get_max_drawdown(self) -> Dict[str, float]:
        """
        Calculate maximum drawdown for each stock

        Max Drawdown = (Trough Value - Peak Value) / Peak Value

        Returns:
            Dictionary with max drawdown for each stock
        """
        cumulative_returns = self.get_cumulative_returns()
        max_drawdowns = {}

        for ticker in cumulative_returns.columns:
            cum_returns = cumulative_returns[ticker]
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / (1 + running_max)
            max_drawdowns[ticker] = drawdown.min()

        return max_drawdowns

    def get_value_at_risk(self, confidence_level: float = 0.95) -> pd.Series:
        """
        Calculate Value at Risk (VaR) using historical method

        VaR answers: "What is the maximum loss with X% confidence?"

        Args:
            confidence_level: Confidence level (default 95%)

        Returns:
            Series with VaR for each stock
        """
        var = self.daily_returns.quantile(1 - confidence_level)
        return var

    def get_conditional_var(self, confidence_level: float = 0.95) -> pd.Series:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall

        CVaR = Average of all losses beyond VaR

        Args:
            confidence_level: Confidence level (default 95%)

        Returns:
            Series with CVaR for each stock
        """
        var = self.get_value_at_risk(confidence_level)
        cvar = {}

        for ticker in self.daily_returns.columns:
            returns = self.daily_returns[ticker]
            cvar[ticker] = returns[returns <= var[ticker]].mean()

        return pd.Series(cvar)

    # ==================== CORRELATION ANALYSIS ====================

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Calculate correlation matrix between all stocks

        Returns:
            DataFrame with correlation coefficients
        """
        return self.daily_returns.corr()

    def get_covariance_matrix(self, annualized: bool = True) -> pd.DataFrame:
        """
        Calculate covariance matrix

        Args:
            annualized: If True, annualize the covariance

        Returns:
            DataFrame with covariance values
        """
        cov_matrix = self.daily_returns.cov()

        if annualized:
            cov_matrix = cov_matrix * 252

        return cov_matrix

    def get_rolling_correlation(self, ticker1: str, ticker2: str, window: int = 30) -> pd.Series:
        """
        Calculate rolling correlation between two stocks

        Args:
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            window: Rolling window size (default 30 days)

        Returns:
            Series with rolling correlation values
        """
        rolling_corr = self.daily_returns[ticker1].rolling(window=window).corr(
            self.daily_returns[ticker2]
        )
        return rolling_corr

    def get_rolling_beta(self, ticker: str, market_ticker: str = 'SPY', window: int = 60) -> pd.Series:
        """
        Calculate rolling beta

        Args:
            ticker: Stock ticker
            market_ticker: Market benchmark
            window: Rolling window size (default 60 days)

        Returns:
            Series with rolling beta values
        """
        if market_ticker not in self.prices.columns:
            market_data = yf.download(market_ticker,
                                      start=self.prices.index[0],
                                      end=self.prices.index[-1],
                                      progress=False)['Adj Close']
            market_returns = market_data.pct_change().dropna()
        else:
            market_returns = self.daily_returns[market_ticker]

        # Calculate rolling covariance and variance
        rolling_cov = self.daily_returns[ticker].rolling(window=window).cov(market_returns)
        rolling_var = market_returns.rolling(window=window).var()
        rolling_beta = rolling_cov / rolling_var

        return rolling_beta

    # ==================== COMPREHENSIVE SUMMARY ====================

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics for all stocks

        Returns:
            DataFrame with all key metrics
        """
        summary = pd.DataFrame()

        # Returns
        summary['Total Return'] = self.get_total_return()
        summary['Annualized Return'] = self.get_annualized_returns()
        summary['Daily Avg Return'] = self.daily_returns.mean()

        # Risk
        summary['Volatility (Annual)'] = self.get_volatility(annualized=True)
        summary['Sharpe Ratio'] = self.get_sharpe_ratio()
        summary['Sortino Ratio'] = self.get_sortino_ratio()

        # Drawdown
        max_dd = self.get_max_drawdown()
        summary['Max Drawdown'] = pd.Series(max_dd)

        # VaR
        summary['VaR (95%)'] = self.get_value_at_risk(0.95)
        summary['CVaR (95%)'] = self.get_conditional_var(0.95)

        # Beta (if SPY data available or can be downloaded)
        try:
            summary['Beta'] = self.get_beta()
        except:
            summary['Beta'] = np.nan

        # Additional metrics
        summary['Skewness'] = self.daily_returns.skew()
        summary['Kurtosis'] = self.daily_returns.kurtosis()

        return summary

    # ==================== PERFORMANCE METRICS ====================

    def get_information_ratio(self, benchmark_ticker: str = 'SPY') -> pd.Series:
        """
        Calculate Information Ratio

        IR = (Portfolio Return - Benchmark Return) / Tracking Error

        Args:
            benchmark_ticker: Benchmark ticker (default SPY)

        Returns:
            Series with information ratio for each stock
        """
        if benchmark_ticker not in self.prices.columns:
            benchmark_data = yf.download(benchmark_ticker,
                                         start=self.prices.index[0],
                                         end=self.prices.index[-1],
                                         progress=False)['Adj Close']
            benchmark_returns = benchmark_data.pct_change().dropna()
        else:
            benchmark_returns = self.daily_returns[benchmark_ticker]

        information_ratios = {}
        for ticker in self.daily_returns.columns:
            if ticker == benchmark_ticker:
                information_ratios[ticker] = 0.0
            else:
                excess_returns = self.daily_returns[ticker] - benchmark_returns
                tracking_error = excess_returns.std()
                information_ratios[ticker] = (excess_returns.mean() / tracking_error) * np.sqrt(252)

        return pd.Series(information_ratios)

    def get_calmar_ratio(self) -> pd.Series:
        """
        Calculate Calmar Ratio

        Calmar Ratio = Annualized Return / Maximum Drawdown

        Returns:
            Series with Calmar ratio for each stock
        """
        annual_returns = self.get_annualized_returns()
        max_drawdowns = pd.Series(self.get_max_drawdown())

        calmar = annual_returns / abs(max_drawdowns)
        return calmar

    def get_omega_ratio(self, threshold: float = 0.0) -> pd.Series:
        """
        Calculate Omega Ratio

        Omega = Probability Weighted Gains / Probability Weighted Losses

        Args:
            threshold: Return threshold (default 0%)

        Returns:
            Series with Omega ratio for each stock
        """
        omega = {}

        for ticker in self.daily_returns.columns:
            returns = self.daily_returns[ticker]
            gains = returns[returns > threshold] - threshold
            losses = threshold - returns[returns < threshold]

            omega[ticker] = gains.sum() / losses.sum() if losses.sum() != 0 else np.inf

        return pd.Series(omega)


# ==================== UTILITY FUNCTIONS ====================

def calculate_portfolio_metrics(weights: Dict[str, float],
                                price_data: pd.DataFrame,
                                risk_free_rate: float = 0.04) -> Dict:
    """
    Calculate metrics for a portfolio with given weights

    Args:
        weights: Dictionary of ticker: weight pairs (weights should sum to 1)
        price_data: DataFrame with price data
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary with portfolio metrics
    """
    fm = FinancialMetrics(price_data, risk_free_rate)

    # Convert weights to array
    weight_array = np.array([weights.get(ticker, 0) for ticker in price_data.columns])

    # Portfolio returns
    portfolio_returns = (fm.daily_returns * weight_array).sum(axis=1)

    # Portfolio metrics
    portfolio_metrics = {
        'annual_return': portfolio_returns.mean() * 252,
        'annual_volatility': portfolio_returns.std() * np.sqrt(252),
        'sharpe_ratio': (portfolio_returns.mean() * 252 - risk_free_rate) / (portfolio_returns.std() * np.sqrt(252)),
        'max_drawdown': ((1 + portfolio_returns).cumprod().cummax() - (1 + portfolio_returns).cumprod()).max(),
        'var_95': portfolio_returns.quantile(0.05),
        'cvar_95': portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.05)].mean()
    }

    return portfolio_metrics


def compare_stocks(tickers: list, price_data: pd.DataFrame, metric: str = 'sharpe_ratio') -> pd.Series:
    """
    Compare stocks based on a specific metric

    Args:
        tickers: List of ticker symbols
        price_data: DataFrame with price data
        metric: Metric to compare ('sharpe_ratio', 'returns', 'volatility', etc.)

    Returns:
        Series with comparison values, sorted
    """
    fm = FinancialMetrics(price_data)

    metric_map = {
        'sharpe_ratio': fm.get_sharpe_ratio(),
        'returns': fm.get_annualized_returns(),
        'volatility': fm.get_volatility(),
        'max_drawdown': pd.Series(fm.get_max_drawdown()),
        'var': fm.get_value_at_risk(),
        'sortino_ratio': fm.get_sortino_ratio()
    }

    if metric not in metric_map:
        raise ValueError(f"Unknown metric: {metric}")

    comparison = metric_map[metric]
    return comparison.sort_values(ascending=False)
