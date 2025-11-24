import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from analysis import FinancialMetrics, compare_stocks
from data_loader import result, tickers_list

"""
Risk-Return Analysis Dashboard

Comprehensive financial metrics dashboard including:
- Returns analysis
- Risk metrics
- Correlation analysis
- Performance comparisons
"""

# Page config
st.set_page_config(
    page_title="Risk-Return Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_price_data():
    """Load price data from result DataFrame"""
    if result is None or result.empty:
        return None

    # Extract Close prices for all tickers
    tickers = result.columns.get_level_values(1).unique().tolist()
    price_df = pd.DataFrame()

    for ticker in tickers:
        price_df[ticker] = result['Close'][ticker]

    return price_df


@st.cache_data
def calculate_metrics(price_data, risk_free_rate):
    """Calculate all financial metrics"""
    fm = FinancialMetrics(price_data, risk_free_rate)
    return fm


def create_risk_return_scatter(fm: FinancialMetrics):
    """Create risk-return scatter plot"""
    returns = fm.get_annualized_returns()
    volatility = fm.get_volatility(annualized=True)
    sharpe = fm.get_sharpe_ratio()

    fig = go.Figure()

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=volatility * 100,
        y=returns * 100,
        mode='markers+text',
        marker=dict(
            size=sharpe * 20,  # Size based on Sharpe ratio
            color=sharpe,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio"),
            line=dict(width=2, color='white')
        ),
        text=returns.index,
        textposition="top center",
        textfont=dict(size=12, color='white'),
        hovertemplate='<b>%{text}</b><br>' +
                      'Return: %{y:.2f}%<br>' +
                      'Volatility: %{x:.2f}%<br>' +
                      '<extra></extra>'
    ))

    fig.update_layout(
        title='Risk-Return Profile',
        xaxis_title='Annualized Volatility (%)',
        yaxis_title='Annualized Return (%)',
        template='plotly_dark',
        hovermode='closest',
        height=600,
        showlegend=False
    )

    return fig


def create_cumulative_returns_chart(fm: FinancialMetrics):
    """Create cumulative returns chart"""
    cum_returns = fm.get_cumulative_returns()

    fig = go.Figure()

    colors = ['#00ff88', '#667eea', '#ff3366', '#ffa500', '#00d4ff']

    for i, ticker in enumerate(cum_returns.columns):
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns[ticker] * 100,
            name=ticker,
            line=dict(width=2.5, color=colors[i % len(colors)]),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Date: %{x}<br>' +
                          'Return: %{y:.2f}%<br>' +
                          '<extra></extra>'
        ))

    fig.update_layout(
        title='Cumulative Returns',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template='plotly_dark',
        hovermode='x unified',
        height=500,
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        )
    )

    return fig


def create_correlation_heatmap(fm: FinancialMetrics):
    """Create correlation heatmap"""
    corr_matrix = fm.get_correlation_matrix()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title='Correlation Matrix',
        template='plotly_dark',
        height=500,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )

    return fig


def create_rolling_correlation_chart(fm: FinancialMetrics, ticker1, ticker2, window=30):
    """Create rolling correlation chart"""
    rolling_corr = fm.get_rolling_correlation(ticker1, ticker2, window)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rolling_corr.index,
        y=rolling_corr,
        fill='tozeroy',
        name=f'{ticker1} vs {ticker2}',
        line=dict(color='#667eea', width=2),
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)

    fig.update_layout(
        title=f'Rolling {window}-Day Correlation: {ticker1} vs {ticker2}',
        xaxis_title='Date',
        yaxis_title='Correlation',
        template='plotly_dark',
        height=400,
        yaxis=dict(range=[-1, 1])
    )

    return fig


def create_drawdown_chart(fm: FinancialMetrics):
    """Create drawdown chart"""
    cum_returns = fm.get_cumulative_returns()

    fig = go.Figure()

    colors = ['#00ff88', '#667eea', '#ff3366', '#ffa500', '#00d4ff']

    for i, ticker in enumerate(cum_returns.columns):
        cum_ret = cum_returns[ticker]
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / (1 + running_max) * 100

        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown,
            name=ticker,
            fill='tozeroy',
            line=dict(width=2, color=colors[i % len(colors)]),
            fillcolor=f'rgba{tuple(list(int(colors[i % len(colors)].lstrip("#")[j:j + 2], 16) for j in (0, 2, 4)) + [0.3])}'
        ))

    fig.update_layout(
        title='Drawdown Analysis',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_dark',
        height=500,
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        )
    )

    return fig


def create_returns_distribution(fm: FinancialMetrics):
    """Create returns distribution histogram"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(fm.daily_returns.columns)
    )

    colors = ['#00ff88', '#667eea', '#ff3366', '#ffa500']

    for i, ticker in enumerate(fm.daily_returns.columns):
        row = i // 2 + 1
        col = i % 2 + 1

        returns = fm.daily_returns[ticker] * 100

        fig.add_trace(
            go.Histogram(
                x=returns,
                name=ticker,
                marker_color=colors[i % len(colors)],
                opacity=0.7,
                nbinsx=50
            ),
            row=row, col=col
        )

        # Add VaR line
        var_95 = fm.get_value_at_risk(0.95)[ticker] * 100
        fig.add_vline(
            x=var_95,
            line_dash="dash",
            line_color="red",
            row=row, col=col,
            annotation_text=f"VaR 95%: {var_95:.2f}%"
        )

    fig.update_layout(
        title='Daily Returns Distribution',
        template='plotly_dark',
        height=600,
        showlegend=False
    )

    fig.update_xaxes(title_text="Daily Return (%)")
    fig.update_yaxes(title_text="Frequency")

    return fig


def create_rolling_volatility_chart(fm: FinancialMetrics, window=30):
    """Create rolling volatility chart"""
    fig = go.Figure()

    colors = ['#00ff88', '#667eea', '#ff3366', '#ffa500', '#00d4ff']

    for i, ticker in enumerate(fm.daily_returns.columns):
        rolling_vol = fm.daily_returns[ticker].rolling(window=window).std() * np.sqrt(252) * 100

        fig.add_trace(go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol,
            name=ticker,
            line=dict(width=2, color=colors[i % len(colors)])
        ))

    fig.update_layout(
        title=f'Rolling {window}-Day Volatility (Annualized)',
        xaxis_title='Date',
        yaxis_title='Volatility (%)',
        template='plotly_dark',
        height=500,
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        )
    )

    return fig


def main():
    st.markdown("""
        <div style='text-align: center; padding: 20px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; margin-bottom: 30px; box-shadow: 0 8px 16px rgba(0,0,0,0.3);'>
            <h1 style='color: white; font-size: 3rem; margin: 0;'>
                📊 Risk-Return Analysis Dashboard
            </h1>
            <p style='color: rgba(255,255,255,0.9); font-size: 1.2rem; margin: 10px 0 0 0;'>
                Comprehensive Financial Metrics & Performance Analysis
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    risk_free_rate = st.sidebar.slider(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.1
    ) / 100

    rolling_window = st.sidebar.slider(
        "Rolling Window (days)",
        min_value=10,
        max_value=90,
        value=30,
        step=5
    )

    # Load data
    price_data = load_price_data()

    if price_data is None or price_data.empty:
        st.error("❌ No price data available. Please load data first.")
        return

    # Calculate metrics
    fm = calculate_metrics(price_data, risk_free_rate)

    # Get summary statistics
    summary_stats = fm.get_summary_statistics()

    # Display key metrics
    st.subheader("📈 Performance Summary")

    cols = st.columns(len(price_data.columns))
    for i, ticker in enumerate(price_data.columns):
        with cols[i]:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>{ticker}</h3>
                    <p><strong>Return:</strong> {summary_stats.loc[ticker, 'Annualized Return'] * 100:.2f}%</p>
                    <p><strong>Volatility:</strong> {summary_stats.loc[ticker, 'Volatility (Annual)'] * 100:.2f}%</p>
                    <p><strong>Sharpe:</strong> {summary_stats.loc[ticker, 'Sharpe Ratio']:.2f}</p>
                    <p><strong>Max DD:</strong> {summary_stats.loc[ticker, 'Max Drawdown'] * 100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Risk-Return Scatter
    st.subheader("🎯 Risk-Return Profile")
    st.plotly_chart(create_risk_return_scatter(fm), use_container_width=True)

    st.markdown("---")

    # Cumulative Returns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Cumulative Returns")
        st.plotly_chart(create_cumulative_returns_chart(fm), use_container_width=True)

    with col2:
        st.subheader("📉 Drawdown Analysis")
        st.plotly_chart(create_drawdown_chart(fm), use_container_width=True)

    st.markdown("---")

    # Correlation Analysis
    st.subheader("🔗 Correlation Analysis")

    col3, col4 = st.columns([1, 1])

    with col3:
        st.plotly_chart(create_correlation_heatmap(fm), use_container_width=True)

    with col4:
        st.write("**Rolling Correlation**")
        tickers = list(price_data.columns)
        if len(tickers) >= 2:
            ticker1 = st.selectbox("Select First Stock", tickers, index=0)
            ticker2 = st.selectbox("Select Second Stock", tickers, index=1)
            st.plotly_chart(
                create_rolling_correlation_chart(fm, ticker1, ticker2, rolling_window),
                use_container_width=True
            )

    st.markdown("---")

    # Volatility Analysis
    st.subheader("📈 Volatility Analysis")
    st.plotly_chart(create_rolling_volatility_chart(fm, rolling_window), use_container_width=True)

    st.markdown("---")

    # Returns Distribution
    st.subheader("📊 Returns Distribution & VaR")
    st.plotly_chart(create_returns_distribution(fm), use_container_width=True)

    st.markdown("---")

    # Detailed Statistics Table
    st.subheader("📋 Detailed Statistics")

    # Format the summary stats for display
    display_stats = summary_stats.copy()

    # Format percentage columns
    pct_cols = ['Total Return', 'Annualized Return', 'Daily Avg Return',
                'Volatility (Annual)', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)']
    for col in pct_cols:
        if col in display_stats.columns:
            display_stats[col] = display_stats[col].apply(lambda x: f"{x * 100:.2f}%")

    # Format ratio columns
    ratio_cols = ['Sharpe Ratio', 'Sortino Ratio', 'Beta']
    for col in ratio_cols:
        if col in display_stats.columns:
            display_stats[col] = display_stats[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    st.dataframe(display_stats, use_container_width=True)

    # Download button
    csv = summary_stats.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Statistics as CSV",
        data=csv,
        file_name="financial_metrics.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()