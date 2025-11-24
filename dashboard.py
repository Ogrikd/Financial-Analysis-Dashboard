import streamlit as st
import pandas as pd
import numpy as np
from data_loader import result, data_loader, tickers_list
from indicator import (
    smas_indicator, ema_indicator, rsi_indicator, mcad_indicator,
    stochastic_oscillator_talib, calculate_bollinger_bands,
    calculate_atr, calculate_obv, calculate_vwap
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from pattern_recognition import PatternRecognition, get_pattern_summary, format_pattern_results

    PATTERN_RECOGNITION_AVAILABLE = True
except ImportError:
    PATTERN_RECOGNITION_AVAILABLE = False
    st.warning("⚠️ Pattern recognition module not found. Some features will be disabled.")

"""
Interactive Financial Analysis Dashboard with Multiple Technical Indicators and Pattern Recognition
Uses pre-calculated indicators from indicator.py

Run with:
streamlit run dashboard.py
"""

# Page configuration
st.set_page_config(
    page_title="Financial Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
        background-color: #0e1117;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMetric label {
        color: #ffffff !important;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.5rem !important;
        font-weight: 700;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #ffffff !important;
    }
    .pattern-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00d4ff;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        color: white;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #ffffff;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #667eea;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="metric-container"] > label {
        font-weight: 700 !important;
        color: white !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="metric-container"] > div {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def get_tickers_from_data():
    """Extract available tickers from the loaded result DataFrame"""
    if result is None or result.empty:
        return []
    return result.columns.get_level_values(1).unique().tolist()


@st.cache_data(ttl=3600)
def calculate_all_indicators():
    """Calculate all indicators once and cache them"""
    tickers = get_tickers_from_data()

    indicators = {}

    for ticker in tickers:
        indicators[ticker] = {}

        # Get basic price data
        indicators[ticker]['close'] = result['Close'][ticker]
        indicators[ticker]['open'] = result['Open'][ticker]
        indicators[ticker]['high'] = result['High'][ticker]
        indicators[ticker]['low'] = result['Low'][ticker]
        indicators[ticker]['volume'] = result['Volume'][ticker]

        # Calculate SMAs
        smas = smas_indicator()
        indicators[ticker]['sma_20'] = smas[ticker]['SMA_20']
        indicators[ticker]['sma_50'] = smas[ticker]['SMA_50']
        indicators[ticker]['sma_200'] = smas[ticker]['SMA_200']

        # Calculate EMAs
        ema = ema_indicator()
        indicators[ticker]['ema_12'] = ema[ticker]['EMA_12']
        indicators[ticker]['ema_26'] = ema[ticker]['EMA_26']

        # Calculate RSI
        rsi = rsi_indicator()
        indicators[ticker]['rsi'] = rsi[ticker]

        # Calculate MACD
        macd = mcad_indicator()
        indicators[ticker]['macd'] = macd[ticker]['MACD']
        indicators[ticker]['macd_signal'] = macd[ticker]['Signal']
        indicators[ticker]['macd_histogram'] = macd[ticker]['Histogram']

        # Calculate Bollinger Bands
        bollinger = calculate_bollinger_bands()
        indicators[ticker]['bb_upper'] = bollinger[ticker]['Upper Band']
        indicators[ticker]['bb_lower'] = bollinger[ticker]['Lower Band']
        indicators[ticker]['bb_middle'] = bollinger[ticker]['SMA']

        # Calculate Stochastic
        stochastic = stochastic_oscillator_talib()
        indicators[ticker]['stoch_k'] = stochastic[ticker]['%K']
        indicators[ticker]['stoch_d'] = stochastic[ticker]['%D']

        # Calculate ATR
        atr = calculate_atr()
        indicators[ticker]['atr'] = atr[ticker]

        # Calculate OBV
        obv = calculate_obv()
        indicators[ticker]['obv_ma'] = obv[ticker]

        # Calculate VWAP
        vwap = calculate_vwap()
        indicators[ticker]['vwap'] = vwap[ticker]['vwap']

    return indicators


@st.cache_data(ttl=3600)
def calculate_pattern_recognition(ticker):
    """Calculate pattern recognition for a specific ticker"""
    if not PATTERN_RECOGNITION_AVAILABLE:
        return None
    # Create DataFrame for pattern recognition
    ticker_df = pd.DataFrame({
        'Open': result['Open'][ticker],
        'High': result['High'][ticker],
        'Low': result['Low'][ticker],
        'Close': result['Close'][ticker],
        'Volume': result['Volume'][ticker]
    })

    pr = PatternRecognition(ticker_df)
    return pr.analyze_all_patterns()


def create_interactive_chart(ticker, indicators_data, indicators_to_show, pattern_data=None):
    data = indicators_data[ticker]

    # Count subplot rows
    row_heights = [0.5]
    rows = 1
    specs = [[{"secondary_y": False}]]

    for indicator in ['Volume', 'ATR', 'OBV', 'RSI', 'MACD', 'Stochastic']:
        if indicator in indicators_to_show:
            rows += 1
            row_heights.append(0.12)
            specs.append([{"secondary_y": False}])

    # Create figure
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        specs=specs
    )

    # Main price chart (always row 1)
    price_row = 1
    current_row = 1

    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=data['close'].index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff3366',
            increasing_fillcolor='#00ff88',
            decreasing_fillcolor='#ff3366'
        ),
        row=price_row, col=1
    )

    # Pattern markers
    if pattern_data and 'candlestick_patterns' in pattern_data:
        patterns = pattern_data['candlestick_patterns']

        pattern_configs = [
            ('Doji', 'circle', 12, '#ffd700', '#ff8c00', 1.02, 'high', '🟡'),
            ('Hammer', 'triangle-up', 14, '#00ff88', '#00cc66', 0.98, 'low', '🔨'),
            ('Shooting_Star', 'triangle-down', 14, '#ff3366', '#cc0033', 1.02, 'high', '⭐'),
            ('Bullish_Engulfing', 'star', 16, '#00ffff', '#00cccc', 0.98, 'low', '📈'),
            ('Bearish_Engulfing', 'star', 16, '#ff6b6b', '#cc0000', 1.02, 'high', '📉')
        ]

        for pat_name, symbol, size, color, border, mult, price_type, emoji in pattern_configs:
            if pat_name in patterns.columns:
                dates = patterns[patterns[pat_name]].index
                if len(dates) > 0:
                    prices = data[price_type].loc[dates] * mult
                    fig.add_trace(
                        go.Scatter(
                            x=dates, y=prices,
                            mode='markers',
                            marker=dict(symbol=symbol, size=size, color=color,
                                        line=dict(color=border, width=2)),
                            name=f'{emoji} {pat_name.replace("_", " ")}',
                            hovertemplate=f'<b>{pat_name.replace("_", " ")}</b><br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                        ),
                        row=price_row, col=1
                    )

    # Support/Resistance
    if pattern_data and 'support_resistance' in pattern_data:
        sr = pattern_data['support_resistance']

        for i, level in enumerate(sr['resistance'][:3]):
            fig.add_hline(
                y=level, line_dash="dash", line_color="#ff3366",
                line_width=2, opacity=0.7, row=price_row, col=1,
                annotation_text=f"R{i + 1}: ${level:.2f}",
                annotation_position="right",
                annotation=dict(font=dict(size=11, color="#ff3366", family="Arial Black"))
            )

        for i, level in enumerate(sr['support'][:3]):
            fig.add_hline(
                y=level, line_dash="dash", line_color="#00ff88",
                line_width=2, opacity=0.7, row=price_row, col=1,
                annotation_text=f"S{i + 1}: ${level:.2f}",
                annotation_position="right",
                annotation=dict(font=dict(size=11, color="#00ff88", family="Arial Black"))
            )

    # Trend Lines
    if pattern_data and 'trend_lines' in pattern_data:
        tl = pattern_data['trend_lines']

        for i, line in enumerate(tl['uptrend'][:2]):
            x_vals = [data['close'].index[line['start_idx']], data['close'].index[line['end_idx']]]
            y_vals = [line['start_price'], line['end_price']]
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines',
                    line=dict(color='#00ff88', width=3, dash='solid'),
                    name=f'📈 Uptrend {i + 1}',
                    hovertemplate=f'<b>Uptrend {i + 1}</b><br>Touches: {line["touches"]}<extra></extra>'
                ),
                row=price_row, col=1
            )

        for i, line in enumerate(tl['downtrend'][:2]):
            x_vals = [data['close'].index[line['start_idx']], data['close'].index[line['end_idx']]]
            y_vals = [line['start_price'], line['end_price']]
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines',
                    line=dict(color='#ff3366', width=3, dash='solid'),
                    name=f'📉 Downtrend {i + 1}',
                    hovertemplate=f'<b>Downtrend {i + 1}</b><br>Touches: {line["touches"]}<extra></extra>'
                ),
                row=price_row, col=1
            )

    # Add indicators to price chart
    if 'VWAP' in indicators_to_show:
        fig.add_trace(
            go.Scatter(x=pd.Series(data['vwap']).index, y=pd.Series(data['vwap']).values, name='VWAP',
                       line=dict(color='#ffa500', width=2.5, dash='dash')),
            row=price_row, col=1
        )

    if 'SMA' in indicators_to_show:
        fig.add_trace(go.Scatter(x=data['sma_20'].index, y=data['sma_20'], name='SMA(20)',
                                 line=dict(color='#667eea', width=2)), row=price_row, col=1)
        fig.add_trace(go.Scatter(x=data['sma_50'].index, y=data['sma_50'], name='SMA(50)',
                                 line=dict(color='#764ba2', width=2)), row=price_row, col=1)
        fig.add_trace(go.Scatter(x=data['sma_200'].index, y=data['sma_200'], name='SMA(200)',
                                 line=dict(color='#f093fb', width=2, dash='dot')), row=price_row, col=1)

    if 'EMA' in indicators_to_show:
        fig.add_trace(go.Scatter(x=data['ema_12'].index, y=data['ema_12'], name='EMA(12)',
                                 line=dict(color='#00d4ff', width=2, dash='dash')), row=price_row, col=1)
        fig.add_trace(go.Scatter(x=data['ema_26'].index, y=data['ema_26'], name='EMA(26)',
                                 line=dict(color='#ff00ff', width=2, dash='dash')), row=price_row, col=1)

    if 'Bollinger Bands' in indicators_to_show:
        fig.add_trace(go.Scatter(x=data['bb_upper'].index, y=data['bb_upper'], name='BB Upper',
                                 line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dot')), row=price_row, col=1)
        fig.add_trace(go.Scatter(x=data['bb_lower'].index, y=data['bb_lower'], name='BB Lower',
                                 line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dot'),
                                 fill='tonexty', fillcolor='rgba(128,128,128,0.15)'), row=price_row, col=1)

    # Move to indicator rows
    current_row += 1

    # Volume
    if 'Volume' in indicators_to_show:
        colors = ['#ff3366' if data['close'].iloc[i] < data['open'].iloc[i] else '#00ff88'
                  for i in range(len(data['close']))]
        fig.add_trace(go.Bar(x=data['volume'].index, y=data['volume'], name='Volume',
                             marker_color=colors, opacity=0.7), row=current_row, col=1)
        fig.update_yaxes(title_text="Volume", row=current_row, col=1, gridcolor='rgba(128,128,128,0.2)')
        current_row += 1

    # ATR
    if 'ATR' in indicators_to_show:
        fig.add_trace(go.Scatter(x=pd.Series(data['atr']).index, y=pd.Series(data['atr']).values, name='ATR',
                                 line=dict(color='#a855f7', width=2), fill='tozeroy',
                                 fillcolor='rgba(168,85,247,0.3)'), row=current_row, col=1)
        fig.update_yaxes(title_text="ATR", row=current_row, col=1, gridcolor='rgba(128,128,128,0.2)')
        current_row += 1

    # OBV
    if 'OBV' in indicators_to_show:
        fig.add_trace(go.Scatter(x=pd.Series(data['obv_ma']).index, y=pd.Series(data['obv_ma']).values, name='OBV MA(20)',
                                 line=dict(color='#ffa500', width=2, dash='dash')), row=current_row, col=1)
        fig.update_yaxes(title_text="OBV", row=current_row, col=1, gridcolor='rgba(128,128,128,0.2)')
        current_row += 1

    # RSI
    if 'RSI' in indicators_to_show:
        fig.add_trace(go.Scatter(x=data['rsi'].index, y=data['rsi'], name='RSI',
                                 line=dict(color='#667eea', width=2.5)), row=current_row, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,51,102,0.1)", layer="below",
                      line_width=0, row=current_row, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,255,136,0.1)", layer="below",
                      line_width=0, row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff3366", opacity=0.6, row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00ff88", opacity=0.6, row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", row=current_row, col=1, range=[0, 100], gridcolor='rgba(128,128,128,0.2)')
        current_row += 1

    # MACD
    if 'MACD' in indicators_to_show:
        fig.add_trace(go.Scatter(x=data['macd'].index, y=data['macd'], name='MACD',
                                 line=dict(color='#667eea', width=2.5)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=data['macd_signal'].index, y=data['macd_signal'], name='Signal',
                                 line=dict(color='#ff3366', width=2.5)), row=current_row, col=1)
        histogram = data['macd_histogram'].dropna()
        colors_macd = ['#00ff88' if val > 0 else '#ff3366' for val in histogram]
        fig.add_trace(go.Bar(x=histogram.index, y=histogram, name='Histogram',
                             marker_color=colors_macd, opacity=0.6), row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_row, col=1, gridcolor='rgba(128,128,128,0.2)')
        current_row += 1

    # Stochastic
    if 'Stochastic' in indicators_to_show:
        fig.add_trace(go.Scatter(x=data['stoch_k'].index, y=data['stoch_k'], name='%K',
                                 line=dict(color='#667eea', width=2.5)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=data['stoch_d'].index, y=data['stoch_d'], name='%D',
                                 line=dict(color='#ff3366', width=2.5)), row=current_row, col=1)
        fig.add_hrect(y0=80, y1=100, fillcolor="rgba(255,51,102,0.1)", layer="below",
                      line_width=0, row=current_row, col=1)
        fig.add_hrect(y0=0, y1=20, fillcolor="rgba(0,255,136,0.1)", layer="below",
                      line_width=0, row=current_row, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="#ff3366", opacity=0.6, row=current_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="#00ff88", opacity=0.6, row=current_row, col=1)
        fig.update_yaxes(title_text="Stochastic", row=current_row, col=1, range=[0, 100],
                         gridcolor='rgba(128,128,128,0.2)')

    # Layout
    fig.update_layout(
        title=dict(text=f'{ticker} Technical Analysis with Pattern Recognition',
                   font=dict(size=24, color='#ffffff', family='Arial Black'), x=0.5, xanchor='center'),
        xaxis_rangeslider_visible=False,
        height=220 * rows,
        showlegend=True,
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='#ffffff', size=12),
        legend=dict(bgcolor='rgba(30,30,46,0.8)', bordercolor='#667eea', borderwidth=2, font=dict(size=11))
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

    return fig


def display_metrics(ticker, indicators_data):
    data = indicators_data[ticker]

    col1, col2, col3, col4, col5 = st.columns(5)

    current_price = data['close'].iloc[-1]
    prev_price = data['close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100

    with col1:
        st.metric(label="💵 Current Price", value=f"${current_price:.2f}",
                  delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)")

    with col2:
        high_52w = data['high'].tail(252).max()
        low_52w = data['low'].tail(252).min()
        st.metric(label="📊 52W High", value=f"${high_52w:.2f}")
        st.markdown(f"<p style='text-align: center; color: #a0a0a0; font-size: 0.85rem;'>52W Low: ${low_52w:.2f}</p>",
                    unsafe_allow_html=True)

    with col3:
        volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].tail(20).mean()
        volume_change = ((volume - avg_volume) / avg_volume) * 100
        st.metric(label="📈 Volume", value=f"{volume / 1e6:.2f}M", delta=f"{volume_change:+.1f}% vs avg")

    with col4:
        vwap = pd.Series(data['vwap']).iloc[-1]
        vwap_diff = ((current_price - vwap) / vwap) * 100
        st.metric(label="🎯 VWAP", value=f"${vwap:.2f}", delta=f"{vwap_diff:+.2f}%")

    with col5:
        atr = pd.Series(data['atr']).dropna().iloc[-1]
        atr_pct = (atr / current_price) * 100
        st.metric(label="⚡ ATR (14)", value=f"${atr:.2f}", delta=f"{atr_pct:.2f}% volatility")


def display_pattern_recognition_results(pattern_data):
    if not pattern_data:
        st.info("Pattern recognition data not available")
        return

    st.subheader("🔍 Pattern Recognition Results")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**📊 Candlestick Patterns (Last 50 bars)**")
        patterns = pattern_data['candlestick_patterns'].tail(50)

        for pat_name in ['Doji', 'Hammer', 'Shooting Star', 'Bullish Engulfing', 'Bearish Engulfing']:
            col_name = pat_name.replace(' ', '_')
            count = patterns[col_name].sum() if col_name in patterns.columns else 0

            if count > 0:
                latest_date = patterns[patterns[col_name]].index[-1]
                st.markdown(f"""
                    <div class="pattern-box">
                        <strong>{pat_name}</strong>: {int(count)} occurrences<br>
                        Latest: {latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date)}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.write(f"✗ {pat_name}: None detected")

    with col2:
        st.write("**📈 Chart Patterns**")
        h_s = pattern_data['head_and_shoulders']
        inv_h_s = pattern_data['inverse_head_and_shoulders']
        st.write(f"**Head and Shoulders (Bearish):** {len(h_s)} pattern(s)")
        st.write(f"**Inverse Head and Shoulders (Bullish):** {len(inv_h_s)} pattern(s)")

    st.write("**🎯 Support and Resistance Levels**")
    col3, col4 = st.columns(2)

    with col3:
        st.write("**Support:**")
        for i, level in enumerate(pattern_data['support_resistance']['support'][:5]):
            st.write(f"S{i + 1}: ${level:.2f}")

    with col4:
        st.write("**Resistance:**")
        for i, level in enumerate(pattern_data['support_resistance']['resistance'][:5]):
            st.write(f"R{i + 1}: ${level:.2f}")


def main():
    st.markdown("""
        <div style='text-align: center; padding: 20px 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; margin-bottom: 30px; box-shadow: 0 8px 16px rgba(0,0,0,0.3);'>
            <h1 style='color: white; font-size: 3rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                📈 Financial Analysis Dashboard
            </h1>
            <p style='color: rgba(255,255,255,0.9); font-size: 1.2rem; margin: 10px 0 0 0;'>
                Advanced Technical Analysis with Pattern Recognition
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.sidebar.header("⚙️ Configuration")

    available_tickers = get_tickers_from_data()
    if not available_tickers:
        st.error("❌ No data loaded. Please ensure data_loader.py has loaded the data.")
        return

    ticker = st.sidebar.selectbox("Select Stock", available_tickers)

    st.sidebar.subheader("📊 Select Indicators")
    indicators = {
        'VWAP': st.sidebar.checkbox('VWAP', value=True),
        'SMA': st.sidebar.checkbox('Moving Averages (SMA)', value=True),
        'EMA': st.sidebar.checkbox('Exponential Moving Averages (EMA)', value=False),
        'Bollinger Bands': st.sidebar.checkbox('Bollinger Bands', value=False),
        'Volume': st.sidebar.checkbox('Volume', value=True),
        'ATR': st.sidebar.checkbox('ATR (Volatility)', value=True),
        'OBV': st.sidebar.checkbox('OBV (On-Balance Volume)', value=True),
        'RSI': st.sidebar.checkbox('RSI', value=True),
        'MACD': st.sidebar.checkbox('MACD', value=True),
        'Stochastic': st.sidebar.checkbox('Stochastic Oscillator', value=False),
    }

    selected_indicators = [k for k, v in indicators.items() if v]

    st.sidebar.subheader("🔍 Pattern Recognition")
    show_patterns = st.sidebar.checkbox('Show Candlestick Patterns', value=True)
    show_sr_levels = st.sidebar.checkbox('Show Support/Resistance', value=True)
    show_trend_lines = st.sidebar.checkbox('Show Trend Lines', value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Technical Analysis", "Risk-Return Metrics"],
        index=0
    )

    if st.sidebar.button("🔄 Recalculate All", type="primary"):
        st.cache_data.clear()
        st.rerun()

    with st.spinner('Calculating indicators...'):
        indicators_data = calculate_all_indicators()

    pattern_data = None
    if PATTERN_RECOGNITION_AVAILABLE and (show_patterns or show_sr_levels or show_trend_lines):
        with st.spinner('Detecting patterns...'):
            pattern_data = calculate_pattern_recognition(ticker)

    # ==================== PAGE ROUTING ====================

    if page == "Technical Analysis":
        show_technical_analysis_page(ticker, indicators_data, selected_indicators, pattern_data, available_tickers)
    elif page == "Risk-Return Metrics":
        show_risk_return_page(indicators_data, available_tickers)


def show_technical_analysis_page(ticker, indicators_data, selected_indicators, pattern_data, available_tickers):
    """Display the technical analysis page"""
    if ticker in indicators_data:
        st.subheader(f"📊 {ticker} Key Metrics")
        display_metrics(ticker, indicators_data)
        st.markdown("---")

        if pattern_data and PATTERN_RECOGNITION_AVAILABLE:
            display_pattern_recognition_results(pattern_data)
            st.markdown("---")

        st.subheader(f"📈 {ticker} Price Chart with Indicators & Patterns")
        fig = create_interactive_chart(ticker, indicators_data, selected_indicators, pattern_data)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Detailed Analysis")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Statistics", "🎯 Signals", "📈 Trends", "🔍 Patterns", "💾 Data"])

        data = indicators_data[ticker]

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Price Statistics**")
                stats_df = pd.DataFrame({
                    'Metric': ['Open', 'High', 'Low', 'Close', 'Volume'],
                    'Value': [
                        f"${data['open'].iloc[-1]:.2f}",
                        f"${data['high'].iloc[-1]:.2f}",
                        f"${data['low'].iloc[-1]:.2f}",
                        f"${data['close'].iloc[-1]:.2f}",
                        f"{data['volume'].iloc[-1]:,.0f}"
                    ]
                })
                st.dataframe(stats_df, hide_index=True, use_container_width=True)

            with col2:
                st.write("**Indicator Values**")
                current_values = {'Indicator': [], 'Current Value': []}

                if 'VWAP' in selected_indicators:
                    current_values['Indicator'].append('VWAP')
                    current_values['Current Value'].append(f"${pd.Series(data['vwap']).iloc[-1]:.2f}")

                if 'ATR' in selected_indicators:
                    current_values['Indicator'].append('ATR(14)')
                    current_values['Current Value'].append(f"${pd.Series(data['atr']).dropna().iloc[-1]:.2f}")

                if 'RSI' in selected_indicators:
                    current_values['Indicator'].append('RSI(14)')
                    current_values['Current Value'].append(f"{data['rsi'].dropna().iloc[-1]:.2f}")

                if current_values['Indicator']:
                    st.dataframe(pd.DataFrame(current_values), hide_index=True, use_container_width=True)

        with tab2:
            st.write("**Trading Signals**")

            if 'VWAP' in selected_indicators:
                if data['close'].iloc[-1] > pd.Series(data['vwap']).iloc[-1]:
                    st.success("✅ Price is ABOVE VWAP - Bullish")
                else:
                    st.error("⚠️ Price is BELOW VWAP - Bearish")

            if 'RSI' in selected_indicators:
                rsi_val = data['rsi'].dropna().iloc[-1]
                if rsi_val > 70:
                    st.warning(f"⚠️ RSI: {rsi_val:.2f} - Overbought")
                elif rsi_val < 30:
                    st.info(f"💡 RSI: {rsi_val:.2f} - Oversold")
                else:
                    st.success(f"✅ RSI: {rsi_val:.2f} - Neutral")

            if 'MACD' in selected_indicators:
                macd_val = data['macd'].dropna().iloc[-1]
                signal_val = data['macd_signal'].dropna().iloc[-1]
                if macd_val > signal_val:
                    st.success(f"✅ MACD: Bullish Crossover")
                else:
                    st.error(f"⚠️ MACD: Bearish Crossover")

            if 'OBV' in selected_indicators:
                if pd.Series(data['obv_ma']).iloc[-1] > pd.Series(data['obv_ma']).iloc[-20]:
                    st.success("✅ OBV Trend: Rising (Accumulation)")
                else:
                    st.error("⚠️ OBV Trend: Falling (Distribution)")

            if 'Stochastic' in selected_indicators:
                stoch_k = data['stoch_k'].dropna().iloc[-1]
                if stoch_k > 80:
                    st.warning(f"⚠️ Stochastic %K: {stoch_k:.2f} - Overbought")
                elif stoch_k < 20:
                    st.info(f"💡 Stochastic %K: {stoch_k:.2f} - Oversold")
                else:
                    st.success(f"✅ Stochastic %K: {stoch_k:.2f} - Neutral")

            if pattern_data and PATTERN_RECOGNITION_AVAILABLE:
                st.markdown("---")
                st.write("**Pattern-Based Signals**")
                patterns = pattern_data['candlestick_patterns'].tail(5)

                if patterns['Hammer'].any():
                    st.success("🔨 Hammer pattern detected - Potential bullish reversal")
                if patterns['Shooting_Star'].any():
                    st.warning("⭐ Shooting Star detected - Potential bearish reversal")
                if patterns['Bullish_Engulfing'].any():
                    st.success("📈 Bullish Engulfing - Strong buy signal")
                if patterns['Bearish_Engulfing'].any():
                    st.error("📉 Bearish Engulfing - Strong sell signal")

        with tab3:
            st.write("**Trend Analysis**")

            if 'SMA' in selected_indicators:
                sma_20 = data['sma_20'].dropna().iloc[-1]
                sma_50 = data['sma_50'].dropna().iloc[-1]
                current_price = data['close'].iloc[-1]

                if current_price > sma_20 > sma_50:
                    st.success("📈 Strong Uptrend: Price > SMA(20) > SMA(50)")
                elif current_price < sma_20 < sma_50:
                    st.error("📉 Strong Downtrend: Price < SMA(20) < SMA(50)")
                else:
                    st.info("↔️ Mixed/Ranging Market")

            if 'ATR' in selected_indicators:
                atr_current = pd.Series(data['atr']).dropna().iloc[-1]
                atr_mean = pd.Series(data['atr']).dropna().mean()

                if atr_current > atr_mean * 1.5:
                    st.warning("⚡ High Volatility Detected")
                elif atr_current < atr_mean * 0.7:
                    st.info("😴 Low Volatility Period")
                else:
                    st.success("✅ Normal Volatility")

            if pattern_data and PATTERN_RECOGNITION_AVAILABLE:
                st.markdown("---")
                st.write("**Trend Line Analysis**")
                uptrends = pattern_data['trend_lines']['uptrend']
                downtrends = pattern_data['trend_lines']['downtrend']

                if len(uptrends) > len(downtrends):
                    st.success(f"📈 More uptrend lines ({len(uptrends)} vs {len(downtrends)}) - Bullish bias")
                elif len(downtrends) > len(uptrends):
                    st.error(f"📉 More downtrend lines ({len(downtrends)} vs {len(uptrends)}) - Bearish bias")
                else:
                    st.info("↔️ Equal trend lines - Neutral/Consolidation")

        with tab4:
            if pattern_data and PATTERN_RECOGNITION_AVAILABLE:
                st.write("**Comprehensive Pattern Analysis**")

                col1, col2, col3 = st.columns(3)

                with col1:
                    total_candle = sum([
                        pattern_data['candlestick_patterns']['Doji'].sum(),
                        pattern_data['candlestick_patterns']['Hammer'].sum(),
                        pattern_data['candlestick_patterns']['Shooting_Star'].sum(),
                        pattern_data['candlestick_patterns']['Bullish_Engulfing'].sum(),
                        pattern_data['candlestick_patterns']['Bearish_Engulfing'].sum()
                    ])
                    st.metric("Candlestick Patterns", int(total_candle))

                with col2:
                    total_chart = len(pattern_data['head_and_shoulders']) + len(
                        pattern_data['inverse_head_and_shoulders'])
                    st.metric("Chart Patterns", total_chart)

                with col3:
                    total_sr = len(pattern_data['support_resistance']['support']) + len(
                        pattern_data['support_resistance']['resistance'])
                    st.metric("S/R Levels", total_sr)

                st.markdown("---")
                st.write("**Recent Candlestick Patterns (Last 20 bars)**")
                recent_patterns = pattern_data['candlestick_patterns'].tail(20)
                pattern_summary = []

                for idx in recent_patterns.index:
                    patterns_found = []
                    for pat in ['Doji', 'Hammer', 'Shooting_Star', 'Bullish_Engulfing', 'Bearish_Engulfing']:
                        if recent_patterns.loc[idx, pat]:
                            patterns_found.append(pat.replace('_', ' '))

                    if patterns_found:
                        pattern_summary.append({
                            'Date': idx,
                            'Patterns': ', '.join(patterns_found),
                            'Price': f"${data['close'].loc[idx]:.2f}"
                        })

                if pattern_summary:
                    st.dataframe(pd.DataFrame(pattern_summary), use_container_width=True)
                else:
                    st.info("No patterns detected in recent bars")
            else:
                st.info(
                    "Pattern recognition not available. Please ensure pattern_recognition.py is in the same directory.")

        with tab5:
            st.write("**Raw Data (Last 50 rows)**")

            display_df = pd.DataFrame({
                'Date': data['close'].index,
                'Open': data['open'].values,
                'High': data['high'].values,
                'Low': data['low'].values,
                'Close': data['close'].values,
                'Volume': data['volume'].values
            })

            st.dataframe(display_df.tail(50), use_container_width=True)

            csv = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Data as CSV",
                data=csv,
                file_name=f'{ticker}_data.csv',
                mime='text/csv',
            )

    else:
        st.error("❌ Unable to load indicator data for selected ticker.")

    st.markdown("<br><br>", unsafe_allow_html=True)


def show_risk_return_page(indicators_data, available_tickers):
    """Display the risk-return analysis page"""

    st.subheader("📊 Risk-Return Metrics & Analysis")

    # Import financial metrics
    try:
        from analysis import FinancialMetrics
    except ImportError:
        st.error("❌ financial_metrics.py not found. Please ensure it's in the same directory.")
        return

    # Prepare price data
    price_data = pd.DataFrame()
    for ticker in available_tickers:
        price_data[ticker] = indicators_data[ticker]['close']

    # Risk-free rate input
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.1
        ) / 100

    with col2:
        rolling_window = st.number_input(
            "Rolling Window (days)",
            min_value=10,
            max_value=90,
            value=30,
            step=5
        )

    # Calculate metrics
    with st.spinner('Calculating financial metrics...'):
        fm = FinancialMetrics(price_data, risk_free_rate)
        summary_stats = fm.get_summary_statistics()

    # Performance Cards
    st.markdown("### 📈 Performance Summary")
    cols = st.columns(len(available_tickers))
    for i, ticker in enumerate(available_tickers):
        with cols[i]:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; color: white; margin: 10px 0;'>
                    <h3 style='margin: 0 0 10px 0;'>{ticker}</h3>
                    <p style='margin: 5px 0;'><strong>Return:</strong> {summary_stats.loc[ticker, 'Annualized Return'] * 100:.2f}%</p>
                    <p style='margin: 5px 0;'><strong>Volatility:</strong> {summary_stats.loc[ticker, 'Volatility (Annual)'] * 100:.2f}%</p>
                    <p style='margin: 5px 0;'><strong>Sharpe:</strong> {summary_stats.loc[ticker, 'Sharpe Ratio']:.2f}</p>
                    <p style='margin: 5px 0;'><strong>Max DD:</strong> {summary_stats.loc[ticker, 'Max Drawdown'] * 100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Risk-Return Profile",
        "📈 Performance Charts",
        "🔗 Correlation Analysis",
        "📉 Risk Metrics",
        "📋 Detailed Statistics"
    ])

    with tab1:
        st.markdown("### 🎯 Risk-Return Scatter Plot")
        st.write("Bubble size represents Sharpe Ratio - larger bubbles indicate better risk-adjusted returns")

        # Risk-Return Scatter
        returns = fm.get_annualized_returns()
        volatility = fm.get_volatility(annualized=True)
        sharpe = fm.get_sharpe_ratio()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=volatility * 100,
            y=returns * 100,
            mode='markers+text',
            marker=dict(
                size=abs(sharpe) * 30,
                color=sharpe,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio"),
                line=dict(width=2, color='white')
            ),
            text=returns.index,
            textposition="top center",
            textfont=dict(size=14, color='white', family='Arial Black'),
            hovertemplate='<b>%{text}</b><br>' +
                          'Return: %{y:.2f}%<br>' +
                          'Volatility: %{x:.2f}%<br>' +
                          '<extra></extra>'
        ))

        fig.update_layout(
            xaxis_title='Annualized Volatility (%)',
            yaxis_title='Annualized Return (%)',
            template='plotly_dark',
            hovermode='closest',
            height=600,
            showlegend=False,
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Ranking table
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**🏆 Best Sharpe Ratio**")
            best_sharpe = sharpe.sort_values(ascending=False)
            for i, (ticker, value) in enumerate(best_sharpe.items(), 1):
                st.write(f"{i}. {ticker}: {value:.2f}")

        with col2:
            st.write("**📈 Highest Returns**")
            best_returns = returns.sort_values(ascending=False)
            for i, (ticker, value) in enumerate(best_returns.items(), 1):
                st.write(f"{i}. {ticker}: {value * 100:.2f}%")

        with col3:
            st.write("**🛡️ Lowest Volatility**")
            lowest_vol = volatility.sort_values(ascending=True)
            for i, (ticker, value) in enumerate(lowest_vol.items(), 1):
                st.write(f"{i}. {ticker}: {value * 100:.2f}%")

    with tab2:
        # Cumulative Returns
        st.markdown("### 📊 Cumulative Returns")
        cum_returns = fm.get_cumulative_returns()

        fig = go.Figure()
        colors = ['#00ff88', '#667eea', '#ff3366', '#ffa500', '#00d4ff']

        for i, ticker in enumerate(cum_returns.columns):
            fig.add_trace(go.Scatter(
                x=cum_returns.index,
                y=cum_returns[ticker] * 100,
                name=ticker,
                line=dict(width=3, color=colors[i % len(colors)]),
                hovertemplate='<b>%{fullData.name}</b><br>Return: %{y:.2f}%<extra></extra>'
            ))

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            template='plotly_dark',
            hovermode='x unified',
            height=500,
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='white', borderwidth=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Drawdown Chart
        st.markdown("### 📉 Drawdown Analysis")

        fig = go.Figure()

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
                hovertemplate='<b>%{fullData.name}</b><br>Drawdown: %{y:.2f}%<extra></extra>'
            ))

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_dark',
            height=500,
            hovermode='x unified',
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='white', borderwidth=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Rolling Volatility
        st.markdown(f"### 📈 Rolling {rolling_window}-Day Volatility")

        fig = go.Figure()

        for i, ticker in enumerate(fm.daily_returns.columns):
            rolling_vol = fm.daily_returns[ticker].rolling(window=rolling_window).std() * np.sqrt(252) * 100

            fig.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                name=ticker,
                line=dict(width=2, color=colors[i % len(colors)])
            ))

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Volatility (%)',
            template='plotly_dark',
            height=400,
            hovermode='x unified',
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='white', borderwidth=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔗 Correlation Matrix")
            corr_matrix = fm.get_correlation_matrix()

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 14, "color": "white"},
                colorbar=dict(title="Correlation")
            ))

            fig.update_layout(
                template='plotly_dark',
                height=500,
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117',
                xaxis=dict(side='bottom'),
                yaxis=dict(autorange='reversed')
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(f"### 📊 Rolling {rolling_window}-Day Correlation")

            if len(available_tickers) >= 2:
                ticker1 = st.selectbox("Select First Stock", available_tickers, index=0, key="corr1")
                ticker2 = st.selectbox("Select Second Stock", available_tickers, index=1, key="corr2")

                rolling_corr = fm.get_rolling_correlation(ticker1, ticker2, rolling_window)

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
                    xaxis_title='Date',
                    yaxis_title='Correlation',
                    template='plotly_dark',
                    height=400,
                    yaxis=dict(range=[-1, 1]),
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#0e1117'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Correlation interpretation
                current_corr = rolling_corr.dropna().iloc[-1]
                if current_corr > 0.7:
                    st.success(f"✅ Strong positive correlation ({current_corr:.2f}) - stocks move together")
                elif current_corr < -0.7:
                    st.error(f"⚠️ Strong negative correlation ({current_corr:.2f}) - stocks move opposite")
                elif abs(current_corr) < 0.3:
                    st.info(f"ℹ️ Weak correlation ({current_corr:.2f}) - stocks move independently")
                else:
                    st.warning(f"📊 Moderate correlation ({current_corr:.2f})")

    with tab4:
        st.markdown("### 📊 Returns Distribution & Value at Risk")

        # Create distribution plots
        num_stocks = len(fm.daily_returns.columns)
        rows = (num_stocks + 1) // 2
        cols = 2

        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"{ticker} Daily Returns" for ticker in fm.daily_returns.columns]
        )

        colors = ['#00ff88', '#667eea', '#ff3366', '#ffa500']

        for i, ticker in enumerate(fm.daily_returns.columns):
            row = i // cols + 1
            col = i % cols + 1

            returns = fm.daily_returns[ticker] * 100

            fig.add_trace(
                go.Histogram(
                    x=returns,
                    name=ticker,
                    marker_color=colors[i % len(colors)],
                    opacity=0.7,
                    nbinsx=50,
                    showlegend=False
                ),
                row=row, col=col
            )

            # Add VaR line
            var_95 = fm.get_value_at_risk(0.95)[ticker] * 100
            fig.add_vline(
                x=var_95,
                line_dash="dash",
                line_color="red",
                line_width=2,
                row=row, col=col,
                annotation_text=f"VaR: {var_95:.2f}%",
                annotation=dict(font=dict(size=10, color="red"))
            )

        fig.update_layout(
            template='plotly_dark',
            height=300 * rows,
            showlegend=False,
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117'
        )

        fig.update_xaxes(title_text="Daily Return (%)")
        fig.update_yaxes(title_text="Frequency")

        st.plotly_chart(fig, use_container_width=True)

        # Risk metrics comparison
        st.markdown("### 📊 Risk Metrics Comparison")

        col1, col2 = st.columns(2)

        with col1:
            # VaR and CVaR
            var_data = pd.DataFrame({
                'VaR (95%)': fm.get_value_at_risk(0.95) * 100,
                'CVaR (95%)': fm.get_conditional_var(0.95) * 100
            })

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=var_data.index,
                y=var_data['VaR (95%)'],
                name='VaR (95%)',
                marker_color='#ff3366'
            ))

            fig.add_trace(go.Bar(
                x=var_data.index,
                y=var_data['CVaR (95%)'],
                name='CVaR (95%)',
                marker_color='#cc0033'
            ))

            fig.update_layout(
                title='Value at Risk Comparison',
                yaxis_title='Loss (%)',
                template='plotly_dark',
                height=400,
                barmode='group',
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117'
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Max Drawdown
            max_dd = pd.Series(fm.get_max_drawdown()) * 100

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=max_dd.index,
                y=abs(max_dd),
                marker_color='#667eea',
                text=max_dd.apply(lambda x: f"{x:.1f}%"),
                textposition='outside'
            ))

            fig.update_layout(
                title='Maximum Drawdown',
                yaxis_title='Drawdown (%)',
                template='plotly_dark',
                height=400,
                showlegend=False,
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117'
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown("### 📋 Comprehensive Statistics")

        # Format display
        display_stats = summary_stats.copy()

        # Format percentage columns
        pct_cols = ['Total Return', 'Annualized Return', 'Daily Avg Return',
                    'Volatility (Annual)', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)']
        for col in pct_cols:
            if col in display_stats.columns:
                display_stats[col] = display_stats[col].apply(lambda x: f"{x * 100:.2f}%")

        # Format ratio columns
        ratio_cols = ['Sharpe Ratio', 'Sortino Ratio', 'Beta', 'Skewness', 'Kurtosis']
        for col in ratio_cols:
            if col in display_stats.columns:
                display_stats[col] = display_stats[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

        st.dataframe(display_stats, use_container_width=True, height=400)

        # Download button
        csv = summary_stats.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Statistics as CSV",
            data=csv,
            file_name="financial_metrics.csv",
            mime="text/csv"
        )

        # Additional metrics explanation
        with st.expander("📖 Metrics Explanation"):
            st.markdown("""
            **Returns Metrics:**
            - **Total Return**: Overall percentage gain/loss
            - **Annualized Return**: Average yearly return
            - **Daily Avg Return**: Average daily percentage change

            **Risk Metrics:**
            - **Volatility**: Standard deviation of returns (higher = more risky)
            - **Sharpe Ratio**: Risk-adjusted return (>1 is good, >2 is excellent)
            - **Sortino Ratio**: Like Sharpe but only considers downside volatility
            - **Beta**: Sensitivity to market movements (1 = moves with market)
            - **Max Drawdown**: Largest peak-to-trough decline
            - **VaR (95%)**: Maximum expected loss with 95% confidence
            - **CVaR (95%)**: Average loss beyond VaR threshold

            **Distribution Metrics:**
            - **Skewness**: Asymmetry of returns (negative = more downside risk)
            - **Kurtosis**: Tail risk (higher = more extreme moves)
            """)


if __name__ == "__main__":
    main()