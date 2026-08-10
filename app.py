import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="PSX Quant Engine", layout="wide", page_icon="⚡")

# Custom CSS for UI Polish
st.markdown("""
    <style>
    .ticker-header { font-size: 26px; font-weight: 700; color: #1E293B; margin-bottom: 0px; }
    .price-subhead { font-size: 14px; color: #64748B; margin-bottom: 15px; }
    .big-signal-buy { font-size: 22px; font-weight: bold; color: #10B981; background-color: #ECFDF5; padding: 10px 18px; border-radius: 8px; border: 1px solid #10B981; text-align: center; }
    .big-signal-wait { font-size: 22px; font-weight: bold; color: #F59E0B; background-color: #FFFBEB; padding: 10px 18px; border-radius: 8px; border: 1px solid #F59E0B; text-align: center; }
    .big-signal-avoid { font-size: 22px; font-weight: bold; color: #EF4444; background-color: #FEF2F2; padding: 10px 18px; border-radius: 8px; border: 1px solid #EF4444; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. UNTOUCHED BACKEND CALCULATIONS ENGINE
# ==========================================
def process_data(df):
    if len(df) < 20:
        return df

    # A. Moving Averages
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean() if len(df) >= 200 else np.nan
    
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # B. Dynamic ATR (14-period)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # C. RSI (14-period)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-1 * delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # D. MACD Calculation
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']

    # E. Wilder ADX & Directional Indicators
    up_move = df['High'].diff()
    down_move = df['Low'].shift() - df['Low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    alpha = 1 / 14
    atr_wilder = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr_wilder)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr_wilder)
    
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    df['ADX'] = dx.ewm(alpha=alpha, adjust=False).mean()

    # F. Relative Volume (RVOL)
    df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
    df['RVOL'] = df['Volume'] / df['Vol_SMA20']

    # G. Multi-Period Support & Resistance
    df['Resistance_20'] = df['High'].shift(1).rolling(20).max()
    df['Support_20'] = df['Low'].shift(1).rolling(20).min()

    return df

# ==========================================
# 2. UNTOUCHED QUANT DECISION ENGINE
# ==========================================
def generate_quant_decision(df, atr_buffer_mult=0.15):
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    price = float(latest['Close'])
    prev_close = float(prev['Close'])
    pdh = float(prev['High'])
    pdl = float(prev['Low'])
    atr = float(latest['ATR']) if pd.notnull(latest['ATR']) and latest['ATR'] > 0 else (price * 0.02)
    rsi = float(latest['RSI']) if pd.notnull(latest['RSI']) else 50.0
    rvol = float(latest['RVOL']) if pd.notnull(latest['RVOL']) else 1.0
    adx = float(latest['ADX']) if pd.notnull(latest['ADX']) else 15.0
    plus_di = float(latest['Plus_DI']) if pd.notnull(latest['Plus_DI']) else 0.0
    minus_di = float(latest['Minus_DI']) if pd.notnull(latest['Minus_DI']) else 0.0
    
    resistance = float(latest['Resistance_20']) if pd.notnull(latest['Resistance_20']) else float(df['High'].max())
    support = float(latest['Support_20']) if pd.notnull(latest['Support_20']) else float(df['Low'].min())

    sma20 = latest['SMA_20']
    sma50 = latest['SMA_50']
    sma200 = latest['SMA_200']
    ema20 = latest['EMA_20']

    # Trend Status
    if pd.notnull(sma20) and pd.notnull(sma50):
        if price > sma20 and sma20 >= sma50:
            trend_status = "Strong Bullish"
        elif price > sma20 and sma20 < sma50:
            trend_status = "Moderate Bullish"
        elif price < sma20 and sma20 < sma50:
            trend_status = "Bearish"
        else:
            trend_status = "Neutral"
    else:
        trend_status = "Neutral"

    is_directional_bullish = plus_di > minus_di

    # Breakout Verification
    atr_buffer = atr_buffer_mult * atr
    is_price_above_res_buffer = price > (resistance + atr_buffer)
    is_vol_confirmed = rvol >= 1.2
    is_trend_confirmed = "Bullish" in trend_status
    is_adx_confirmed = (adx >= 20) and is_directional_bullish
    is_above_sma20 = pd.notnull(sma20) and (price > sma20)

    is_false_breakout = (latest['High'] > resistance) and (price < resistance)

    if is_price_above_res_buffer and is_vol_confirmed and is_above_sma20 and is_trend_confirmed and is_adx_confirmed:
        breakout_status = "CONFIRMED BREAKOUT"
    elif is_price_above_res_buffer and not is_vol_confirmed:
        breakout_status = "WEAK BREAKOUT"
    elif price >= resistance and (rvol >= 1.0 or adx >= 20):
        breakout_status = "BREAKOUT READY"
    elif is_false_breakout:
        breakout_status = "FALSE BREAKOUT RISK"
    else:
        breakout_status = "NO BREAKOUT"

    # Pullback Verification
    near_support_zone = (pd.notnull(ema20) and abs(price - ema20) / price <= 0.018) or (abs(price - support) / price <= 0.02)
    is_rejection_bounce = (latest['Close'] > latest['Open']) or (price > float(prev['Low']))
    macd_not_deteriorating = latest['MACD_Hist'] >= prev['MACD_Hist']

    reasons = []

    if breakout_status in ["CONFIRMED BREAKOUT", "BREAKOUT READY"]:
        setup_type = "BREAKOUT"
        main_signal = "BUY"
        reasons.append(f"Strong price breakout above resistance level ({resistance:.2f}).")
        reasons.append(f"High volume ({rvol:.1f}x normal) and momentum support the move.")
    elif is_trend_confirmed and near_support_zone and is_rejection_bounce and rsi <= 58 and macd_not_deteriorating:
        setup_type = "PULLBACK"
        main_signal = "BUY"
        reasons.append(f"Price pulled back to key support ({support:.2f}) in a primary uptrend.")
        reasons.append("Recent candle shows price bounce/recovery confirmation.")
    elif is_trend_confirmed and latest['MACD_Hist'] > 0 and rvol > 1.0 and is_directional_bullish:
        setup_type = "MOMENTUM"
        main_signal = "BUY"
        reasons.append("Positive momentum with increasing volume above key moving averages.")
    else:
        setup_type = "NO SETUP"
        if is_false_breakout:
            main_signal = "AVOID"
            reasons.append("False Breakout Risk: Price crossed resistance intraday but closed below it.")
        elif breakout_status == "WEAK BREAKOUT":
            main_signal = "WAIT"
            reasons.append("Weak Breakout: Price crossed resistance but lacks volume confirmation.")
        elif "Bearish" in trend_status:
            main_signal = "AVOID"
            reasons.append("Stock is in a downtrend. Avoid buying against the overall trend.")
        else:
            main_signal = "WAIT"
            reasons.append("Market is consolidating. No low-risk buy entry confirmed right now.")

    # Conditional Execution Levels
    if setup_type == "BREAKOUT":
        conditional_buy = round(max(resistance, pdh) + atr_buffer, 2)
        stop_loss = round(resistance - (0.8 * atr), 2)
        target1 = round(conditional_buy + (1.5 * atr), 2)
        target2 = round(conditional_buy + (3.0 * atr), 2)
    elif setup_type == "PULLBACK":
        conditional_buy = round(price + (0.2 * atr), 2)
        structural_support = min(ema20, support) if pd.notnull(ema20) else support
        stop_loss = round(structural_support - (0.5 * atr), 2)
        target1 = round(conditional_buy + (1.8 * atr), 2)
        target2 = round(conditional_buy + (3.2 * atr), 2)
    elif setup_type == "MOMENTUM":
        conditional_buy = round(pdh + atr_buffer, 2)
        stop_loss = round(price - (1.2 * atr), 2)
        target1 = round(conditional_buy + (1.5 * atr), 2)
        target2 = round(conditional_buy + (2.8 * atr), 2)
    else:
        conditional_buy = round(pdh + atr_buffer, 2)
        stop_loss = round(price - (1.5 * atr), 2)
        target1 = round(price + (1.5 * atr), 2)
        target2 = round(price + (2.5 * atr), 2)

    # Enforce SL below Conditional Buy
    if stop_loss >= conditional_buy:
        stop_loss = round(conditional_buy - (1.2 * atr), 2)

    # Risk / Reward Filter
    risk_per_share = max(conditional_buy - stop_loss, 0.01)
    rr_target1 = round((target1 - conditional_buy) / risk_per_share, 2)
    rr_target2 = round((target2 - conditional_buy) / risk_per_share, 2)

    if rr_target1 < 1.5 and main_signal == "BUY":
        main_signal = "WAIT"
        reasons.append("Trade invalidated because Risk-to-Reward ratio is below 1:1.5.")

    # Technical Score Factors for Advanced Section
    score = 0
    factors = []
    if "Bullish" in trend_status:
        score += 25
        factors.append({"Factor": "Trend Alignment", "Points": "+25", "Details": trend_status})
    else:
        factors.append({"Factor": "Trend Alignment", "Points": "+0", "Details": trend_status})

    if is_adx_confirmed:
        score += 15
        factors.append({"Factor": "ADX Strength", "Points": "+15", "Details": f"ADX={adx:.1f}, +DI > -DI"})
    else:
        factors.append({"Factor": "ADX Strength", "Points": "+0", "Details": f"ADX={adx:.1f}"})

    if latest['MACD_Line'] > latest['MACD_Signal']:
        score += 20
        factors.append({"Factor": "MACD Crossover", "Points": "+20", "Details": "Bullish"})
    else:
        factors.append({"Factor": "MACD Crossover", "Points": "+0", "Details": "Bearish / Neutral"})

    if rvol >= 1.2:
        score += 20
        factors.append({"Factor": "Volume Surge", "Points": "+20", "Details": f"{rvol:.1f}x Volume"})
    else:
        factors.append({"Factor": "Volume Surge", "Points": "+0", "Details": f"{rvol:.1f}x Volume"})

    if 40 <= rsi <= 65:
        score += 20
        factors.append({"Factor": "RSI Level", "Points": "+20", "Details": f"Healthy ({rsi:.1f})"})
    else:
        factors.append({"Factor": "RSI Level", "Points": "+0", "Details": f"Extreme ({rsi:.1f})"})

    return {
        "Price": price,
        "Prev Close": prev_close,
        "PDH": pdh,
        "PDL": pdl,
        "Resistance": resistance,
        "Support": support,
        "RVOL": rvol,
        "RSI": rsi,
        "ADX": adx,
        "Plus_DI": plus_di,
        "Minus_DI": minus_di,
        "SMA20": sma20,
        "SMA50": sma50,
        "SMA200": sma200,
        "EMA9": latest['EMA_9'],
        "EMA20": ema20,
        "MACD_Line": latest['MACD_Line'],
        "MACD_Signal": latest['MACD_Signal'],
        "MACD_Hist": latest['MACD_Hist'],
        "Trend Status": trend_status,
        "Main Signal": main_signal,
        "Setup Type": setup_type,
        "Technical Score": score,
        "Conditional Buy": conditional_buy,
        "Stop Loss": stop_loss,
        "Target 1": target1,
        "Target 2": target2,
        "RR Target 1": rr_target1,
        "RR Target 2": rr_target2,
        "Reasons": reasons,
        "Factors": factors,
        "Risk Per Share": risk_per_share
    }

# ==========================================
# 3. POLISHED STREAMLIT USER INTERFACE
# ==========================================
st.sidebar.header("🔍 Stock & Account Setup")
symbol_input = st.sidebar.text_input("PSX Ticker", value="SYS").strip().upper()
trading_capital = st.sidebar.number_input("Capital (PKR)", value=100000, step=10000)
risk_pct = st.sidebar.number_input("Risk Per Trade (%)", value=1.0, step=0.25, max_value=5.0)

if symbol_input:
    data = yf.Ticker(f"{symbol_input}.KA").history(period="1y")
    
    if data.empty or len(data) < 20:
        st.error(f"⚠️ Insufficient data for '{symbol_input}'. Please check the ticker.")
    else:
        df = process_data(data)
        q = generate_quant_decision(df)

        # 1. STOCK / CURRENT PRICE (Polished Heading Size)
        top_col1, top_col2 = st.columns([2.5, 1.5])
        with top_col1:
            st.markdown(f'<div class="ticker-header">{symbol_input} — PKR {q["Price"]:.2f}</div>', unsafe_allow_html=True)
            change = q['Price'] - q['Prev Close']
            change_pct = (change / q['Prev Close']) * 100
            st.markdown(f'<div class="price-subhead">Prev Close: PKR {q["Prev Close"]:.2f} | Change: {change:+.2f} ({change_pct:+.2f}%)</div>', unsafe_allow_html=True)

        # 2. MAIN SIGNAL
        with top_col2:
            if q['Main Signal'] == "BUY":
                st.markdown(f'<div class="big-signal-buy">🟢 SIGNAL: BUY ({q["Setup Type"]})</div>', unsafe_allow_html=True)
            elif q['Main Signal'] == "WAIT":
                st.markdown('<div class="big-signal-wait">🟡 SIGNAL: WAIT</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="big-signal-avoid">🔴 SIGNAL: AVOID</div>', unsafe_allow_html=True)

        st.markdown("---")

        # 3. SIMPLE REASON FOR SIGNAL & 4. TRADE PLAN
        col_reason, col_plan = st.columns([1, 1])

        with col_reason:
            st.subheader("💡 Why this Signal?")
            for idx, r in enumerate(q['Reasons'], 1):
                st.write(f"**{idx}.** {r}")

        with col_plan:
            st.subheader("🎯 Trade Plan")
            tp1, tp2 = st.columns(2)
            tp1.metric("Conditional Buy Above", f"PKR {q['Conditional Buy']:.2f}")
            tp2.metric("Stop Loss", f"PKR {q['Stop Loss']:.2f}")
            
            tp3, tp4, tp5 = st.columns(3)
            tp3.metric("Target 1", f"PKR {q['Target 1']:.2f}")
            tp4.metric("Target 2", f"PKR {q['Target 2']:.2f}")
            tp5.metric("Risk / Reward", f"1 : {q['RR Target 1']}")

        st.markdown("---")

        # 5. MARKET STATUS & 6. POSITION SIZE
        col_status, col_pos = st.columns([1, 1])

        with col_status:
            st.subheader("📊 Market Status")
            ms1, ms2 = st.columns(2)
            ms1.metric("Trend", q['Trend Status'])
            ms2.metric("Volume", "High Volume" if q['RVOL'] >= 1.2 else "Normal Volume")
            
            ms3, ms4 = st.columns(2)
            ms3.metric("Support", f"PKR {q['Support']:.2f}")
            ms4.metric("Resistance", f"PKR {q['Resistance']:.2f}")

        with col_pos:
            st.subheader("🧮 Position Size")
            max_rupee_risk = (trading_capital * risk_pct) / 100.0
            qty = int(max_rupee_risk / q['Risk Per Share']) if q['Risk Per Share'] > 0 else 0
            investment = qty * q['Conditional Buy']

            ps1, ps2, ps3 = st.columns(3)
            ps1.metric("Shares to Buy", f"{qty:,}")
            ps2.metric("Investment Value", f"PKR {investment:,.0f}")
            ps3.metric("Max Loss Risk", f"PKR {max_rupee_risk:,.0f}")

        st.markdown("---")

        # 7. ONE COMPACT & CLEAN PRICE CHART
        st.subheader("📈 Price Chart")
        
        fig = go.Figure()

        # Candlestick Trace
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"
        ))
        
        # Structural Levels
        fig.add_trace(go.Scatter(x=df.index, y=df['Resistance_20'], mode='lines', name='Resistance', line=dict(color='red', dash='dash', width=1.2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['Support_20'], mode='lines', name='Support', line=dict(color='green', dash='dash', width=1.2)))
        
        # Current Price Highlight Line
        fig.add_hline(y=q['Price'], line_color="#0284C7", line_dash="dot", line_width=1.5, annotation_text=f"Current: {q['Price']:.2f}", annotation_position="top right")

        # Trade Plan Levels Highlighted Only When Relevant
        if q['Main Signal'] == "BUY":
            fig.add_hline(y=q['Conditional Buy'], line_color="#2563EB", line_dash="solid", line_width=1.5, annotation_text=f"Buy Above: {q['Conditional Buy']:.2f}", annotation_position="bottom right")
            fig.add_hline(y=q['Stop Loss'], line_color="#DC2626", line_dash="solid", line_width=1.5, annotation_text=f"SL: {q['Stop Loss']:.2f}", annotation_position="bottom right")
            fig.add_hline(y=q['Target 1'], line_color="#16A34A", line_dash="dashdot", line_width=1.2, annotation_text=f"T1: {q['Target 1']:.2f}", annotation_position="top right")
            fig.add_hline(y=q['Target 2'], line_color="#059669", line_dash="dashdot", line_width=1.2, annotation_text=f"T2: {q['Target 2']:.2f}", annotation_position="top right")

        # Compact Height & Clean Padding Layout
        fig.update_layout(
            height=380,
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            margin=dict(l=10, r=10, t=25, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ADVANCED TECHNICAL ANALYSIS (Collapsed Expander)
        with st.expander("🔬 Advanced Technical Analysis", expanded=False):
            st.write("#### Technical Score Breakdown")
            st.metric("Overall Score", f"{q['Technical Score']} / 100")
            st.table(pd.DataFrame(q['Factors']))

            st.markdown("---")
            st.write("#### Detailed Indicator Metrics")
            ind1, ind2, ind3, ind4 = st.columns(4)
            ind1.metric("RSI (14)", f"{q['RSI']:.1f}")
            ind2.metric("ADX", f"{q['ADX']:.1f}")
            ind3.metric("+DI / -DI", f"{q['Plus_DI']:.1f} / {q['Minus_DI']:.1f}")
            ind4.metric("RVOL", f"{q['RVOL']:.2f}x")

            ind5, ind6, ind7, ind8 = st.columns(4)
            ind5.metric("SMA 20", f"{q['SMA20']:.2f}" if pd.notnull(q['SMA20']) else "N/A")
            ind6.metric("SMA 50", f"{q['SMA50']:.2f}" if pd.notnull(q['SMA50']) else "N/A")
            ind7.metric("EMA 20", f"{q['EMA20']:.2f}" if pd.notnull(q['EMA20']) else "N/A")
            ind8.metric("MACD Hist", f"{q['MACD_Hist']:.2f}")
