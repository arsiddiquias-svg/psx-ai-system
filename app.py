import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="PSX Quant Engine", layout="wide", page_icon="🤖")

st.title("🤖 PSX Technical Signal & Quant Engine")
st.caption("Data-Driven Trade Architecture: Wilder ADX/DI, Structural Stop Loss, Strict Pullback Engine & Multi-Panel Charts")

# ==========================================
# 1. ROBUST INDICATOR CALCULATIONS ENGINE
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

    # E. Standard Wilder ADX Calculation
    up_move = df['High'].diff()
    down_move = df['Low'].shift() - df['Low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Wilder Exponential Smoothing
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

    # G. Multi-Period Support & Resistance (Zero Look-Ahead Bias)
    df['Resistance_20'] = df['High'].shift(1).rolling(20).max()
    df['Support_20'] = df['Low'].shift(1).rolling(20).min()

    return df

# ==========================================
# 2. QUANT DECISION & SETUP ENGINE
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

    # --- A. Trend Classification ---
    if pd.notnull(sma20) and pd.notnull(sma50):
        if price > sma20 and sma20 >= sma50:
            trend_status = "STRONG BULLISH"
        elif price > sma20 and sma20 < sma50:
            trend_status = "MODERATE BULLISH / RECOVERY"
        elif price < sma20 and sma20 < sma50:
            trend_status = "BEARISH"
        else:
            trend_status = "SIDEWAYS / NEUTRAL"
    else:
        trend_status = "NEUTRAL"

    # ADX Strength & Direction Status
    is_directional_bullish = plus_di > minus_di
    if adx >= 25:
        adx_label = f"Strong Trend (ADX={adx:.1f})"
    elif 15 <= adx < 25:
        adx_label = f"Moderate Trend (ADX={adx:.1f})"
    else:
        adx_label = f"Weak Trend (ADX={adx:.1f})"

    # --- B. Breakout Engine ---
    atr_buffer = atr_buffer_mult * atr
    is_price_above_res_buffer = price > (resistance + atr_buffer)
    is_vol_confirmed = rvol >= 1.2
    is_trend_confirmed = "BULLISH" in trend_status
    is_adx_confirmed = (adx >= 20) and is_directional_bullish
    is_above_sma20 = pd.notnull(sma20) and (price > sma20)

    # False Breakout vs Weak Breakout Test
    is_false_breakout = (latest['High'] > resistance) and (price < resistance)

    if is_price_above_res_buffer and is_vol_confirmed and is_above_sma20 and is_trend_confirmed and is_adx_confirmed:
        breakout_status = "CONFIRMED BREAKOUT"
    elif is_price_above_res_buffer and not is_vol_confirmed:
        breakout_status = "WEAK BREAKOUT / WAIT"
    elif price >= resistance and (rvol >= 1.0 or adx >= 20):
        breakout_status = "BREAKOUT READY / WATCH"
    elif is_false_breakout:
        breakout_status = "FALSE BREAKOUT RISK"
    elif latest['High'] >= resistance or price >= resistance:
        breakout_status = "TESTING RESISTANCE"
    else:
        breakout_status = "NO BREAKOUT"

    # --- C. Strict Pullback Engine ---
    near_support_zone = (pd.notnull(ema20) and abs(price - ema20) / price <= 0.018) or (abs(price - support) / price <= 0.02)
    is_rejection_bounce = (latest['Close'] > latest['Open']) or (price > float(prev['Low']))
    macd_not_deteriorating = latest['MACD_Hist'] >= prev['MACD_Hist']

    reasons = []

    if breakout_status in ["CONFIRMED BREAKOUT", "BREAKOUT READY / WATCH"]:
        setup_type = "BREAKOUT"
        reasons.append(f"Price holding above 20D Resistance ({resistance:.2f}) with ATR buffer ({atr_buffer:.2f}).")
        reasons.append(f"RVOL ({rvol:.2f}x) and +DI ({plus_di:.1f}) > -DI ({minus_di:.1f}) confirm bullish momentum.")
    elif is_trend_confirmed and near_support_zone and is_rejection_bounce and rsi <= 58 and macd_not_deteriorating:
        setup_type = "PULLBACK"
        reasons.append(f"Stock in primary uptrend pulled back to key zone near EMA20/Support ({support:.2f}).")
        reasons.append(f"Price bounce confirmation with healthy RSI ({rsi:.1f}) offers favorable risk profile.")
    elif is_trend_confirmed and latest['MACD_Hist'] > 0 and rvol > 1.0 and is_directional_bullish:
        setup_type = "MOMENTUM"
        reasons.append("MACD Histogram expanding with price trading above key short-term EMAs.")
        reasons.append("Directional Index (+DI > -DI) supports trend continuation.")
    else:
        setup_type = "NO TRADE"
        if is_false_breakout:
            reasons.append(f"False Breakout Warning: Intraday high reached {latest['High']:.2f} but price closed below resistance.")
        elif breakout_status == "WEAK BREAKOUT / WAIT":
            reasons.append(f"Weak Breakout: Price crossed resistance but lacks Volume ({rvol:.2f}x < 1.2x) or ADX confirmation.")
        else:
            reasons.append("Structure lacks high-probability breakout or pullback confirmation at current levels.")

    # --- D. Conditional Entry, Target & Structural Stop Loss ---
    if setup_type == "BREAKOUT":
        conditional_buy = round(max(resistance, pdh) + atr_buffer, 2)
        # Structural SL below broken resistance
        stop_loss = round(resistance - (0.8 * atr), 2)
        target1 = round(conditional_buy + (1.5 * atr), 2)
        target2 = round(conditional_buy + (3.0 * atr), 2)
        next_bias = "BULLISH"
        condition_msg = f"BUY ABOVE PKR {conditional_buy} ONLY IF BREAKOUT CONFIRMED WITH VOLUME"

    elif setup_type == "PULLBACK":
        conditional_buy = round(price + (0.2 * atr), 2)
        # Structural SL below nearest key support / EMA20
        structural_support = min(ema20, support) if pd.notnull(ema20) else support
        stop_loss = round(structural_support - (0.5 * atr), 2)
        target1 = round(conditional_buy + (1.8 * atr), 2)
        target2 = round(conditional_buy + (3.2 * atr), 2)
        next_bias = "BULLISH"
        condition_msg = f"BUY ABOVE PKR {conditional_buy} ON RECOVERY BOUNCE CONFIRMATION"

    elif setup_type == "MOMENTUM":
        conditional_buy = round(pdh + atr_buffer, 2)
        stop_loss = round(price - (1.2 * atr), 2)
        target1 = round(conditional_buy + (1.5 * atr), 2)
        target2 = round(conditional_buy + (2.8 * atr), 2)
        next_bias = "BULLISH"
        condition_msg = f"BUY ABOVE PKR {conditional_buy} IF MOMENTUM CONTINUES"

    else:
        conditional_buy = round(pdh + atr_buffer, 2)
        stop_loss = round(price - (1.5 * atr), 2)
        target1 = round(price + (1.5 * atr), 2)
        target2 = round(price + (2.5 * atr), 2)
        next_bias = "NEUTRAL / BEARISH" if is_false_breakout else "NEUTRAL"
        condition_msg = "WAIT / AVOID — NO CLEAR CONDITIONAL TRIGGER"

    # Enforce SL strictly below Conditional Buy
    if stop_loss >= conditional_buy:
        stop_loss = round(conditional_buy - (1.2 * atr), 2)

    # Risk / Reward Filters & Quality Rating
    risk_per_share = max(conditional_buy - stop_loss, 0.01)
    rr_target1 = round((target1 - conditional_buy) / risk_per_share, 2)
    rr_target2 = round((target2 - conditional_buy) / risk_per_share, 2)

    if rr_target1 < 1.5:
        setup_quality = "POOR (R:R < 1.5) — WAIT"
        if setup_type != "NO TRADE":
            condition_msg = "WAIT / AVOID — RISK/REWARD UNATTRACTIVE (< 1:1.5)"
            reasons.append("Trade invalidated due to unattractive Risk-to-Reward ratio on Target 1.")
    elif setup_type in ["BREAKOUT", "PULLBACK"] and rvol >= 1.2:
        setup_quality = "HIGH QUALITY"
    elif setup_type != "NO TRADE":
        setup_quality = "MODERATE QUALITY"
    else:
        setup_quality = "NO SETUP"

    # --- E. Transparent Technical Score Breakdown ---
    score = 0
    factors = []

    if "BULLISH" in trend_status:
        score += 25
        factors.append({"Factor": "Trend Alignment", "Points": "+25", "Details": trend_status})
    else:
        factors.append({"Factor": "Trend Alignment", "Points": "+0", "Details": trend_status})

    if is_adx_confirmed:
        score += 15
        factors.append({"Factor": "ADX & +DI Alignment", "Points": "+15", "Details": f"{adx_label}, +DI > -DI"})
    else:
        factors.append({"Factor": "ADX & +DI Alignment", "Points": "+0", "Details": f"{adx_label}"})

    if latest['MACD_Line'] > latest['MACD_Signal']:
        score += 20
        factors.append({"Factor": "MACD Indicator", "Points": "+20", "Details": "Bullish Crossover"})
    else:
        factors.append({"Factor": "MACD Indicator", "Points": "+0", "Details": "Bearish / Neutral"})

    if rvol >= 1.2:
        score += 20
        factors.append({"Factor": "Relative Volume", "Points": "+20", "Details": f"High RVOL ({rvol:.2f}x)"})
    else:
        factors.append({"Factor": "Relative Volume", "Points": "+0", "Details": f"Normal RVOL ({rvol:.2f}x)"})

    if 40 <= rsi <= 65:
        score += 20
        factors.append({"Factor": "RSI Quality", "Points": "+20", "Details": f"Healthy ({rsi:.1f})"})
    elif rsi < 40:
        score += 10
        factors.append({"Factor": "RSI Quality", "Points": "+10", "Details": f"Oversold ({rsi:.1f})"})
    else:
        factors.append({"Factor": "RSI Quality", "Points": "+0", "Details": f"Overbought ({rsi:.1f})"})

    return {
        "Price": price,
        "Prev Close": prev_close,
        "PDH": pdh,
        "PDL": pdl,
        "Resistance": resistance,
        "Support": support,
        "RVOL": rvol,
        "ADX Label": adx_label,
        "Plus DI": plus_di,
        "Minus DI": minus_di,
        "SMA200": sma200,
        "Trend Status": trend_status,
        "Breakout Status": breakout_status,
        "Setup Type": setup_type,
        "Setup Quality": setup_quality,
        "Next Bias": next_bias,
        "Technical Score": score,
        "Conditional Buy Above": conditional_buy,
        "Stop Loss": stop_loss,
        "Target 1": target1,
        "Target 2": target2,
        "RR Target 1": rr_target1,
        "RR Target 2": rr_target2,
        "Condition Msg": condition_msg,
        "Reasons": reasons,
        "Factors": factors,
        "Risk Per Share": risk_per_share
    }

# ==========================================
# 3. STREAMLIT USER INTERFACE & LAYOUT
# ==========================================
st.sidebar.header("⚙️ Parameters & Position Sizing")
symbol_input = st.sidebar.text_input("Enter PSX Ticker", value="SYS").strip().upper()
period_choice = st.sidebar.selectbox("History Period", ["3mo", "6mo", "1y", "2y"], index=2)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Risk & Position Sizing")
trading_capital = st.sidebar.number_input("Trading Capital (PKR)", value=100000, step=10000)
risk_pct = st.sidebar.number_input("Risk Per Trade (%)", value=1.0, step=0.25, max_value=5.0)
max_allocation_pct = st.sidebar.number_input("Max Capital Allocation (%)", value=25.0, step=5.0, max_value=100.0)

if symbol_input:
    fetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S PKT")
    data = yf.Ticker(f"{symbol_input}.KA").history(period=period_choice)
    
    if data.empty or len(data) < 20:
        st.error(f"⚠️ Insufficient data for '{symbol_input}'. Minimum 20 trading sessions required.")
    else:
        if len(data) < 50:
            st.warning(f"⚠️ **Limited Data Warning:** Only {len(data)} rows loaded. Standard calculations require more history.")
        
        df = process_data(data)
        q = generate_quant_decision(df)

        last_data_date = df.index[-1].strftime("%Y-%m-%d")
        days_diff = (datetime.now() - df.index[-1].tz_localize(None)).days

        if days_diff > 3:
            st.warning(
                f"⚠️ **Data Currency Alert:** Market data last updated on **{last_data_date}** ({days_diff} days old). "
                "Verify session status before placing conditional orders."
            )

        if q['Breakout Status'] == "FALSE BREAKOUT RISK":
            st.error("🚨 **FALSE BREAKOUT RISK:** Price tested resistance intraday but failed to close above it. High probability of bull trap!")

        # --- EXECUTIVE DASHBOARD HEADER ---
        st.markdown(f"### Conditional Next-Session Setup: **:{'green' if q['Next Bias'] == 'BULLISH' else 'orange' if q['Next Bias'] == 'NEUTRAL' else 'red'}[{q['Setup Type']} — {q['Next Bias']}]**")
        st.info(f"📋 **Conditional Trigger Rule:** {q['Condition Msg']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Technical Score", f"{q['Technical Score']} / 100")
        c2.metric("Setup Quality", q['Setup Quality'])
        c3.metric("Breakout Status", q['Breakout Status'])
        c4.metric("Trend Classification", q['Trend Status'])

        if q['Technical Score'] >= 75 and "POOR" in q['Setup Quality']:
            st.caption("ℹ️ **Engine Audit Note:** Strong technical trend, but current entry has poor Risk-to-Reward. Waiting for pullback or confirmed breakout buffer.")

        st.markdown("---")

        # --- CONDITIONAL EXECUTION PARAMETERS ---
        st.subheader("🎯 Conditional Next-Session Execution Parameters")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Conditional Buy Above", f"PKR {q['Conditional Buy Above']}")
        m2.metric("Stop Loss (Strict)", f"PKR {q['Stop Loss']}")
        m3.metric("Target 1", f"PKR {q['Target 1']}", delta=f"1 : {q['RR Target 1']} R:R")
        m4.metric("Target 2", f"PKR {q['Target 2']}", delta=f"1 : {q['RR Target 2']} R:R")
        m5.metric("Relative Vol (RVOL)", f"{q['RVOL']:.2f}x")

        st.markdown("---")

        # --- POSITION SIZING & RISK ALLOCATION ---
        st.subheader("🧮 Position Sizing & Allocation Management")
        max_rupee_risk = (trading_capital * risk_pct) / 100.0
        max_capital_allowed = (trading_capital * max_allocation_pct) / 100.0

        qty_by_risk = int(max_rupee_risk / q['Risk Per Share']) if q['Risk Per Share'] > 0 else 0
        qty_by_cap = int(max_capital_allowed / q['Conditional Buy Above']) if q['Conditional Buy Above'] > 0 else 0
        
        final_qty = min(qty_by_risk, qty_by_cap)
        position_val = round(final_qty * q['Conditional Buy Above'], 2)
        actual_rupee_risk = round(final_qty * q['Risk Per Share'], 2)
        actual_risk_pct = round((actual_rupee_risk / trading_capital) * 100, 2) if trading_capital > 0 else 0.0

        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Max Rupee Risk", f"PKR {max_rupee_risk:,.2f}")
        p2.metric("Risk Per Share", f"PKR {q['Risk Per Share']:.2f}")
        p3.metric("Allowed Shares", f"{final_qty:,}")
        p4.metric("Position Value", f"PKR {position_val:,.2f}")
        p5.metric("Actual Risk %", f"{actual_risk_pct}% ({actual_rupee_risk:,.0f} PKR)")

        st.markdown("---")

        # --- MARKET LEVELS & THESIS ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("📊 Key Market Levels")
            sma200_str = f"{q['SMA200']:.2f}" if pd.notnull(q['SMA200']) else "Insufficient History"
            
            levels_df = pd.DataFrame([
                {"Level": "Current Close Price", "Value (PKR)": f"{q['Price']:.2f}"},
                {"Level": "Previous Close", "Value (PKR)": f"{q['Prev Close']:.2f}"},
                {"Level": "Previous Day High (PDH)", "Value (PKR)": f"{q['PDH']:.2f}"},
                {"Level": "Previous Day Low (PDL)", "Value (PKR)": f"{q['PDL']:.2f}"},
                {"Level": "20-Day Resistance", "Value (PKR)": f"{q['Resistance']:.2f}"},
                {"Level": "20-Day Support", "Value (PKR)": f"{q['Support']:.2f}"},
                {"Level": "200-Day SMA", "Value (PKR)": sma200_str},
            ])
            st.table(levels_df)

        with col_right:
            st.subheader("💡 Technical Trade Thesis")
            for idx, r in enumerate(q['Reasons'], 1):
                st.write(f"**{idx}.** {r}")
            
            with st.expander("🔍 **View Factor Score Audit**"):
                st.table(pd.DataFrame(q['Factors']))

        # --- MULTI-PANEL PLOTLY CHART ---
        st.markdown("---")
        st.subheader("📈 Multi-Indicator Technical Chart Engine")
        
        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True, 
            row_heights=[0.45, 0.20, 0.18, 0.17], 
            vertical_spacing=0.03,
            subplot_titles=(
                "Price, SMAs, EMAs & Support/Resistance", 
                "MACD Indicator (Line, Signal, Histogram)", 
                "Standard Wilder ADX & Directional Indicators (+DI / -DI)",
                "Volume & 20-Period Volume SMA (RVOL)"
            )
        )

        # Panel 1: Price & Overlay Indicators
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='blue', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange', width=1)), row=1, col=1)
        if pd.notnull(q['SMA200']):
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], mode='lines', name='SMA 200', line=dict(color='purple', width=1.5)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], mode='lines', name='EMA 9', line=dict(color='cyan', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA 20', line=dict(color='magenta', width=1, dash='dot')), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['Resistance_20'], mode='lines', name='20D Resistance', line=dict(color='red', dash='dash', width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Support_20'], mode='lines', name='20D Support', line=dict(color='green', dash='dash', width=1.2)), row=1, col=1)

        # PDH / PDL Horizontal Markers on Latest Candle
        fig.add_hline(y=q['PDH'], line_dash="dot", line_color="darkred", annotation_text="PDH", row=1, col=1)
        fig.add_hline(y=q['PDL'], line_dash="dot", line_color="darkgreen", annotation_text="PDL", row=1, col=1)

        # Panel 2: MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Line'], mode='lines', name='MACD', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='Signal', line=dict(color='orange')), row=2, col=1)
        colors_hist = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram', marker_color=colors_hist), row=2, col=1)

        # Panel 3: Standard ADX & +DI/-DI
        fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], mode='lines', name='ADX', line=dict(color='black', width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Plus_DI'], mode='lines', name='+DI', line=dict(color='green', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Minus_DI'], mode='lines', name='-DI', line=dict(color='red', width=1)), row=3, col=1)

        # Panel 4: Volume & RVOL
        colors_vol = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=colors_vol), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Vol_SMA20'], mode='lines', name='20D Vol SMA', line=dict(color='black', width=1)), row=4, col=1)

        fig.update_layout(height=850, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
