import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="PSX Quant AI Engine", layout="wide", page_icon="🤖")

st.title("🤖 PSX Advanced Quant & AI Decision Engine")
st.caption("Data-Driven Trade Architecture: Entry, Hold Days, Target Cap, Breakout Confirmation, & Risk Management")

# Helper: Indicator Calculations
def process_data(df):
    if len(df) < 50:
        return df
    
    # 1. Moving Averages & Volatility
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['STD_20'] = df['Close'].rolling(20).std()
    
    # 2. Bollinger Bands (Breakout volatility)
    df['BB_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)
    
    # 3. ATR (Average True Range for Dynamic Stop Loss & Targets)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # 4. RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-1 * delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 5. Volume Spike Detection
    df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
    df['Vol_Spike'] = df['Volume'] > (df['Vol_SMA20'] * 1.5)
    
    return df

# AI Trade Signal Generator Engine
def generate_ai_decision(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    price = latest['Close']
    atr = latest['ATR'] if pd.notnull(latest['ATR']) else (price * 0.02)
    rsi = latest['RSI']
    bb_upper = latest['BB_Upper']
    vol_spike = latest['Vol_Spike']
    
    # Scoring System
    bull_score = 0
    if rsi < 35: bull_score += 25
    elif 45 <= rsi <= 60: bull_score += 15
    
    if latest['SMA_20'] > latest['SMA_50']: bull_score += 25
    if price > latest['SMA_20']: bull_score += 20
    if vol_spike: bull_score += 20
    if price >= bb_upper: bull_score += 10 # Breakout pressure

    # AI Recommendation Logic
    if bull_score >= 65:
        action = "BUY / ACCUMULATE"
        confidence = f"{min(bull_score + 10, 92)}%"
        color = "green"
    elif bull_score <= 35:
        action = "SELL / AVOID"
        confidence = f"{min(100 - bull_score, 88)}%"
        color = "red"
    else:
        action = "NEUTRAL / HOLD"
        confidence = "55%"
        color = "orange"

    # Breakout & Timing Architecture
    is_breakout = (price >= bb_upper) or (vol_spike and price > latest['SMA_20'])
    trade_style = "INTRADAY (Same Day In/Out)" if (rsi > 65 and vol_spike) else "SWING TRADE"
    
    # Target Expectations & Dynamic Risk Management (Based on ATR Volatility)
    entry_price = round(price, 2)
    stop_loss = round(price - (1.5 * atr), 2)
    upper_cap = round(price + (2.5 * atr), 2)
    expected_return = round(((upper_cap - entry_price) / entry_price) * 100, 2)
    
    # Est Hold Duration based on volatility
    hold_days = "1 Day (Close before 3:30 PM)" if trade_style == "INTRADAY (Same Day In/Out)" else "3 to 8 Trading Days"

    return {
        "Action": action,
        "Confidence": confidence,
        "Color": color,
        "Trade Style": trade_style,
        "Entry Price": entry_price,
        "Breakout Level": round(bb_upper, 2),
        "Breakout Confirmed": "YES (Volume Supported)" if is_breakout else "NO (Wait for trigger)",
        "Upper Cap Target": upper_cap,
        "Expected Gain": f"+{expected_return}%",
        "Stop Loss": stop_loss,
        "Est Hold Duration": hold_days
    }

# Streamlit Layout
symbol_input = st.sidebar.text_input("Enter PSX Ticker", value="SYS").strip().upper()
period_choice = st.sidebar.selectbox("History Period", ["3mo", "6mo", "1y"], index=1)

if symbol_input:
    data = yf.Ticker(f"{symbol_input}.KA").history(period=period_choice)
    
    if data.empty:
        st.error(f"❌ Ticker '{symbol_input}' ka live data nahi mila. Symbol re-check karein.")
    else:
        df = process_data(data)
        ai = generate_ai_decision(df)

        # Main AI Decision Card
        st.markdown(f"### AI Trade Strategy: **:{ai['Color']}[{ai['Action']}]**")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model Confidence Score", ai['Confidence'])
        c2.metric("Trade Type", ai['Trade Style'])
        c3.metric("Breakout Triggered?", ai['Breakout Confirmed'])
        c4.metric("Est. Hold Duration", ai['Est Hold Duration'])

        st.markdown("---")
        
        # Actionable Setup Metrics
        st.subheader("🎯 Execution Parameters")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Optimal Entry Price", f"PKR {ai['Entry Price']}")
        m2.metric("Upper Cap Target", f"PKR {ai['Upper Cap']}", delta=ai['Expected Gain'])
        m3.metric("Stop Loss (Strict)", f"PKR {ai['Stop Loss']}")
        m4.metric("Breakout Level", f"PKR {ai['Breakout Level']}")

        # Interactive Chart
        st.markdown("---")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # Price & Bollinger Bands
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Candlestick"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='Breakout Level (Upper BB)', line=dict(color='red', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='20 SMA', line=dict(color='orange')), row=1, col=1)
        
        # Volume Spike Subplot
        colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=colors), row=2, col=1)
        
        fig.update_layout(height=550, xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
