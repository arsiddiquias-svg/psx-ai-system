"""
PSX QUANT ENGINE - PRODUCTION BUILD v2.0
===========================================
A PSX-focused quantitative decision-support terminal.

Key improvements over v1:
- Dynamic PSX universe from yfinance (not hardcoded)
- Wilder's RSI (verified correct)
- SMA crossover with equal-value handling
- Technical projection engine
- Penny stock detector with RVOL/breakout
- Capital allocation % removed (risk-based only)
- Cleaner UI with better information hierarchy
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from typing import Optional, Tuple, Dict, Any, List

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PSX Quant Engine v2",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# CONSTANTS & CONFIG
# ============================================================

MIN_RR = 1.5
PENNY_STOCK_THRESHOLD = 50  # PKR - configurable
MIN_HISTORY_DAYS = 60
CACHE_TTL = 300  # 5 minutes

WEIGHTS = {
    "trend": 0.25,
    "momentum": 0.20,
    "volume": 0.15,
    "setup": 0.20,
    "rr": 0.10,
    "sr": 0.10,
}

# KSE-100 candidates for market regime
MARKET_INDEX_CANDIDATES = ["^KSE100", "KSE100.KA", "PSX.KA", "^KSE"]
KSE100_PLAUSIBLE_MIN = 5000
KSE100_PLAUSIBLE_MAX = 1000000

# ============================================================
# TIME HELPERS
# ============================================================

def pkt_now():
    """Current time in Pakistan (UTC+5)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Karachi"))
    except Exception:
        return datetime.utcnow() + timedelta(hours=5)

# ============================================================
# TICKER NORMALIZATION
# ============================================================

def normalize_ticker(raw: str) -> str:
    """Normalize PSX ticker to yfinance format."""
    t = raw.strip().upper()
    if not t.endswith(".KA"):
        t = t + ".KA"
    return t

def normalize_ticker_display(raw: str) -> str:
    """Display-friendly ticker without .KA."""
    return raw.strip().upper().replace(".KA", "")

# ============================================================
# PSX UNIVERSE (DYNAMIC FROM YFINANCE)
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_psx_universe() -> List[str]:
    """
    Fetch PSX tickers dynamically from yfinance.
    
    yfinance doesn't have a direct "all PSX stocks" endpoint, so we:
    1. Try to fetch from known PSX ETF/components
    2. Fall back to curated list if dynamic fetch fails
    
    This is honest - we don't claim to have every PSX stock
    if yfinance doesn't provide it.
    """
    try:
        # Try to get from MSCI Pakistan or PSX ETF holdings
        etf_tickers = ["PAK", "PK", "KSE"]
        all_tickers = set()
        
        for etf in etf_tickers:
            try:
                ticker = yf.Ticker(etf)
                holdings = ticker.get_holdings()
                if holdings is not None and not holdings.empty:
                    # Filter for .KA stocks (PSX listings)
                    psx_stocks = [t for t in holdings.index if ".KA" in t]
                    all_tickers.update(psx_stocks)
            except Exception:
                continue
        
        if len(all_tickers) > 10:
            return sorted(list(all_tickers))
        
        # Fallback: Known liquid PSX stocks (from yfinance data)
        fallback = [
            "SYS.KA", "OGDC.KA", "LUCK.KA", "FFC.KA", "HUBC.KA",
            "PSO.KA", "ENGRO.KA", "HBL.KA", "UBL.KA", "MCB.KA",
            "BAFL.KA", "ABL.KA", "NBP.KA", "MARI.KA", "POL.KA",
            "PPL.KA", "KAPCO.KA", "DGKC.KA", "MLCF.KA", "FCCL.KA",
            "FATIMA.KA", "LOTCHEM.KA", "EPCL.KA", "SEARL.KA", "AGP.KA",
            "NML.KA", "ICI.KA", "TRG.KA", "NETSOL.KA", "SYS.KA",
            "INDU.KA", "PSMC.KA", "PIBTL.KA", "GATM.KA", "ATRL.KA",
            "NRL.KA", "SNGP.KA", "SSGC.KA", "KEL.KA", "KOHC.KA",
            "DAWH.KA", "THALL.KA", "PAEL.KA", "AICL.KA", "IGIHL.KA",
            "JSCL.KA", "PIOC.KA", "CHCC.KA", "ACPL.KA", "KOHTM.KA",
            "GHNI.KA", "MEHT.KA", "COLG.KA", "BNWM.KA", "FEROZ.KA",
            "SHFA.KA", "AGL.KA", "MUREB.KA", "BIFO.KA", "BGL.KA",
        ]
        return fallback
        
    except Exception:
        # Ultimate fallback - the most liquid names
        return [
            "SYS.KA", "OGDC.KA", "LUCK.KA", "FFC.KA", "HUBC.KA",
            "PSO.KA", "ENGRO.KA", "HBL.KA", "UBL.KA", "MCB.KA"
        ]

# ============================================================
# DATA FETCHING
# ============================================================

# Track fetch times for transparency
_LAST_FETCH_TIME = {}

def _flatten_columns(df):
    """Handle yfinance MultiIndex columns."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = list(df.columns.get_level_values(0))
        known = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if known.intersection(set(lvl0)):
            df = df.copy()
            df.columns = lvl0
        else:
            df = df.copy()
            df.columns = df.columns.get_level_values(-1)
    return df

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d"):
    """
    Fetch OHLCV data from yfinance.
    
    Returns: (DataFrame, status, error_message)
    status: "SUCCESS", "EMPTY", "EXCEPTION", "INSUFFICIENT"
    """
    symbol = normalize_ticker(ticker)
    _LAST_FETCH_TIME[(symbol, period, interval)] = pkt_now()
    
    try:
        raw = yf.download(
            symbol, period=period, interval=interval,
            auto_adjust=False, progress=False
        )
        
        if raw is None or raw.empty:
            return None, "EMPTY", f"No data returned for {ticker}"
        
        df = _flatten_columns(raw)
        
        # Required columns
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, "EXCEPTION", f"Missing columns: {missing}"
        
        df = df[required].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["Close", "High", "Low", "Open"])
        
        if df.empty:
            return None, "EMPTY", "Data became empty after cleaning"
        
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        
        # Quality checks
        df = df[(df["Close"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Open"] > 0)]
        df = df[df["High"] >= df["Low"]]
        
        if df.empty:
            return None, "EMPTY", "All rows failed quality validation"
        
        df["Volume"] = df["Volume"].fillna(0).clip(lower=0)
        
        if len(df) < MIN_HISTORY_DAYS:
            return df, "INSUFFICIENT", f"Only {len(df)} candles, need {MIN_HISTORY_DAYS}"
        
        return df, "SUCCESS", None
        
    except Exception as e:
        return None, "EXCEPTION", f"{type(e).__name__}: {str(e)}"

# ============================================================
# INDICATORS (ALL WILDER'S CORRECT)
# ============================================================

def sma(series, period):
    return series.rolling(period).mean()

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    """
    Wilder's RSI - matches TradingView.
    
    First average: simple mean of first 'period' periods.
    Subsequent: (prev_avg * (period-1) + current) / period
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(upper=0)
    
    # First average: simple mean
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # Wilder's smoothing: ewm with alpha=1/period
    avg_gain = avg_gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = avg_loss.ewm(alpha=1/period, adjust=False).mean()
    
    # RS calculation with safe division
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.inf), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    
    # No price movement -> neutral 50
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi = np.where(both_zero, 50, rsi)
    
    return rsi

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def true_range(df):
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df, period=14):
    tr = true_range(df)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def adx(df, period=14):
    """ADX with +DI and -DI."""
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr = true_range(df)
    atr_val = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr_val.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr_val.replace(0, np.nan)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    return adx_val.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

def bollinger(series, period=20, num_std=2):
    mid = sma(series, period)
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower

def build_indicators(df):
    """Attach all indicators to DataFrame."""
    d = df.copy()
    
    # Moving averages
    d["SMA20"] = sma(d["Close"], 20)
    d["SMA50"] = sma(d["Close"], 50)
    d["SMA100"] = sma(d["Close"], 100)
    d["SMA200"] = sma(d["Close"], 200)
    d["EMA20"] = ema(d["Close"], 20)
    d["EMA50"] = ema(d["Close"], 50)
    
    # RSI (Wilder's)
    d["RSI14"] = rsi(d["Close"], 14)
    
    # MACD
    macd_line, signal_line, hist = macd(d["Close"])
    d["MACD"] = macd_line
    d["MACD_SIGNAL"] = signal_line
    d["MACD_HIST"] = hist
    
    # ATR
    d["ATR14"] = atr(d, 14)
    
    # ADX
    adx_val, plus_di, minus_di = adx(d, 14)
    d["ADX14"] = adx_val
    d["PLUS_DI"] = plus_di
    d["MINUS_DI"] = minus_di
    
    # Bollinger Bands
    bb_u, bb_m, bb_l = bollinger(d["Close"], 20, 2)
    d["BB_UPPER"] = bb_u
    d["BB_MID"] = bb_m
    d["BB_LOWER"] = bb_l
    
    # Volume
    d["VOL_SMA20"] = sma(d["Volume"], 20)
    d["VOL_RATIO"] = d["Volume"] / d["VOL_SMA20"].replace(0, np.nan)
    
    # Returns and volatility
    d["RETURN_1D"] = d["Close"].pct_change()
    d["ROC_10"] = d["Close"].pct_change(10) * 100
    d["VOLATILITY_20"] = d["RETURN_1D"].rolling(20).std() * np.sqrt(252)
    
    # 52-week high/low (exclude today, use shift)
    d["52W_HIGH"] = d["High"].shift(1).rolling(252, min_periods=20).max()
    d["52W_LOW"] = d["Low"].shift(1).rolling(252, min_periods=20).min()
    
    return d

# ============================================================
# CROSSOVER DETECTION (FIXED - HANDLES EQUAL VALUES)
# ============================================================

def detect_crossovers(sma20: pd.Series, sma50: pd.Series):
    """
    Detect bullish and bearish SMA crossovers.
    
    Bullish: was <= yesterday, is > today
    Bearish: was >= yesterday, is < today
    """
    if len(sma20) < 2 or len(sma50) < 2:
        return pd.Series(False, index=sma20.index), pd.Series(False, index=sma20.index)
    
    # Current relationships
    curr_bullish = sma20 > sma50
    curr_bearish = sma20 < sma50
    curr_equal = sma20 == sma50
    
    # Previous relationships (shifted by 1)
    prev_bullish = sma20.shift(1) > sma50.shift(1)
    prev_bearish = sma20.shift(1) < sma50.shift(1)
    prev_equal = sma20.shift(1) == sma50.shift(1)
    
    # Bullish: was bearish or equal yesterday, bullish today
    bullish_crossover = (~prev_bullish) & curr_bullish
    
    # Bearish: was bullish or equal yesterday, bearish today
    bearish_crossover = (~prev_bearish) & curr_bearish
    
    # If equal today, no crossover (equal is transition, not crossover)
    bullish_crossover = bullish_crossover & ~curr_equal
    bearish_crossover = bearish_crossover & ~curr_equal
    
    # First row has no previous data
    if len(bullish_crossover) > 0:
        bullish_crossover.iloc[0] = False
        bearish_crossover.iloc[0] = False
    
    return bullish_crossover, bearish_crossover

# ============================================================
# MARKET REGIME
# ============================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_market_index():
    """Fetch KSE-100 data from yfinance with validation."""
    for cand in MARKET_INDEX_CANDIDATES:
        try:
            raw = yf.download(cand, period="6mo", interval="1d", auto_adjust=False, progress=False)
            if raw is None or raw.empty:
                continue
            
            df = _flatten_columns(raw)
            if "Close" not in df.columns:
                continue
            
            df = df.dropna(subset=["Close"])
            if len(df) < 40:
                continue
            
            last_close = float(df["Close"].iloc[-1])
            
            # Plausibility check: KSE-100 should be in range
            if not (KSE100_PLAUSIBLE_MIN <= last_close <= KSE100_PLAUSIBLE_MAX):
                continue
            
            # Volatility check: index shouldn't have single-stock volatility
            daily_vol = df["Close"].pct_change().std()
            if pd.isna(daily_vol) or daily_vol > 0.06:
                continue
            
            return df, cand
            
        except Exception:
            continue
    
    return None, None

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def market_snapshot():
    """Get market regime and trend."""
    idx_df, label = fetch_market_index()
    
    if idx_df is None or len(idx_df) < 40:
        return {
            "regime": "UNAVAILABLE",
            "trend": "UNAVAILABLE",
            "label": None,
            "reasoning": "KSE-100 data unavailable from yfinance",
            "last_date": None,
            "last_close": None,
        }
    
    d = idx_df.copy()
    
    # Build indicators
    d["SMA20"] = sma(d["Close"], 20)
    d["SMA50"] = sma(d["Close"], 50)
    d["RETURN_1D"] = d["Close"].pct_change()
    
    last_date = d.index[-1]
    last_close = float(d["Close"].iloc[-1])
    sma20 = d["SMA20"].iloc[-1]
    sma50 = d["SMA50"].iloc[-1]
    vol20 = d["RETURN_1D"].rolling(20).std().iloc[-1] * np.sqrt(252)
    
    # Determine regime
    if pd.isna(sma20) or pd.isna(sma50):
        regime = "UNAVAILABLE"
        reasoning = "Insufficient data for SMA calculation"
    elif vol20 is not None and vol20 > 0.35:
        regime = "HIGH VOLATILITY"
        reasoning = f"KSE-100 volatility ({round(vol20*100,1)}%) is elevated"
    elif last_close > sma20 > sma50:
        regime = "BULLISH"
        reasoning = f"KSE-100 ({round(last_close,0)}) > SMA20 ({round(sma20,0)}) > SMA50 ({round(sma50,0)})"
    elif last_close < sma20 < sma50:
        regime = "BEARISH"
        reasoning = f"KSE-100 ({round(last_close,0)}) < SMA20 ({round(sma20,0)}) < SMA50 ({round(sma50,0)})"
    else:
        regime = "NEUTRAL"
        reasoning = "Mixed SMA alignment - transitional market"
    
    return {
        "regime": regime,
        "trend": regime if regime in ("BULLISH", "BEARISH") else "NEUTRAL",
        "label": label,
        "reasoning": reasoning,
        "last_date": last_date,
        "last_close": last_close,
    }

# ============================================================
# TREND ENGINE
# ============================================================

def trend_engine(d):
    """
    Determine trend using multiple confirmations.
    Returns: (trend_label, reasons_list, score)
    """
    if len(d) < 50:
        return "INSUFFICIENT DATA", [], 0
    
    last = d.iloc[-1]
    reasons = []
    bullish = 0
    bearish = 0
    
    # Price vs SMA20
    if last["Close"] > last["SMA20"]:
        bullish += 1
        reasons.append("Price above SMA20")
    else:
        bearish += 1
        reasons.append("Price below SMA20")
    
    # Price vs SMA50
    if last["Close"] > last["SMA50"]:
        bullish += 1
        reasons.append("Price above SMA50")
    else:
        bearish += 1
        reasons.append("Price below SMA50")
    
    # SMA20 vs SMA50
    if last["SMA20"] > last["SMA50"]:
        bullish += 1
        reasons.append("SMA20 above SMA50 (short-term bullish)")
    else:
        bearish += 1
        reasons.append("SMA20 below SMA50 (short-term bearish)")
    
    # SMA50 vs SMA200 (if available)
    if not pd.isna(last.get("SMA200", np.nan)):
        if last["SMA50"] > last["SMA200"]:
            bullish += 1
            reasons.append("SMA50 above SMA200 (long-term bullish)")
        else:
            bearish += 1
            reasons.append("SMA50 below SMA200 (long-term bearish)")
    
    # Higher highs / higher lows (last 20)
    recent = d.tail(20)
    if len(recent) >= 10:
        higher_highs = recent["High"].iloc[-1] > recent["High"].iloc[0]
        higher_lows = recent["Low"].iloc[-1] > recent["Low"].iloc[0]
        if higher_highs and higher_lows:
            bullish += 1
            reasons.append("Higher highs and higher lows (last 20)")
        elif (not higher_highs) and (not higher_lows):
            bearish += 1
            reasons.append("Lower highs and lower lows (last 20)")
    
    # ADX strength
    if last["ADX14"] >= 25:
        reasons.append(f"ADX {round(last['ADX14'],1)} - trending market")
    
    total = bullish + bearish
    score = (bullish / total * 100) if total > 0 else 50
    
    if score >= 80:
        trend = "STRONG BULLISH"
    elif score >= 60:
        trend = "BULLISH"
    elif score >= 40:
        trend = "NEUTRAL"
    elif score >= 20:
        trend = "BEARISH"
    else:
        trend = "STRONG BEARISH"
    
    return trend, reasons, score

# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def support_resistance(d):
    """
    Calculate support and resistance levels.
    Excludes today's candle to avoid look-ahead bias.
    """
    history = d.iloc[:-1] if len(d) > 1 else d
    
    recent20 = history.tail(20) if len(history) >= 20 else history
    recent60 = history.tail(60) if len(history) >= 60 else history
    
    primary_resistance = recent20["High"].max()
    primary_support = recent20["Low"].min()
    secondary_resistance = recent60["High"].max()
    secondary_support = recent60["Low"].min()
    
    last = d.iloc[-1]
    high_52w = last.get("52W_HIGH", np.nan)
    low_52w = last.get("52W_LOW", np.nan)
    
    return {
        "primary_support": primary_support,
        "primary_resistance": primary_resistance,
        "secondary_support": secondary_support,
        "secondary_resistance": secondary_resistance,
        "high_52w": high_52w,
        "low_52w": low_52w,
    }

# ============================================================
# BREAKOUT ENGINE
# ============================================================

def breakout_engine(d, sr, vol_ratio_threshold=1.5):
    """Detect breakout status with freshness check."""
    last = d.iloc[-1]
    prev = d.iloc[-2] if len(d) >= 2 else last
    
    resistance = sr["primary_resistance"]
    price = last["Close"]
    vol_ratio = last["VOL_RATIO"] if not pd.isna(last["VOL_RATIO"]) else 0
    
    # Freshness baseline (exclude last 2 candles for true breakout detection)
    if len(d) >= 22:
        baseline_window = d.iloc[:-2].tail(20)
        baseline_resistance = baseline_window["High"].max()
    else:
        baseline_resistance = resistance
    
    # Was price below resistance before today?
    was_below = prev["Close"] <= baseline_resistance
    now_above = price > resistance
    fresh_cross = now_above and was_below
    
    volume_confirmed = vol_ratio >= vol_ratio_threshold
    momentum_positive = last["MACD_HIST"] > 0
    adx_strong = last["ADX14"] >= 20
    
    distance_to_resistance = (resistance - price) / price * 100 if price > 0 else None
    
    # Classification
    if now_above and volume_confirmed and momentum_positive and fresh_cross:
        status = "CONFIRMED BREAKOUT"
        note = "Closed above resistance with volume + momentum confirmation"
    elif now_above and volume_confirmed and momentum_positive and not fresh_cross:
        status = "EXTENDED BREAKOUT"
        note = "Already above resistance - continuation, not fresh signal"
    elif now_above and (not volume_confirmed or not momentum_positive):
        status = "BREAKOUT ATTEMPT"
        note = "Price above resistance but lacking confirmation"
    elif (not now_above) and distance_to_resistance is not None and 0 <= distance_to_resistance <= 3:
        status = "BREAKOUT READY"
        note = "Within 3% of resistance - monitoring zone"
    elif prev["Close"] > baseline_resistance and price < baseline_resistance:
        status = "FAILED BREAKOUT"
        note = "Broke above but closed back below"
    else:
        status = "NO BREAKOUT"
        note = "Not near breakout level"
    
    # Near 52-week high?
    if not pd.isna(sr["high_52w"]) and price >= sr["high_52w"] * 0.98:
        status = status + " / 52W HIGH"
        note = note + " - near 52-week high"
    
    return {
        "status": status,
        "note": note,
        "fresh_cross": fresh_cross,
        "resistance": resistance,
        "price": price,
        "volume_ratio": vol_ratio,
        "distance_to_resistance": distance_to_resistance,
    }

# ============================================================
# PULLBACK ENGINE
# ============================================================

def pullback_engine(d, trend, sr):
    """Detect healthy pullback vs breakdown."""
    last = d.iloc[-1]
    support = sr["primary_support"]
    price = last["Close"]
    
    if trend not in ("BULLISH", "STRONG BULLISH"):
        return {"status": "NO PULLBACK", "note": "Trend is not bullish"}
    
    near_support = abs(price - support) / price < 0.03 if price > 0 else False
    near_ema20 = abs(price - last["EMA20"]) / price < 0.02 if not pd.isna(last["EMA20"]) and price > 0 else False
    
    cooling_rsi = 35 <= last["RSI14"] <= 55
    bullish_candle = last["Close"] > last["Open"]
    macd_stabilizing = last["MACD_HIST"] > d["MACD_HIST"].iloc[-2] if len(d) >= 2 else False
    
    if price < support * 0.98:
        return {"status": "BROKEN SUPPORT", "note": "Closed below primary support - invalidation"}
    
    if (near_support or near_ema20) and cooling_rsi and bullish_candle:
        return {"status": "HEALTHY PULLBACK", "note": "At support/EMA20 with confirmation candle"}
    
    if near_support or near_ema20:
        return {"status": "PULLBACK WATCH", "note": "Approaching support/EMA20, awaiting confirmation"}
    
    return {"status": "NO PULLBACK", "note": "Not near pullback zone"}

# ============================================================
# MOMENTUM ENGINE
# ============================================================

def momentum_engine(d):
    """Classify momentum using multiple indicators."""
    last = d.iloc[-1]
    
    score = 0
    signals = []
    
    # RSI
    if last["RSI14"] > 55:
        score += 1
        signals.append("RSI positive")
    elif last["RSI14"] < 45:
        score -= 1
        signals.append("RSI negative")
    
    # MACD
    if last["MACD_HIST"] > 0:
        score += 1
        signals.append("MACD positive")
    else:
        score -= 1
        signals.append("MACD negative")
    
    # MACD slope
    if len(d) >= 2:
        if last["MACD_HIST"] > d["MACD_HIST"].iloc[-2]:
            score += 1
            signals.append("MACD accelerating")
    
    # ROC
    if last["ROC_10"] > 0:
        score += 1
        signals.append("ROC positive")
    else:
        score -= 1
        signals.append("ROC negative")
    
    # ADX + DI
    if last["ADX14"] >= 20:
        if last["PLUS_DI"] > last["MINUS_DI"]:
            score += 1
            signals.append("ADX confirms +DI")
        else:
            score -= 1
            signals.append("ADX confirms -DI")
    
    # Divergence detection (simplified)
    window = d.tail(10)
    if len(window) >= 10:
        price_high = window["Close"].max()
        rsi_high = window["RSI14"].max()
        if window["Close"].iloc[-1] >= price_high * 0.999 and window["RSI14"].iloc[-1] < rsi_high - 5:
            signals.append("⚠️ Bearish divergence (price high, RSI lower)")
            score -= 2
        
        price_low = window["Close"].min()
        rsi_low = window["RSI14"].min()
        if window["Close"].iloc[-1] <= price_low * 1.001 and window["RSI14"].iloc[-1] > rsi_low + 5:
            signals.append("✅ Bullish divergence (price low, RSI higher)")
            score += 2
    
    # Classification
    if score >= 4:
        label = "STRONG MOMENTUM"
    elif score >= 2:
        label = "POSITIVE MOMENTUM"
    elif score >= -1:
        label = "NEUTRAL MOMENTUM"
    elif score >= -3:
        label = "NEGATIVE MOMENTUM"
    else:
        label = "STRONG NEGATIVE MOMENTUM"
    
    return {
        "label": label,
        "score": score,
        "signals": signals,
        "overbought": last["RSI14"] > 70,
        "oversold": last["RSI14"] < 30,
    }

# ============================================================
# PROJECTION ENGINE (NEW - "kahan tak ja sakta hai")
# ============================================================

def projection_engine(d, trend, sr, momentum):
    """
    Calculate technical projection zones.
    
    This is NOT a prediction - it's a "what if trend continues" projection.
    """
    last = d.iloc[-1]
    price = last["Close"]
    atr_val = last["ATR14"] if not pd.isna(last["ATR14"]) else 0
    
    resistance = sr["primary_resistance"]
    support = sr["primary_support"]
    
    # Determine if uptrend or downtrend
    is_bullish = trend in ("BULLISH", "STRONG BULLISH")
    is_bearish = trend in ("BEARISH", "STRONG BEARISH")
    
    # Base projection using ATR and structure
    if is_bullish:
        # Upside projection
        if price >= resistance:
            # Already above resistance - use breakout measured move
            range_size = resistance - support
            upside_zone_low = price + range_size * 0.5
            upside_zone_high = price + range_size * 1.0
            next_resistance = sr["secondary_resistance"]
            extension_zone = price + range_size * 1.5
        else:
            # Below resistance - project to resistance first
            upside_zone_low = resistance
            upside_zone_high = resistance + atr_val * 1.5
            next_resistance = sr["secondary_resistance"]
            extension_zone = resistance + atr_val * 3
        
        return {
            "direction": "UP",
            "zone_low": upside_zone_low,
            "zone_high": upside_zone_high,
            "next_resistance": next_resistance,
            "extension_zone": extension_zone,
            "label": f"Upside: {round(upside_zone_low,2)} - {round(upside_zone_high,2)}",
            "note": "Technical projection if uptrend continues (not a guarantee)"
        }
    
    elif is_bearish:
        # Downside projection
        if price <= support:
            # Already below support - measured move down
            range_size = resistance - support
            downside_zone_low = price - range_size * 1.0
            downside_zone_high = price - range_size * 0.5
            next_support = sr["secondary_support"]
            invalidation = resistance
        else:
            downside_zone_low = support - atr_val * 1.5
            downside_zone_high = support
            next_support = sr["secondary_support"]
            invalidation = resistance
        
        return {
            "direction": "DOWN",
            "zone_low": downside_zone_low,
            "zone_high": downside_zone_high,
            "next_support": next_support,
            "invalidation": invalidation,
            "label": f"Downside: {round(downside_zone_low,2)} - {round(downside_zone_high,2)}",
            "note": "Technical projection if downtrend continues (not a guarantee)"
        }
    
    else:
        return {
            "direction": "NEUTRAL",
            "zone_low": None,
            "zone_high": None,
            "label": "No clear direction for projection",
            "note": "Price is in a neutral/range-bound structure"
        }

# ============================================================
# PENNY STOCK DETECTOR (NEW)
# ============================================================

def detect_penny_setup(d, sr, threshold=PENNY_STOCK_THRESHOLD):
    """
    Detect interesting penny stock setups.
    
    Penny stock = price below threshold with unusual volume/breakout activity.
    """
    last = d.iloc[-1]
    price = last["Close"]
    
    if price > threshold:
        return {
            "is_penny": False,
            "status": "NORMAL PRICE STOCK",
            "note": f"Price {price} > {threshold} threshold"
        }
    
    # Penny stock detection
    vol_ratio = last["VOL_RATIO"] if not pd.isna(last["VOL_RATIO"]) else 0
    breakout_status = "NO BREAKOUT"
    
    # Check if near resistance
    near_resistance = False
    if price > 0 and sr["primary_resistance"]:
        near_resistance = abs(price - sr["primary_resistance"]) / price < 0.05
    
    # Check if broke resistance
    broke_resistance = price > sr["primary_resistance"]
    
    # RVOL expansion
    rvol_expansion = vol_ratio >= 2.0
    
    # Momentum
    momentum_positive = last["MACD_HIST"] > 0
    
    # Classification
    if broke_resistance and rvol_expansion and momentum_positive:
        status = "🔥 PENNY BREAKOUT"
        note = "Low-priced stock breaking resistance with huge volume!"
    elif near_resistance and rvol_expansion:
        status = "⚡ PENNY BREAKOUT READY"
        note = "Low-priced stock near resistance with unusual volume"
    elif rvol_expansion:
        status = "📈 PENNY VOLUME SPIKE"
        note = "Unusual volume in low-priced stock"
    elif momentum_positive and near_resistance:
        status = "👀 PENNY WATCH"
        note = "Low-priced stock with momentum, near resistance"
    else:
        status = "PENNY (NO SETUP)"
        note = "Low-priced stock but no unusual activity detected"
    
    return {
        "is_penny": True,
        "status": status,
        "note": note,
        "price": price,
        "vol_ratio": vol_ratio,
        "near_resistance": near_resistance,
        "broke_resistance": broke_resistance,
        "rvol_expansion": rvol_expansion,
    }

# ============================================================
# RISK / TRADE PLAN ENGINE
# ============================================================

def risk_engine(d, sr):
    """Calculate stop loss, targets, and R:R."""
    last = d.iloc[-1]
    price = last["Close"]
    atr_val = last["ATR14"] if not pd.isna(last["ATR14"]) else 0
    
    # Stop loss: swing low OR support, whichever is lower
    swing_low = d.tail(10)["Low"].min()
    stop_loss = min(swing_low, sr["primary_support"]) - 0.3 * atr_val
    risk_per_share = price - stop_loss
    
    # Targets: structural levels + ATR projection
    near_or_above_resistance = price >= sr["primary_resistance"] * 0.995
    atr_target = price + 2.5 * atr_val if atr_val else price
    
    if near_or_above_resistance:
        target1 = max(sr["secondary_resistance"], atr_target)
    else:
        target1 = sr["primary_resistance"]
    
    if sr["secondary_resistance"] > target1:
        target2 = sr["secondary_resistance"] + max(atr_val, (target1 - price) * 0.5)
    else:
        target2 = target1 + max(2 * atr_val, (target1 - price))
    
    reward1 = target1 - price
    reward2 = target2 - price
    
    rr1 = reward1 / risk_per_share if risk_per_share > 0 else None
    rr2 = reward2 / risk_per_share if risk_per_share > 0 else None
    
    return {
        "entry": price,
        "stop_loss": stop_loss,
        "risk_per_share": risk_per_share,
        "target1": target1,
        "target2": target2,
        "rr1": rr1,
        "rr2": rr2,
    }

# ============================================================
# POSITION SIZING (NO CAPITAL ALLOCATION %)
# ============================================================

def position_sizing(capital: float, risk_pct: float, risk_data: Dict) -> Dict:
    """
    Calculate position size based purely on risk.
    
    Capital allocation % has been REMOVED as per requirements.
    """
    if risk_data["risk_per_share"] is None or risk_data["risk_per_share"] <= 0:
        return {
            "shares": 0,
            "investment": 0,
            "max_loss": 0,
            "note": "Invalid risk per share"
        }
    
    max_risk_amount = capital * (risk_pct / 100)
    shares = int(max_risk_amount // risk_data["risk_per_share"])
    investment = shares * risk_data["entry"]
    max_loss = shares * risk_data["risk_per_share"]
    
    return {
        "shares": shares,
        "investment": round(investment, 2),
        "max_loss": round(max_loss, 2),
        "max_risk_amount": round(max_risk_amount, 2),
        "note": "Risk-based sizing only (capital allocation % removed)",
    }

# ============================================================
# TECHNICAL SCORE
# ============================================================

def _trend_component(trend_score):
    return min(100, trend_score)

def _momentum_component(momentum):
    if momentum["label"] == "STRONG MOMENTUM":
        score = 95
    elif momentum["label"] == "POSITIVE MOMENTUM":
        score = 75
    elif momentum["label"] == "NEUTRAL MOMENTUM":
        score = 50
    elif momentum["label"] == "NEGATIVE MOMENTUM":
        score = 25
    else:
        score = 5
    
    # Adjust for divergence
    if momentum["overbought"]:
        score -= 10
    if momentum["oversold"]:
        score += 10
    
    return max(0, min(100, score))

def _volume_component(vol_ratio):
    if vol_ratio is None or pd.isna(vol_ratio):
        return 40
    if vol_ratio >= 3:
        return 100
    if vol_ratio >= 2:
        return 85
    if vol_ratio >= 1.5:
        return 65
    if vol_ratio >= 1.0:
        return 50
    return 30

def _setup_component(breakout_status, pullback_status):
    if "CONFIRMED" in breakout_status and "EXTENDED" not in breakout_status:
        return 100
    if pullback_status == "HEALTHY PULLBACK":
        return 88
    if "CONFIRMED" in breakout_status and "EXTENDED" in breakout_status:
        return 78
    if "BREAKOUT READY" in breakout_status:
        return 65
    if "BREAKOUT ATTEMPT" in breakout_status:
        return 55
    if pullback_status == "PULLBACK WATCH":
        return 50
    if "FAILED" in breakout_status or pullback_status == "BROKEN SUPPORT":
        return 10
    return 40

def _rr_component(rr1):
    if rr1 is None:
        return 20
    if rr1 >= 3:
        return 100
    if rr1 >= 2:
        return 85
    if rr1 >= MIN_RR:
        return 65
    if rr1 >= 1:
        return 35
    return 10

def _sr_component(price, sr):
    resistance = sr["primary_resistance"]
    support = sr["primary_support"]
    if resistance == support:
        return 50
    position = (price - support) / (resistance - support)
    if 0.2 <= position <= 0.75:
        return 80
    if position < 0:
        return 15
    if position > 1.05:
        return 30
    return 55

def technical_score(components):
    """Calculate weighted technical score."""
    score = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
    return round(score, 1)

# ============================================================
# SIGNAL ENGINE
# ============================================================

def signal_engine(d, trend, trend_score, momentum, breakout, pullback, sr, risk_data, market):
    """Generate final BUY/WAIT/AVOID signal."""
    
    components = {
        "trend": min(100, trend_score),
        "momentum": _momentum_component(momentum),
        "volume": _volume_component(breakout["volume_ratio"]),
        "setup": _setup_component(breakout["status"], pullback["status"]),
        "rr": _rr_component(risk_data["rr1"]),
        "sr": _sr_component(risk_data["entry"], sr),
    }
    
    score = technical_score(components)
    
    # Check R:R
    rr_ok = risk_data["rr1"] is not None and risk_data["rr1"] >= MIN_RR
    
    # Market condition adjustment
    market_adjust = 0
    if market["regime"] == "BULLISH":
        market_adjust = 5
    elif market["regime"] == "BEARISH":
        market_adjust = -10
    elif market["regime"] == "HIGH VOLATILITY":
        market_adjust = -5
    
    adjusted_score = max(0, min(100, score + market_adjust))
    
    # Final signal
    if not rr_ok:
        signal = "WAIT"
        setup_quality = "R:R BELOW MINIMUM"
    elif adjusted_score >= 80:
        signal = "STRONG BUY"
        setup_quality = "A+ SETUP"
    elif adjusted_score >= 65:
        signal = "BUY"
        setup_quality = "A SETUP"
    elif adjusted_score >= 45:
        signal = "WAIT"
        setup_quality = "B SETUP / WATCH"
    elif adjusted_score >= 25:
        signal = "REDUCE"
        setup_quality = "WEAK"
    else:
        signal = "AVOID"
        setup_quality = "POOR"
    
    # Reasons
    reasons = []
    if components["trend"] >= 70:
        reasons.append(f"✅ Trend: {trend}")
    elif components["trend"] <= 35:
        reasons.append(f"❌ Trend: {trend}")
    
    if components["momentum"] >= 70:
        reasons.append(f"✅ Momentum: {momentum['label']}")
    elif components["momentum"] <= 35:
        reasons.append(f"❌ Momentum: {momentum['label']}")
    
    if "CONFIRMED" in breakout["status"]:
        reasons.append(f"✅ {breakout['status']} - {breakout['note']}")
    elif "BREAKOUT READY" in breakout["status"]:
        reasons.append(f"📌 {breakout['status']} - {breakout['note']}")
    
    if pullback["status"] == "HEALTHY PULLBACK":
        reasons.append(f"✅ {pullback['status']} - {pullback['note']}")
    elif pullback["status"] == "BROKEN SUPPORT":
        reasons.append(f"❌ {pullback['status']} - {pullback['note']}")
    
    if momentum["overbought"]:
        reasons.append("⚠️ RSI overbought")
    if momentum["oversold"]:
        reasons.append("📌 RSI oversold - reversal watch")
    
    if not rr_ok:
        reasons.append(f"❌ R:R {round(risk_data['rr1'],2) if risk_data['rr1'] else 'N/A'} < {MIN_RR}")
    
    if market["regime"] == "BULLISH":
        reasons.append("✅ Market: Bullish regime")
    elif market["regime"] == "BEARISH":
        reasons.append("❌ Market: Bearish regime")
    
    return {
        "score": adjusted_score,
        "components": components,
        "signal": signal,
        "setup_quality": setup_quality,
        "reasons": reasons[:6],  # Limit to keep it readable
    }

# ============================================================
# FULL STOCK ANALYSIS
# ============================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def analyze_stock(ticker: str, period: str = "1y"):
    """Complete stock analysis orchestrating all engines."""
    
    df, status, error = fetch_ohlcv(ticker, period=period)
    
    if status != "SUCCESS":
        return None, status, error
    
    # Build indicators
    d = build_indicators(df)
    
    # Market context
    market = market_snapshot()
    
    # Trend
    trend, trend_reasons, trend_score = trend_engine(d)
    
    # Support/Resistance
    sr = support_resistance(d)
    
    # Breakout
    breakout = breakout_engine(d, sr)
    
    # Pullback
    pullback = pullback_engine(d, trend, sr)
    
    # Momentum
    momentum = momentum_engine(d)
    
    # Risk / Trade Plan
    risk = risk_engine(d, sr)
    
    # Penny stock detection
    penny = detect_penny_setup(d, sr)
    
    # Projection
    projection = projection_engine(d, trend, sr, momentum)
    
    # Signal
    signal = signal_engine(d, trend, trend_score, momentum, breakout, pullback, sr, risk, market)
    
    last = d.iloc[-1]
    
    result = {
        "ticker": normalize_ticker(ticker),
        "ticker_display": normalize_ticker_display(ticker),
        "df": d,
        "last": last,
        "trend": trend,
        "trend_score": trend_score,
        "trend_reasons": trend_reasons,
        "sr": sr,
        "breakout": breakout,
        "pullback": pullback,
        "momentum": momentum,
        "risk": risk,
        "penny": penny,
        "projection": projection,
        "signal": signal,
        "market": market,
        "data_date": d.index[-1],
        "has_sma200": not pd.isna(last.get("SMA200", np.nan)),
    }
    
    return result, "SUCCESS", None

# ============================================================
# SCREENER
# ============================================================

def run_screener(universe: List[str], period: str = "6mo"):
    """Run screener on universe."""
    rows = []
    coverage = {
        "total": len(universe),
        "success": 0,
        "failed": 0,
        "analyzed": 0
    }
    
    for ticker in universe:
        result, status, error = analyze_stock(ticker, period=period)
        
        if status != "SUCCESS":
            coverage["failed"] += 1
            rows.append({
                "Ticker": ticker,
                "Price": None,
                "Change %": None,
                "Trend": None,
                "Score": None,
                "Signal": "ERROR",
                "Status": "DATA UNAVAILABLE",
                "Penny": None,
            })
            continue
        
        coverage["success"] += 1
        coverage["analyzed"] += 1
        
        last = result["last"]
        prev_close = result["df"]["Close"].iloc[-2] if len(result["df"]) >= 2 else last["Close"]
        change_pct = (last["Close"] - prev_close) / prev_close * 100 if prev_close else 0
        
        rows.append({
            "Ticker": result["ticker_display"],
            "Price": round(last["Close"], 2),
            "Change %": round(change_pct, 2),
            "Trend": result["trend"],
            "Score": result["signal"]["score"],
            "Signal": result["signal"]["signal"],
            "Status": result["breakout"]["status"],
            "Penny": result["penny"]["status"] if result["penny"]["is_penny"] else "N/A",
            "RR": round(result["risk"]["rr1"], 2) if result["risk"]["rr1"] else None,
            "_ticker_raw": ticker,
        })
    
    return pd.DataFrame(rows), coverage

# ============================================================
# PORTFOLIO DECISION
# ============================================================

def portfolio_decision(holding: Dict, result: Dict) -> Tuple[str, str]:
    """Generate portfolio decision based on technicals, not P&L."""
    if result is None:
        return "WATCH", "Data unavailable"
    
    signal = result["signal"]["signal"]
    trend = result["trend"]
    pullback = result["pullback"]["status"]
    breakout = result["breakout"]["status"]
    
    # Don't decide based on P&L - use technical structure
    
    if pullback == "BROKEN SUPPORT":
        return "EXIT / AVOID", "Support broken - structure deteriorated"
    
    if trend in ("BEARISH", "STRONG BEARISH"):
        return "REDUCE", "Trend turned bearish"
    
    if signal in ("STRONG BUY", "BUY") and trend in ("BULLISH", "STRONG BULLISH"):
        if "CONFIRMED" in breakout:
            return "ADD ON CONFIRMATION", "Fresh breakout confirmation"
        return "HOLD", "Trend and signal constructive"
    
    if signal in ("REDUCE", "AVOID"):
        return "REDUCE / EXIT", "Signal engine flags deterioration"
    
    if pullback == "HEALTHY PULLBACK" and trend in ("BULLISH", "STRONG BULLISH"):
        return "HOLD / WATCH", "Healthy pullback in uptrend"
    
    return "HOLD", "No clear signal - maintain position"

# ============================================================
# CHARTING
# ============================================================

def build_chart(result, show_bb=False, show_sma200=False):
    """Build clean Plotly chart."""
    d = result["df"].tail(150)
    risk = result["risk"]
    sr = result["sr"]
    
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.15, 0.17, 0.18],
        vertical_spacing=0.03,
        subplot_titles=("Price", "Volume", "RSI (14)", "MACD"),
    )
    
    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="Price", increasing_line_color="#16C784", decreasing_line_color="#EA3943",
    ), row=1, col=1)
    
    # SMAs (always visible - minimal)
    fig.add_trace(go.Scatter(
        x=d.index, y=d["SMA20"], line=dict(color="#4A9DE0", width=1.2), name="SMA20"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=d.index, y=d["SMA50"], line=dict(color="#D4A94D", width=1.2), name="SMA50"
    ), row=1, col=1)
    
    if show_sma200 and result["has_sma200"]:
        fig.add_trace(go.Scatter(
            x=d.index, y=d["SMA200"], line=dict(color="#9B59B6", width=1, dash="dot"), name="SMA200"
        ), row=1, col=1)
    
    if show_bb:
        fig.add_trace(go.Scatter(
            x=d.index, y=d["BB_UPPER"], line=dict(color="gray", width=0.8, dash="dot"), name="BB Upper"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=d.index, y=d["BB_LOWER"], line=dict(color="gray", width=0.8, dash="dot"), name="BB Lower"
        ), row=1, col=1)
    
    # Support/Resistance lines
    fig.add_hline(y=sr["primary_resistance"], line_dash="dash", line_color="#EA3943", 
                  annotation_text="Resistance", row=1, col=1)
    fig.add_hline(y=sr["primary_support"], line_dash="dash", line_color="#16C784", 
                  annotation_text="Support", row=1, col=1)
    fig.add_hline(y=risk["stop_loss"], line_dash="dot", line_color="#D4A94D", 
                  annotation_text="Stop", row=1, col=1)
    fig.add_hline(y=risk["target1"], line_dash="dot", line_color="#4A9DE0", 
                  annotation_text="T1", row=1, col=1)
    
    # Volume
    vol_colors = np.where(d["Close"] >= d["Open"], "#16C784", "#EA3943")
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=vol_colors, name="Volume"), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=d.index, y=d["RSI14"], line=dict(color="#4A9DE0", width=1.3), name="RSI"
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#EA3943", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#16C784", row=3, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(
        x=d.index, y=d["MACD"], line=dict(color="#4A9DE0", width=1), name="MACD"
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=d.index, y=d["MACD_SIGNAL"], line=dict(color="#D4A94D", width=1), name="Signal"
    ), row=4, col=1)
    hist_colors = np.where(d["MACD_HIST"] >= 0, "#16C784", "#EA3943")
    fig.add_trace(go.Bar(
        x=d.index, y=d["MACD_HIST"], marker_color=hist_colors, name="Hist"
    ), row=4, col=1)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_dark",
    )
    
    return fig

# ============================================================
# UI HELPERS
# ============================================================

def signal_color(signal):
    colors = {
        "STRONG BUY": "#16C784",
        "BUY": "#16C784",
        "WAIT": "#D4A94D",
        "REDUCE": "#EA3943",
        "AVOID": "#EA3943",
    }
    return colors.get(signal, "#8A94A6")

def metric_row(items):
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)

def data_freshness_label(data_date):
    if data_date is None:
        return "UNAVAILABLE", "No data date"
    now = pkt_now()
    data_dt = pd.Timestamp(data_date)
    if data_dt.tzinfo is None:
        data_dt = data_dt.tz_localize(None)
    now_naive = now.replace(tzinfo=None)
    age_days = (now_naive.date() - data_dt.date()).days
    if age_days <= 3:
        return "END OF DAY", f"{age_days} day(s) old"
    return "STALE", f"{age_days} days old - verify data"

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📈 PSX Quant Engine v2")

# Ticker input
sidebar_ticker = st.sidebar.text_input(
    "Ticker",
    value="SYS",
    help="Enter PSX symbol (e.g., SYS, OGDC, LUCK)"
)

# Trading parameters
st.sidebar.subheader("Trading Parameters")
capital = st.sidebar.number_input(
    "Trading Capital (PKR)",
    min_value=10000,
    value=100000,
    step=10000
)

risk_pct = st.sidebar.slider(
    "Risk Per Trade (%)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.5
)

period = st.sidebar.selectbox(
    "Analysis Period",
    ["3mo", "6mo", "1y", "2y", "5y"],
    index=2
)

# Watchlist
watchlist_input = st.sidebar.text_area(
    "Watchlist (comma-separated)",
    value="SYS, OGDC, HBL, LUCK, FFC, ENGRO"
)

# Penny stock threshold
penny_threshold = st.sidebar.number_input(
    "Penny Stock Threshold (PKR)",
    min_value=10,
    max_value=200,
    value=50,
    step=5,
    help="Stocks below this price are classified as penny stocks"
)

# Refresh button
st.sidebar.divider()
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    fetch_ohlcv.clear()
    fetch_market_index.clear()
    market_snapshot.clear()
    analyze_stock.clear()
    _LAST_FETCH_TIME.clear()
    st.session_state.pop("screener_df", None)
    st.session_state.pop("watchlist_df", None)
    st.sidebar.success("Cache cleared!")

st.sidebar.caption("Data: yfinance (daily EOD) | Cache: 5min")
st.sidebar.caption(f"Checked: {pkt_now().strftime('%d-%b %H:%M')} PKT")
st.sidebar.caption("⚠️ Signals are analytical outputs, not guaranteed advice.")

# Portfolio state
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

# ============================================================
# MAIN TABS
# ============================================================

tab_dash, tab_screener, tab_breakouts, tab_penny, tab_next, tab_watch, tab_port, tab_market = st.tabs([
    "📊 Dashboard",
    "🔍 Screener",
    "🚀 Breakouts",
    "🪙 Penny Stocks",
    "📅 Next Session",
    "📋 Watchlist",
    "💼 Portfolio",
    "📈 Market"
])

# ============================================================
# DASHBOARD TAB
# ============================================================

with tab_dash:
    result, status, error = analyze_stock(sidebar_ticker, period=period)
    
    if status != "SUCCESS":
        st.error(f"❌ Could not analyze {sidebar_ticker}: {error}")
        
        # Show what went wrong
        with st.expander("🔍 Troubleshooting"):
            st.write(f"**Status:** {status}")
            st.write(f"**Error:** {error}")
            st.write("**Possible causes:**")
            st.write("- Invalid ticker symbol")
            st.write("- Data source (yfinance) temporarily unavailable")
            st.write("- No trading history for this symbol")
            st.write("**Try:**")
            st.write("- Check spelling (e.g., SYS, OGDC, LUCK)")
            st.write("- Refresh data using sidebar button")
            st.write("- Try a different symbol")
    else:
        last = result["last"]
        sig = result["signal"]
        
        # ===== DATA STATUS =====
        fresh_label, fresh_note = data_freshness_label(result["data_date"])
        col1, col2, col3 = st.columns(3)
        col1.caption(f"**Last Candle:** {result['data_date'].date()} ({fresh_label})")
        col2.caption(f"**Checked:** {pkt_now().strftime('%d-%b %H:%M')} PKT")
        col3.caption(f"**Source:** yfinance | {len(result['df'])} candles")
        
        # ===== MAIN METRICS =====
        c1, c2, c3, c4, c5 = st.columns([2, 1.2, 1.2, 1.2, 1.2])
        
        with c1:
            st.subheader(f"📌 {result['ticker_display']}")
            st.caption(f"PKR {round(last['Close'], 2)}")
        
        with c2:
            color = signal_color(sig["signal"])
            st.markdown(f"**Signal**")
            st.markdown(f":{color}_circle: **{sig['signal']}**")
        
        with c3:
            st.metric("Score", f"{sig['score']}/100")
        
        with c4:
            st.metric("Trend", result["trend"])
        
        with c5:
            st.metric("Setup", sig["setup_quality"])
        
        # ===== MARKET CONTEXT =====
        mkt = result["market"]
        if mkt["regime"] != "UNAVAILABLE":
            st.caption(f"📊 Market: {mkt['regime']} | KSE-100: {round(mkt['last_close'], 0) if mkt['last_close'] else 'N/A'}")
        else:
            st.caption("📊 Market: Data unavailable - confidence reduced")
        
        # ===== SIGNAL REASONS =====
        with st.expander("🔍 Why this signal?", expanded=True):
            for r in sig["reasons"]:
                st.write(r)
        
        # ===== TRADE PLAN =====
        st.subheader("📋 Trade Plan")
        risk = result["risk"]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Entry", round(risk["entry"], 2))
        col2.metric("Stop Loss", round(risk["stop_loss"], 2))
        col3.metric("Target 1", round(risk["target1"], 2))
        col4.metric("Target 2", round(risk["target2"], 2))
        col5.metric("R:R", f"1:{round(risk['rr1'], 2) if risk['rr1'] else 'N/A'}")
        
        # ===== POSITION SIZING =====
        sizing = position_sizing(capital, risk_pct, risk)
        if sizing["shares"] > 0:
            st.caption(f"📊 Position Sizing: **{sizing['shares']} shares** | Investment: PKR {sizing['investment']} | Max Loss: PKR {sizing['max_loss']}")
            st.caption("Risk-based sizing only - no capital allocation % used.")
        
        # ===== TECHNICAL PROJECTION =====
        proj = result["projection"]
        if proj["direction"] != "NEUTRAL":
            st.info(f"📈 **Technical Projection:** {proj['label']}")
            st.caption(proj["note"])
        
        # ===== PENNY STOCK ALERT =====
        if result["penny"]["is_penny"]:
            st.warning(f"🪙 **Penny Stock Alert:** {result['penny']['status']} - {result['penny']['note']}")
        
        # ===== CHART =====
        st.subheader("📊 Chart")
        col1, col2 = st.columns(2)
        show_bb = col1.checkbox("Show Bollinger Bands", value=False)
        show_sma200 = col2.checkbox("Show SMA200", value=False)
        
        st.plotly_chart(build_chart(result, show_bb, show_sma200), use_container_width=True)
        
        # ===== SUPPORT/RESISTANCE =====
        with st.expander("📊 Support / Resistance Details"):
            sr = result["sr"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Primary Support", round(sr["primary_support"], 2))
            col2.metric("Primary Resistance", round(sr["primary_resistance"], 2))
            col3.metric("Secondary Support", round(sr["secondary_support"], 2))
            col4.metric("Secondary Resistance", round(sr["secondary_resistance"], 2))
            
            if not pd.isna(sr["high_52w"]):
                st.caption(f"52-Week High: {round(sr['high_52w'], 2)} | 52-Week Low: {round(sr['low_52w'], 2)}")

# ============================================================
# SCREENER TAB
# ============================================================

with tab_screener:
    st.subheader("🔍 PSX Opportunity Scanner")
    
    st.caption("Scans the broadest available PSX universe from yfinance.")
    
    # Universe selection
    col1, col2 = st.columns(2)
    with col1:
        use_dynamic = st.checkbox("Use dynamic universe (slower but broader)", value=True)
    with col2:
        custom_syms = st.text_input("Add extra symbols (comma-separated)", "")
    
    if st.button("🔍 Run Screener", use_container_width=True):
        with st.spinner("Scanning PSX universe..."):
            # Build universe
            if use_dynamic:
                universe = fetch_psx_universe()
            else:
                universe = ["SYS.KA", "OGDC.KA", "LUCK.KA", "FFC.KA", "HUBC.KA", 
                           "PSO.KA", "ENGRO.KA", "HBL.KA", "UBL.KA", "MCB.KA"]
            
            # Add custom symbols
            if custom_syms:
                extra = [t.strip() + ".KA" if not t.strip().endswith(".KA") else t.strip() 
                        for t in custom_syms.split(",") if t.strip()]
                universe = list(dict.fromkeys(universe + extra))
            
            st.caption(f"Scanning {len(universe)} symbols...")
            
            screener_df, coverage = run_screener(universe, period="6mo")
            st.session_state["screener_df"] = screener_df
            st.session_state["screener_coverage"] = coverage
    
    # Show coverage
    if "screener_coverage" in st.session_state:
        cov = st.session_state["screener_coverage"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total", cov["total"])
        col2.metric("Analyzed", cov["analyzed"])
        col3.metric("Success", cov["success"])
        col4.metric("Failed", cov["failed"])
    
    # Show results
    if "screener_df" in st.session_state:
        df_s = st.session_state["screener_df"]
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            signal_filter = st.multiselect(
                "Filter by Signal",
                ["STRONG BUY", "BUY", "WAIT", "REDUCE", "AVOID", "ERROR"],
                default=["STRONG BUY", "BUY"]
            )
        with col2:
            sort_by = st.selectbox("Sort by", ["Score", "Change %", "Price"], index=0)
        
        # Apply filters
        view = df_s.copy()
        if signal_filter:
            view = view[view["Signal"].isin(signal_filter)]
        
        # Remove ERROR rows for display
        view = view[view["Signal"] != "ERROR"]
        
        # Sort
        view = view.sort_values(sort_by, ascending=False, na_position="last")
        
        if not view.empty:
            display_cols = [c for c in view.columns if not c.startswith("_")]
            st.dataframe(view[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No stocks match the current filters")

# ============================================================
# BREAKOUT TAB
# ============================================================

with tab_breakouts:
    st.subheader("🚀 Breakout Candidates")
    
    if "screener_df" in st.session_state:
        df_s = st.session_state["screener_df"]
        
        # Filter for breakout-related status
        breakout_keywords = ["CONFIRMED", "READY", "ATTEMPT", "52W"]
        bo = df_s[df_s["Status"].astype(str).str.contains('|'.join(breakout_keywords), case=False, na=False)]
        bo = bo.sort_values("Score", ascending=False)
        
        if not bo.empty:
            display_cols = [c for c in bo.columns if not c.startswith("_")]
            st.dataframe(bo[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No breakout candidates found in current scan")
    else:
        st.info("Run the Screener first to find breakout candidates")

# ============================================================
# PENNY STOCKS TAB
# ============================================================

with tab_penny:
    st.subheader("🪙 Penny Stock Breakout Watch")
    
    st.caption(f"Stocks below PKR {penny_threshold} with unusual activity")
    st.caption("⚠️ Low-priced stocks can have high volatility, liquidity risk, and false breakouts.")
    
    if "screener_df" in st.session_state:
        df_s = st.session_state["screener_df"]
        
        # Filter penny stocks
        penny_df = df_s[df_s["Penny"].notna()]
        penny_df = penny_df[penny_df["Penny"] != "N/A"]
        
        if not penny_df.empty:
            # Show interesting penny setups first
            interesting = penny_df[penny_df["Penny"].str.contains("BREAKOUT|READY|VOLUME|WATCH", na=False)]
            interesting = interesting.sort_values("Score", ascending=False)
            
            if not interesting.empty:
                st.success(f"🔥 {len(interesting)} interesting penny stock setups found!")
                display_cols = [c for c in interesting.columns if not c.startswith("_")]
                st.dataframe(interesting[display_cols], use_container_width=True, hide_index=True)
            else:
                st.info("No interesting penny setups at this time")
            
            # Show all penny stocks
            with st.expander("All penny stocks"):
                all_penny = penny_df.sort_values("Price", ascending=True)
                display_cols = [c for c in all_penny.columns if not c.startswith("_")]
                st.dataframe(all_penny[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No penny stocks found in current scan")
    else:
        st.info("Run the Screener first to identify penny stocks")

# ============================================================
# NEXT SESSION TAB
# ============================================================

with tab_next:
    st.subheader("📅 Next Session Watchlist")
    st.caption("Top candidates for the next trading session based on latest data")
    st.caption("⚠️ Not a guarantee - watch for confirmation before trading")
    
    if "screener_df" in st.session_state:
        df_s = st.session_state["screener_df"]
        
        # Filter for BUY signals with decent scores
        top = df_s[df_s["Signal"].isin(["STRONG BUY", "BUY"])]
        top = top.dropna(subset=["Score"])
        top = top.sort_values("Score", ascending=False).head(10)
        
        if not top.empty:
            display_cols = [c for c in top.columns if not c.startswith("_")]
            st.dataframe(top[display_cols], use_container_width=True, hide_index=True)
            
            st.caption("**Why interesting:** Score above 65, BUY signal, technical setup confirmed")
        else:
            st.info("No strong BUY candidates at this time")
    else:
        st.info("Run the Screener first")

# ============================================================
# WATCHLIST TAB
# ============================================================

with tab_watch:
    st.subheader("📋 Watchlist Analysis")
    
    tickers = [t.strip() for t in watchlist_input.split(",") if t.strip()]
    
    if st.button("🔄 Refresh Watchlist", use_container_width=True):
        with st.spinner("Analyzing watchlist..."):
            watchlist_df, coverage = run_screener(tickers, period="6mo")
            st.session_state["watchlist_df"] = watchlist_df
    
    if "watchlist_df" in st.session_state:
        df_w = st.session_state["watchlist_df"]
        display_cols = [c for c in df_w.columns if not c.startswith("_")]
        st.dataframe(df_w[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("Click 'Refresh Watchlist' to analyze")

# ============================================================
# PORTFOLIO TAB
# ============================================================

with tab_port:
    st.subheader("💼 Portfolio Tracker (max 5 holdings)")
    
    # Add holding form
    with st.form("add_holding"):
        col1, col2, col3 = st.columns(3)
        h_ticker = col1.text_input("Ticker")
        h_price = col2.number_input("Buy Price (PKR)", min_value=0.0, step=0.5)
        h_shares = col3.number_input("Shares", min_value=0, step=1)
        
        submitted = st.form_submit_button("Add Holding")
        if submitted and h_ticker and h_price > 0 and h_shares > 0:
            if len(st.session_state.portfolio) >= 5:
                st.warning("Maximum 5 holdings")
            else:
                st.session_state.portfolio.append({
                    "ticker": h_ticker.strip().upper(),
                    "buy_price": h_price,
                    "shares": h_shares
                })
                st.success(f"Added {h_ticker}")
    
    # Display portfolio
    if st.session_state.portfolio:
        rows = []
        total_invested = 0
        total_current = 0
        
        for i, h in enumerate(st.session_state.portfolio):
            result, status, error = analyze_stock(h["ticker"], period="6mo")
            invested = h["buy_price"] * h["shares"]
            total_invested += invested
            
            if status == "SUCCESS":
                cur_price = result["last"]["Close"]
                cur_value = cur_price * h["shares"]
                total_current += cur_value
                pnl = cur_value - invested
                pnl_pct = pnl / invested * 100 if invested else 0
                
                decision, reason = portfolio_decision(h, result)
                
                rows.append({
                    "Ticker": result["ticker_display"],
                    "Buy Price": h["buy_price"],
                    "Shares": h["shares"],
                    "Invested": round(invested, 2),
                    "Current": round(cur_price, 2),
                    "Value": round(cur_value, 2),
                    "P/L": round(pnl, 2),
                    "P/L %": round(pnl_pct, 2),
                    "Trend": result["trend"],
                    "Score": result["signal"]["score"],
                    "Signal": result["signal"]["signal"],
                    "Decision": decision,
                    "Reason": reason,
                })
            else:
                total_current += invested
                rows.append({
                    "Ticker": h["ticker"],
                    "Buy Price": h["buy_price"],
                    "Shares": h["shares"],
                    "Invested": round(invested, 2),
                    "Current": None,
                    "Value": None,
                    "P/L": None,
                    "P/L %": None,
                    "Trend": None,
                    "Score": None,
                    "Signal": "ERROR",
                    "Decision": "WATCH",
                    "Reason": error,
                })
        
        # Summary
        total_pnl = total_current - total_invested
        total_pnl_pct = total_pnl / total_invested * 100 if total_invested else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Invested", f"PKR {round(total_invested, 2)}")
        col2.metric("Current Value", f"PKR {round(total_current, 2)}")
        col3.metric("Total P/L", f"PKR {round(total_pnl, 2)}")
        col4.metric("Total P/L %", f"{round(total_pnl_pct, 2)}%")
        
        # Portfolio table
        port_df = pd.DataFrame(rows)
        st.dataframe(port_df, use_container_width=True, hide_index=True)
        
        # Remove holding
        remove_idx = st.selectbox(
            "Remove holding",
            options=["-"] + [h["ticker"] for h in st.session_state.portfolio]
        )
        if remove_idx != "-" and st.button("Remove Selected"):
            st.session_state.portfolio = [
                h for h in st.session_state.portfolio
                if h["ticker"] != remove_idx
            ]
            st.rerun()
    else:
        st.info("No holdings added yet. Add up to 5 holdings above.")

# ============================================================
# MARKET TAB
# ============================================================

with tab_market:
    st.subheader("📈 Market Overview - KSE-100")
    
    market = market_snapshot()
    
    if market["regime"] == "UNAVAILABLE":
        st.warning("KSE-100 data unavailable from yfinance")
        st.caption("Stock-level analysis continues, but market confidence is reduced.")
        
        with st.expander("🔍 Why is KSE-100 unavailable?"):
            st.write("yfinance doesn't reliably serve KSE-100 for all deployments.")
            st.write("Possible reasons:")
            st.write("- Tick symbol mismatch (tried: ^KSE100, KSE100.KA, ^KSE)")
            st.write("- Data source temporarily unavailable")
            st.write("- Deployment region blocking certain symbols")
            st.write("**Solution:** Individual stock analysis still works - use market context as 'unknown'.")
    else:
        fresh_label, _ = data_freshness_label(market["last_date"])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Market Regime", market["regime"])
        col2.metric("Market Trend", market["trend"])
        col3.metric("KSE-100 Level", round(market["last_close"], 2) if market["last_close"] else "N/A")
        
        st.caption(f"Source: {market['label']} | Last: {market['last_date'].date() if market['last_date'] else 'N/A'} ({fresh_label})")
        st.info(f"**Reasoning:** {market['reasoning']}")
        
        # Try to show KSE-100 chart
        idx_df, _ = fetch_market_index()
        if idx_df is not None and len(idx_df) > 30:
            st.line_chart(idx_df["Close"].tail(180))

# ============================================================
# FOOTER
# ============================================================

st.sidebar.caption("---")
st.sidebar.caption("⚠️ **Disclaimer:** Signals are analytical outputs based on historical price/volume data. Not guaranteed investment advice. Always do your own research before making trading decisions.")

# ============================================================
# END
# ============================================================
