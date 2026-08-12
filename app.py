"""
PSX QUANT ENGINE - Full Production Build (Phase 2)
=====================================================
A PSX-focused quantitative decision-support terminal.
Sections are isolated so the data provider can be swapped later
without touching the analytical engine.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="PSX Quant Engine", layout="wide")

# ============================================================
# CONFIG / UNIVERSE
# ============================================================
# Editable starting universe of liquid PSX names. Expand as needed.
PSX_UNIVERSE = [
    "OGDC", "PPL", "POL", "MARI", "PSO", "HUBC", "KAPCO", "ENGRO", "FFC",
    "FATIMA", "LUCK", "DGKC", "MLCF", "FCCL", "HBL", "UBL", "MCB", "BAFL",
    "ABL", "MEBL", "BAHL", "NBP", "ICI", "NML", "SYS", "TRG", "NETSOL",
    "INDU", "PSMC", "PIBTL", "SEARL", "AGP", "EPCL", "LOTCHEM", "GATM",
]

MIN_RR = 1.5  # minimum acceptable risk/reward for a trade to be considered actionable

WEIGHTS = {
    "trend": 0.20,
    "momentum": 0.15,
    "volume": 0.15,
    "setup": 0.20,
    "rr": 0.15,
    "sr": 0.10,
    "regime": 0.05,
}

MARKET_INDEX_CANDIDATES = ["^KSE100", "KSE100.KA", "PSX"]

# ============================================================
# SECTION: DATA PROVIDER
# (only section a future/alternate provider needs to replace,
#  as long as it keeps returning the same (df, status, error) contract)
# ============================================================

def normalize_ticker(raw):
    t = raw.strip().upper()
    if not t.endswith(".KA"):
        t = t + ".KA"
    return t


def _flatten_columns(df):
    """Handles yfinance MultiIndex columns, e.g. ('Close','SYS.KA'), which
    otherwise leave Close/Volume as None even when rows are populated."""
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = list(df.columns.get_level_values(0))
        known = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if known.intersection(set(lvl0)):
            df = df.copy()
            df.columns = lvl0
        else:
            df = df.copy()
            df.columns = df.columns.get_level_values(-1)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlcv(ticker, period="1y", interval="1d"):
    """Returns (dataframe_or_None, status, error_or_None).
    status in {SUCCESS, EMPTY, EXCEPTION}"""
    symbol = normalize_ticker(ticker)
    try:
        raw = yf.download(symbol, period=period, interval=interval,
                           auto_adjust=False, progress=False)
        if raw is None or raw.empty:
            return None, "EMPTY", "No data returned by provider."

        df = _flatten_columns(raw)
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, "EXCEPTION", "Missing columns after normalization: " + str(missing)

        df = df[required].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["Close", "High", "Low", "Open"])
        if df.empty:
            return None, "EMPTY", "Data became empty after cleaning."

        df.index = pd.to_datetime(df.index)
        df["Volume"] = df["Volume"].fillna(0)
        return df, "SUCCESS", None
    except Exception as e:
        return None, "EXCEPTION", type(e).__name__ + ": " + str(e)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_index():
    """Attempts a few known index ticker candidates. Returns (df_or_None, label_or_None)."""
    for cand in MARKET_INDEX_CANDIDATES:
        try:
            raw = yf.download(cand, period="6mo", interval="1d", auto_adjust=False, progress=False)
            if raw is not None and not raw.empty:
                df = _flatten_columns(raw)
                if "Close" in df.columns:
                    return df, cand
        except Exception:
            continue
    return None, None


# ============================================================
# SECTION: INDICATORS
# ============================================================

def sma(series, period):
    return series.rolling(period).mean()


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


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
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def adx(df, period=14):
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    atr_val = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_val.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx_val.fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def bollinger(series, period=20, num_std=2):
    mid = sma(series, period)
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def build_indicators(df):
    """Attaches every indicator series to a copy of df and returns it."""
    d = df.copy()
    d["SMA20"] = sma(d["Close"], 20)
    d["SMA50"] = sma(d["Close"], 50)
    d["SMA100"] = sma(d["Close"], 100)
    d["SMA200"] = sma(d["Close"], 200)
    d["EMA20"] = ema(d["Close"], 20)
    d["EMA50"] = ema(d["Close"], 50)
    d["RSI14"] = rsi(d["Close"], 14)
    macd_line, signal_line, hist = macd(d["Close"])
    d["MACD"] = macd_line
    d["MACD_SIGNAL"] = signal_line
    d["MACD_HIST"] = hist
    d["ATR14"] = atr(d, 14)
    adx_val, plus_di, minus_di = adx(d, 14)
    d["ADX14"] = adx_val
    d["PLUS_DI"] = plus_di
    d["MINUS_DI"] = minus_di
    bb_u, bb_m, bb_l = bollinger(d["Close"], 20, 2)
    d["BB_UPPER"] = bb_u
    d["BB_MID"] = bb_m
    d["BB_LOWER"] = bb_l
    d["VOL_SMA20"] = sma(d["Volume"], 20)
    d["VOL_RATIO"] = d["Volume"] / d["VOL_SMA20"].replace(0, np.nan)
    d["RETURN_1D"] = d["Close"].pct_change()
    d["VOLATILITY_20"] = d["RETURN_1D"].rolling(20).std() * np.sqrt(252)
    d["ROC_10"] = d["Close"].pct_change(10) * 100
    d["52W_HIGH"] = d["Close"].rolling(252, min_periods=20).max()
    d["52W_LOW"] = d["Close"].rolling(252, min_periods=20).min()
    return d


def min_history_ok(d, needed=210):
    """SMA200 etc need real history; flag if insufficient."""
    return len(d) >= needed


# ============================================================
# SECTION: MARKET REGIME
# ============================================================

def market_regime():
    idx_df, label = fetch_market_index()
    if idx_df is None or len(idx_df) < 60:
        return "UNAVAILABLE", None, "Market index data unavailable from provider."
    d = idx_df.copy()
    d["SMA20"] = sma(d["Close"], 20)
    d["SMA50"] = sma(d["Close"], 50)
    last = d.iloc[-1]
    vol20 = d["Close"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)

    if pd.isna(last["SMA20"]) or pd.isna(last["SMA50"]):
        return "UNAVAILABLE", label, "Insufficient index history."

    if vol20 is not None and vol20 > 0.35:
        return "HIGH VOLATILITY", label, None
    if last["Close"] > last["SMA20"] > last["SMA50"]:
        return "BULLISH", label, None
    if last["Close"] < last["SMA20"] < last["SMA50"]:
        return "BEARISH", label, None
    return "NEUTRAL", label, None


# ============================================================
# SECTION: TREND ENGINE
# ============================================================

def trend_engine(d):
    if len(d) < 55:
        return "INSUFFICIENT DATA", []
    last = d.iloc[-1]
    reasons = []
    bullish_points = 0
    bearish_points = 0

    has200 = not pd.isna(last.get("SMA200", np.nan))

    if last["Close"] > last["SMA20"]:
        bullish_points += 1; reasons.append("Price above SMA20")
    else:
        bearish_points += 1; reasons.append("Price below SMA20")

    if last["Close"] > last["SMA50"]:
        bullish_points += 1; reasons.append("Price above SMA50")
    else:
        bearish_points += 1; reasons.append("Price below SMA50")

    if has200:
        if last["Close"] > last["SMA200"]:
            bullish_points += 1; reasons.append("Price above SMA200")
        else:
            bearish_points += 1; reasons.append("Price below SMA200")
        if last["SMA50"] > last["SMA200"]:
            bullish_points += 1; reasons.append("SMA50 above SMA200 (golden trend)")
        else:
            bearish_points += 1; reasons.append("SMA50 below SMA200 (death trend)")

    recent = d.tail(20)
    higher_lows = recent["Low"].iloc[-1] > recent["Low"].iloc[0]
    higher_highs = recent["High"].iloc[-1] > recent["High"].iloc[0]
    if higher_highs and higher_lows:
        bullish_points += 1; reasons.append("Higher highs and higher lows (last 20 sessions)")
    elif (not higher_highs) and (not higher_lows):
        bearish_points += 1; reasons.append("Lower highs and lower lows (last 20 sessions)")

    if last["ADX14"] >= 25:
        reasons.append("ADX " + str(round(last["ADX14"], 1)) + " indicates a trending market")

    total = bullish_points + bearish_points
    score_ratio = bullish_points / total if total else 0.5

    if score_ratio >= 0.85:
        trend = "STRONG UPTREND"
    elif score_ratio >= 0.6:
        trend = "UPTREND"
    elif score_ratio <= 0.15:
        trend = "STRONG DOWNTREND"
    elif score_ratio <= 0.4:
        trend = "DOWNTREND"
    else:
        trend = "SIDEWAYS"

    return trend, reasons


# ============================================================
# SECTION: SUPPORT / RESISTANCE
# ============================================================

def support_resistance(d):
    recent20 = d.tail(20)
    recent60 = d.tail(60) if len(d) >= 60 else d

    primary_resistance = recent20["High"].max()
    primary_support = recent20["Low"].min()
    secondary_resistance = recent60["High"].max()
    secondary_support = recent60["Low"].min()

    return {
        "primary_support": primary_support,
        "primary_resistance": primary_resistance,
        "secondary_support": secondary_support,
        "secondary_resistance": secondary_resistance,
    }


# ============================================================
# SECTION: BREAKOUT ENGINE
# ============================================================

def breakout_engine(d, sr, vol_ratio_threshold=1.5):
    last = d.iloc[-1]
    prev = d.iloc[-2] if len(d) >= 2 else last
    resistance = sr["primary_resistance"]
    price = last["Close"]
    vol_ratio = last["VOL_RATIO"] if not pd.isna(last["VOL_RATIO"]) else 0

    was_below = prev["Close"] <= resistance
    now_above = price > resistance
    volume_confirmed = vol_ratio >= vol_ratio_threshold
    momentum_positive = last["MACD_HIST"] > 0

    distance_pct = (resistance - price) / price * 100 if price else None

    if now_above and volume_confirmed and momentum_positive:
        status = "BREAKOUT CONFIRMED"
    elif now_above and not (volume_confirmed and momentum_positive):
        status = "BREAKOUT WATCH (needs volume/momentum confirmation)"
    elif (not now_above) and prev["Close"] > resistance and price < resistance:
        status = "FAILED BREAKOUT"
    elif distance_pct is not None and 0 <= distance_pct <= 3:
        status = "BREAKOUT WATCH"
    else:
        status = "NO BREAKOUT"

    return {
        "status": status,
        "resistance": resistance,
        "price": price,
        "volume_ratio": vol_ratio,
        "distance_to_resistance_pct": distance_pct,
    }


# ============================================================
# SECTION: PULLBACK ENGINE
# ============================================================

def pullback_engine(d, trend, sr):
    last = d.iloc[-1]
    support = sr["primary_support"]
    price = last["Close"]

    if trend not in ("UPTREND", "STRONG UPTREND"):
        return {"status": "NO PULLBACK", "note": "Trend is not currently bullish."}

    near_support = abs(price - support) / price < 0.03 if price else False
    near_ema20 = abs(price - last["EMA20"]) / price < 0.02 if not pd.isna(last["EMA20"]) and price else False
    cooling_rsi = 35 <= last["RSI14"] <= 55
    bullish_candle = last["Close"] > last["Open"]

    if price < support * 0.98:
        return {"status": "BROKEN SUPPORT", "note": "Price closed meaningfully below primary support."}

    if (near_support or near_ema20) and cooling_rsi and bullish_candle:
        return {"status": "HEALTHY PULLBACK", "note": "Price near support/EMA20 with cooling RSI and a bullish recovery candle."}
    if near_support or near_ema20:
        return {"status": "PULLBACK WATCH", "note": "Price approaching support/EMA20, awaiting confirmation candle."}
    return {"status": "NO PULLBACK", "note": "Price not currently near a pullback zone."}


# ============================================================
# SECTION: MOMENTUM ENGINE
# ============================================================

def momentum_engine(d):
    last = d.iloc[-1]
    prev5 = d.tail(5)

    score = 0
    if last["RSI14"] > 55:
        score += 1
    if last["RSI14"] < 45:
        score -= 1
    if last["MACD_HIST"] > 0:
        score += 1
    else:
        score -= 1
    if last["MACD_HIST"] > d["MACD_HIST"].iloc[-2]:
        score += 1
    if last["ROC_10"] > 0:
        score += 1
    else:
        score -= 1

    # simple divergence check: price higher high, RSI lower high over last ~10 bars
    window = d.tail(10)
    bearish_div = window["Close"].iloc[-1] >= window["Close"].max() * 0.999 and window["RSI14"].iloc[-1] < window["RSI14"].max() - 5
    bullish_div = window["Close"].iloc[-1] <= window["Close"].min() * 1.001 and window["RSI14"].iloc[-1] > window["RSI14"].min() + 5

    if score >= 3:
        label = "STRONG POSITIVE"
    elif score >= 1:
        label = "POSITIVE"
    elif score <= -3:
        label = "STRONG NEGATIVE"
    elif score <= -1:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    overbought = last["RSI14"] > 70
    oversold = last["RSI14"] < 30

    return {
        "label": label,
        "overbought": overbought,
        "oversold": oversold,
        "bearish_divergence": bool(bearish_div),
        "bullish_divergence": bool(bullish_div),
    }


# ============================================================
# SECTION: RISK ENGINE (stop / targets / R:R)
# ============================================================

def risk_engine(d, sr):
    last = d.iloc[-1]
    price = last["Close"]
    atr_val = last["ATR14"] if not pd.isna(last["ATR14"]) else 0
    swing_low = d.tail(10)["Low"].min()

    stop_loss = min(swing_low, sr["primary_support"]) - 0.3 * atr_val
    risk_per_share = price - stop_loss

    # If price is already at/above the primary resistance (breakout territory),
    # that level is too close to current price to serve as a target - fall back
    # to secondary resistance, and use an ATR-based projection as a floor.
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
# SECTION: POSITION SIZING
# ============================================================

def position_sizing(capital, risk_pct, risk_data, max_alloc_pct=25):
    if risk_data["risk_per_share"] is None or risk_data["risk_per_share"] <= 0:
        return {"shares": 0, "investment": 0, "max_loss": 0, "note": "Invalid or non-positive risk per share."}

    max_risk_amount = capital * (risk_pct / 100)
    risk_based_shares = int(max_risk_amount // risk_data["risk_per_share"])

    max_alloc_amount = capital * (max_alloc_pct / 100)
    capital_based_shares = int(max_alloc_amount // risk_data["entry"]) if risk_data["entry"] > 0 else 0

    final_shares = max(0, min(risk_based_shares, capital_based_shares))
    investment = final_shares * risk_data["entry"]
    max_loss = final_shares * risk_data["risk_per_share"]

    return {
        "risk_based_shares": risk_based_shares,
        "capital_based_shares": capital_based_shares,
        "shares": final_shares,
        "investment": round(investment, 2),
        "max_loss": round(max_loss, 2),
        "max_risk_amount": round(max_risk_amount, 2),
    }


# ============================================================
# SECTION: SIGNAL ENGINE (composite score + final signal)
# ============================================================

def _trend_component(trend):
    mapping = {
        "STRONG UPTREND": 100, "UPTREND": 78, "SIDEWAYS": 50,
        "DOWNTREND": 22, "STRONG DOWNTREND": 0, "INSUFFICIENT DATA": 40,
    }
    return mapping.get(trend, 40)


def _momentum_component(mom):
    mapping = {"STRONG POSITIVE": 100, "POSITIVE": 72, "NEUTRAL": 50, "NEGATIVE": 28, "STRONG NEGATIVE": 0}
    val = mapping.get(mom["label"], 50)
    if mom["overbought"]:
        val -= 10
    if mom["bearish_divergence"]:
        val -= 15
    if mom["bullish_divergence"]:
        val += 10
    return max(0, min(100, val))


def _volume_component(vol_ratio):
    if vol_ratio is None or pd.isna(vol_ratio):
        return 40
    if vol_ratio >= 2:
        return 100
    if vol_ratio >= 1.5:
        return 85
    if vol_ratio >= 1.0:
        return 60
    return 30


def _setup_component(breakout_status, pullback_status):
    if breakout_status == "BREAKOUT CONFIRMED":
        return 100
    if pullback_status == "HEALTHY PULLBACK":
        return 88
    if breakout_status == "BREAKOUT WATCH" or "WATCH" in breakout_status:
        return 65
    if pullback_status == "PULLBACK WATCH":
        return 55
    if breakout_status == "FAILED BREAKOUT" or pullback_status == "BROKEN SUPPORT":
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
    # sweet spot: not too extended near resistance, not broken below support
    if 0.2 <= position <= 0.75:
        return 80
    if position < 0:
        return 15
    if position > 1.05:
        return 30
    return 55


def _regime_component(regime):
    mapping = {"BULLISH": 100, "NEUTRAL": 55, "HIGH VOLATILITY": 35, "BEARISH": 15, "UNAVAILABLE": 50}
    return mapping.get(regime, 50)


def signal_engine(d, trend, momentum, breakout, pullback, sr, risk_data, regime):
    components = {
        "trend": _trend_component(trend),
        "momentum": _momentum_component(momentum),
        "volume": _volume_component(breakout["volume_ratio"]),
        "setup": _setup_component(breakout["status"], pullback["status"]),
        "rr": _rr_component(risk_data["rr1"]),
        "sr": _sr_component(risk_data["entry"], sr),
        "regime": _regime_component(regime),
    }
    score = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
    score = round(score, 1)

    rr_ok = risk_data["rr1"] is not None and risk_data["rr1"] >= MIN_RR

    if not rr_ok:
        signal = "WAIT"
        setup_quality = "INVALID (R:R below " + str(MIN_RR) + ")"
    elif score >= 80:
        signal = "STRONG BUY"; setup_quality = "A+ SETUP"
    elif score >= 70:
        signal = "BUY"; setup_quality = "A SETUP"
    elif score >= 50:
        signal = "WAIT"; setup_quality = "B SETUP / WATCH"
    elif score >= 30:
        signal = "SELL"; setup_quality = "WEAK"
    else:
        signal = "STRONG SELL"; setup_quality = "AVOID"

    reasons = []
    if components["trend"] >= 70:
        reasons.append("+ Trend: " + trend)
    elif components["trend"] <= 35:
        reasons.append("- Trend: " + trend)
    if components["momentum"] >= 70:
        reasons.append("+ Momentum " + momentum["label"])
    elif components["momentum"] <= 35:
        reasons.append("- Momentum " + momentum["label"])
    if breakout["status"] == "BREAKOUT CONFIRMED":
        reasons.append("+ Breakout confirmed with volume " + str(round(breakout["volume_ratio"], 2)) + "x average")
    if pullback["status"] == "HEALTHY PULLBACK":
        reasons.append("+ Healthy pullback to support/EMA20")
    if pullback["status"] == "BROKEN SUPPORT":
        reasons.append("- Support broken")
    if momentum["overbought"]:
        reasons.append("- RSI overbought")
    if momentum["oversold"]:
        reasons.append("+ RSI oversold (potential reversal watch)")
    if not rr_ok:
        rr_display = round(risk_data["rr1"], 2) if risk_data["rr1"] is not None else "N/A"
        reasons.append("- Risk/Reward " + str(rr_display) + " below minimum " + str(MIN_RR))
    if regime == "BEARISH":
        reasons.append("- Market regime bearish, confidence reduced")
    elif regime == "BULLISH":
        reasons.append("+ Market regime bullish, confidence increased")

    return {
        "score": score,
        "components": components,
        "signal": signal,
        "setup_quality": setup_quality,
        "reasons": reasons,
    }


# ============================================================
# SECTION: FULL STOCK ANALYSIS (orchestrates all engines)
# ============================================================

def analyze_stock(ticker, period="1y"):
    df, status, error = fetch_ohlcv(ticker, period=period)
    if status != "SUCCESS":
        return None, status, error

    if len(df) < 30:
        return None, "INSUFFICIENT", "Only " + str(len(df)) + " candles available; need at least 30."

    d = build_indicators(df)
    regime, regime_label, regime_note = market_regime()
    trend, trend_reasons = trend_engine(d)
    sr = support_resistance(d)
    breakout = breakout_engine(d, sr)
    pullback = pullback_engine(d, trend, sr)
    momentum = momentum_engine(d)
    risk_data = risk_engine(d, sr)
    signal_data = signal_engine(d, trend, momentum, breakout, pullback, sr, risk_data, regime)

    has_sma200 = min_history_ok(d, 210)

    result = {
        "ticker": normalize_ticker(ticker),
        "df": d,
        "last": d.iloc[-1],
        "trend": trend,
        "trend_reasons": trend_reasons,
        "sr": sr,
        "breakout": breakout,
        "pullback": pullback,
        "momentum": momentum,
        "risk": risk_data,
        "signal": signal_data,
        "regime": regime,
        "regime_label": regime_label,
        "has_sma200": has_sma200,
        "data_date": d.index[-1],
    }
    return result, "SUCCESS", None


# ============================================================
# SECTION: POSITION SIZING WRAPPER FOR UI
# ============================================================

def sizing_for_result(result, capital, risk_pct, max_alloc_pct):
    return position_sizing(capital, risk_pct, result["risk"], max_alloc_pct)


# ============================================================
# SECTION: SCREENER
# ============================================================

def run_screener(universe, period="6mo"):
    rows = []
    for tkr in universe:
        result, status, error = analyze_stock(tkr, period=period)
        if status != "SUCCESS":
            rows.append({
                "Ticker": normalize_ticker(tkr), "Price": None, "Change %": None,
                "Trend": None, "RSI": None, "Vol Ratio": None, "Score": None,
                "Breakout": None, "Signal": "DATA ERROR", "Entry": None,
                "Stop": None, "Target1": None, "R:R": None, "Note": error,
            })
            continue

        last = result["last"]
        prev_close = result["df"]["Close"].iloc[-2] if len(result["df"]) >= 2 else last["Close"]
        change_pct = (last["Close"] - prev_close) / prev_close * 100 if prev_close else 0

        rows.append({
            "Ticker": result["ticker"],
            "Price": round(last["Close"], 2),
            "Change %": round(change_pct, 2),
            "Trend": result["trend"],
            "RSI": round(last["RSI14"], 1),
            "Vol Ratio": round(result["breakout"]["volume_ratio"], 2),
            "Score": result["signal"]["score"],
            "Breakout": result["breakout"]["status"],
            "Signal": result["signal"]["signal"],
            "Entry": round(result["risk"]["entry"], 2),
            "Stop": round(result["risk"]["stop_loss"], 2),
            "Target1": round(result["risk"]["target1"], 2),
            "R:R": round(result["risk"]["rr1"], 2) if result["risk"]["rr1"] else None,
            "Note": "",
        })
    return pd.DataFrame(rows)


# ============================================================
# SECTION: CHARTING
# ============================================================

def build_chart(result, show_bb=False):
    d = result["df"].tail(150)
    risk = result["risk"]
    sr = result["sr"]

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.15, 0.17, 0.18],
        vertical_spacing=0.03,
        subplot_titles=("Price", "Volume", "RSI (14)", "MACD"),
    )

    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="Price", increasing_line_color="#16C784", decreasing_line_color="#EA3943",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=d.index, y=d["SMA20"], line=dict(color="#4A9DE0", width=1), name="SMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA50"], line=dict(color="#D4A94D", width=1), name="SMA50"), row=1, col=1)
    if result["has_sma200"]:
        fig.add_trace(go.Scatter(x=d.index, y=d["SMA200"], line=dict(color="#9B59B6", width=1), name="SMA200"), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_UPPER"], line=dict(color="gray", width=1, dash="dot"), name="BB Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_LOWER"], line=dict(color="gray", width=1, dash="dot"), name="BB Lower"), row=1, col=1)

    fig.add_hline(y=sr["primary_resistance"], line_dash="dash", line_color="#EA3943", annotation_text="Resistance", row=1, col=1)
    fig.add_hline(y=sr["primary_support"], line_dash="dash", line_color="#16C784", annotation_text="Support", row=1, col=1)
    fig.add_hline(y=risk["stop_loss"], line_dash="dot", line_color="#D4A94D", annotation_text="Stop", row=1, col=1)
    fig.add_hline(y=risk["target1"], line_dash="dot", line_color="#4A9DE0", annotation_text="T1", row=1, col=1)

    vol_colors = np.where(d["Close"] >= d["Open"], "#16C784", "#EA3943")
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=vol_colors, name="Volume"), row=2, col=1)

    fig.add_trace(go.Scatter(x=d.index, y=d["RSI14"], line=dict(color="#4A9DE0", width=1.3), name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#EA3943", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#16C784", row=3, col=1)

    fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], line=dict(color="#4A9DE0", width=1), name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD_SIGNAL"], line=dict(color="#D4A94D", width=1), name="Signal"), row=4, col=1)
    hist_colors = np.where(d["MACD_HIST"] >= 0, "#16C784", "#EA3943")
    fig.add_trace(go.Bar(x=d.index, y=d["MACD_HIST"], marker_color=hist_colors, name="Hist"), row=4, col=1)

    fig.update_layout(
        height=780, showlegend=True, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_dark",
    )
    return fig


# ============================================================
# SECTION: PORTFOLIO DECISION ENGINE
# ============================================================

def portfolio_decision(holding, result):
    if result is None:
        return "WATCH", "Live data unavailable for this holding right now."

    signal = result["signal"]["signal"]
    trend = result["trend"]
    pullback_status = result["pullback"]["status"]
    price = result["last"]["Close"]
    pnl_pct = (price - holding["buy_price"]) / holding["buy_price"] * 100 if holding["buy_price"] else 0

    if pullback_status == "BROKEN SUPPORT" or trend in ("DOWNTREND", "STRONG DOWNTREND"):
        return "REDUCE / EXIT", "Support broken or trend turned bearish; structure has deteriorated regardless of P/L."

    if signal in ("STRONG BUY", "BUY") and trend in ("UPTREND", "STRONG UPTREND"):
        if pnl_pct > 0:
            return "HOLD / TRAIL STOP", "Trend remains bullish and position is profitable; trail the stop rather than exit early."
        return "HOLD", "Trend and signal remain constructive even though position is not yet profitable."

    if result["breakout"]["status"] == "BREAKOUT CONFIRMED" and pnl_pct > 0:
        return "ADD ON CONFIRMATION", "Fresh confirmed breakout with volume support while already in profit."

    if signal in ("SELL", "STRONG SELL"):
        return "REDUCE / EXIT", "Signal engine flags deteriorating technical setup."

    return "HOLD / WATCH", "No major breakdown or fresh confirmation; maintain position and monitor."


# ============================================================
# SECTION: UI HELPERS
# ============================================================

def signal_color(signal):
    return {
        "STRONG BUY": "#16C784", "BUY": "#16C784", "WAIT": "#D4A94D",
        "SELL": "#EA3943", "STRONG SELL": "#EA3943",
    }.get(signal, "#8A94A6")


def metric_row(items):
    cols = st.columns(len(items))
    for c, (label, value) in zip(cols, items):
        c.metric(label, value)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("PSX Quant Engine")
sidebar_ticker = st.sidebar.text_input("Ticker", value="SYS")
capital = st.sidebar.number_input("Capital (PKR)", min_value=10000, value=100000, step=10000)
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5)
max_alloc_pct = st.sidebar.slider("Max Capital Allocation (%)", 5, 100, 25, 5)
period = st.sidebar.selectbox("Analysis Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
watchlist_input = st.sidebar.text_area("Watchlist (comma-separated)", value="SYS, OGDC, HBL, LUCK, FFC")

st.sidebar.caption("Signals are analytical outputs based on historical price/volume data - not guaranteed investment advice.")

if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

# ============================================================
# MAIN TABS
# ============================================================

tab_dash, tab_screener, tab_breakouts, tab_next, tab_watch, tab_port, tab_market, tab_adv = st.tabs(
    ["Dashboard", "Screener", "Breakouts", "Next Session", "Watchlist", "Portfolio", "Market Overview", "Advanced Analysis"]
)

# ---------------- DASHBOARD ----------------
with tab_dash:
    result, status, error = analyze_stock(sidebar_ticker, period=period)

    if status != "SUCCESS":
        st.error("Could not analyze " + normalize_ticker(sidebar_ticker) + ": " + str(error))
    else:
        last = result["last"]
        prev_close = result["df"]["Close"].iloc[-2] if len(result["df"]) >= 2 else last["Close"]
        change = last["Close"] - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0
        sig = result["signal"]

        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c1:
            st.subheader(result["ticker"] + "  ·  PKR " + str(round(last["Close"], 2)))
            st.caption(("+" if change >= 0 else "") + str(round(change, 2)) + " (" + str(round(change_pct, 2)) + "%) · Data: " + str(result["data_date"].date()))
        with c2:
            st.markdown("**Signal**")
            st.markdown(":large_" + ("green" if "BUY" in sig["signal"] else "yellow" if sig["signal"] == "WAIT" else "red") + "_circle: " + sig["signal"])
        with c3:
            st.metric("Technical Score", str(sig["score"]) + "/100")
        with c4:
            st.metric("Trend", result["trend"])

        st.caption("Setup quality: " + sig["setup_quality"] + " · Market regime: " + result["regime"])

        with st.expander("Why this signal?", expanded=True):
            for r in sig["reasons"]:
                st.write(r)
            if not result["has_sma200"]:
                st.caption("Note: insufficient history for a reliable SMA200 read on this period.")

        st.markdown("#### Trade Plan")
        risk = result["risk"]
        rr1_display = round(risk["rr1"], 2) if risk["rr1"] else "N/A"
        metric_row([
            ("Entry", str(round(risk["entry"], 2))),
            ("Stop Loss", str(round(risk["stop_loss"], 2))),
            ("Target 1", str(round(risk["target1"], 2))),
            ("Target 2", str(round(risk["target2"], 2))),
            ("R:R (T1)", "1 : " + str(rr1_display)),
        ])

        sizing = sizing_for_result(result, capital, risk_pct, max_alloc_pct)
        metric_row([
            ("Shares to Buy", str(sizing["shares"])),
            ("Investment", "PKR " + str(sizing["investment"])),
            ("Max Loss", "PKR " + str(sizing["max_loss"])),
        ])

        st.markdown("#### Chart")
        show_bb = st.checkbox("Show Bollinger Bands", value=False)
        st.plotly_chart(build_chart(result, show_bb), use_container_width=True)

        st.markdown("#### Market Status")
        sr = result["sr"]
        metric_row([
            ("Support", str(round(sr["primary_support"], 2))),
            ("Resistance", str(round(sr["primary_resistance"], 2))),
            ("Breakout Status", result["breakout"]["status"]),
            ("Pullback Status", result["pullback"]["status"]),
            ("Momentum", result["momentum"]["label"]),
        ])

# ---------------- SCREENER ----------------
with tab_screener:
    st.markdown("#### PSX Screener")
    st.caption("Universe: " + str(len(PSX_UNIVERSE)) + " configured liquid PSX names (editable in code). Not the full PSX listing.")
    run_btn = st.button("Run Screener")
    if run_btn:
        with st.spinner("Scanning universe..."):
            screener_df = run_screener(PSX_UNIVERSE, period="6mo")
        st.session_state["screener_df"] = screener_df

    if "screener_df" in st.session_state:
        df_s = st.session_state["screener_df"]
        filt = st.multiselect("Filter by Signal", ["STRONG BUY", "BUY", "WAIT", "SELL", "STRONG SELL", "DATA ERROR"], default=[])
        sort_by = st.selectbox("Sort by", ["Score", "Vol Ratio", "R:R", "Change %"], index=0)
        view = df_s.copy()
        if filt:
            view = view[view["Signal"].isin(filt)]
        view = view.sort_values(sort_by, ascending=False, na_position="last")
        st.dataframe(view, use_container_width=True, hide_index=True)

# ---------------- BREAKOUTS ----------------
with tab_breakouts:
    st.markdown("#### Breakout Scanner")
    if "screener_df" in st.session_state:
        df_s = st.session_state["screener_df"]
        bo = df_s[df_s["Breakout"].astype(str).str.contains("BREAKOUT", na=False)]
        bo = bo.sort_values("Score", ascending=False)
        st.dataframe(bo, use_container_width=True, hide_index=True)
    else:
        st.info("Run the Screener tab first to populate breakout candidates.")

# ---------------- NEXT SESSION ----------------
with tab_next:
    st.markdown("#### Next Session Opportunities")
    st.caption("Top candidates for the next trading session based on the latest available data. Not a certainty.")
    if "screener_df" in st.session_state:
        df_s = st.session_state["screener_df"].dropna(subset=["Score"])
        top = df_s.sort_values("Score", ascending=False).head(10)
        st.dataframe(top, use_container_width=True, hide_index=True)
    else:
        st.info("Run the Screener tab first.")

# ---------------- WATCHLIST ----------------
with tab_watch:
    st.markdown("#### Watchlist")
    tickers = [t.strip() for t in watchlist_input.split(",") if t.strip()]
    if st.button("Refresh Watchlist"):
        wl_df = run_screener(tickers, period="6mo")
        st.session_state["watchlist_df"] = wl_df
    if "watchlist_df" in st.session_state:
        st.dataframe(st.session_state["watchlist_df"], use_container_width=True, hide_index=True)

# ---------------- PORTFOLIO ----------------
with tab_port:
    st.markdown("#### Portfolio Tracker (max 5 holdings)")

    with st.form("add_holding"):
        c1, c2, c3 = st.columns(3)
        h_ticker = c1.text_input("Ticker")
        h_price = c2.number_input("Buy Price", min_value=0.0, step=0.5)
        h_shares = c3.number_input("Shares", min_value=0, step=1)
        submitted = st.form_submit_button("Add Holding")
        if submitted and h_ticker and h_price > 0 and h_shares > 0:
            if len(st.session_state.portfolio) >= 5:
                st.warning("Maximum 5 holdings allowed. Remove one before adding another.")
            else:
                st.session_state.portfolio.append({"ticker": h_ticker.strip().upper(), "buy_price": h_price, "shares": h_shares})

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
                    "Ticker": normalize_ticker(h["ticker"]), "Buy Price": h["buy_price"], "Shares": h["shares"],
                    "Invested": round(invested, 2), "Current": round(cur_price, 2), "Value": round(cur_value, 2),
                    "P/L": round(pnl, 2), "P/L %": round(pnl_pct, 2), "Trend": result["trend"],
                    "Signal": result["signal"]["signal"], "Score": result["signal"]["score"],
                    "Decision": decision, "Reason": reason,
                })
            else:
                total_current += invested
                rows.append({
                    "Ticker": normalize_ticker(h["ticker"]), "Buy Price": h["buy_price"], "Shares": h["shares"],
                    "Invested": round(invested, 2), "Current": None, "Value": None, "P/L": None, "P/L %": None,
                    "Trend": None, "Signal": "DATA ERROR", "Score": None, "Decision": "WATCH", "Reason": error,
                })

        total_pnl = total_current - total_invested
        total_pnl_pct = total_pnl / total_invested * 100 if total_invested else 0
        metric_row([
            ("Total Invested", "PKR " + str(round(total_invested, 2))),
            ("Current Value", "PKR " + str(round(total_current, 2))),
            ("Total P/L", "PKR " + str(round(total_pnl, 2))),
            ("Total P/L %", str(round(total_pnl_pct, 2)) + "%"),
        ])

        port_df = pd.DataFrame(rows)
        st.dataframe(port_df, use_container_width=True, hide_index=True)

        remove_idx = st.selectbox("Remove holding (by ticker)", options=["-"] + [h["ticker"] for h in st.session_state.portfolio])
        if remove_idx != "-" and st.button("Remove Selected"):
            st.session_state.portfolio = [h for h in st.session_state.portfolio if h["ticker"] != remove_idx]
            st.rerun()
    else:
        st.info("No holdings added yet.")

# ---------------- MARKET OVERVIEW ----------------
with tab_market:
    st.markdown("#### Market Overview")
    regime, regime_label, regime_note = market_regime()
    if regime == "UNAVAILABLE":
        st.warning("Market index data unavailable from provider. Stock-level analysis continues to function independently.")
    else:
        st.metric("Market Regime (" + str(regime_label) + ")", regime)
        idx_df, _ = fetch_market_index()
        if idx_df is not None:
            st.line_chart(idx_df["Close"].tail(180))

# ---------------- ADVANCED ANALYSIS ----------------
with tab_adv:
    st.markdown("#### Advanced Technical Analysis")
    result, status, error = analyze_stock(sidebar_ticker, period=period)
    if status != "SUCCESS":
        st.error("Could not analyze " + normalize_ticker(sidebar_ticker) + ": " + str(error))
    else:
        last = result["last"]
        d = result["df"]

        with st.expander("Trend Structure", expanded=True):
            st.write("Trend: " + result["trend"])
            for r in result["trend_reasons"]:
                st.write("- " + r)

        with st.expander("Moving Averages"):
            metric_row([
                ("SMA20", str(round(last["SMA20"], 2))),
                ("SMA50", str(round(last["SMA50"], 2))),
                ("SMA100", str(round(last["SMA100"], 2)) if not pd.isna(last["SMA100"]) else "insufficient data"),
                ("SMA200", str(round(last["SMA200"], 2)) if result["has_sma200"] else "insufficient history"),
            ])

        with st.expander("Momentum (RSI / MACD / ROC)"):
            metric_row([
                ("RSI14", str(round(last["RSI14"], 1))),
                ("MACD Hist", str(round(last["MACD_HIST"], 3))),
                ("ROC 10", str(round(last["ROC_10"], 2)) + "%"),
            ])
            st.write("Classification: " + result["momentum"]["label"])
            if result["momentum"]["bullish_divergence"]:
                st.write("Bullish divergence detected (price low vs RSI higher low).")
            if result["momentum"]["bearish_divergence"]:
                st.write("Bearish divergence detected (price high vs RSI lower high).")

        with st.expander("Volatility / ATR / Bollinger"):
            metric_row([
                ("ATR14", str(round(last["ATR14"], 2))),
                ("Volatility (ann.)", str(round(last["VOLATILITY_20"] * 100, 1)) + "%" if not pd.isna(last["VOLATILITY_20"]) else "N/A"),
                ("BB Upper", str(round(last["BB_UPPER"], 2)) if not pd.isna(last["BB_UPPER"]) else "N/A"),
                ("BB Lower", str(round(last["BB_LOWER"], 2)) if not pd.isna(last["BB_LOWER"]) else "N/A"),
            ])

        with st.expander("Support / Resistance / Breakout / Pullback"):
            sr = result["sr"]
            st.write("Primary Support: " + str(round(sr["primary_support"], 2)) + " · Secondary Support: " + str(round(sr["secondary_support"], 2)))
            st.write("Primary Resistance: " + str(round(sr["primary_resistance"], 2)) + " · Secondary Resistance: " + str(round(sr["secondary_resistance"], 2)))
            st.write("Breakout: " + result["breakout"]["status"])
            st.write("Pullback: " + result["pullback"]["status"] + " - " + result["pullback"]["note"])

        with st.expander("Score Breakdown"):
            comp = result["signal"]["components"]
            for k, v in comp.items():
                st.write(k.upper() + ": " + str(round(v, 1)) + "/100  (weight " + str(int(WEIGHTS[k] * 100)) + "%)")
            st.write("Composite Score: " + str(result["signal"]["score"]) + "/100")
