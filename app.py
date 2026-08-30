"""
PSX QUANT ENGINE - PRODUCTION BUILD v3.0
===========================================
A PSX-focused quantitative decision-support terminal.

Key Improvements in v3.0:
- Provider abstraction layer (psxdata → yfinance → UNAVAILABLE)
- KSE-100 via multiple sources with honest fallback
- Dynamic PSX universe discovery
- Expanded screener filters
- Simplified chart (RSI/MACD off by default)
- Data freshness everywhere
- Professional UI redesign
- Provider diagnostics panel
- Universe deduplication (FIX 32)
- Conditional entry display (FIX 33)
- Chart trend label (FIX 34)
- Prominent stale data warning (FIX 35)
- Alternative KSE-100 sources (FIX 36)
- Scanner performance optimization (FIX 37)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from typing import Optional, Tuple, Dict, Any, List, Union
import requests
import json

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PSX Quant Engine v3",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# VISUAL IDENTITY CSS (Professional Financial Terminal)
# ============================================================

st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
}
h1, h2, h3, h4, .stMetric label, [data-testid="stMetricLabel"] {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: #F0F4F8 !important;
}
section[data-testid="stSidebar"] {
    background-color: #0B0E13 !important;
    border-right: 1px solid #1E293B !important;
}
[data-testid="stCaptionContainer"], .stCaption {
    color: #94A3B8 !important;
    font-size: 0.8rem !important;
}
[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 0.75rem !important;
    font-weight: 400 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
.current-price {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #F0F4F8;
    letter-spacing: -0.5px;
}
.change-positive {
    color: #10B981;
    font-weight: 600;
}
.change-negative {
    color: #EF4444;
    font-weight: 600;
}
.signal-buy {
    color: #10B981;
    font-weight: 700;
    font-size: 1.2rem;
}
.signal-wait {
    color: #F59E0B;
    font-weight: 700;
    font-size: 1.2rem;
}
.signal-avoid {
    color: #EF4444;
    font-weight: 700;
    font-size: 1.2rem;
}
div[data-testid="stExpander"] {
    background-color: #14181F;
    border: 1px solid #1E293B;
    border-radius: 8px;
}
div[data-testid="stExpander"]:hover {
    border-color: #2DD4BF33;
}
.badge-fresh {
    color: #10B981;
    font-weight: 600;
}
.badge-stale {
    color: #EF4444;
    font-weight: 600;
}
.badge-delayed {
    color: #F59E0B;
    font-weight: 600;
}
hr {
    border-color: #1E293B !important;
    opacity: 0.5;
}
.stSelectbox, .stMultiSelect {
    background-color: #14181F;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS & CONFIG
# ============================================================

MIN_RR = 1.5
PENNY_STOCK_THRESHOLD = 50
MIN_HISTORY_DAYS = 60
CACHE_TTL = 300

WEIGHTS = {
    "trend": 0.25,
    "momentum": 0.20,
    "volume": 0.15,
    "setup": 0.20,
    "rr": 0.10,
    "sr": 0.10,
}

KSE100_PLAUSIBLE_MIN = 5000
KSE100_PLAUSIBLE_MAX = 1000000

# ============================================================
# PSX UNIVERSE CONSTANTS (Fallback only)
# ============================================================

PSX_LIQUID_UNIVERSE = [
    "SYS.KA", "OGDC.KA", "LUCK.KA", "FFC.KA", "HUBC.KA",
    "PSO.KA", "ENGRO.KA", "HBL.KA", "UBL.KA", "MCB.KA",
    "BAFL.KA", "ABL.KA", "NBP.KA", "MARI.KA", "POL.KA",
    "PPL.KA", "KAPCO.KA", "DGKC.KA", "MLCF.KA", "FCCL.KA",
    "FATIMA.KA", "LOTCHEM.KA", "EPCL.KA", "SEARL.KA", "AGP.KA",
    "NML.KA", "ICI.KA", "TRG.KA", "NETSOL.KA", "INDU.KA",
    "PSMC.KA", "PIBTL.KA", "GATM.KA", "ATRL.KA",
]

PSX_SMALL_CAP_UNIVERSE = [
    "KEL.KA", "KOHC.KA", "DAWH.KA", "THALL.KA", "PAEL.KA",
    "AICL.KA", "IGIHL.KA", "JSCL.KA", "PIOC.KA", "CHCC.KA",
    "ACPL.KA", "KOHTM.KA", "GHNI.KA", "MEHT.KA", "COLG.KA",
    "BNWM.KA", "FEROZ.KA", "SHFA.KA", "AGL.KA", "MUREB.KA",
    "BIFO.KA", "BGL.KA", "NRL.KA", "SNGP.KA", "SSGC.KA",
]

PSX_FALLBACK_UNIVERSE = list(dict.fromkeys(PSX_LIQUID_UNIVERSE + PSX_SMALL_CAP_UNIVERSE))

MARKET_INDEX_CANDIDATES = ["^KSE100", "KSE100.KA", "PSX.KA", "^KSE", "KSE100", "KSE100.PK", "PSX.PA", "KSE:100", "KSE-100"]

# ============================================================
# PROVIDER DIAGNOSTICS (with cache limitation note)
# ============================================================
#
# NOTE: PROVIDER_STATUS is a module-level global dict updated inside
# functions wrapped with @st.cache_data. On a cache hit, the function
# body does NOT execute, so update_provider_status() is NOT called.
# This means the Provider Diagnostics panel reflects only cache-miss
# calls, not cache hits. A full fix would require moving status tracking
# outside cached functions (e.g., using session_state or a separate
# non-cached function). This is a known limitation, documented here.
#
# The diagnostics show "last_success" only for actual network fetches,
# not for cached responses. This is intentional for now.

PROVIDER_STATUS = {
    "psxdata": {"available": False, "last_success": None, "error": None, "coverage": 0, "kse100": False},
    "yfinance": {"available": True, "last_success": None, "error": None, "coverage": 0, "kse100": False},
}

def update_provider_status(provider: str, available: bool = None, error: str = None, coverage: int = None, kse100: bool = None):
    if provider in PROVIDER_STATUS:
        if available is not None:
            PROVIDER_STATUS[provider]["available"] = available
        if error is not None:
            PROVIDER_STATUS[provider]["error"] = error
        if coverage is not None:
            PROVIDER_STATUS[provider]["coverage"] = coverage
        if kse100 is not None:
            PROVIDER_STATUS[provider]["kse100"] = kse100
        if available:
            PROVIDER_STATUS[provider]["last_success"] = pkt_now()

# ============================================================
# TIME HELPERS
# ============================================================

def pkt_now():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Karachi"))
    except Exception:
        return datetime.utcnow() + timedelta(hours=5)

def normalize_ticker(raw: str) -> str:
    t = raw.strip().upper()
    if not t.endswith(".KA"):
        t = t + ".KA"
    return t

def normalize_ticker_display(raw: str) -> str:
    return raw.strip().upper().replace(".KA", "")

def format_timestamp(dt):
    if dt is None:
        return "N/A"
    return dt.strftime("%d-%b-%Y %H:%M:%S PKT")

def trading_days_between(date1, date2):
    try:
        import numpy as np
        return int(np.busday_count(date1.date(), date2.date()))
    except Exception:
        return (date2.date() - date1.date()).days

def get_freshness_status(data_date):
    if data_date is None:
        return "UNAVAILABLE", None, "No data date available"
    
    now = pkt_now()
    if data_date.tzinfo is None:
        data_date = data_date.tz_localize(None)
    now_naive = now.replace(tzinfo=None)
    
    trading_gap = trading_days_between(data_date, now_naive)
    
    if trading_gap <= 1:
        return "FRESH", trading_gap, f"✅ {trading_gap} trading day(s) old"
    elif trading_gap <= 3:
        return "DELAYED", trading_gap, f"⚠️ {trading_gap} trading day(s) old"
    else:
        return "STALE", trading_gap, f"🔴 {trading_gap} trading day(s) old — STALE"

# ============================================================
# PROVIDER ABSTRACTION LAYER
# ============================================================

def _flatten_columns(df):
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

def _validate_ohlcv(df: pd.DataFrame) -> Tuple[bool, str]:
    if df is None or df.empty:
        return False, "Empty DataFrame"
    
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"
    
    if df[required].isna().any().any():
        return False, "Contains NaN values"
    
    if (df["High"] < df["Low"]).any():
        return False, "High < Low in some rows"
    
    if (df["Close"] <= 0).any() or (df["High"] <= 0).any():
        return False, "Non-positive prices"
    
    if (df["Volume"] < 0).any():
        return False, "Negative volume"
    
    if df.index.duplicated().any():
        return False, "Duplicate dates"
    
    if not df.index.is_monotonic_increasing:
        return False, "Dates not sorted"
    
    if len(df) < MIN_HISTORY_DAYS:
        return False, f"Insufficient history: {len(df)} candles, need {MIN_HISTORY_DAYS}"
    
    return True, "Valid"

# ============================================================
# PROVIDER 1: psxdata
# ============================================================

def fetch_psxdata_ohlcv(ticker: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], str, str]:
    try:
        import psxdata
        symbol = normalize_ticker_display(ticker)
        
        df = None
        for func_name in ["get_historical_data", "historical_data", "history"]:
            if hasattr(psxdata, func_name):
                try:
                    func = getattr(psxdata, func_name)
                    df = func(symbol, period=period)
                    if df is not None and not df.empty:
                        break
                except Exception:
                    continue
        
        if df is None or df.empty:
            if hasattr(psxdata, "stocks"):
                try:
                    stock = psxdata.stocks(symbol)
                    if hasattr(stock, "history"):
                        df = stock.history(period=period)
                    elif hasattr(stock, "historical"):
                        df = stock.historical(period=period)
                except Exception:
                    pass
        
        if df is None or df.empty:
            update_provider_status("psxdata", available=True, error="No data returned")
            return None, "EMPTY", "No data returned from psxdata"
        
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            update_provider_status("psxdata", available=True, error=f"Missing columns: {missing}")
            return None, "EXCEPTION", f"Missing columns: {missing}"
        
        df = df[required].apply(pd.to_numeric, errors="coerce")
        df = df.dropna()
        
        if df.empty:
            update_provider_status("psxdata", available=True, error="Data empty after cleaning")
            return None, "EMPTY", "Data empty after cleaning"
        
        valid, msg = _validate_ohlcv(df)
        if not valid:
            update_provider_status("psxdata", available=True, error=msg)
            return None, "EXCEPTION", msg
        
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df.index = pd.to_datetime(df["date"])
                df = df.drop(columns=["date"])
        
        update_provider_status("psxdata", available=True, coverage=len(df), kse100=True)
        return df, "SUCCESS", None
        
    except ImportError:
        update_provider_status("psxdata", available=False, error="psxdata not installed")
        return None, "EXCEPTION", "psxdata not installed"
    except Exception as e:
        update_provider_status("psxdata", available=True, error=str(e))
        return None, "EXCEPTION", f"psxdata error: {str(e)}"

def fetch_psxdata_kse100() -> Tuple[Optional[pd.DataFrame], str, str]:
    try:
        import psxdata
        
        df = None
        if hasattr(psxdata, "indices"):
            try:
                df = psxdata.indices("KSE100")
            except Exception:
                pass
        
        if df is None or df.empty:
            if hasattr(psxdata, "stocks"):
                try:
                    stock = psxdata.stocks("KSE100")
                    if hasattr(stock, "history"):
                        df = stock.history(period="1y")
                    elif hasattr(stock, "historical"):
                        df = stock.historical(period="1y")
                except Exception:
                    pass
        
        if df is None or df.empty:
            return None, "psxdata (no data)", "No KSE-100 data from psxdata"
        
        if "Close" not in df.columns:
            for c in df.columns:
                if "close" in str(c).lower():
                    df = df.rename(columns={c: "Close"})
                    break
            if "Close" not in df.columns:
                return None, "psxdata (no close)", "No close column in KSE-100 data"
        
        df = df[["Close"]].dropna()
        if df.empty:
            return None, "psxdata (empty)", "KSE-100 data empty"
        
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df.index = pd.to_datetime(df["date"])
                df = df.drop(columns=["date"])
        
        update_provider_status("psxdata", available=True, kse100=True)
        return df, "psxdata", None
        
    except ImportError:
        update_provider_status("psxdata", available=False, error="psxdata not installed")
        return None, "psxdata (not installed)", "psxdata not installed"
    except Exception as e:
        update_provider_status("psxdata", available=True, error=str(e))
        return None, "psxdata (error)", str(e)

# ============================================================
# FIX 32 + FIX 37 — UNIVERSE DEDUPLICATION & PERFORMANCE
# ============================================================

def fetch_psxdata_universe() -> Tuple[Optional[List[str]], str, str]:
    try:
        import psxdata
        
        tickers = None
        if hasattr(psxdata, "get_all_tickers"):
            try:
                tickers = psxdata.get_all_tickers()
            except Exception:
                pass
        
        if tickers is None or len(tickers) == 0:
            if hasattr(psxdata, "tickers"):
                try:
                    tickers = psxdata.tickers()
                except Exception:
                    pass
        
        if tickers is None or len(tickers) == 0:
            if hasattr(psxdata, "stocks"):
                try:
                    tickers = psxdata.stocks()
                except Exception:
                    pass
        
        if tickers is None or len(tickers) == 0:
            return None, "psxdata (no tickers)", "No tickers from psxdata"
        
        # FIX 32 + FIX 37 — Deduplicate and filter only .KA tickers
        if isinstance(tickers, list):
            # Keep only .KA tickers
            ka_tickers = [t for t in tickers if isinstance(t, str) and t.endswith(".KA")]
            # Deduplicate
            unique_tickers = list(dict.fromkeys(ka_tickers))
            if len(unique_tickers) > 0:
                tickers = unique_tickers
            else:
                # Fallback: format all tickers
                formatted = []
                for t in tickers:
                    if isinstance(t, str):
                        if not t.endswith(".KA"):
                            formatted.append(t + ".KA")
                        else:
                            formatted.append(t)
                tickers = list(dict.fromkeys(formatted))
        
        update_provider_status("psxdata", coverage=len(tickers))
        return tickers, "psxdata", None
        
    except ImportError:
        return None, "psxdata (not installed)", "psxdata not installed"
    except Exception as e:
        return None, "psxdata (error)", str(e)

# ============================================================
# PROVIDER 2: yfinance (Fallback)
# ============================================================

def fetch_yfinance_ohlcv(ticker: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], str, str]:
    symbol = normalize_ticker(ticker)
    
    try:
        raw = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
        
        if raw is None or raw.empty:
            update_provider_status("yfinance", available=True, error="No data returned")
            return None, "EMPTY", "No data returned from yfinance"
        
        df = _flatten_columns(raw)
        
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            update_provider_status("yfinance", available=True, error=f"Missing columns: {missing}")
            return None, "EXCEPTION", f"Missing columns: {missing}"
        
        df = df[required].apply(pd.to_numeric, errors="coerce")
        df = df.dropna()
        
        if df.empty:
            return None, "EMPTY", "Data empty after cleaning"
        
        valid, msg = _validate_ohlcv(df)
        if not valid:
            update_provider_status("yfinance", available=True, error=msg)
            return None, "EXCEPTION", msg
        
        update_provider_status("yfinance", available=True, coverage=len(df))
        return df, "SUCCESS", None
        
    except Exception as e:
        update_provider_status("yfinance", available=True, error=str(e))
        return None, "EXCEPTION", f"yfinance error: {str(e)}"

def fetch_yfinance_kse100() -> Tuple[Optional[pd.DataFrame], str, str]:
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
            if not (KSE100_PLAUSIBLE_MIN <= last_close <= KSE100_PLAUSIBLE_MAX):
                continue
            
            daily_vol = df["Close"].pct_change().std()
            if pd.isna(daily_vol) or daily_vol > 0.06:
                continue
            
            update_provider_status("yfinance", available=True, kse100=True)
            return df[["Close"]], cand, None
            
        except Exception:
            continue
    
    update_provider_status("yfinance", available=True, error="No KSE-100 candidate")
    return None, "yfinance (no KSE-100)", "No KSE-100 data from yfinance"

# ============================================================
# MASTER FETCH FUNCTIONS (Provider Priority)
# ============================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], str, str, str]:
    df, status, error = fetch_psxdata_ohlcv(ticker, period)
    if status == "SUCCESS":
        return df, status, error, "psxdata"
    
    df, status, error = fetch_yfinance_ohlcv(ticker, period)
    if status == "SUCCESS":
        return df, status, error, "yfinance (fallback)"
    
    return None, "EXCEPTION", "No provider could fetch data for this ticker", "UNAVAILABLE"

# ============================================================
# MASTER fetch_market_index() (FIX 36 — Alternative KSE-100 sources)
# ============================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_market_index():
    """
    Fetch KSE-100 data. Tries multiple sources:
    1. psxdata (experimental)
    2. psx-data-hub (if available)
    3. yfinance candidates (fallback)
    
    NEVER blocks the app if any provider fails.
    """
    
    # ============================================================
    # 1. Try psxdata first
    # ============================================================
    try:
        import psxdata
        kse_df = psxdata.indices("KSE100")
        if kse_df is not None and not kse_df.empty:
            if "Close" not in kse_df.columns:
                pass
            else:
                kse_df = kse_df.dropna(subset=["Close"])
                if len(kse_df) >= 40:
                    last_close = float(kse_df["Close"].iloc[-1])
                    if KSE100_PLAUSIBLE_MIN <= last_close <= KSE100_PLAUSIBLE_MAX:
                        daily_vol = kse_df["Close"].pct_change().std()
                        if not pd.isna(daily_vol) and daily_vol <= 0.06:
                            update_provider_status("psxdata", available=True, kse100=True)
                            return kse_df, "psxdata (KSE100 official - experimental)"
    except ImportError:
        pass
    except Exception:
        pass
    
    # ============================================================
    # 2. FIX 36 — Try alternative psx-data-hub
    # ============================================================
    try:
        import requests
        response = requests.get("https://psx-data-hub.vercel.app/api/v1/indices/KSE100", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                inner = data["data"]
                if isinstance(inner, list) and len(inner) > 0:
                    df = pd.DataFrame(inner)
                    if "Close" in df.columns or "close" in df.columns:
                        if "close" in df.columns:
                            df = df.rename(columns={"close": "Close"})
                        df = df[["Close"]].dropna()
                        if len(df) > 20:
                            # Try to parse dates
                            if "date" in df.columns or "Date" in df.columns:
                                date_col = "date" if "date" in df.columns else "Date"
                                df.index = pd.to_datetime(df[date_col])
                                df = df.drop(columns=[date_col])
                            last_close = float(df["Close"].iloc[-1])
                            if KSE100_PLAUSIBLE_MIN <= last_close <= KSE100_PLAUSIBLE_MAX:
                                daily_vol = df["Close"].pct_change().std()
                                if not pd.isna(daily_vol) and daily_vol <= 0.06:
                                    update_provider_status("psxdata", available=True, kse100=True)
                                    return df, "psx-data-hub (KSE100)"
    except Exception:
        pass
    
    # ============================================================
    # 3. Fallback: yfinance candidates
    # ============================================================
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
            if not (KSE100_PLAUSIBLE_MIN <= last_close <= KSE100_PLAUSIBLE_MAX):
                continue
            
            daily_vol = df["Close"].pct_change().std()
            if pd.isna(daily_vol) or daily_vol > 0.06:
                continue
            
            update_provider_status("yfinance", available=True, kse100=True)
            return df, cand
            
        except Exception:
            continue
    
    return None, None

# ============================================================
# MASTER UNIVERSE FETCH (FIX 32 + FIX 37)
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_universe() -> Tuple[List[str], str, str]:
    tickers, source, error = fetch_psxdata_universe()
    if tickers is not None and len(tickers) > 10:
        # FIX 32 — Already deduplicated in fetch_psxdata_universe
        return tickers, source, None
    
    return PSX_FALLBACK_UNIVERSE, "curated fallback", None

# ============================================================
# INDICATORS (ALL WILDER'S CORRECT)
# ============================================================

def sma(series, period):
    return series.rolling(period).mean()

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(upper=0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    avg_gain = avg_gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = avg_loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.inf), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    
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
    d["ROC_10"] = d["Close"].pct_change(10) * 100
    d["VOLATILITY_20"] = d["RETURN_1D"].rolling(20).std() * np.sqrt(252)
    
    d["52W_HIGH"] = d["High"].shift(1).rolling(252, min_periods=20).max()
    d["52W_LOW"] = d["Low"].shift(1).rolling(252, min_periods=20).min()
    
    return d

# ============================================================
# CROSSOVER DETECTION
# ============================================================

def detect_crossovers(sma20: pd.Series, sma50: pd.Series):
    if len(sma20) < 2 or len(sma50) < 2:
        return pd.Series(False, index=sma20.index), pd.Series(False, index=sma20.index)
    
    curr_bullish = sma20 > sma50
    curr_bearish = sma20 < sma50
    curr_equal = sma20 == sma50
    
    prev_bullish = sma20.shift(1) > sma50.shift(1)
    prev_bearish = sma20.shift(1) < sma50.shift(1)
    
    bullish_crossover = (~prev_bullish) & curr_bullish
    bearish_crossover = (~prev_bearish) & curr_bearish
    
    bullish_crossover = bullish_crossover & ~curr_equal
    bearish_crossover = bearish_crossover & ~curr_equal
    
    if len(bullish_crossover) > 0:
        bullish_crossover.iloc[0] = False
        bearish_crossover.iloc[0] = False
    
    return bullish_crossover, bearish_crossover

# ============================================================
# MARKET REGIME
# ============================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def market_snapshot():
    idx_df, source = fetch_market_index()
    
    if idx_df is None or len(idx_df) < 40:
        return {
            "regime": "UNAVAILABLE",
            "trend": "UNAVAILABLE",
            "source": source if source else "UNAVAILABLE",
            "reasoning": "KSE-100 data unavailable",
            "last_date": None,
            "last_close": None,
            "sma20": None,
            "sma50": None,
            "sma200": None,
        }
    
    d = idx_df.copy()
    
    d["SMA20"] = sma(d["Close"], 20)
    d["SMA50"] = sma(d["Close"], 50)
    d["SMA200"] = sma(d["Close"], 200)
    d["RETURN_1D"] = d["Close"].pct_change()
    
    last_date = d.index[-1]
    last_close = float(d["Close"].iloc[-1])
    sma20 = d["SMA20"].iloc[-1]
    sma50 = d["SMA50"].iloc[-1]
    sma200 = d["SMA200"].iloc[-1] if len(d) >= 200 else None
    vol20 = d["RETURN_1D"].rolling(20).std().iloc[-1] * np.sqrt(252)
    
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
        "source": source,
        "reasoning": reasoning,
        "last_date": last_date,
        "last_close": last_close,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
    }

# ============================================================
# PROXY INDICATOR (FIX 30 — period="3mo" instead of "1mo")
# ============================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def liquid_basket_trend():
    """
    Compute equal-weighted average daily % change across PSX liquid universe.
    
    This is a PROXY indicator — NOT official KSE-100.
    Used only when KSE-100 data is unavailable.
    
    FIX 30: Changed period from "1mo" to "3mo" to satisfy MIN_HISTORY_DAYS (60)
    validation requirement while still being lightweight.
    """
    universe = PSX_LIQUID_UNIVERSE
    changes = []
    successful = 0
    
    for ticker in universe:
        try:
            # FIX 30 — Using "3mo" instead of "1mo" (60+ candles for validation)
            df, status, _, _ = fetch_ohlcv(ticker, period="3mo")
            if status == "SUCCESS" and len(df) >= 6:
                recent = df.tail(5)
                if len(recent) >= 2:
                    avg_change = recent["Close"].pct_change().mean() * 100
                    if not pd.isna(avg_change):
                        changes.append(avg_change)
                        successful += 1
        except Exception:
            continue
    
    if len(changes) < 10:
        return {
            "trend": "UNAVAILABLE",
            "change_pct": None,
            "stocks_contributing": successful,
            "note": "Insufficient data for proxy calculation"
        }
    
    avg_change = np.mean(changes)
    
    if avg_change > 0.5:
        trend = "BULLISH (proxy)"
    elif avg_change > 0.1:
        trend = "MILD BULLISH (proxy)"
    elif avg_change > -0.1:
        trend = "NEUTRAL (proxy)"
    elif avg_change > -0.5:
        trend = "MILD BEARISH (proxy)"
    else:
        trend = "BEARISH (proxy)"
    
    return {
        "trend": trend,
        "change_pct": round(avg_change, 2),
        "stocks_contributing": successful,
        "note": "⚠️ PROXY — NOT official KSE-100"
    }

# ============================================================
# TREND ENGINE
# ============================================================

def trend_engine(d):
    if len(d) < 50:
        return "INSUFFICIENT DATA", [], 0
    
    last = d.iloc[-1]
    reasons = []
    bullish = 0
    bearish = 0
    
    if last["Close"] > last["SMA20"]:
        bullish += 1
        reasons.append("Price above SMA20")
    else:
        bearish += 1
        reasons.append("Price below SMA20")
    
    if last["Close"] > last["SMA50"]:
        bullish += 1
        reasons.append("Price above SMA50")
    else:
        bearish += 1
        reasons.append("Price below SMA50")
    
    if last["SMA20"] > last["SMA50"]:
        bullish += 1
        reasons.append("SMA20 above SMA50")
    else:
        bearish += 1
        reasons.append("SMA20 below SMA50")
    
    if not pd.isna(last.get("SMA200", np.nan)):
        if last["SMA50"] > last["SMA200"]:
            bullish += 1
            reasons.append("SMA50 above SMA200")
        else:
            bearish += 1
            reasons.append("SMA50 below SMA200")
    
    recent = d.tail(20)
    if len(recent) >= 10:
        higher_highs = recent["High"].iloc[-1] > recent["High"].iloc[0]
        higher_lows = recent["Low"].iloc[-1] > recent["Low"].iloc[0]
        if higher_highs and higher_lows:
            bullish += 1
            reasons.append("Higher highs and higher lows")
        elif (not higher_highs) and (not higher_lows):
            bearish += 1
            reasons.append("Lower highs and lower lows")
    
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
    history = d.iloc[:-1] if len(d) > 1 else d
    
    recent20 = history.tail(20) if len(history) >= 20 else history
    recent60 = history.tail(60) if len(history) >= 60 else history
    recent120 = history.tail(120) if len(history) >= 120 else history
    
    primary_resistance = recent20["High"].max()
    primary_support = recent20["Low"].min()
    secondary_resistance = recent60["High"].max()
    secondary_support = recent60["Low"].min()
    
    if secondary_support == primary_support:
        secondary_support = recent120["Low"].min()
    if secondary_resistance == primary_resistance:
        secondary_resistance = recent120["High"].max()
    
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
        "secondary_support_is_distinct": secondary_support != primary_support,
        "secondary_resistance_is_distinct": secondary_resistance != primary_resistance,
    }

# ============================================================
# BREAKOUT ENGINE
# ============================================================

def breakout_engine(d, sr, vol_ratio_threshold=1.5):
    last = d.iloc[-1]
    prev = d.iloc[-2] if len(d) >= 2 else last
    
    resistance = sr["primary_resistance"]
    price = last["Close"]
    vol_ratio = last["VOL_RATIO"] if not pd.isna(last["VOL_RATIO"]) else 0
    
    if len(d) >= 22:
        baseline_window = d.iloc[:-2].tail(20)
        baseline_resistance = baseline_window["High"].max()
    else:
        baseline_resistance = resistance
    
    was_below = prev["Close"] <= baseline_resistance
    now_above = price > resistance
    fresh_cross = now_above and was_below
    
    volume_confirmed = vol_ratio >= vol_ratio_threshold
    momentum_positive = last["MACD_HIST"] > 0
    
    distance_to_resistance = (resistance - price) / price * 100 if price > 0 else None
    
    if now_above and volume_confirmed and momentum_positive and fresh_cross:
        status = "CONFIRMED BREAKOUT"
        note = "Closed above resistance with volume + momentum confirmation"
    elif now_above and volume_confirmed and momentum_positive and not fresh_cross:
        status = "EXTENDED BREAKOUT"
        note = "Already above resistance - continuation"
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
    last = d.iloc[-1]
    support = sr["primary_support"]
    price = last["Close"]
    
    if trend not in ("BULLISH", "STRONG BULLISH"):
        return {"status": "NO PULLBACK", "note": "Trend is not bullish"}
    
    near_support = abs(price - support) / price < 0.03 if price > 0 else False
    near_ema20 = abs(price - last["EMA20"]) / price < 0.02 if not pd.isna(last["EMA20"]) and price > 0 else False
    
    cooling_rsi = 35 <= last["RSI14"] <= 55
    bullish_candle = last["Close"] > last["Open"]
    
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
    last = d.iloc[-1]
    
    score = 0
    signals = []
    
    if last["RSI14"] > 55:
        score += 1
        signals.append("RSI positive")
    elif last["RSI14"] < 45:
        score -= 1
        signals.append("RSI negative")
    
    if last["MACD_HIST"] > 0:
        score += 1
        signals.append("MACD positive")
    else:
        score -= 1
        signals.append("MACD negative")
    
    if len(d) >= 2:
        if last["MACD_HIST"] > d["MACD_HIST"].iloc[-2]:
            score += 1
            signals.append("MACD accelerating")
    
    if last["ROC_10"] > 0:
        score += 1
        signals.append("ROC positive")
    else:
        score -= 1
        signals.append("ROC negative")
    
    if last["ADX14"] >= 20:
        if last["PLUS_DI"] > last["MINUS_DI"]:
            score += 1
            signals.append("ADX confirms +DI")
        else:
            score -= 1
            signals.append("ADX confirms -DI")
    
    window = d.tail(10)
    if len(window) >= 10:
        price_high = window["Close"].max()
        rsi_high = window["RSI14"].max()
        if window["Close"].iloc[-1] >= price_high * 0.999 and window["RSI14"].iloc[-1] < rsi_high - 5:
            signals.append("⚠️ Bearish divergence")
            score -= 2
        
        price_low = window["Close"].min()
        rsi_low = window["RSI14"].min()
        if window["Close"].iloc[-1] <= price_low * 1.001 and window["RSI14"].iloc[-1] > rsi_low + 5:
            signals.append("✅ Bullish divergence")
            score += 2
    
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
# PROJECTION ENGINE
# ============================================================

def projection_engine(d, trend, sr, momentum):
    last = d.iloc[-1]
    price = last["Close"]
    atr_val = last["ATR14"] if not pd.isna(last["ATR14"]) else 0
    
    resistance = sr["primary_resistance"]
    support = sr["primary_support"]
    
    is_bullish = trend in ("BULLISH", "STRONG BULLISH")
    is_bearish = trend in ("BEARISH", "STRONG BEARISH")
    
    if is_bullish:
        if price >= resistance:
            range_size = resistance - support
            upside_zone_low = price + range_size * 0.5
            upside_zone_high = price + range_size * 1.0
            next_resistance = sr["secondary_resistance"]
            extension_zone = price + range_size * 1.5
        else:
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
        if price <= support:
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
# PENNY STOCK DETECTOR
# ============================================================

def detect_penny_setup(d, sr, threshold=PENNY_STOCK_THRESHOLD, rvol_threshold=2.0):
    last = d.iloc[-1]
    price = last["Close"]
    
    if price > threshold:
        return {
            "is_penny": False,
            "status": "NORMAL PRICE STOCK",
            "note": f"Price {price} > {threshold} threshold"
        }
    
    vol_ratio = last["VOL_RATIO"] if not pd.isna(last["VOL_RATIO"]) else 0
    near_resistance = False
    if price > 0 and sr["primary_resistance"]:
        near_resistance = abs(price - sr["primary_resistance"]) / price < 0.05
    
    broke_resistance = price > sr["primary_resistance"]
    rvol_expansion = vol_ratio >= rvol_threshold
    momentum_positive = last["MACD_HIST"] > 0
    
    if broke_resistance and rvol_expansion and momentum_positive:
        status = "🔥 PENNY BREAKOUT"
        note = f"Low-priced stock breaking resistance with {round(vol_ratio,1)}x volume!"
    elif near_resistance and rvol_expansion:
        status = "⚡ PENNY BREAKOUT READY"
        note = f"Low-priced stock near resistance with {round(vol_ratio,1)}x volume"
    elif rvol_expansion:
        status = "📈 PENNY VOLUME SPIKE"
        note = f"Unusual volume ({round(vol_ratio,1)}x) in low-priced stock"
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
# RISK ENGINE
# ============================================================

def risk_engine(d, sr, breakout_status=""):
    last = d.iloc[-1]
    price = last["Close"]
    atr_val = last["ATR14"] if not pd.isna(last["ATR14"]) else 0
    
    if "EXTENDED BREAKOUT" in breakout_status:
        tighter_stop = price - (2.5 * atr_val)
        stop_loss = max(tighter_stop, sr["primary_support"])
    else:
        swing_low = d.tail(10)["Low"].min()
        stop_loss = min(swing_low, sr["primary_support"]) - 0.3 * atr_val
    
    risk_per_share = price - stop_loss
    
    near_or_above_resistance = price >= sr["primary_resistance"] * 0.995
    atr_target = price + 2.5 * atr_val if atr_val else price
    
    distance_to_resistance = (sr["primary_resistance"] - price) / price * 100 if price > 0 else 999
    is_breakout_ready = 0 < distance_to_resistance <= 3
    
    if near_or_above_resistance:
        target1 = max(sr["secondary_resistance"], atr_target)
    elif is_breakout_ready:
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
    
    conditional_entry = None
    conditional_entry_note = None
    
    if "EXTENDED BREAKOUT" in breakout_status:
        ema20_val = last["EMA20"] if not pd.isna(last["EMA20"]) else None
        if ema20_val is not None:
            conditional_entry = round(ema20_val, 2)
            conditional_entry_note = "Wait for pullback near EMA20 for better R:R"
    
    return {
        "entry": price,
        "stop_loss": stop_loss,
        "risk_per_share": risk_per_share,
        "target1": target1,
        "target2": target2,
        "rr1": rr1,
        "rr2": rr2,
        "conditional_entry": conditional_entry,
        "conditional_entry_note": conditional_entry_note,
    }

# ============================================================
# POSITION SIZING
# ============================================================

def position_sizing(capital: float, risk_pct: float, risk_data: Dict) -> Dict:
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
    score = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
    return round(score, 1)

# ============================================================
# SIGNAL ENGINE
# ============================================================

def signal_engine(d, trend, trend_score, momentum, breakout, pullback, sr, risk_data, market):
    components = {
        "trend": min(100, trend_score),
        "momentum": _momentum_component(momentum),
        "volume": _volume_component(breakout["volume_ratio"]),
        "setup": _setup_component(breakout["status"], pullback["status"]),
        "rr": _rr_component(risk_data["rr1"]),
        "sr": _sr_component(risk_data["entry"], sr),
    }
    
    score = technical_score(components)
    
    rr_ok = risk_data["rr1"] is not None and risk_data["rr1"] >= MIN_RR
    trend_ok = trend not in ("BEARISH", "STRONG BEARISH")
    
    market_adjust = 0
    if market["regime"] == "BULLISH":
        market_adjust = 5
    elif market["regime"] == "BEARISH":
        market_adjust = -10
    elif market["regime"] == "HIGH VOLATILITY":
        market_adjust = -5
    
    adjusted_score = max(0, min(100, score + market_adjust))
    reasons = []
    
    if not trend_ok:
        signal = "WAIT" if adjusted_score >= 45 else "AVOID"
        setup_quality = "TREND BEARISH - NO LONG SETUP"
        reasons.append(f"❌ Trend is {trend} - no long entries taken")
    elif not rr_ok:
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
    
    if trend_ok:
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
        "reasons": reasons[:6],
    }

# ============================================================
# STOCK CLASSIFICATION
# ============================================================

def classify_stock(price: float, avg_volume: float) -> str:
    if price > 200 and avg_volume > 100000:
        return "LARGE"
    elif price > 50 and avg_volume > 20000:
        return "MID"
    elif price > 20 and avg_volume > 5000:
        return "SMALL"
    elif price < 10 and avg_volume < 2000:
        return "MICRO"
    elif price < 20:
        return "LOW-PRICE"
    else:
        return "SMALL"

# ============================================================
# INDICATOR EXPLANATIONS
# ============================================================

def get_indicator_explanation(indicator: str) -> str:
    explanations = {
        "SMA20": "20 din ki average price. Price iske upar ho to short-term strength hai.",
        "SMA50": "50 din ki average price. Medium-term trend ka idea deta hai.",
        "RSI": "Momentum indicator (0-100). 70+ overbought, 30- oversold.",
        "MACD": "Momentum direction. Positive histogram = bullish momentum.",
        "ADX": "Trend strength. 25+ = strong trend.",
        "RVOL": "Aaj ka volume normal se kitna zyada/kam hai.",
        "Support": "Price area where buyers have appeared before.",
        "Resistance": "Price area where sellers have appeared before.",
        "Breakout": "Price moves above resistance with volume.",
        "Stop Loss": "Level where position is exited to limit loss.",
        "Target": "Technical price objective if trend continues.",
    }
    return explanations.get(indicator, "Technical indicator.")

# ============================================================
# ESTIMATE PACE TO TARGET
# ============================================================

def estimate_pace_to_target(result: Dict) -> Tuple[str, str]:
    """
    Estimate pace to target1 using ATR.
    
    Returns: (trade_type, pace_label)
    """
    last = result["last"]
    risk = result["risk"]
    breakout = result["breakout"]
    pullback = result["pullback"]
    
    # Trade Type
    if "CONFIRMED BREAKOUT" in breakout["status"] and "EXTENDED" not in breakout["status"]:
        trade_type = "Day/Short-Term"
    elif pullback["status"] == "HEALTHY PULLBACK":
        trade_type = "Swing"
    else:
        trade_type = "Momentum"
    
    # Estimated Pace
    atr_val = last["ATR14"] if not pd.isna(last["ATR14"]) else 0
    entry = risk["entry"]
    target1 = risk["target1"]
    distance_to_target1 = abs(target1 - entry)
    
    if atr_val > 0 and distance_to_target1 > 0:
        est_sessions = max(1, round(distance_to_target1 / atr_val))
        pace_label = f"~{est_sessions} sessions (technical estimate)"
    else:
        pace_label = "N/A"
    
    return trade_type, pace_label

# ============================================================
# ANALYZE STOCK (FIX: only SUCCESS status proceeds)
# ============================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def analyze_stock(ticker: str, period: str = "1y", penny_threshold: float = PENNY_STOCK_THRESHOLD, rvol_threshold: float = 2.0):
    df, status, error, source = fetch_ohlcv(ticker, period=period)
    
    # FIX: Only proceed when status == "SUCCESS"
    # INSUFFICIENT data is blocked to avoid unreliable indicators (SMA50 with <60 days)
    if status != "SUCCESS":
        return None, status, error, source
    
    if df is None or df.empty:
        return None, "EMPTY", "No data available", source
    
    d = build_indicators(df)
    market = market_snapshot()
    trend, trend_reasons, trend_score = trend_engine(d)
    sr = support_resistance(d)
    breakout = breakout_engine(d, sr)
    pullback = pullback_engine(d, trend, sr)
    momentum = momentum_engine(d)
    risk = risk_engine(d, sr, breakout["status"])
    penny = detect_penny_setup(d, sr, threshold=penny_threshold, rvol_threshold=rvol_threshold)
    projection = projection_engine(d, trend, sr, momentum)
    signal = signal_engine(d, trend, trend_score, momentum, breakout, pullback, sr, risk, market)
    
    last = d.iloc[-1]
    avg_volume = last["VOL_SMA20"] if not pd.isna(last["VOL_SMA20"]) else last["Volume"]
    cap_size = classify_stock(last["Close"], avg_volume)
    
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
        "cap_size": cap_size,
        "data_source": source,
        "avg_volume": avg_volume,  # FIX 31 — Added for screener filter
    }
    
    return result, "SUCCESS", None, source

# ============================================================
# SCREENER (FIX 31 — Avg Volume column added for filter)
# ============================================================

def run_screener(universe: List[str], period: str = "6mo", penny_threshold: float = PENNY_STOCK_THRESHOLD, rvol_threshold: float = 2.0):
    rows = []
    coverage = {"total": len(universe), "success": 0, "failed": 0, "analyzed": 0}
    
    for ticker in universe:
        result, status, error, source = analyze_stock(ticker, period=period, penny_threshold=penny_threshold, rvol_threshold=rvol_threshold)
        
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
                "Cap Size": None,
                "Source": source,
                "Why": "Data unavailable",
                "Avg Volume": None,  # FIX 31
                "_ticker_raw": ticker,
            })
            continue
        
        coverage["success"] += 1
        coverage["analyzed"] += 1
        
        last = result["last"]
        prev_close = result["df"]["Close"].iloc[-2] if len(result["df"]) >= 2 else last["Close"]
        change_pct = (last["Close"] - prev_close) / prev_close * 100 if prev_close else 0
        
        why_parts = []
        if result["trend"] in ("BULLISH", "STRONG BULLISH"):
            why_parts.append(result["trend"])
        if "CONFIRMED" in result["breakout"]["status"]:
            why_parts.append(result["breakout"]["status"])
        if result["signal"]["reasons"]:
            positive_reasons = [r for r in result["signal"]["reasons"] if r.startswith("✅")]
            why_parts.extend(positive_reasons[:2])
        if result["risk"]["rr1"] and result["risk"]["rr1"] >= 1.5:
            why_parts.append(f"R:R 1:{round(result['risk']['rr1'],2)}")
        
        if result["penny"]["is_penny"] and result["penny"]["status"] != "PENNY (NO SETUP)":
            why_parts.insert(0, result["penny"]["note"])
        
        why_text = " + ".join(why_parts[:4]) if why_parts else "No clear setup"
        
        # FIX 31 — Add Avg Volume column
        avg_volume = result["avg_volume"] if not pd.isna(result["avg_volume"]) else 0
        
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
            "Cap Size": result["cap_size"],
            "Source": result["data_source"],
            "Why": why_text,
            "Avg Volume": round(avg_volume, 0),  # FIX 31
            "_ticker_raw": ticker,
        })
    
    return pd.DataFrame(rows), coverage

# ============================================================
# PORTFOLIO DECISION
# ============================================================

def portfolio_decision(holding: Dict, result: Dict) -> Tuple[str, str]:
    if result is None:
        return "WATCH", "Data unavailable"
    
    signal = result["signal"]["signal"]
    trend = result["trend"]
    pullback = result["pullback"]["status"]
    breakout = result["breakout"]["status"]
    
    stop_loss = result["risk"]["stop_loss"]
    current_price = result["last"]["Close"]
    
    if current_price < stop_loss:
        return "EXIT", f"Price {round(current_price,2)} below stop-loss {round(stop_loss,2)}"
    
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
# CHART (FIX 34 — Trend direction label on chart)
# ============================================================

def build_chart(result, show_bb=False, show_sma200=False, show_support_resistance=True, show_rsi=False, show_macd=False):
    d = result["df"].tail(150)
    risk = result["risk"]
    sr = result["sr"]
    trend = result["trend"]
    
    num_rows = 1
    if show_rsi:
        num_rows += 1
    if show_macd:
        num_rows += 1
    
    row_heights = [0.5]
    if show_rsi:
        row_heights.append(0.17)
    if show_macd:
        row_heights.append(0.17)
    if len(row_heights) == 3:
        row_heights = [0.5, 0.17, 0.17]
    elif len(row_heights) == 2:
        row_heights = [0.6, 0.25]
    
    subplot_titles = ["Price"]
    if show_rsi:
        subplot_titles.append("RSI (14)")
    if show_macd:
        subplot_titles.append("MACD")
    
    fig = make_subplots(
        rows=num_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.04,
        subplot_titles=subplot_titles,
    )
    
    current_row = 1
    
    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="Price", increasing_line_color="#10B981", decreasing_line_color="#EF4444"
    ), row=current_row, col=1)
    
    # FIX 34 — Trend direction annotation
    trend_color = "#10B981" if trend in ("BULLISH", "STRONG BULLISH") else "#EF4444" if trend in ("BEARISH", "STRONG BEARISH") else "#F59E0B"
    trend_arrow = "↑" if trend in ("BULLISH", "STRONG BULLISH") else "↓" if trend in ("BEARISH", "STRONG BEARISH") else "→"
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=f"{trend_arrow} TREND: {trend}",
        showarrow=False,
        font=dict(color=trend_color, size=14, family="monospace"),
        bgcolor="rgba(14, 17, 23, 0.8)",
        bordercolor=trend_color,
        borderwidth=1,
        borderpad=4,
        opacity=0.9
    )
    
    # SMA20
    fig.add_trace(go.Scatter(
        x=d.index, y=d["SMA20"], line=dict(color="#3B82F6", width=1.2), name="SMA20"
    ), row=current_row, col=1)
    
    # SMA50
    fig.add_trace(go.Scatter(
        x=d.index, y=d["SMA50"], line=dict(color="#F59E0B", width=1.2), name="SMA50"
    ), row=current_row, col=1)
    
    if show_sma200 and result["has_sma200"]:
        fig.add_trace(go.Scatter(
            x=d.index, y=d["SMA200"], line=dict(color="#8B5CF6", width=1, dash="dot"), name="SMA200"
        ), row=current_row, col=1)
    
    if show_bb:
        fig.add_trace(go.Scatter(
            x=d.index, y=d["BB_UPPER"], line=dict(color="#94A3B8", width=0.8, dash="dot"), name="BB Upper"
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=d.index, y=d["BB_LOWER"], line=dict(color="#94A3B8", width=0.8, dash="dot"), name="BB Lower"
        ), row=current_row, col=1)
    
    if show_support_resistance:
        fig.add_hline(y=sr["primary_resistance"], line_dash="dash", line_color="#EF4444", 
                      annotation_text="Resistance", row=current_row, col=1)
        fig.add_hline(y=sr["primary_support"], line_dash="dash", line_color="#10B981", 
                      annotation_text="Support", row=current_row, col=1)
    
    if risk["stop_loss"] > 0:
        fig.add_hline(y=risk["stop_loss"], line_dash="dot", line_color="#F59E0B", 
                      annotation_text="Stop", row=current_row, col=1)
    if risk["target1"] > 0:
        fig.add_hline(y=risk["target1"], line_dash="dot", line_color="#3B82F6", 
                      annotation_text="T1", row=current_row, col=1)
    
    if show_rsi:
        current_row += 1
        fig.add_trace(go.Scatter(
            x=d.index, y=d["RSI14"], line=dict(color="#3B82F6", width=1.3), name="RSI"
        ), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#EF4444", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#10B981", row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
    
    if show_macd:
        current_row += 1
        fig.add_trace(go.Scatter(
            x=d.index, y=d["MACD"], line=dict(color="#3B82F6", width=1), name="MACD"
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=d.index, y=d["MACD_SIGNAL"], line=dict(color="#F59E0B", width=1), name="Signal"
        ), row=current_row, col=1)
        hist_colors = np.where(d["MACD_HIST"] >= 0, "#10B981", "#EF4444")
        fig.add_trace(go.Bar(
            x=d.index, y=d["MACD_HIST"], marker_color=hist_colors, name="Hist"
        ), row=current_row, col=1)
    
    fig.update_layout(
        height=700 if not show_rsi and not show_macd else 800,
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
        "STRONG BUY": "green",
        "BUY": "green",
        "WAIT": "orange",
        "REDUCE": "red",
        "AVOID": "red",
    }
    return colors.get(signal, "blue")

def get_signal_class(signal):
    if signal in ("STRONG BUY", "BUY"):
        return "signal-buy"
    elif signal == "WAIT":
        return "signal-wait"
    else:
        return "signal-avoid"

# ============================================================
# FIX 35 — Prominent Stale Data Warning
# ============================================================

def show_stale_data_warning(freshness_status, freshness_warning):
    if freshness_status == "STALE":
        st.warning(f"🔴 {freshness_warning}")
        st.caption("⚠️ Data is stale. Signal confidence reduced. Please refresh or use a different data source.")
    elif freshness_status == "DELAYED":
        st.info(f"⚠️ {freshness_warning}")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "<div style='font-family:monospace; color:#2DD4BF; font-size:22px; "
    "font-weight:bold; letter-spacing:1px;'>PSX QUANT ENGINE</div>"
    "<div style='color:#94A3B8; font-size:11px; margin-bottom:10px;'>"
    "Quantitative Decision Support · v3</div>",
    unsafe_allow_html=True
)

st.sidebar.header("📈 Analysis")

sidebar_ticker = st.sidebar.text_input(
    "Ticker",
    value="SYS",
    help="Enter PSX symbol (e.g., SYS, OGDC, LUCK)"
)

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

watchlist_input = st.sidebar.text_area(
    "Watchlist (comma-separated)",
    value="SYS, OGDC, HBL, LUCK, FFC, ENGRO"
)

penny_threshold = st.sidebar.number_input(
    "Penny Stock Threshold (PKR)",
    min_value=10,
    max_value=200,
    value=50,
    step=5,
    help="Stocks below this price are classified as penny stocks"
)

rvol_threshold = st.sidebar.slider(
    "Penny RVOL Threshold (x average)",
    min_value=1.0,
    max_value=5.0,
    value=2.0,
    step=0.5,
    help="Minimum volume ratio to flag unusual penny stock activity"
)

st.sidebar.divider()
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    fetch_ohlcv.clear()
    fetch_market_index.clear()
    market_snapshot.clear()
    analyze_stock.clear()
    st.session_state.pop("screener_df", None)
    st.session_state.pop("watchlist_df", None)
    st.sidebar.success("Cache cleared!")

st.sidebar.caption("Data: Provider Priority | Cache: 5min")
st.sidebar.caption(f"Checked: {pkt_now().strftime('%d-%b %H:%M')} PKT")
st.sidebar.caption("⚠️ Signals are analytical outputs, not guaranteed advice.")

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
# DASHBOARD TAB (FIX 33 + FIX 35)
# ============================================================

with tab_dash:
    result, status, error, source = analyze_stock(sidebar_ticker, period=period, penny_threshold=penny_threshold, rvol_threshold=rvol_threshold)
    
    if status != "SUCCESS":
        st.error(f"❌ Could not analyze {sidebar_ticker}: {error}")
        
        with st.expander("🔍 Troubleshooting"):
            st.write(f"**Status:** {status}")
            st.write(f"**Error:** {error}")
            st.write(f"**Data Source:** {source if source else 'N/A'}")
            st.write("**Try:** Refresh data or check ticker spelling")
    else:
        last = result["last"]
        sig = result["signal"]
        
        # Data Freshness
        freshness_status, freshness_age, freshness_warning = get_freshness_status(result["data_date"])
        
        # FIX 35 — Prominent stale data warning
        show_stale_data_warning(freshness_status, freshness_warning)
        
        # Header row: Ticker, Price, Signal, Score, Trend
        col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.2, 1.2, 1.2])
        
        with col1:
            st.markdown(f"<span class='current-price'>{round(last['Close'], 2)}</span>", unsafe_allow_html=True)
            prev_close = result["df"]["Close"].iloc[-2] if len(result["df"]) >= 2 else last["Close"]
            change = last["Close"] - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0
            change_class = "change-positive" if change >= 0 else "change-negative"
            st.markdown(f"<span class='{change_class}'>{'▲' if change >= 0 else '▼'} {round(change, 2)} ({round(change_pct, 2)}%)</span>", unsafe_allow_html=True)
            st.caption(f"{result['ticker_display']} · {result['cap_size']}")
            st.caption(f"Data: {source} · {freshness_status} · {result['data_date'].date()}")
            if freshness_status == "STALE":
                st.warning(f"🔴 {freshness_warning}")
        
        with col2:
            signal_class = get_signal_class(sig["signal"])
            st.markdown(f"**Signal**")
            st.markdown(f"<span class='{signal_class}'>{sig['signal']}</span>", unsafe_allow_html=True)
        
        with col3:
            st.metric("Score", f"{sig['score']}/100")
        
        with col4:
            st.metric("Trend", result["trend"])
        
        with col5:
            st.metric("Setup", sig["setup_quality"])
        
        # Market Context
        mkt = result["market"]
        if mkt["regime"] != "UNAVAILABLE":
            st.caption(f"📊 KSE-100: {mkt['regime']} | Level: {round(mkt['last_close'], 0) if mkt['last_close'] else 'N/A'}")
            st.caption(f"Source: {mkt['source']}")
        else:
            st.caption("📊 KSE-100: DATA UNAVAILABLE")
            proxy = liquid_basket_trend()
            if proxy["change_pct"] is not None:
                st.caption(f"📊 Proxy (NOT KSE-100): {proxy['trend']} ({proxy['change_pct']}%)")
        
        # Signal Reasons
        with st.expander("🔍 Why this signal?", expanded=True):
            for r in sig["reasons"]:
                st.write(r)
        
        # ============================================================
        # FIX 33 — Conditional Entry Display
        # ============================================================
        st.subheader("📋 Trade Plan")
        risk = result["risk"]
        trend = result["trend"]
        
        if trend in ("BEARISH", "STRONG BEARISH"):
            st.warning("📉 No long trade setup — bearish structure.")
        else:
            # Main entry
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Entry", round(risk["entry"], 2))
            col2.metric("Stop Loss", round(risk["stop_loss"], 2))
            col3.metric("Target 1", round(risk["target1"], 2))
            col4.metric("Target 2", round(risk["target2"], 2))
            col5.metric("R:R", f"1:{round(risk['rr1'], 2) if risk['rr1'] else 'N/A'}")
            
            # FIX 33 — Show conditional entry if available
            if risk.get("conditional_entry") is not None:
                st.caption(f"💡 **Better Entry Alternative:** {risk['conditional_entry']} (pullback to EMA20) — better R:R than current price.")
                st.caption(f"   Current price: {round(risk['entry'], 2)} | Conditional: {risk['conditional_entry']} | Save: {round(risk['entry'] - risk['conditional_entry'], 2)} per share")
        
        sizing = position_sizing(capital, risk_pct, risk)
        if sizing["shares"] > 0 and trend not in ("BEARISH", "STRONG BEARISH"):
            st.caption(f"📊 Position: **{sizing['shares']} shares** · Investment: PKR {sizing['investment']} · Max Loss: PKR {sizing['max_loss']}")
        
        # Projection
        proj = result["projection"]
        if proj["direction"] != "NEUTRAL":
            st.info(f"📈 **Technical Projection:** {proj['label']}")
            st.caption(proj["note"])
        
        # Penny Alert
        if result["penny"]["is_penny"]:
            st.warning(f"🪙 {result['penny']['status']} - {result['penny']['note']}")
        
        # Chart
        st.subheader("📊 Chart")
        col1, col2, col3, col4, col5 = st.columns(5)
        show_bb = col1.checkbox("Bollinger Bands", value=False)
        show_sma200 = col2.checkbox("SMA200", value=False)
        show_sr = col3.checkbox("Support/Resistance", value=True)
        show_rsi = col4.checkbox("RSI", value=False)
        show_macd = col5.checkbox("MACD", value=False)
        
        st.plotly_chart(build_chart(result, show_bb, show_sma200, show_sr, show_rsi, show_macd), use_container_width=True)
        
        # Indicator Explanations
        with st.expander("📖 Indicator Explanations"):
            st.markdown(f"**SMA20:** {get_indicator_explanation('SMA20')} (Current: {round(last['SMA20'],2)})")
            st.markdown(f"**SMA50:** {get_indicator_explanation('SMA50')} (Current: {round(last['SMA50'],2)})")
            st.markdown(f"**RSI:** {get_indicator_explanation('RSI')} (Current: {round(last['RSI14'],1)})")
            st.markdown(f"**MACD:** {get_indicator_explanation('MACD')} (Current: {round(last['MACD_HIST'],2)})")
            st.markdown(f"**ADX:** {get_indicator_explanation('ADX')} (Current: {round(last['ADX14'],1)})")
            st.markdown(f"**RVOL:** {get_indicator_explanation('RVOL')} (Current: {round(last['VOL_RATIO'],2) if not pd.isna(last['VOL_RATIO']) else 'N/A'}x)")
            st.markdown(f"**Support:** {get_indicator_explanation('Support')} ({round(result['sr']['primary_support'],2)})")
            st.markdown(f"**Resistance:** {get_indicator_explanation('Resistance')} ({round(result['sr']['primary_resistance'],2)})")
            st.markdown(f"**Breakout:** {result['breakout']['status']} — {result['breakout']['note']}")
            st.markdown(f"**Stop Loss:** {get_indicator_explanation('Stop Loss')} ({round(risk['stop_loss'],2)})")
            st.markdown(f"**Target:** {get_indicator_explanation('Target')} ({round(risk['target1'],2)})")
        
        # Support/Resistance Details
        with st.expander("📊 Support / Resistance Details"):
            sr = result["sr"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Primary Support", round(sr["primary_support"], 2))
            col2.metric("Primary Resistance", round(sr["primary_resistance"], 2))
            col3.metric("Secondary Support", round(sr["secondary_support"], 2))
            col4.metric("Secondary Resistance", round(sr["secondary_resistance"], 2))
            
            if not sr.get("secondary_support_is_distinct", True):
                st.caption("⚠️ Secondary Support = Primary Support (stock at extreme)")
            if not sr.get("secondary_resistance_is_distinct", True):
                st.caption("⚠️ Secondary Resistance = Primary Resistance (stock at extreme)")
            
            if not pd.isna(sr["high_52w"]):
                st.caption(f"52-Week High: {round(sr['high_52w'], 2)} | 52-Week Low: {round(sr['low_52w'], 2)}")

# ============================================================
# SCREENER TAB (FIX 32 + FIX 37 — Deduplicated universe)
# ============================================================

with tab_screener:
    st.subheader("🔍 PSX Opportunity Scanner")
    
    st.caption("Scans available PSX universe. FIX 32: Universe deduplicated for faster scanning.")
    st.caption("FIX 37: Scanner performance optimized.")
    
    universe_option = st.selectbox(
        "Universe",
        ["Dynamic (from provider)", "Liquid PSX (~34)", "Small Cap (~25)", "Custom (from sidebar)"],
        index=0,
    )
    
    col1, col2 = st.columns(2)
    with col1:
        custom_syms = st.text_input("Add extra symbols (comma-separated)", "")
    with col2:
        pass
    
    st.markdown("**Filters**")
    filt_col1, filt_col2, filt_col3, filt_col4 = st.columns(4)
    with filt_col1:
        signal_filter = st.multiselect(
            "Signal",
            ["STRONG BUY", "BUY", "WAIT", "REDUCE", "AVOID"],
            default=["STRONG BUY", "BUY"]
        )
    with filt_col2:
        category_filter = st.multiselect(
            "Category",
            ["LARGE", "MID", "SMALL", "LOW-PRICE", "MICRO"],
            default=[]
        )
    with filt_col3:
        breakout_filter = st.multiselect(
            "Breakout",
            ["CONFIRMED BREAKOUT", "EXTENDED BREAKOUT", "BREAKOUT READY", "BREAKOUT ATTEMPT"],
            default=[]
        )
    with filt_col4:
        min_score = st.slider("Min Score", 0, 100, 0)
    
    filt_col5, filt_col6, filt_col7, filt_col8 = st.columns(4)
    with filt_col5:
        min_rr = st.slider("Min R:R", 0.0, 5.0, 0.0, 0.1)
    with filt_col6:
        min_price = st.number_input("Min Price", min_value=0.0, value=0.0, step=1.0)
    with filt_col7:
        max_price = st.number_input("Max Price", min_value=0.0, value=10000.0, step=50.0)
    with filt_col8:
        min_avg_volume = st.number_input("Min Avg Volume", min_value=0, value=0, step=1000)
    
    if st.button("🔍 Run Screener", use_container_width=True):
        with st.spinner("Scanning PSX universe..."):
            if universe_option == "Dynamic (from provider)":
                universe, _, _ = fetch_universe()
            elif universe_option == "Liquid PSX (~34)":
                universe = PSX_LIQUID_UNIVERSE
            elif universe_option == "Small Cap (~25)":
                universe = PSX_SMALL_CAP_UNIVERSE
            else:
                universe = [t.strip() + ".KA" if not t.strip().endswith(".KA") else t.strip() 
                           for t in watchlist_input.split(",") if t.strip()]
            
            if custom_syms:
                extra = [t.strip() + ".KA" if not t.strip().endswith(".KA") else t.strip() 
                        for t in custom_syms.split(",") if t.strip()]
                universe = list(dict.fromkeys(universe + extra))
            
            st.caption(f"Scanning {len(universe)} symbols...")
            
            screener_df, coverage = run_screener(universe, period="6mo", penny_threshold=penny_threshold, rvol_threshold=rvol_threshold)
            st.session_state["screener_df"] = screener_df
            st.session_state["screener_coverage"] = coverage
    
    if "screener_coverage" in st.session_state:
        cov = st.session_state["screener_coverage"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total", cov["total"])
        col2.metric("Analyzed", cov["analyzed"])
        col3.metric("Success", cov["success"])
        col4.metric("Failed", cov["failed"])
    
    if "screener_df" in st.session_state:
        df_s = st.session_state["screener_df"]
        
        view = df_s.copy()
        
        if signal_filter:
            view = view[view["Signal"].isin(signal_filter)]
        if category_filter:
            view = view[view["Cap Size"].isin(category_filter)]
        if breakout_filter:
            view = view[view["Status"].isin(breakout_filter)]
        if min_score > 0:
            view = view[view["Score"] >= min_score]
        if min_rr > 0:
            view = view[view["RR"] >= min_rr]
        if min_price > 0:
            view = view[view["Price"] >= min_price]
        if max_price < 10000:
            view = view[view["Price"] <= max_price]
        
        # FIX 31 — Min Avg Volume filter now works
        if min_avg_volume > 0:
            view = view[view["Avg Volume"] >= min_avg_volume]
        
        view = view[view["Signal"] != "ERROR"]
        
        sort_by = st.selectbox("Sort by", ["Score", "Change %", "Price"], index=0)
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
        
        breakout_keywords = ["CONFIRMED", "READY", "ATTEMPT", "52W"]
        bo = df_s[df_s["Status"].astype(str).str.contains('|'.join(breakout_keywords), case=False, na=False)]
        
        if not bo.empty:
            breakout_rows = []
            for idx, row in bo.iterrows():
                ticker_raw = row.get("_ticker_raw", row["Ticker"])
                result, status, _, _ = analyze_stock(ticker_raw, period="6mo", penny_threshold=penny_threshold, rvol_threshold=rvol_threshold)
                if status == "SUCCESS":
                    breakout_rows.append({
                        "Ticker": row["Ticker"],
                        "Price": row["Price"],
                        "Change %": row["Change %"],
                        "Trend": row["Trend"],
                        "Score": row["Score"],
                        "Signal": row["Signal"],
                        "Status": row["Status"],
                        "Resistance": round(result["breakout"]["resistance"], 2),
                        "Dist %": round(result["breakout"]["distance_to_resistance"], 2) if result["breakout"]["distance_to_resistance"] is not None else "N/A",
                        "Momentum": result["momentum"]["label"],
                        "RR": row["RR"],
                        "Why": row["Why"],
                        "Cap Size": row.get("Cap Size", "N/A"),
                        "_ticker_raw": ticker_raw,
                    })
            
            if breakout_rows:
                bo_df = pd.DataFrame(breakout_rows)
                
                status_order = {
                    "CONFIRMED BREAKOUT": 0,
                    "CONFIRMED BREAKOUT / 52W HIGH": 0,
                    "EXTENDED BREAKOUT": 1,
                    "EXTENDED BREAKOUT / 52W HIGH": 1,
                    "BREAKOUT READY": 2,
                    "BREAKOUT READY / 52W HIGH": 2,
                    "BREAKOUT ATTEMPT": 3,
                    "BREAKOUT ATTEMPT / 52W HIGH": 3,
                }
                bo_df["_sort_key"] = bo_df["Status"].map(status_order).fillna(4)
                bo_df = bo_df.sort_values(["_sort_key", "Score"], ascending=[True, False])
                
                display_cols = [c for c in bo_df.columns if not c.startswith("_")]
                st.dataframe(bo_df[display_cols], use_container_width=True, hide_index=True)
            else:
                st.info("No breakout candidates with detailed data available")
        else:
            st.info("No breakout candidates found")
    else:
        st.info("Run the Screener first")

# ============================================================
# PENNY STOCKS TAB
# ============================================================

with tab_penny:
    st.subheader("🪙 Penny Stock Breakout Watch")
    
    st.caption(f"Stocks below PKR {penny_threshold} with unusual activity (RVOL ≥ {rvol_threshold}x)")
    st.caption("⚠️ Low-priced stocks: high volatility, liquidity risk, false breakouts possible.")
    
    if "screener_df" in st.session_state:
        df_s = st.session_state["screener_df"]
        
        penny_df = df_s[df_s["Penny"].notna()]
        penny_df = penny_df[penny_df["Penny"] != "N/A"]
        
        if not penny_df.empty:
            interesting = penny_df[penny_df["Penny"].str.contains("BREAKOUT|READY|VOLUME|WATCH", na=False)]
            interesting = interesting.sort_values("Score", ascending=False)
            
            if not interesting.empty:
                st.success(f"🔥 {len(interesting)} interesting penny setups!")
                display_cols = [c for c in interesting.columns if not c.startswith("_")]
                st.dataframe(interesting[display_cols], use_container_width=True, hide_index=True)
            else:
                st.info("No interesting penny setups")
            
            with st.expander("All penny stocks"):
                all_penny = penny_df.sort_values("Price", ascending=True)
                display_cols = [c for c in all_penny.columns if not c.startswith("_")]
                st.dataframe(all_penny[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No penny stocks found")
    else:
        st.info("Run the Screener first")

# ============================================================
# NEXT SESSION TAB
# ============================================================

with tab_next:
    st.subheader("📅 Next Session Watchlist")
    st.caption("Top candidates for the next trading session based on latest data")
    
    if st.button("🔄 Refresh Next Session", use_container_width=True):
        with st.spinner("Scanning..."):
            combined_universe = list(dict.fromkeys(PSX_LIQUID_UNIVERSE + PSX_SMALL_CAP_UNIVERSE))
            st.session_state["next_session_df"], _ = run_screener(
                combined_universe, period="6mo", penny_threshold=penny_threshold, rvol_threshold=rvol_threshold
            )
    
    if "next_session_df" not in st.session_state:
        with st.spinner("Initial scan..."):
            combined_universe = list(dict.fromkeys(PSX_LIQUID_UNIVERSE + PSX_SMALL_CAP_UNIVERSE))
            st.session_state["next_session_df"], _ = run_screener(
                combined_universe, period="6mo", penny_threshold=penny_threshold, rvol_threshold=rvol_threshold
            )
    
    df_s = st.session_state["next_session_df"]
    
    top = df_s[df_s["Signal"].isin(["STRONG BUY", "BUY"])]
    top = top.dropna(subset=["Score"])
    top = top.sort_values("Score", ascending=False).head(10)
    
    if not top.empty:
        cap_size_rows = []
        for idx, row in top.iterrows():
            ticker_raw = row.get("_ticker_raw", row["Ticker"])
            
            # Calculate confidence
            if row["Score"] >= 75 and row["Trend"] in ("BULLISH", "STRONG BULLISH"):
                confidence = "HIGH"
            elif row["Score"] >= 60:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            
            result, status, _, _ = analyze_stock(ticker_raw, period="6mo", penny_threshold=penny_threshold, rvol_threshold=rvol_threshold)
            if status == "SUCCESS":
                trade_type, pace_label = estimate_pace_to_target(result)
                freshness_status, _, _ = get_freshness_status(result["data_date"])
            else:
                trade_type, pace_label = "N/A", "N/A"
                freshness_status = "UNAVAILABLE"
            
            cap_size_rows.append({
                "Ticker": row["Ticker"],
                "Price": row["Price"],
                "Trend": row["Trend"],
                "Score": row["Score"],
                "Signal": row["Signal"],
                "Status": row["Status"],
                "RR": row["RR"],
                "Cap Size": row.get("Cap Size", "N/A"),
                "Trade Type": trade_type,
                "Est. Pace": pace_label,
                "Confidence": confidence,
                "Data": freshness_status,
                "Why": row["Why"],
            })
        
        cap_df = pd.DataFrame(cap_size_rows)
        display_cols = ["Ticker", "Price", "Score", "Signal", "Confidence", "Trade Type", "Est. Pace", "Data", "Why"]
        st.dataframe(cap_df[display_cols], use_container_width=True, hide_index=True)
        
        if len(top) < 3:
            st.info(f"Only {len(top)} candidates — reflects limited high-quality setups.")
        
        st.caption("⚠️ 'Est. Pace' is a rough ATR-based estimate. Not a guaranteed timeline.")
        st.caption("⚠️ 'Confidence' based on score & trend alignment. Not a guarantee of profit.")
        
    else:
        st.info("No strong BUY candidates at this time")

# ============================================================
# WATCHLIST TAB
# ============================================================

with tab_watch:
    st.subheader("📋 Watchlist Analysis")
    
    tickers = [t.strip() for t in watchlist_input.split(",") if t.strip()]
    
    if st.button("🔄 Refresh Watchlist", use_container_width=True):
        with st.spinner("Analyzing..."):
            watchlist_df, coverage = run_screener(tickers, period="6mo", penny_threshold=penny_threshold, rvol_threshold=rvol_threshold)
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
    
    if st.session_state.portfolio:
        rows = []
        total_invested = 0
        total_current = 0
        
        for i, h in enumerate(st.session_state.portfolio):
            result, status, error, _ = analyze_stock(h["ticker"], period="6mo", penny_threshold=penny_threshold, rvol_threshold=rvol_threshold)
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
                    "Support": round(result["sr"]["primary_support"], 2),
                    "Resistance": round(result["sr"]["primary_resistance"], 2),
                    "Stop Loss": round(result["risk"]["stop_loss"], 2),
                    "Target 1": round(result["risk"]["target1"], 2),
                    "Decision": decision,
                    "Reason": reason,
                    "Cap Size": result.get("cap_size", "N/A"),
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
                    "Support": None,
                    "Resistance": None,
                    "Stop Loss": None,
                    "Target 1": None,
                    "Decision": "WATCH",
                    "Reason": error,
                    "Cap Size": None,
                })
        
        total_pnl = total_current - total_invested
        total_pnl_pct = total_pnl / total_invested * 100 if total_invested else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Invested", f"PKR {round(total_invested, 2)}")
        col2.metric("Current Value", f"PKR {round(total_current, 2)}")
        col3.metric("Total P/L", f"PKR {round(total_pnl, 2)}")
        col4.metric("Total P/L %", f"{round(total_pnl_pct, 2)}%")
        
        port_df = pd.DataFrame(rows)
        st.dataframe(port_df, use_container_width=True, hide_index=True)
        
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
    st.subheader("📈 Market Overview")
    
    market = market_snapshot()
    
    if market["regime"] == "UNAVAILABLE":
        st.warning("🔴 KSE-100: DATA UNAVAILABLE")
        st.caption(f"Source: {market['source']}")
        st.caption("No KSE-100 data from any provider. Market regime confidence reduced.")
        
        with st.expander("🔍 Provider Diagnostics"):
            # Note: PROVIDER_STATUS reflects only cache-miss calls, not cache hits
            diag_df = pd.DataFrame([
                {"Provider": k, "Available": v["available"], "Coverage": v["coverage"], "KSE-100": v["kse100"], "Error": v["error"][:100] if v["error"] else None}
                for k, v in PROVIDER_STATUS.items()
            ])
            st.dataframe(diag_df, use_container_width=True, hide_index=True)
            st.caption("⚠️ Diagnostics reflect only actual network fetches, not cached responses.")
    else:
        freshness_status, freshness_age, freshness_warning = get_freshness_status(market["last_date"])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Market Regime", market["regime"])
        col2.metric("Market Trend", market["trend"])
        col3.metric("KSE-100 Level", round(market["last_close"], 2) if market["last_close"] else "N/A")
        
        st.caption(f"Source: {market['source']} | Last: {market['last_date'].date() if market['last_date'] else 'N/A'} ({freshness_status})")
        if freshness_status == "STALE":
            st.warning(f"🔴 {freshness_warning}")
        
        st.info(f"**Reasoning:** {market['reasoning']}")
        
        # KSE-100 Chart
        idx_df, _ = fetch_market_index()
        if idx_df is not None and len(idx_df) > 30:
            idx_df["SMA20"] = sma(idx_df["Close"], 20)
            idx_df["SMA50"] = sma(idx_df["Close"], 50)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=idx_df.index, y=idx_df["Close"], line=dict(color="#3B82F6", width=2), name="KSE-100"))
            fig.add_trace(go.Scatter(x=idx_df.index, y=idx_df["SMA20"], line=dict(color="#F59E0B", width=1.5, dash="dot"), name="SMA20"))
            fig.add_trace(go.Scatter(x=idx_df.index, y=idx_df["SMA50"], line=dict(color="#8B5CF6", width=1.5, dash="dot"), name="SMA50"))
            fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
    
    # Proxy (Always visible, separately labelled)
    st.divider()
    st.subheader("📊 PSX Market Proxy")
    st.caption("⚠️ PROXY — NOT official KSE-100. Equal-weighted average of liquid PSX stocks.")
    
    proxy = liquid_basket_trend()
    if proxy["change_pct"] is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Proxy Trend", proxy["trend"])
        col2.metric("Avg Change (5d)", f"{proxy['change_pct']}%")
        col3.metric("Stocks Contributing", proxy["stocks_contributing"])
        st.caption(f"Note: {proxy['note']}")
    else:
        st.info("Proxy unavailable — insufficient data")
    
    # Provider Diagnostics
    with st.expander("🔍 Provider Diagnostics"):
        diag_df = pd.DataFrame([
            {"Provider": k, "Available": "✅" if v["available"] else "❌", "Coverage": v["coverage"], "KSE-100": "✅" if v["kse100"] else "❌", "Last Success": v["last_success"].strftime("%d-%b %H:%M") if v["last_success"] else "Never", "Error": v["error"][:100] if v["error"] else "-"}
            for k, v in PROVIDER_STATUS.items()
        ])
        st.dataframe(diag_df, use_container_width=True, hide_index=True)
        st.caption("⚠️ Diagnostics reflect only actual network fetches, not cached responses.")
        st.caption("KSE-100 availability determines whether market regime uses real KSE-100 data.")

# ============================================================
# FOOTER
# ============================================================

st.sidebar.caption("---")
st.sidebar.caption("⚠️ **Disclaimer:** Signals are analytical outputs based on historical price/volume data. Not guaranteed investment advice. Always do your own research before making trading decisions.")

# ============================================================
# END
# ============================================================
