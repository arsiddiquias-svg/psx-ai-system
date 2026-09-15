"""
PSX QUANT ENGINE - v8.0 (Fixed)
================================
PSX-focused quantitative decision-support terminal.

FEATURES:
- PSX official data portal integration (dps.psx.com.pk)
- Price reconciliation between PSX and yfinance (date-aware)
- Technical indicators: RSI, SMA/EMA, MACD, ATR, ADX, Bollinger
- Trend, breakout, pullback, momentum, projection engines
- Screener with simplified filters + penny toggle
- Watchlist (session-state based, inside tab)
- Portfolio tracker (max 5 holdings, partial valuation aware)
- Top Picks (Day Trade + Swing Trade + Hold)
- Light theme
- Chart projection arrows

DATA SOURCES (truthful):
- KSE-100 index: PSX official EOD -> yfinance fallback
- Stock OHLCV:   psxdata (optional) -> yfinance
- Market-watch:  PSX official HTML
- Reconciliation: PSX live vs yfinance (date-aware)

STARTUP POLICY:
- Heavy scans (Screener, Next Session, Market Proxy, Top Picks) are
  USER-TRIGGERED ONLY via explicit buttons.
- No automatic universe scans on first render.

DEPLOYMENT:
- Python 3.11+ (tested on 3.12)
- See requirements.txt
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import time
import requests
from io import StringIO
from typing import Optional, Tuple, Dict, Any, List, Union

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PSX Quant Engine v8.0",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# CSS - LIGHT THEME
# ============================================================

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #FAFBFC 0%, #F4F6F8 100%); color: #1A1A2E; }
h1, h2, h3, h4 { font-family: 'Inter', 'Segoe UI', sans-serif !important; font-weight: 600 !important; color: #16213E !important; }
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 16px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04); }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { color: #64748B !important; font-size: 0.72rem !important; font-weight: 500 !important; text-transform: uppercase !important; letter-spacing: 0.6px !important; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: #0F172A !important; font-family: 'Inter', 'Segoe UI', sans-serif !important; font-size: 1.55rem !important; font-weight: 700 !important; }
section[data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E2E8F0 !important; }
section[data-testid="stSidebar"] * { color: #1E293B !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *, .stCaption, .stCaption * { color: #64748B !important; font-size: 0.8rem !important; }
.current-price { font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 2.8rem; font-weight: 800; color: #0F172A; letter-spacing: -0.6px; }
.change-positive { color: #059669; font-weight: 700; }
.change-negative { color: #DC2626; font-weight: 700; }
.signal-buy   { color: #059669; font-weight: 800; font-size: 1.2rem; }
.signal-wait  { color: #D97706; font-weight: 800; font-size: 1.2rem; }
.signal-avoid { color: #DC2626; font-weight: 800; font-size: 1.2rem; }
div[data-testid="stExpander"] { background-color: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 10px !important; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important; }
div[data-testid="stExpander"] summary { color: #1E293B !important; font-weight: 600 !important; }
.stButton > button { background-color: #FFFFFF !important; color: #1E293B !important; border: 1px solid #CBD5E1 !important; border-radius: 8px !important; font-weight: 600 !important; transition: all 0.15s ease; }
.stButton > button:hover { background-color: #F1F5F9 !important; border-color: #94A3B8 !important; color: #0F172A !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: transparent; border-bottom: 2px solid #E2E8F0; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #64748B !important; font-weight: 600 !important; padding: 8px 16px !important; border-radius: 8px 8px 0 0 !important; }
.stTabs [aria-selected="true"] { color: #2563EB !important; background: #EFF6FF !important; border-bottom: 2px solid #2563EB !important; }
[data-testid="stDataFrame"] { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; overflow: hidden; }
.stTextInput input, .stNumberInput input, .stTextArea textarea, .stSelectbox div, .stMultiSelect div { background-color: #FFFFFF !important; border-color: #CBD5E1 !important; color: #1E293B !important; }
[data-testid="stAlert"] { border-radius: 10px; border-left-width: 4px; }
hr { border-color: #E2E8F0 !important; opacity: 1 !important; }
h2, h3 { margin-top: 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================

MIN_RR = 1.5
PENNY_STOCK_THRESHOLD = 50
MIN_HISTORY_DAYS = 60
CACHE_TTL = 300

WEIGHTS = {"trend": 0.25, "momentum": 0.20, "volume": 0.15, "setup": 0.20, "rr": 0.10, "sr": 0.10}

KSE100_PLAUSIBLE_MIN = 5000
KSE100_PLAUSIBLE_MAX = 1000000
KSE100_CANDIDATES = ["^KSE100", "KSE100.KA"]

# ============================================================
# PSX UNIVERSE (Fallback only)
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

# ============================================================
# PROVIDER STATUS
# ============================================================

PROVIDER_STATUS = {
    "psx_official": {"available": False, "last_success": None, "error": None, "coverage": 0, "kse100": False, "last_fetch_attempt": None},
    "psxdata":      {"available": False, "last_success": None, "error": None, "coverage": 0, "kse100": False, "last_fetch_attempt": None},
    "yfinance":     {"available": True,  "last_success": None, "error": None, "coverage": 0, "kse100": False, "last_fetch_attempt": None},
}

def update_provider_status(provider: str, available: bool = None, error: str = None,
                           coverage: int = None, kse100: bool = None):
    if provider in PROVIDER_STATUS:
        if available is not None: PROVIDER_STATUS[provider]["available"] = available
        if error is not None: PROVIDER_STATUS[provider]["error"] = error
        if coverage is not None: PROVIDER_STATUS[provider]["coverage"] = coverage
        if kse100 is not None: PROVIDER_STATUS[provider]["kse100"] = kse100
        if available: PROVIDER_STATUS[provider]["last_success"] = pkt_now()
        PROVIDER_STATUS[provider]["last_fetch_attempt"] = pkt_now()

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
    if not t.endswith(".KA"): t = t + ".KA"
    return t

def normalize_ticker_display(raw: str) -> str:
    return raw.strip().upper().replace(".KA", "")

def trading_days_between(date1, date2):
    """Mon-Fri business-day count. PSX holidays NOT accounted for — approximate."""
    try:
        return int(np.busday_count(date1.date(), date2.date()))
    except Exception:
        return (date2.date() - date1.date()).days

def get_freshness_status(data_date):
    if data_date is None:
        return "UNAVAILABLE", None, "No data date available"
    now = pkt_now()
    if data_date.tzinfo is None: data_date = data_date.tz_localize(None)
    now_naive = now.replace(tzinfo=None)
    trading_gap = trading_days_between(data_date, now_naive)
    if trading_gap <= 1:
        return "FRESH", trading_gap, f"~{trading_gap} business day(s) old (approx, no PSX holiday calendar)"
    elif trading_gap <= 3:
        return "DELAYED", trading_gap, f"~{trading_gap} business days old (approx)"
    else:
        return "STALE", trading_gap, f"~{trading_gap} business days old — STALE (approx)"

# ============================================================
# PSX OFFICIAL DATA PORTAL
# ============================================================

PSX_BASE = "https://dps.psx.com.pk"
PSX_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 PSXQuantEngine/8.0 (personal research)")

_LAST_PSX_REQUEST_TS = 0.0
_PSX_MIN_INTERVAL = 1.0

def _psx_get(url: str, timeout: int = 15) -> Optional[requests.Response]:
    global _LAST_PSX_REQUEST_TS
    elapsed = time.time() - _LAST_PSX_REQUEST_TS
    if elapsed < _PSX_MIN_INTERVAL:
        time.sleep(_PSX_MIN_INTERVAL - elapsed)
    try:
        resp = requests.get(url, headers={
            "User-Agent": PSX_USER_AGENT,
            "Accept": "text/html,application/json,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }, timeout=timeout)
        _LAST_PSX_REQUEST_TS = time.time()
        return resp
    except Exception:
        _LAST_PSX_REQUEST_TS = time.time()
        return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_psx_official_index_json(index_symbol: str = "KSE100") -> Tuple[Optional[pd.DataFrame], str, str]:
    """PSX EOD index. Format: {"data": [[ts, close, vol, open], ...]}. Daily."""
    url = f"{PSX_BASE}/timeseries/eod/{index_symbol}"
    try:
        resp = _psx_get(url)
        if resp is None or resp.status_code != 200:
            return None, "EMPTY", f"HTTP {resp.status_code if resp else 'no-response'}"
        data = resp.json()
        if isinstance(data, dict): data = data.get("data", [])
        rows = []
        if isinstance(data, list):
            for item in data:
                try:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        ts, px = item[0], item[1]
                        dt = pd.to_datetime(ts) if isinstance(ts, str) else pd.to_datetime(int(ts), unit="s")
                        rows.append({"Date": dt, "Close": float(px)})
                    elif isinstance(item, dict):
                        ts = item.get("ts_utc") or item.get("timestamp") or item.get("time")
                        px = item.get("close") or item.get("price")
                        if ts is not None and px is not None:
                            dt = pd.to_datetime(ts) if isinstance(ts, str) else pd.to_datetime(int(ts), unit="s")
                            rows.append({"Date": dt, "Close": float(px)})
                except Exception:
                    continue
        if not rows:
            return None, "EMPTY", "EOD JSON mein koi parseable row nahi"
        df = pd.DataFrame(rows).set_index("Date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df = df.dropna(subset=["Close"])
        if len(df) < 40:
            return None, "EMPTY", f"History kam: {len(df)} rows"
        if len(df) >= 2:
            median_gap = df.index.to_series().diff().dt.total_seconds().median()
            if pd.notna(median_gap) and median_gap < 3600:
                return None, "EXCEPTION", f"Intraday lag raha hai (gap {median_gap:.0f}s)"
        update_provider_status("psx_official", available=True, coverage=len(df), kse100=(index_symbol == "KSE100"))
        return df, "SUCCESS", None
    except Exception as e:
        update_provider_status("psx_official", available=False, error=str(e)[:120])
        return None, "EXCEPTION", f"PSX EOD error: {str(e)[:120]}"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_psx_official_market_watch() -> Tuple[Optional[pd.DataFrame], str, str]:
    """Market-watch HTML scrape. Defensive parsing."""
    url = f"{PSX_BASE}/market-watch"
    try:
        resp = _psx_get(url)
        if resp is None or resp.status_code != 200:
            return None, "EMPTY", f"HTTP {resp.status_code if resp else 'no-response'}"
        tables = pd.read_html(StringIO(resp.text))
        if not tables: return None, "EMPTY", "HTML mein koi table nahi"
        df = max(tables, key=lambda t: t.shape[0] * t.shape[1])
        rename_map = {}
        for c in df.columns:
            cl = str(c).strip().lower()
            if "symbol" in cl:                 rename_map[c] = "Symbol"
            elif "sector" in cl:               rename_map[c] = "Sector"
            elif "listed" in cl:               rename_map[c] = "Listed_In"
            elif "ldcp" in cl:                 rename_map[c] = "LDCP"
            elif cl == "open":                 rename_map[c] = "Open"
            elif cl == "high":                 rename_map[c] = "High"
            elif cl == "low":                  rename_map[c] = "Low"
            elif "current" in cl:              rename_map[c] = "Current"
            elif "change" in cl and "%" in cl: rename_map[c] = "Change_Pct"
            elif "change" in cl:               rename_map[c] = "Change"
            elif "volume" in cl:               rename_map[c] = "Volume"
        df = df.rename(columns=rename_map)
        if "Symbol" not in df.columns or "Current" not in df.columns:
            return None, "EXCEPTION", f"Columns missing: {list(df.columns)[:8]}"
        for col in ["LDCP", "Open", "High", "Low", "Current", "Change", "Change_Pct", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.replace("%", "").str.strip(), errors="coerce")
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        df = df[df["Symbol"].str.len() > 0].dropna(subset=["Current"])
        update_provider_status("psx_official", available=True, coverage=len(df))
        return df, "SUCCESS", None
    except Exception as e:
        return None, "EXCEPTION", f"Market-watch error: {str(e)[:120]}"

# ============================================================
# OHLCV PROVIDERS
# ============================================================

def fetch_psxdata_ohlcv(ticker: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], str, str]:
    """Experimental psxdata provider."""
    try:
        import psxdata
        symbol = normalize_ticker_display(ticker)
        period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}.get(period, 365)
        start_date = (date.today() - timedelta(days=period_days)).strftime("%Y-%m-%d")
        end_date = date.today().strftime("%Y-%m-%d")
        df = psxdata.stocks(symbol, start=start_date, end=end_date)
        if df is None or df.empty: return None, "EMPTY", "No data from psxdata"
        col_map = {}
        for c in df.columns:
            cl = str(c).lower()
            if cl in ["open", "o"]: col_map[c] = "Open"
            elif cl in ["high", "h"]: col_map[c] = "High"
            elif cl in ["low", "l"]: col_map[c] = "Low"
            elif cl in ["close", "c", "price", "adj close"]: col_map[c] = "Close"
            elif cl in ["volume", "vol", "v"]: col_map[c] = "Volume"
            elif "date" in cl or "time" in cl: col_map[c] = "Date"
        if col_map: df = df.rename(columns=col_map)
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing: return None, "EXCEPTION", f"Missing columns: {missing}"
        df[required] = df[required].apply(pd.to_numeric, errors="coerce")
        df = df.dropna()
        if df.empty: return None, "EMPTY", "Empty after cleaning"
        if "Date" in df.columns:
            df.index = pd.to_datetime(df["Date"]); df = df.drop(columns=["Date"])
        valid, msg = _validate_ohlcv(df)
        if not valid: return None, "EXCEPTION", msg
        update_provider_status("psxdata", available=True, coverage=len(df))
        return df, "SUCCESS", None
    except ImportError:
        return None, "EMPTY", "psxdata not installed"
    except Exception as e:
        return None, "EMPTY", f"psxdata error: {str(e)}"

def fetch_yfinance_ohlcv(ticker: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], str, str]:
    symbol = normalize_ticker(ticker)
    try:
        raw = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
        if raw is None or raw.empty: return None, "EMPTY", "No data from yfinance"
        df = _flatten_columns(raw)
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing: return None, "EXCEPTION", f"Missing columns: {missing}"
        df = df[required].apply(pd.to_numeric, errors="coerce").dropna()
        if df.empty: return None, "EMPTY", "Empty after cleaning"
        valid, msg = _validate_ohlcv(df)
        if not valid: return None, "EXCEPTION", msg
        update_provider_status("yfinance", available=True, coverage=len(df))
        return df, "SUCCESS", None
    except Exception as e:
        # FIX: On exception, provider is NOT available.
        update_provider_status("yfinance", available=False, error=str(e))
        return None, "EXCEPTION", f"yfinance error: {str(e)}"

def _flatten_columns(df):
    if df is None or df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = list(df.columns.get_level_values(0))
        known = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        df = df.copy()
        if known.intersection(set(lvl0)):
            df.columns = lvl0
        else:
            df.columns = df.columns.get_level_values(-1)
    return df

def _validate_ohlcv(df: pd.DataFrame) -> Tuple[bool, str]:
    """Full OHLCV integrity check."""
    if df is None or df.empty: return False, "Empty DataFrame"
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing: return False, f"Missing columns: {missing}"
    if df[required].isna().any().any(): return False, "Contains NaN"
    if (df["High"] < df["Low"]).any(): return False, "High < Low"
    if (df["High"] < df["Open"]).any(): return False, "High < Open"
    if (df["High"] < df["Close"]).any(): return False, "High < Close"
    if (df["Low"] > df["Open"]).any(): return False, "Low > Open"
    if (df["Low"] > df["Close"]).any(): return False, "Low > Close"
    if (df["Close"] <= 0).any() or (df["High"] <= 0).any() or (df["Open"] <= 0).any():
        return False, "Non-positive prices"
    if (df["Volume"] < 0).any(): return False, "Negative volume"
    if df.index.duplicated().any(): return False, "Duplicate dates"
    if not df.index.is_monotonic_increasing: return False, "Dates not sorted"
    try:
        today_pkt = pd.Timestamp(pkt_now().date())
        if (df.index > today_pkt).any(): return False, "Contains future dates"
    except Exception:
        pass
    if len(df) < MIN_HISTORY_DAYS: return False, f"Insufficient history: {len(df)} < {MIN_HISTORY_DAYS}"
    return True, "Valid"

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], str, str, str]:
    """psxdata -> yfinance -> UNAVAILABLE."""
    provider_attempts = []
    df, status, error = fetch_psxdata_ohlcv(ticker, period)
    provider_attempts.append(f"psxdata: {status}")
    if status == "SUCCESS": return df, status, error, "psxdata"
    df, status, error = fetch_yfinance_ohlcv(ticker, period)
    provider_attempts.append(f"yfinance: {status}")
    if status == "SUCCESS": return df, status, error, "yfinance"
    return None, "UNAVAILABLE", " | ".join(provider_attempts), "UNAVAILABLE"

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_market_index():
    """KSE-100: PSX official first, yfinance fallback. Label truthful."""
    df, status, err = fetch_psx_official_index_json("KSE100")
    if status == "SUCCESS" and df is not None and len(df) >= 40:
        last_close = float(df["Close"].iloc[-1])
        if KSE100_PLAUSIBLE_MIN <= last_close <= KSE100_PLAUSIBLE_MAX:
            return df, "psx_official (KSE100 EOD, verified)"
    for cand in KSE100_CANDIDATES:
        try:
            raw = yf.download(cand, period="6mo", interval="1d", auto_adjust=False, progress=False)
            if raw is None or raw.empty: continue
            df = _flatten_columns(raw)
            if "Close" not in df.columns: continue
            df = df.dropna(subset=["Close"])
            if len(df) < 40: continue
            last_close = float(df["Close"].iloc[-1])
            if not (KSE100_PLAUSIBLE_MIN <= last_close <= KSE100_PLAUSIBLE_MAX): continue
            daily_vol = df["Close"].pct_change().std()
            if pd.isna(daily_vol) or daily_vol > 0.06: continue
            update_provider_status("yfinance", available=True, kse100=True)
            return df, f"yfinance fallback ({cand}) — identity via requested symbol, not independently verified"
        except Exception:
            continue
    return None, None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_universe() -> Tuple[List[str], str, str]:
    try:
        import psxdata
        tickers = psxdata.tickers()
        if tickers and len(tickers) > 10:
            formatted = [t if t.endswith(".KA") else t + ".KA" for t in tickers if isinstance(t, str)]
            return list(dict.fromkeys(formatted)), "psxdata (experimental)", None
    except Exception:
        pass
    return PSX_FALLBACK_UNIVERSE, "curated fallback", None

# ============================================================
# RECONCILIATION (date-aware)
# ============================================================

def reconcile_psx_vs_yf(ticker: str, yf_df: pd.DataFrame) -> Dict[str, Any]:
    """PSX vs yfinance comparison, date-aware. Mark not_comparable when dates differ."""
    out = {"mismatch": False, "diff_pct": None, "psx_price": None, "yf_price": None,
           "source_used": "yfinance", "note": None, "not_comparable": False}
    if yf_df is None or len(yf_df) == 0:
        out["note"] = "yfinance data empty"; return out
    yf_last_date = yf_df.index[-1]
    yf_last = float(yf_df["Close"].iloc[-1])
    out["yf_price"] = yf_last
    mw, status, _ = fetch_psx_official_market_watch()
    if status != "SUCCESS" or mw is None:
        out["note"] = "PSX market-watch unavailable"; return out
    sym = normalize_ticker_display(ticker)
    row = mw[mw["Symbol"] == sym]
    if row.empty or pd.isna(row.iloc[0].get("Current")):
        out["note"] = "symbol not in PSX market-watch"; return out
    psx_last = float(row.iloc[0]["Current"])
    out["psx_price"] = psx_last
    today_pkt = pkt_now().date()
    yf_date_pkt = yf_last_date.date() if hasattr(yf_last_date, "date") else yf_last_date
    if yf_date_pkt != today_pkt:
        out["not_comparable"] = True
        out["note"] = (f"Not comparable — yfinance last close is {yf_date_pkt}, "
                       f"PSX market-watch is live for {today_pkt}")
        return out
    if yf_last > 0:
        diff = abs(psx_last - yf_last) / yf_last
        out["diff_pct"] = round(diff * 100, 2)
        if diff > 0.01:
            out["mismatch"] = True
            out["source_used"] = "psx_official"
    return out

# ============================================================
# INDICATORS
# ============================================================

def sma(series, period): return series.rolling(period).mean()
def ema(series, period): return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    """
    Wilder RSI. Single EWM smoothing (matches atr()/adx() pattern).
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.mask((avg_gain == 0) & (avg_loss == 0), 50)
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0)
    return result

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast); ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

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
    """Wilder ADX."""
    up_move = df["High"].diff(); down_move = -df["Low"].diff()
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
    """Standard Bollinger Bands. ddof=1 (TradingView-compatible sample std)."""
    mid = sma(series, period)
    std = series.rolling(period).std(ddof=1)
    return mid + num_std * std, mid, mid - num_std * std

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
    d["MACD"] = macd_line; d["MACD_SIGNAL"] = signal_line; d["MACD_HIST"] = hist
    d["ATR14"] = atr(d, 14)
    adx_val, plus_di, minus_di = adx(d, 14)
    d["ADX14"] = adx_val; d["PLUS_DI"] = plus_di; d["MINUS_DI"] = minus_di
    bb_u, bb_m, bb_l = bollinger(d["Close"], 20, 2)
    d["BB_UPPER"] = bb_u; d["BB_MID"] = bb_m; d["BB_LOWER"] = bb_l
    d["VOL_SMA20"] = d["Volume"].rolling(20).mean().shift(1)
    d["VOL_RATIO"] = d["Volume"] / d["VOL_SMA20"].replace(0, np.nan)
    d["RETURN_1D"] = d["Close"].pct_change()
    d["ROC_10"] = d["Close"].pct_change(10) * 100
    d["VOLATILITY_20"] = d["RETURN_1D"].rolling(20).std() * np.sqrt(252)
    # 52W requires FULL 252 sessions. No partial-year fallback.
    d["52W_HIGH"] = d["High"].shift(1).rolling(252, min_periods=252).max()
    d["52W_LOW"] = d["Low"].shift(1).rolling(252, min_periods=252).min()
    return d

# ============================================================
# TREND / S&R / BREAKOUT / PULLBACK / MOMENTUM / PROJECTION
# ============================================================

def trend_engine(d):
    if len(d) < 50: return "INSUFFICIENT DATA", [], 0
    last = d.iloc[-1]
    reasons = []; bullish = 0; bearish = 0
    if last["Close"] > last["SMA20"]: bullish += 1; reasons.append("Price above SMA20")
    else: bearish += 1; reasons.append("Price below SMA20")
    if last["Close"] > last["SMA50"]: bullish += 1; reasons.append("Price above SMA50")
    else: bearish += 1; reasons.append("Price below SMA50")
    if last["SMA20"] > last["SMA50"]: bullish += 1; reasons.append("SMA20 above SMA50")
    else: bearish += 1; reasons.append("SMA20 below SMA50")
    if not pd.isna(last.get("SMA200", np.nan)):
        if last["SMA50"] > last["SMA200"]: bullish += 1; reasons.append("SMA50 above SMA200")
        else: bearish += 1; reasons.append("SMA50 below SMA200")
    recent = d.tail(20)
    if len(recent) >= 10:
        hh = recent["High"].iloc[-1] > recent["High"].iloc[0]
        hl = recent["Low"].iloc[-1] > recent["Low"].iloc[0]
        if hh and hl:
            bullish += 1
            reasons.append("Trend Structure Proxy: first vs last 20-session high/low aligned up")
        elif (not hh) and (not hl):
            bearish += 1
            reasons.append("Trend Structure Proxy: first vs last 20-session high/low aligned down")
    if last["ADX14"] >= 25: reasons.append(f"ADX {round(last['ADX14'],1)} - trending")
    total = bullish + bearish
    score = (bullish / total * 100) if total > 0 else 50
    if score >= 80: trend = "STRONG BULLISH"
    elif score >= 60: trend = "BULLISH"
    elif score >= 40: trend = "NEUTRAL"
    elif score >= 20: trend = "BEARISH"
    else: trend = "STRONG BEARISH"
    return trend, reasons, score

def support_resistance(d):
    prior = d.iloc[:-1] if len(d) > 1 else d
    r20 = prior.tail(20) if len(prior) >= 20 else prior
    r60 = prior.tail(60) if len(prior) >= 60 else prior
    r120 = prior.tail(120) if len(prior) >= 120 else prior
    primary_resistance = r20["High"].max(); primary_support = r20["Low"].min()
    secondary_resistance = r60["High"].max(); secondary_support = r60["Low"].min()
    if secondary_support == primary_support: secondary_support = r120["Low"].min()
    if secondary_resistance == primary_resistance: secondary_resistance = r120["High"].max()
    last = d.iloc[-1]
    return {"primary_support": primary_support, "primary_resistance": primary_resistance,
            "secondary_support": secondary_support, "secondary_resistance": secondary_resistance,
            "high_52w": last.get("52W_HIGH", np.nan), "low_52w": last.get("52W_LOW", np.nan),
            "secondary_support_is_distinct": secondary_support != primary_support,
            "secondary_resistance_is_distinct": secondary_resistance != primary_resistance}

def breakout_engine(d, sr, vol_ratio_threshold=1.5):
    """No look-ahead. Current candle excluded. Fresh vs continuation vs near."""
    last = d.iloc[-1]
    prev = d.iloc[-2] if len(d) >= 2 else last
    prior = d.iloc[:-1] if len(d) > 1 else d
    baseline_window = prior.tail(20) if len(prior) >= 20 else prior
    baseline_resistance = baseline_window["High"].max()
    resistance = sr["primary_resistance"]
    price = last["Close"]
    vol_ratio = last["VOL_RATIO"] if not pd.isna(last["VOL_RATIO"]) else 0
    was_below = prev["Close"] <= baseline_resistance
    now_above = price > baseline_resistance
    fresh_cross = now_above and was_below
    volume_confirmed = vol_ratio >= vol_ratio_threshold
    momentum_positive = last["MACD_HIST"] > 0
    distance_to_resistance = (resistance - price) / price * 100 if price > 0 else None
    if now_above and volume_confirmed and momentum_positive and fresh_cross:
        status = "CONFIRMED BREAKOUT"; note = "Closed above resistance with volume + momentum"
    elif now_above and volume_confirmed and momentum_positive and not fresh_cross:
        status = "EXTENDED BREAKOUT"; note = "Above resistance - continuation"
    elif now_above and (not volume_confirmed or not momentum_positive):
        status = "BREAKOUT ATTEMPT"; note = "Above resistance but weak confirmation"
    elif (not now_above) and distance_to_resistance is not None and 0 <= distance_to_resistance <= 3:
        status = "BREAKOUT READY"; note = "Within 3% of resistance"
    elif prev["Close"] > baseline_resistance and price < baseline_resistance:
        status = "FAILED BREAKOUT"; note = "Broke above but closed back below"
    else:
        status = "NO BREAKOUT"; note = "Not near breakout level"
    is_near_52w_high = False; is_52w_high_breakout = False
    high_52w = sr.get("high_52w")
    if not pd.isna(high_52w):
        prev_below_52w = prev["Close"] <= high_52w
        curr_above_52w = price >= high_52w
        if curr_above_52w and prev_below_52w:
            is_52w_high_breakout = True
            status = f"{status} / 52W HIGH BREAKOUT"
            note = f"{note} - FRESH breakout above 52W high ({round(high_52w, 2)})"
        elif curr_above_52w and not prev_below_52w:
            is_52w_high_breakout = True
            status = f"{status} / 52W HIGH CONTINUATION"
            note = f"{note} - continuing above 52W high ({round(high_52w, 2)})"
        elif price >= high_52w * 0.98 and price < high_52w:
            is_near_52w_high = True
            status = f"{status} / NEAR 52W HIGH"
            note = f"{note} - within 2% of 52W high ({round(high_52w, 2)})"
    return {"status": status, "note": note, "fresh_cross": fresh_cross,
            "resistance": resistance, "price": price, "volume_ratio": vol_ratio,
            "distance_to_resistance": distance_to_resistance,
            "is_near_52w_high": is_near_52w_high, "is_52w_high_breakout": is_52w_high_breakout}

def pullback_engine(d, trend, sr):
    last = d.iloc[-1]; support = sr["primary_support"]; price = last["Close"]
    if trend not in ("BULLISH", "STRONG BULLISH"):
        return {"status": "NO PULLBACK", "note": "Trend is not bullish"}
    near_support = abs(price - support) / price < 0.03 if price > 0 else False
    near_ema20 = abs(price - last["EMA20"]) / price < 0.02 if not pd.isna(last["EMA20"]) and price > 0 else False
    cooling_rsi = 35 <= last["RSI14"] <= 55
    bullish_candle = last["Close"] > last["Open"]
    if price < support * 0.98: return {"status": "BROKEN SUPPORT", "note": "Closed below primary support"}
    if (near_support or near_ema20) and cooling_rsi and bullish_candle:
        return {"status": "HEALTHY PULLBACK", "note": "At support/EMA20 with confirmation"}
    if near_support or near_ema20:
        return {"status": "PULLBACK WATCH", "note": "Approaching support/EMA20"}
    return {"status": "NO PULLBACK", "note": "Not near pullback zone"}

def momentum_engine(d):
    last = d.iloc[-1]; score = 0; signals = []
    if last["RSI14"] > 55: score += 1; signals.append("RSI positive")
    elif last["RSI14"] < 45: score -= 1; signals.append("RSI negative")
    if last["MACD_HIST"] > 0: score += 1; signals.append("MACD positive")
    else: score -= 1; signals.append("MACD negative")
    if len(d) >= 2 and last["MACD_HIST"] > d["MACD_HIST"].iloc[-2]:
        score += 1; signals.append("MACD accelerating")
    if last["ROC_10"] > 0: score += 1; signals.append("ROC positive")
    else: score -= 1; signals.append("ROC negative")
    if last["ADX14"] >= 20:
        if last["PLUS_DI"] > last["MINUS_DI"]: score += 1; signals.append("ADX confirms +DI")
        else: score -= 1; signals.append("ADX confirms -DI")
    window = d.tail(10)
    if len(window) >= 10:
        price_high = window["Close"].max(); rsi_high = window["RSI14"].max()
        if window["Close"].iloc[-1] >= price_high * 0.999 and window["RSI14"].iloc[-1] < rsi_high - 5:
            signals.append("Bearish divergence"); score -= 2
        price_low = window["Close"].min(); rsi_low = window["RSI14"].min()
        if window["Close"].iloc[-1] <= price_low * 1.001 and window["RSI14"].iloc[-1] > rsi_low + 5:
            signals.append("Bullish divergence"); score += 2
    if score >= 4: label = "STRONG MOMENTUM"
    elif score >= 2: label = "POSITIVE MOMENTUM"
    elif score >= -1: label = "NEUTRAL MOMENTUM"
    elif score >= -3: label = "NEGATIVE MOMENTUM"
    else: label = "STRONG NEGATIVE MOMENTUM"
    return {"label": label, "score": score, "signals": signals,
            "overbought": last["RSI14"] > 70, "oversold": last["RSI14"] < 30}

def projection_engine(d, trend, sr, momentum):
    last = d.iloc[-1]; price = last["Close"]
    atr_val = last["ATR14"] if not pd.isna(last["ATR14"]) else 0
    resistance = sr["primary_resistance"]; support = sr["primary_support"]
    is_bullish = trend in ("BULLISH", "STRONG BULLISH")
    is_bearish = trend in ("BEARISH", "STRONG BEARISH")
    if is_bullish:
        if price >= resistance:
            range_size = resistance - support
            upside_low = price + range_size * 0.5; upside_high = price + range_size * 1.0
            next_res = sr["secondary_resistance"]
        else:
            upside_low = resistance; upside_high = resistance + atr_val * 1.5
            next_res = sr["secondary_resistance"]
        return {"direction": "UP", "zone_low": upside_low, "zone_high": upside_high,
                "next_resistance": next_res,
                "label": f"Upside: {round(upside_low,2)} - {round(upside_high,2)}",
                "note": "Technical projection if uptrend continues"}
    elif is_bearish:
        if price <= support:
            range_size = resistance - support
            downside_low = price - range_size * 1.0; downside_high = price - range_size * 0.5
            next_sup = sr["secondary_support"]
        else:
            downside_low = support - atr_val * 1.5; downside_high = support
            next_sup = sr["secondary_support"]
        return {"direction": "DOWN", "zone_low": downside_low, "zone_high": downside_high,
                "next_support": next_sup,
                "label": f"Downside: {round(downside_low,2)} - {round(downside_high,2)}",
                "note": "Technical projection if downtrend continues"}
    else:
        return {"direction": "NEUTRAL", "zone_low": None, "zone_high": None,
                "label": "No clear direction", "note": "Neutral/range-bound structure"}

def detect_penny_setup(d, sr, threshold=PENNY_STOCK_THRESHOLD, rvol_threshold=2.0):
    last = d.iloc[-1]; price = last["Close"]
    if price > threshold:
        return {"is_penny": False, "status": "NORMAL PRICE STOCK", "note": f"Price {price} > {threshold}"}
    vol_ratio = last["VOL_RATIO"] if not pd.isna(last["VOL_RATIO"]) else 0
    near_resistance = False
    if price > 0 and sr["primary_resistance"]:
        near_resistance = abs(price - sr["primary_resistance"]) / price < 0.05
    broke_resistance = price > sr["primary_resistance"]
    rvol_expansion = vol_ratio >= rvol_threshold
    momentum_positive = last["MACD_HIST"] > 0
    if broke_resistance and rvol_expansion and momentum_positive:
        status = "PENNY BREAKOUT"; note = f"Low-priced breaking resistance with {round(vol_ratio,1)}x volume"
    elif near_resistance and rvol_expansion:
        status = "PENNY BREAKOUT READY"; note = f"Near resistance with {round(vol_ratio,1)}x volume"
    elif rvol_expansion:
        status = "PENNY VOLUME SPIKE"; note = f"Unusual volume ({round(vol_ratio,1)}x)"
    elif momentum_positive and near_resistance:
        status = "PENNY WATCH"; note = "Momentum near resistance"
    else:
        status = "PENNY (NO SETUP)"; note = "Low-priced but no unusual activity"
    return {"is_penny": True, "status": status, "note": note, "price": price,
            "vol_ratio": vol_ratio, "near_resistance": near_resistance,
            "broke_resistance": broke_resistance, "rvol_expansion": rvol_expansion}

def risk_engine(d, sr, breakout_status=""):
    """Returns error field if invalid. Never invents a stop."""
    last = d.iloc[-1]; price = last["Close"]
    atr_val = last["ATR14"] if not pd.isna(last["ATR14"]) else 0
    if price is None or price <= 0:
        return {"entry": None, "stop_loss": None, "risk_per_share": None, "target1": None,
                "target2": None, "rr1": None, "rr2": None, "conditional_entry": None,
                "conditional_entry_note": "Invalid price", "error": "Invalid entry price"}
    if "EXTENDED BREAKOUT" in breakout_status:
        tighter_stop = price - (2.5 * atr_val)
        stop_loss = max(tighter_stop, sr["primary_support"])
    else:
        swing_low = d.tail(10)["Low"].min()
        stop_loss = min(swing_low, sr["primary_support"]) - 0.3 * atr_val
    if stop_loss is None or stop_loss >= price:
        return {"entry": price, "stop_loss": None, "risk_per_share": None, "target1": None,
                "target2": None, "rr1": None, "rr2": None, "conditional_entry": None,
                "conditional_entry_note": "Invalid stop", "error": "Stop-loss must be below entry"}
    risk_per_share = price - stop_loss
    if risk_per_share <= 0:
        return {"entry": price, "stop_loss": stop_loss, "risk_per_share": None, "target1": None,
                "target2": None, "rr1": None, "rr2": None, "conditional_entry": None,
                "conditional_entry_note": "Non-positive risk", "error": "Risk per share must be positive"}
    near_or_above_resistance = price >= sr["primary_resistance"] * 0.995
    atr_target = price + 2.5 * atr_val if atr_val else price
    dist_res = (sr["primary_resistance"] - price) / price * 100 if price > 0 else 999
    is_breakout_ready = 0 < dist_res <= 3
    if near_or_above_resistance or is_breakout_ready:
        target1 = max(sr["secondary_resistance"], atr_target)
    else:
        target1 = sr["primary_resistance"]
    if sr["secondary_resistance"] > target1:
        target2 = sr["secondary_resistance"] + max(atr_val, (target1 - price) * 0.5)
    else:
        target2 = target1 + max(2 * atr_val, (target1 - price))
    reward1 = target1 - price; reward2 = target2 - price
    rr1 = reward1 / risk_per_share if risk_per_share > 0 else None
    rr2 = reward2 / risk_per_share if risk_per_share > 0 else None
    conditional_entry = None; conditional_entry_note = None
    if "EXTENDED BREAKOUT" in breakout_status:
        ema20_val = last["EMA20"] if not pd.isna(last["EMA20"]) else None
        if ema20_val is not None:
            conditional_entry = round(ema20_val, 2)
            conditional_entry_note = "Wait for pullback near EMA20"
    return {"entry": price, "stop_loss": stop_loss, "risk_per_share": risk_per_share,
            "target1": target1, "target2": target2, "rr1": rr1, "rr2": rr2,
            "conditional_entry": conditional_entry,
            "conditional_entry_note": conditional_entry_note, "error": None}

# ============================================================
# SCORING & SIGNAL
# ============================================================

def _momentum_component(momentum):
    if momentum["label"] == "STRONG MOMENTUM": score = 95
    elif momentum["label"] == "POSITIVE MOMENTUM": score = 75
    elif momentum["label"] == "NEUTRAL MOMENTUM": score = 50
    elif momentum["label"] == "NEGATIVE MOMENTUM": score = 25
    else: score = 5
    if momentum["overbought"]: score -= 10
    if momentum["oversold"]: score += 10
    return max(0, min(100, score))

def _volume_component(vol_ratio):
    if vol_ratio is None or pd.isna(vol_ratio): return 40
    if vol_ratio >= 3: return 100
    if vol_ratio >= 2: return 85
    if vol_ratio >= 1.5: return 65
    if vol_ratio >= 1.0: return 50
    return 30

def _setup_component(breakout_status, pullback_status):
    if "CONFIRMED" in breakout_status and "EXTENDED" not in breakout_status: return 100
    if pullback_status == "HEALTHY PULLBACK": return 88
    if "CONFIRMED" in breakout_status and "EXTENDED" in breakout_status: return 78
    if "BREAKOUT READY" in breakout_status: return 65
    if "BREAKOUT ATTEMPT" in breakout_status: return 55
    if pullback_status == "PULLBACK WATCH": return 50
    if "FAILED" in breakout_status or pullback_status == "BROKEN SUPPORT": return 10
    return 40

def _rr_component(rr1):
    if rr1 is None: return 20
    if rr1 >= 3: return 100
    if rr1 >= 2: return 85
    if rr1 >= MIN_RR: return 65
    if rr1 >= 1: return 35
    return 10

def _sr_component(price, sr):
    resistance = sr["primary_resistance"]; support = sr["primary_support"]
    if resistance == support: return 50
    position = (price - support) / (resistance - support)
    if 0.2 <= position <= 0.75: return 80
    if position < 0: return 15
    if position > 1.05: return 30
    return 55

def signal_engine(d, trend, trend_score, momentum, breakout, pullback, sr, risk_data, market):
    components = {
        "trend": min(100, trend_score),
        "momentum": _momentum_component(momentum),
        "volume": _volume_component(breakout["volume_ratio"]),
        "setup": _setup_component(breakout["status"], pullback["status"]),
        "rr": _rr_component(risk_data["rr1"]),
        "sr": _sr_component(risk_data.get("entry") or 0, sr),
    }
    score = round(sum(components[k] * WEIGHTS[k] for k in WEIGHTS), 1)
    rr_ok = risk_data["rr1"] is not None and risk_data["rr1"] >= MIN_RR
    trend_ok = trend not in ("BEARISH", "STRONG BEARISH")
    market_adjust = 0
    if market["regime"] == "BULLISH": market_adjust = 5
    elif market["regime"] == "BEARISH": market_adjust = -10
    elif market["regime"] == "HIGH VOLATILITY": market_adjust = -5
    adjusted_score = max(0, min(100, score + market_adjust))
    reasons = []
    if not trend_ok:
        signal = "WAIT" if adjusted_score >= 45 else "AVOID"
        setup_quality = "TREND BEARISH - NO LONG SETUP"
        reasons.append(f"❌ Trend {trend} hai — long entry nahi")
    elif not rr_ok:
        signal = "WAIT"; setup_quality = "R:R BELOW MINIMUM"
    elif adjusted_score >= 80: signal = "STRONG BUY"; setup_quality = "A+ SETUP"
    elif adjusted_score >= 65: signal = "BUY"; setup_quality = "A SETUP"
    elif adjusted_score >= 45: signal = "WAIT"; setup_quality = "B SETUP / WATCH"
    elif adjusted_score >= 25: signal = "REDUCE"; setup_quality = "WEAK"
    else: signal = "AVOID"; setup_quality = "POOR"
    if trend_ok:
        if components["trend"] >= 70: reasons.append(f"✅ Trend strong hai ({trend}) — upar ja raha hai")
        elif components["trend"] <= 35: reasons.append(f"❌ Trend kamzor hai ({trend}) — neeche ja raha hai")
        if components["momentum"] >= 70: reasons.append("✅ Momentum acha hai — buyers active")
        elif components["momentum"] <= 35: reasons.append("❌ Momentum kamzor — sellers havi hain")
        if "CONFIRMED" in breakout["status"]: reasons.append("✅ Resistance toot gaya — volume ke saath upar close")
        elif "BREAKOUT READY" in breakout["status"]: reasons.append("📌 Resistance ke qareeb hai — watch karo")
        if pullback["status"] == "HEALTHY PULLBACK": reasons.append("✅ Pullback hua hai support pe — achha entry point")
        elif pullback["status"] == "BROKEN SUPPORT": reasons.append("❌ Support toot gaya — structure kharab")
        if momentum["overbought"]: reasons.append("⚠️ RSI 70 se upar — overbought, correction aa sakta hai")
        if momentum["oversold"]: reasons.append("📌 RSI 30 se neeche — oversold, bounce aa sakta hai")
        if not rr_ok:
            rr_val = round(risk_data['rr1'], 2) if risk_data['rr1'] else 'N/A'
            reasons.append(f"❌ Risk:Reward kam hai ({rr_val}) — nuksan zyada mumkin")
        if market["regime"] == "BULLISH": reasons.append("✅ Overall market bullish hai — saath chal raha hai")
        elif market["regime"] == "BEARISH": reasons.append("❌ Overall market bearish hai — akele stock upar nahi rukega")
    return {"score": adjusted_score, "components": components, "signal": signal,
            "setup_quality": setup_quality, "reasons": reasons[:6]}

# ============================================================
# MARKET REGIME
# ============================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def market_snapshot():
    idx_df, source = fetch_market_index()
    if idx_df is None or len(idx_df) < 40:
        return {"regime": "UNAVAILABLE", "trend": "UNAVAILABLE",
                "source": source if source else "UNAVAILABLE",
                "reasoning": "KSE-100 data unavailable",
                "last_date": None, "last_close": None,
                "sma20": None, "sma50": None, "sma200": None}
    d = idx_df.copy()
    d["SMA20"] = sma(d["Close"], 20); d["SMA50"] = sma(d["Close"], 50)
    d["SMA200"] = sma(d["Close"], 200); d["RETURN_1D"] = d["Close"].pct_change()
    last_date = d.index[-1]; last_close = float(d["Close"].iloc[-1])
    sma20 = d["SMA20"].iloc[-1]; sma50 = d["SMA50"].iloc[-1]
    sma200 = d["SMA200"].iloc[-1] if len(d) >= 200 else None
    vol20 = d["RETURN_1D"].rolling(20).std().iloc[-1] * np.sqrt(252)
    if pd.isna(sma20) or pd.isna(sma50):
        regime = "UNAVAILABLE"; reasoning = "Insufficient SMA data"
    elif vol20 is not None and vol20 > 0.35:
        regime = "HIGH VOLATILITY"; reasoning = f"KSE-100 volatility {round(vol20*100,1)}%"
    elif last_close > sma20 > sma50: regime = "BULLISH"; reasoning = "KSE-100 > SMA20 > SMA50"
    elif last_close < sma20 < sma50: regime = "BEARISH"; reasoning = "KSE-100 < SMA20 < SMA50"
    else: regime = "NEUTRAL"; reasoning = "Mixed SMA alignment"
    return {"regime": regime, "trend": regime if regime in ("BULLISH","BEARISH") else "NEUTRAL",
            "source": source, "reasoning": reasoning, "last_date": last_date,
            "last_close": last_close, "sma20": sma20, "sma50": sma50, "sma200": sma200}

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def liquid_basket_trend():
    changes = []; successful = 0
    for ticker in PSX_LIQUID_UNIVERSE:
        try:
            df, status, _, _ = fetch_ohlcv(ticker, period="3mo")
            if status == "SUCCESS" and len(df) >= 6:
                recent = df.tail(5)
                if len(recent) >= 2:
                    avg_change = recent["Close"].pct_change().mean() * 100
                    if not pd.isna(avg_change):
                        changes.append(avg_change); successful += 1
        except Exception:
            continue
    if len(changes) < 10:
        return {"trend": "UNAVAILABLE", "change_pct": None,
                "stocks_contributing": successful, "note": "Insufficient data"}
    avg_change = np.mean(changes)
    if avg_change > 0.5: trend = "BULLISH (proxy)"
    elif avg_change > 0.1: trend = "MILD BULLISH (proxy)"
    elif avg_change > -0.1: trend = "NEUTRAL (proxy)"
    elif avg_change > -0.5: trend = "MILD BEARISH (proxy)"
    else: trend = "BEARISH (proxy)"
    return {"trend": trend, "change_pct": round(avg_change, 2),
            "stocks_contributing": successful,
            "note": "⚠️ PROXY — NOT official KSE-100"}

# ============================================================
# CLASSIFY / ANALYZE / SCREENER
# ============================================================

def classify_stock(price: float, avg_volume: float) -> str:
    if price > 200 and avg_volume > 100000: return "LARGE-LIKE (proxy)"
    elif price > 50 and avg_volume > 20000: return "MID-LIKE (proxy)"
    elif price > 20 and avg_volume > 5000: return "SMALL-LIKE (proxy)"
    elif price < 10 and avg_volume < 2000: return "MICRO (proxy)"
    elif price < 20: return "LOW-PRICE (proxy)"
    else: return "SMALL-LIKE (proxy)"

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def analyze_stock(ticker: str, period: str = "1y",
                  penny_threshold: float = PENNY_STOCK_THRESHOLD,
                  rvol_threshold: float = 2.0,
                  market: Optional[Dict] = None):
    """Market snapshot can be injected to avoid per-stock fetches during scans."""
    df, status, error, source = fetch_ohlcv(ticker, period=period)
    if status != "SUCCESS" or df is None or df.empty:
        return None, status, error, source
    d = build_indicators(df)
    if market is None:
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
        "ticker": normalize_ticker(ticker), "ticker_display": normalize_ticker_display(ticker),
        "df": d, "last": last,
        "trend": trend, "trend_score": trend_score, "trend_reasons": trend_reasons,
        "sr": sr, "breakout": breakout, "pullback": pullback,
        "momentum": momentum, "risk": risk, "penny": penny,
        "projection": projection, "signal": signal, "market": market,
        "data_date": d.index[-1],
        "has_sma200": not pd.isna(last.get("SMA200", np.nan)),
        "cap_size": cap_size, "data_source": source, "avg_volume": avg_volume,
        "reconciliation": reconcile_psx_vs_yf(ticker, df),
    }
    return result, "SUCCESS", None, source

def run_screener(universe: List[str], period: str = "1y",
                 penny_threshold: float = PENNY_STOCK_THRESHOLD,
                 rvol_threshold: float = 2.0,
                 market: Optional[Dict] = None):
    rows = []
    coverage = {"total": len(universe), "success": 0, "failed": 0, "analyzed": 0}
    total = len(universe)
    # Fetch market ONCE per scan (was: once per stock)
    if market is None:
        market = market_snapshot()
    for idx, ticker in enumerate(universe):
        if idx % 10 == 0:
            st.caption(f"📊 Scanning {idx+1}/{total}...")
        result, status, error, source = analyze_stock(
            ticker, period=period, penny_threshold=penny_threshold,
            rvol_threshold=rvol_threshold, market=market)
        if status != "SUCCESS":
            coverage["failed"] += 1
            rows.append({"Ticker": ticker, "Price": None, "Change %": None, "Trend": None,
                "Score": None, "Signal": "ERROR", "Status": "DATA UNAVAILABLE",
                "Penny": None, "Cap Size": None, "Source": source, "Why": "Data unavailable",
                "Avg Volume": None, "RR": None, "_ticker_raw": ticker})
            continue
        coverage["success"] += 1
        coverage["analyzed"] += 1
        last = result["last"]
        prev_close = result["df"]["Close"].iloc[-2] if len(result["df"]) >= 2 else last["Close"]
        change_pct = (last["Close"] - prev_close) / prev_close * 100 if prev_close else 0
        why_parts = []
        if result["trend"] in ("BULLISH", "STRONG BULLISH"): why_parts.append(result["trend"])
        if "CONFIRMED" in result["breakout"]["status"]: why_parts.append(result["breakout"]["status"])
        if result["signal"]["reasons"]:
            pos = [r for r in result["signal"]["reasons"] if r.startswith("✅")]
            why_parts.extend(pos[:2])
        if result["risk"]["rr1"] and result["risk"]["rr1"] >= 1.5:
            why_parts.append(f"R:R 1:{round(result['risk']['rr1'],2)}")
        if result["penny"]["is_penny"] and result["penny"]["status"] != "PENNY (NO SETUP)":
            why_parts.insert(0, result["penny"]["note"])
        why_text = " + ".join(why_parts[:4]) if why_parts else "No clear setup"
        avg_volume = result["avg_volume"] if not pd.isna(result["avg_volume"]) else 0
        rows.append({
            "Ticker": result["ticker_display"], "Price": round(last["Close"], 2),
            "Change %": round(change_pct, 2), "Trend": result["trend"],
            "Score": result["signal"]["score"], "Signal": result["signal"]["signal"],
            "Status": result["breakout"]["status"],
            "Penny": result["penny"]["status"] if result["penny"]["is_penny"] else "N/A",
            "RR": round(result["risk"]["rr1"], 2) if result["risk"]["rr1"] else None,
            "Cap Size": result["cap_size"], "Source": result["data_source"],
            "Why": why_text, "Avg Volume": round(avg_volume, 0), "_ticker_raw": ticker,
        })
    return pd.DataFrame(rows), coverage

# ============================================================
# ESTIMATE / TOP PICKS
# ============================================================

def estimate_pace_to_target(result: Dict) -> Tuple[str, str]:
    last = result["last"]; risk = result["risk"]; breakout = result["breakout"]; pullback = result["pullback"]
    if "CONFIRMED BREAKOUT" in breakout["status"] and "EXTENDED" not in breakout["status"]:
        trade_type = "Day/Short-Term"
    elif pullback["status"] == "HEALTHY PULLBACK": trade_type = "Swing"
    else: trade_type = "Momentum"
    atr_val = last["ATR14"] if not pd.isna(last["ATR14"]) else 0
    entry = risk.get("entry"); target1 = risk.get("target1")
    if not entry or not target1 or atr_val <= 0: return trade_type, "N/A"
    distance = abs(target1 - entry)
    est_sessions = max(1, round(distance / atr_val))
    return trade_type, f"~{est_sessions} sessions (technical estimate)"

def categorize_top_picks(screener_df: pd.DataFrame,
                        penny_threshold: float = PENNY_STOCK_THRESHOLD,
                        rvol_threshold: float = 2.0) -> Dict[str, pd.DataFrame]:
    """Day / Swing / Hold from existing screener data. No re-fetching."""
    empty = pd.DataFrame()
    if screener_df is None or screener_df.empty:
        return {"day": empty, "swing": empty, "hold": empty}
    df = screener_df[screener_df["Signal"] != "ERROR"].copy()
    if df.empty:
        return {"day": empty, "swing": empty, "hold": empty}

    # DAY
    day_mask = (df["Status"].astype(str).str.contains("BREAKOUT|READY|ATTEMPT", case=False, na=False)
                & (df["Score"].fillna(0) >= 55))
    day_df = df[day_mask].copy()
    if not day_df.empty:
        day_df = day_df.sort_values("Score", ascending=False).head(10)
        day_df["Hold Estimate"] = "1-2 sessions"
        day_df["Basis"] = "Momentum + breakout structure"

    # SWING
    swing_mask = (df["Trend"].astype(str).str.contains("BULLISH", case=False, na=False)
                  & (~df["Trend"].astype(str).str.contains("STRONG BULLISH", case=False, na=False))
                  & (df["RR"].fillna(0) >= 1.5) & (df["Score"].fillna(0) >= 45))
    swing_df = df[swing_mask].copy()
    if not swing_df.empty:
        swing_df = swing_df.sort_values("Score", ascending=False).head(10)
        swing_df["Hold Estimate"] = "5-15 sessions (technical estimate)"
        swing_df["Basis"] = "Trend + R:R >= 1.5"

    # HOLD
    hold_mask = (df["Trend"].astype(str).str.contains("STRONG BULLISH", case=False, na=False)
                 & (df["Score"].fillna(0) >= 60) & (df["RR"].fillna(0) >= 1.5))
    hold_df = df[hold_mask].copy()
    if not hold_df.empty:
        hold_df = hold_df.sort_values("Score", ascending=False).head(10)
        hold_df["Hold Estimate"] = "15-40 sessions (technical estimate)"
        hold_df["Basis"] = "Strong trend + healthy R:R"

    return {"day": day_df, "swing": swing_df, "hold": hold_df}

# ============================================================
# PORTFOLIO DECISION
# ============================================================

def portfolio_decision(holding: Dict, result: Dict) -> Tuple[str, str]:
    """Original + technical + active stop. None-safe."""
    if result is None: return "WATCH", "Data unavailable"
    signal = result["signal"]["signal"]; trend = result["trend"]
    pullback = result["pullback"]["status"]; breakout = result["breakout"]["status"]
    momentum = result["momentum"]; risk = result["risk"]
    technical_stop = risk["stop_loss"]
    original_stop = holding.get("original_stop_loss")
    if original_stop is not None and technical_stop is not None:
        active_stop = max(original_stop, technical_stop)
    elif original_stop is not None: active_stop = original_stop
    elif technical_stop is not None: active_stop = technical_stop
    else: return "WATCH", "⚠️ Stop-loss data unavailable — cannot make exit decision"
    current_price = result["last"]["Close"]
    entry_price = holding["buy_price"]
    pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price else 0
    if current_price < active_stop: return "EXIT", f"❌ Price {current_price} < active stop {active_stop}"
    if pullback == "BROKEN SUPPORT": return "EXIT", "❌ Support broken"
    if trend in ("BEARISH", "STRONG BEARISH"): return "REDUCE", f"📉 Trend {trend}"
    if signal in ("STRONG BUY", "BUY") and trend in ("BULLISH", "STRONG BULLISH"):
        if "CONFIRMED" in breakout and pnl_pct > 5:
            return "TRAIL STOP", f"✅ Profit {pnl_pct:.1f}% with breakout — trail stop"
        elif "CONFIRMED" in breakout: return "ADD ON CONFIRMATION", "✅ Breakout — add if risk allows"
        elif pnl_pct > 10: return "TRAIL STOP", f"✅ Profit {pnl_pct:.1f}% — trail stop"
        else: return "HOLD", "✅ Trend + signal constructive"
    if signal in ("REDUCE", "AVOID"):
        if pnl_pct > 0: return "REDUCE", f"⚠️ Signal {signal} while in profit"
        else: return "EXIT", f"❌ Signal {signal} and losing"
    if pullback == "HEALTHY PULLBACK" and trend in ("BULLISH", "STRONG BULLISH"):
        if pnl_pct > 0: return "ADD ON CONFIRMATION", "✅ Healthy pullback"
        else: return "HOLD", "✅ Healthy pullback — maintain"
    if momentum["label"] in ("STRONG NEGATIVE MOMENTUM", "NEGATIVE MOMENTUM") \
            and trend not in ("BULLISH", "STRONG BULLISH"):
        if pnl_pct > 0: return "REDUCE", "⚠️ Negative momentum"
        else: return "EXIT", "❌ Negative momentum + losing"
    return "HOLD", f"📊 No clear signal — stop at {round(active_stop, 2)}"

# ============================================================
# CHART
# ============================================================

def build_chart(result, show_bb=False, show_sma200=False,
                show_support_resistance=True, show_rsi=False, show_macd=False):
    d = result["df"].tail(150)
    risk = result["risk"]; sr = result["sr"]; trend = result["trend"]
    num_rows = 1
    if show_rsi: num_rows += 1
    if show_macd: num_rows += 1
    row_heights = [0.5]
    if show_rsi: row_heights.append(0.17)
    if show_macd: row_heights.append(0.17)
    if len(row_heights) == 3: row_heights = [0.5, 0.17, 0.17]
    elif len(row_heights) == 2: row_heights = [0.6, 0.25]
    subplot_titles = ["Price"]
    if show_rsi: subplot_titles.append("RSI (14)")
    if show_macd: subplot_titles.append("MACD")
    fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.04, subplot_titles=subplot_titles)
    row = 1
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"],
        low=d["Low"], close=d["Close"], name="Price",
        increasing_line_color="#10B981", decreasing_line_color="#EF4444"), row=row, col=1)
    trend_color = "#10B981" if trend in ("BULLISH", "STRONG BULLISH") else \
                  "#EF4444" if trend in ("BEARISH", "STRONG BEARISH") else "#F59E0B"
    trend_arrow = "↑" if trend in ("BULLISH", "STRONG BULLISH") else \
                  "↓" if trend in ("BEARISH", "STRONG BEARISH") else "→"
    fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
        text=f"{trend_arrow} TREND: {trend}", showarrow=False,
        font=dict(color=trend_color, size=14, family="monospace"),
        bgcolor="rgba(255, 255, 255, 0.9)", bordercolor=trend_color,
        borderwidth=1, borderpad=4, opacity=0.95)
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA20"],
        line=dict(color="#3B82F6", width=1.2), name="SMA20"), row=row, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA50"],
        line=dict(color="#F59E0B", width=1.2), name="SMA50"), row=row, col=1)
    if show_sma200 and result["has_sma200"]:
        fig.add_trace(go.Scatter(x=d.index, y=d["SMA200"],
            line=dict(color="#8B5CF6", width=1, dash="dot"), name="SMA200"), row=row, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_UPPER"],
            line=dict(color="#94A3B8", width=0.8, dash="dot"), name="BB Upper"), row=row, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_LOWER"],
            line=dict(color="#94A3B8", width=0.8, dash="dot"), name="BB Lower"), row=row, col=1)
    if show_support_resistance:
        fig.add_hline(y=sr["primary_resistance"], line_dash="dash",
            line_color="#EF4444", annotation_text="Resistance", row=row, col=1)
        fig.add_hline(y=sr["primary_support"], line_dash="dash",
            line_color="#10B981", annotation_text="Support", row=row, col=1)
    if risk.get("stop_loss") and risk["stop_loss"] > 0:
        fig.add_hline(y=risk["stop_loss"], line_dash="dot",
            line_color="#F59E0B", annotation_text="Stop", row=row, col=1)
    if risk.get("target1") and risk["target1"] > 0:
        fig.add_hline(y=risk["target1"], line_dash="dot",
            line_color="#3B82F6", annotation_text="T1", row=row, col=1)
    # Projection
    try:
        if (risk.get("target1") and risk["target1"] > 0 and len(d) >= 2 and risk.get("entry")):
            last_date = d.index[-1]
            future_dates = pd.bdate_range(start=last_date, periods=13, freq="B")[1:]
            current_price = float(d["Close"].iloc[-1]); target_price = float(risk["target1"])
            n = len(future_dates)
            if n > 0:
                future_prices = np.linspace(current_price, target_price, n + 1)[1:]
                breakout_status = result.get("breakout", {}).get("status", "")
                if "CONFIRMED" in breakout_status: basis = "Breakout measured-move"
                elif "EXTENDED" in breakout_status: basis = "Trend continuation"
                else: basis = "R:R Target (T1)"
                arrow_color = "#10B981" if target_price > current_price else "#EF4444"
                proj_x = [last_date] + list(future_dates)
                proj_y = [current_price] + list(future_prices)
                fig.add_trace(go.Scatter(x=proj_x, y=proj_y, mode="lines",
                    line=dict(color=arrow_color, width=2, dash="dash"),
                    name=f"Projection → {round(target_price, 2)}", opacity=0.7,
                    hovertemplate="Projection<br>Date: %{x|%d-%b}<br>Price: %{y:.2f}<extra></extra>"),
                    row=row, col=1)
                fig.add_trace(go.Scatter(x=[future_dates[-1]], y=[future_prices[-1]],
                    mode="markers+text",
                    marker=dict(color=arrow_color, size=10, symbol="triangle-right"),
                    text=[f"→ {round(target_price, 2)} ({basis})"],
                    textposition="middle right", textfont=dict(color=arrow_color, size=11),
                    name="Projected", showlegend=False,
                    hovertemplate=f"<b>Projected: {round(target_price, 2)}</b><br>Basis: {basis}<br>"
                                  f"<i>Technical estimate, not guaranteed</i><extra></extra>"),
                    row=row, col=1)
    except Exception:
        pass
    if show_rsi:
        row += 1
        fig.add_trace(go.Scatter(x=d.index, y=d["RSI14"],
            line=dict(color="#3B82F6", width=1.3), name="RSI"), row=row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#EF4444", row=row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#10B981", row=row, col=1)
        fig.update_yaxes(range=[0, 100], row=row, col=1)
    if show_macd:
        row += 1
        fig.add_trace(go.Scatter(x=d.index, y=d["MACD"],
            line=dict(color="#3B82F6", width=1), name="MACD"), row=row, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["MACD_SIGNAL"],
            line=dict(color="#F59E0B", width=1), name="Signal"), row=row, col=1)
        hist_colors = np.where(d["MACD_HIST"] >= 0, "#10B981", "#EF4444")
        fig.add_trace(go.Bar(x=d.index, y=d["MACD_HIST"],
            marker_color=hist_colors, name="Hist"), row=row, col=1)
    fig.update_layout(height=700 if not show_rsi and not show_macd else 800,
        showlegend=True, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white")
    return fig

# ============================================================
# UI HELPERS
# ============================================================

def get_signal_class(signal):
    if signal in ("STRONG BUY", "BUY"): return "signal-buy"
    elif signal == "WAIT": return "signal-wait"
    else: return "signal-avoid"

def show_stale_data_warning(freshness_status, freshness_warning):
    if freshness_status == "STALE":
        st.warning(f"🔴 {freshness_warning}")
        st.caption("⚠️ Data stale hai. Signal confidence kam hai.")
    elif freshness_status == "DELAYED":
        st.info(f"⚠️ {freshness_warning}")

def get_indicator_explanation(indicator: str) -> str:
    explanations = {
        "SMA20": "20 din ka average price. Price iske upar = short-term mein mazboot.",
        "SMA50": "50 din ka average. Medium-term direction batata hai.",
        "RSI": "0-100 ka meter. 70+ = overbought. 30- = oversold.",
        "MACD": "Do average ka farq. Positive = momentum upar.",
        "ADX": "Trend ki strength. 25+ = strong trend.",
        "RVOL": "Aaj ka volume normal se kitna zyada. 2x = double.",
        "Support": "Jahan buyers aate hain — price wahin rukta hai.",
        "Resistance": "Jahan sellers aate hain — price wahin rukta hai.",
        "Breakout": "Price resistance ke upar jaata hai, volume ke saath.",
        "Stop Loss": "Yahan exit karo loss control ke liye.",
        "Target": "Technical price jahan tak ja sakta hai.",
        "ATR": "Rozana ka average movement.",
    }
    return explanations.get(indicator, "Technical indicator.")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "<div style='font-family:monospace; color:#2563EB; font-size:22px; "
    "font-weight:bold; letter-spacing:1px;'>PSX QUANT ENGINE</div>"
    "<div style='color:#64748B; font-size:11px; margin-bottom:10px;'>"
    "Quantitative Decision Support · v8.0</div>", unsafe_allow_html=True)

st.sidebar.subheader("🔌 Provider Status")
psx_avail = PROVIDER_STATUS.get("psx_official", {}).get("available", False)
if psx_avail:
    kse100_ok = "✅ verified" if PROVIDER_STATUS["psx_official"].get("kse100") else "⚠️ unverified"
    st.sidebar.success(f"✅ PSX Official: Working ({kse100_ok})")
else: st.sidebar.warning("⚠️ PSX Official: Not available")
yf_avail = PROVIDER_STATUS.get("yfinance", {}).get("available", True)
if yf_avail: st.sidebar.success("✅ yfinance: Working (fallback)")
else: st.sidebar.error("❌ yfinance: Failed")
psxd_avail = PROVIDER_STATUS.get("psxdata", {}).get("available", False)
if psxd_avail: st.sidebar.info("ℹ️ psxdata: Working (experimental)")
else: st.sidebar.caption("psxdata: Not installed (optional)")

st.sidebar.divider()
st.sidebar.header("📈 Analysis")
sidebar_ticker = st.sidebar.text_input("Ticker", value="SYS",
    help="PSX symbol (SYS, OGDC, LUCK, etc.)")
period = st.sidebar.selectbox("Analysis Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
penny_threshold = st.sidebar.number_input("Penny Stock Threshold (PKR)",
    min_value=10, max_value=200, value=50, step=5,
    help="Is price se neeche wale penny stocks hain")
rvol_threshold = st.sidebar.slider("Penny RVOL Threshold (x avg)",
    min_value=1.0, max_value=5.0, value=2.0, step=0.5,
    help="Penny stock pe unusual volume threshold")

st.sidebar.divider()
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    fetch_ohlcv.clear()
    fetch_market_index.clear()
    market_snapshot.clear()
    analyze_stock.clear()
    fetch_psx_official_index_json.clear()
    fetch_psx_official_market_watch.clear()
    for k in ["screener_df", "watchlist_df", "next_session_df", "top_picks_df", "proxy_result"]:
        st.session_state.pop(k, None)
    st.sidebar.success("Cache cleared!")

st.sidebar.caption("KSE-100: PSX Official → yfinance | Stock OHLCV: psxdata → yfinance")
st.sidebar.caption("Market-watch: PSX Official HTML | Cache: 5min")
st.sidebar.caption(f"Checked: {pkt_now().strftime('%d-%b %H:%M')} PKT")

# Session state init
if "portfolio" not in st.session_state: st.session_state.portfolio = []
if "watchlist" not in st.session_state:
    st.session_state.watchlist = "SYS, OGDC, HBL, LUCK, FFC, ENGRO"

# ============================================================
# MAIN TABS
# ============================================================

tab_dash, tab_screener, tab_breakouts, tab_next, tab_watch, tab_port, tab_market, tab_top = st.tabs([
    "📊 Dashboard", "🔍 Screener", "🚀 Breakouts", "📅 Next Session",
    "📋 Watchlist", "💼 Portfolio", "📈 Market", "🎯 Top Picks"])

# ============================================================
# DASHBOARD TAB
# ============================================================

with tab_dash:
    result, status, error, source = analyze_stock(
        sidebar_ticker, period=period,
        penny_threshold=penny_threshold, rvol_threshold=rvol_threshold)
    if status != "SUCCESS":
        st.error(f"❌ {sidebar_ticker} analyze nahi ho saka: {error}")
        with st.expander("🔍 Troubleshooting"):
            st.write(f"**Status:** {status}"); st.write(f"**Error:** {error}")
            st.write(f"**Source:** {source if source else 'N/A'}")
            st.write("**Try:** Refresh data ya ticker spelling check karo")
    else:
        last = result["last"]; sig = result["signal"]
        freshness_status, _, freshness_warning = get_freshness_status(result["data_date"])
        show_stale_data_warning(freshness_status, freshness_warning)
        recon = result.get("reconciliation", {})
        if recon.get("mismatch"):
            st.warning(f"⚠️ Price mismatch — PSX: {recon['psx_price']} | "
                       f"yfinance: {recon['yf_price']} | farq {recon['diff_pct']}% "
                       f"(PSX official use ho raha hai)")
        elif recon.get("not_comparable"):
            st.caption(f"ℹ️ Reconciliation: {recon.get('note', 'not comparable')}")
        col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.2, 1.2, 1.2])
        with col1:
            st.markdown(f"<span class='current-price'>{round(last['Close'], 2)}</span>",
                unsafe_allow_html=True)
            prev_close = result["df"]["Close"].iloc[-2] if len(result["df"]) >= 2 else last["Close"]
            change = last["Close"] - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0
            cc = "change-positive" if change >= 0 else "change-negative"
            st.markdown(f"<span class='{cc}'>{'▲' if change >= 0 else '▼'} "
                f"{round(change, 2)} ({round(change_pct, 2)}%)</span>", unsafe_allow_html=True)
            st.caption(f"{result['ticker_display']} · {result['cap_size']}")
            st.caption(f"Data: {source} · {freshness_status} · {result['data_date'].date()}")
        with col2:
            sc = get_signal_class(sig["signal"])
            st.markdown("**Signal**")
            st.markdown(f"<span class='{sc}'>{sig['signal']}</span>", unsafe_allow_html=True)
        with col3: st.metric("Score", f"{sig['score']}/100")
        with col4: st.metric("Trend", result["trend"])
        with col5: st.metric("Setup", sig["setup_quality"])
        mkt = result["market"]
        if mkt["regime"] != "UNAVAILABLE":
            st.caption(f"📊 KSE-100: {mkt['regime']} | Level: "
                f"{round(mkt['last_close'], 0) if mkt['last_close'] else 'N/A'}")
            st.caption(f"Source: {mkt['source']}")
        else:
            st.caption("📊 KSE-100: DATA UNAVAILABLE")
        with st.expander("🔍 Why this signal?", expanded=True):
            for r in sig["reasons"]: st.write(r)
        st.subheader("📋 Trade Plan")
        risk = result["risk"]; trend = result["trend"]
        if risk.get("error"): st.warning(f"⚠️ Risk calculation error: {risk['error']}")
        elif trend in ("BEARISH", "STRONG BEARISH"):
            st.warning("📉 Bearish structure — no long setup.")
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Entry", round(risk["entry"], 2))
            c2.metric("Stop Loss", round(risk["stop_loss"], 2))
            c3.metric("Target 1", round(risk["target1"], 2))
            c4.metric("Target 2", round(risk["target2"], 2))
            c5.metric("R:R", f"1:{round(risk['rr1'], 2) if risk['rr1'] else 'N/A'}")
            if risk.get("conditional_entry") is not None:
                st.caption(f"💡 Better Entry: {risk['conditional_entry']} — {risk['conditional_entry_note']}")
        proj = result["projection"]
        if proj["direction"] != "NEUTRAL":
            st.info(f"📈 **Technical Projection:** {proj['label']}")
            st.caption(proj["note"])
        if result["penny"]["is_penny"]:
            st.warning(f"🪙 {result['penny']['status']} - {result['penny']['note']}")
        st.subheader("📊 Chart")
        c1, c2, c3, c4, c5 = st.columns(5)
        show_bb = c1.checkbox("Bollinger Bands", value=False)
        show_sma200 = c2.checkbox("SMA200", value=False)
        show_sr = c3.checkbox("Support/Resistance", value=True)
        show_rsi = c4.checkbox("RSI", value=False)
        show_macd = c5.checkbox("MACD", value=False)
        st.plotly_chart(build_chart(result, show_bb, show_sma200, show_sr, show_rsi, show_macd),
            use_container_width=True)
        st.caption("📉 **Projection line** = technical estimate. Market forecast NAHI.")
        with st.expander("📖 Indicator Explanations"):
            st.markdown(f"**SMA20:** {get_indicator_explanation('SMA20')} (Current: {round(last['SMA20'],2)})")
            st.markdown(f"**SMA50:** {get_indicator_explanation('SMA50')} (Current: {round(last['SMA50'],2)})")
            st.markdown(f"**RSI:** {get_indicator_explanation('RSI')} (Current: {round(last['RSI14'],1)})")
            st.markdown(f"**MACD:** {get_indicator_explanation('MACD')} (Current: {round(last['MACD_HIST'],2)})")
            st.markdown(f"**ADX:** {get_indicator_explanation('ADX')} (Current: {round(last['ADX14'],1)})")
            rvol_val = round(last['VOL_RATIO'],2) if not pd.isna(last['VOL_RATIO']) else 'N/A'
            st.markdown(f"**RVOL:** {get_indicator_explanation('RVOL')} (Current: {rvol_val}x)")
            st.markdown(f"**Support:** {get_indicator_explanation('Support')} ({round(result['sr']['primary_support'],2)})")
            st.markdown(f"**Resistance:** {get_indicator_explanation('Resistance')} ({round(result['sr']['primary_resistance'],2)})")
            st.markdown(f"**Breakout:** {result['breakout']['status']} — {result['breakout']['note']}")
            st.markdown(f"**Stop Loss:** {get_indicator_explanation('Stop Loss')} ({round(risk['stop_loss'],2) if risk['stop_loss'] else 'N/A'})")
            st.markdown(f"**Target:** {get_indicator_explanation('Target')} ({round(risk['target1'],2) if risk['target1'] else 'N/A'})")
            st.markdown(f"**ATR:** {get_indicator_explanation('ATR')} (Current: {round(last['ATR14'],2)})")
        with st.expander("📊 Support / Resistance Details"):
            sr = result["sr"]
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Primary Support", round(sr["primary_support"], 2))
            cc2.metric("Primary Resistance", round(sr["primary_resistance"], 2))
            cc3.metric("Secondary Support", round(sr["secondary_support"], 2))
            cc4.metric("Secondary Resistance", round(sr["secondary_resistance"], 2))
            if not sr.get("secondary_support_is_distinct", True):
                st.caption("⚠️ Secondary Support = Primary Support")
            if not sr.get("secondary_resistance_is_distinct", True):
                st.caption("⚠️ Secondary Resistance = Primary Resistance")
            if not pd.isna(sr["high_52w"]):
                st.caption(f"52-Week High: {round(sr['high_52w'], 2)} | 52-Week Low: {round(sr['low_52w'], 2)}")
            else: st.caption("N/A — insufficient 52-week history (need ≥252 sessions)")

# ============================================================
# SCREENER TAB
# ============================================================

with tab_screener:
    st.subheader("🔍 PSX Opportunity Scanner")
    st.caption("Data: psxdata → yfinance | Default: first 100 symbols (speed)")
    universe_option = st.selectbox("Universe",
        ["Dynamic (from provider)", "Liquid PSX (~34)", "Small Cap (~25)", "Custom (from watchlist)"],
        index=0)
    col1, col2 = st.columns(2)
    with col1: custom_syms = st.text_input("Add extra symbols (comma-separated)", "")
    with col2: scan_full = st.checkbox("Scan full universe (slow, 10+ min)", value=False,
        help="Default scans first 100 symbols")
    st.markdown("**Filters**")
    p1, p2, p3 = st.columns(3)
    with p1:
        signal_filter = st.multiselect("Signal",
            ["STRONG BUY", "BUY", "WAIT", "REDUCE", "AVOID"],
            default=["STRONG BUY", "BUY"])
    with p2:
        price_range = st.slider("Price Range (PKR)", min_value=0.0, max_value=5000.0,
            value=(0.0, 5000.0), step=10.0)
    with p3: min_score = st.slider("Min Score", 0, 100, 0)
    include_penny = st.checkbox(f"Include penny stocks (below PKR {int(penny_threshold)})",
        value=False, help="Penny stocks high volatility")
    if include_penny:
        st.caption("⚠️ Penny stocks: high volatility, liquidity risk, false breakouts")
    with st.expander("⚙️ Advanced Filters"):
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            category_filter = st.multiselect("Category",
                ["LARGE-LIKE (proxy)", "MID-LIKE (proxy)", "SMALL-LIKE (proxy)",
                 "LOW-PRICE (proxy)", "MICRO (proxy)"], default=[])
        with a2:
            breakout_filter = st.multiselect("Breakout Status",
                ["CONFIRMED BREAKOUT", "EXTENDED BREAKOUT",
                 "BREAKOUT READY", "BREAKOUT ATTEMPT"], default=[])
        with a3: min_rr = st.slider("Min R:R", 0.0, 5.0, 0.0, 0.1)
        with a4: min_avg_volume = st.number_input("Min Avg Volume", min_value=0, value=0, step=1000)
    if st.button("🔍 Run Screener", use_container_width=True):
        with st.spinner("Scanning PSX universe..."):
            if universe_option == "Dynamic (from provider)":
                universe, _, _ = fetch_universe()
            elif universe_option == "Liquid PSX (~34)": universe = PSX_LIQUID_UNIVERSE
            elif universe_option == "Small Cap (~25)": universe = PSX_SMALL_CAP_UNIVERSE
            else:
                universe = [t.strip() + ".KA" if not t.strip().endswith(".KA") else t.strip()
                           for t in st.session_state.watchlist.split(",") if t.strip()]
            if custom_syms:
                extra = [t.strip() + ".KA" if not t.strip().endswith(".KA") else t.strip()
                        for t in custom_syms.split(",") if t.strip()]
                universe = list(dict.fromkeys(universe + extra))
            if not scan_full and len(universe) > 100:
                total_count = len(universe)
                universe = universe[:100]
                st.caption(f"📊 First 100 symbols (of {total_count}).")
            else: st.caption(f"📊 Scanning {len(universe)} symbols...")
            market = market_snapshot()
            screener_df, coverage = run_screener(universe, period="1y",
                penny_threshold=penny_threshold, rvol_threshold=rvol_threshold, market=market)
            st.session_state["screener_df"] = screener_df
            st.session_state["screener_coverage"] = coverage
    if "screener_coverage" in st.session_state:
        cov = st.session_state["screener_coverage"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", cov["total"]); c2.metric("Analyzed", cov["analyzed"])
        c3.metric("Success", cov["success"]); c4.metric("Failed", cov["failed"])
    if "screener_df" in st.session_state:
        df_s = st.session_state["screener_df"]
        view = df_s.copy()
        if signal_filter: view = view[view["Signal"].isin(signal_filter)]
        if min_score > 0: view = view[view["Score"].fillna(0) >= min_score]
        view = view[(view["Price"].fillna(0) >= price_range[0]) &
                    (view["Price"].fillna(0) <= price_range[1])]
        if not include_penny:
            penny_mask = view["Penny"].notna() & (view["Penny"] != "N/A")
            view = view[~penny_mask]
        if category_filter: view = view[view["Cap Size"].isin(category_filter)]
        if breakout_filter:
            view = view[view["Status"].astype(str).apply(
                lambda s: any(bf in s for bf in breakout_filter))]
        if min_rr > 0: view = view[view["RR"].fillna(0) >= min_rr]
        if min_avg_volume > 0: view = view[view["Avg Volume"].fillna(0) >= min_avg_volume]
        view = view[view["Signal"] != "ERROR"]
        sort_by = st.selectbox("Sort by", ["Score", "Change %", "Price"], index=0)
        view = view.sort_values(sort_by, ascending=False, na_position="last")
        total_scanned = len(df_s)
        penny_count = len(view[view["Penny"].notna() & (view["Penny"] != "N/A")])
        st.caption(f"📊 Showing **{len(view)}** of {total_scanned} scanned"
            + (f" ({penny_count} penny stocks)" if include_penny and penny_count > 0 else ""))
        if not view.empty:
            display_cols = [c for c in view.columns if not c.startswith("_")]
            st.dataframe(view[display_cols], use_container_width=True, hide_index=True)
        else: st.info("No stocks match filters")

# ============================================================
# BREAKOUTS TAB
# ============================================================

with tab_breakouts:
    st.subheader("🚀 Breakout Candidates")
    if "screener_df" not in st.session_state:
        st.info("Pehle Screener tab pe 'Run Screener' chalao.")
    else:
        df_s = st.session_state["screener_df"]
        keywords = ["CONFIRMED", "READY", "ATTEMPT", "BREAKOUT"]
        bo = df_s[df_s["Status"].astype(str).str.contains('|'.join(keywords), case=False, na=False)]
        if bo.empty: st.info("Koi breakout candidate nahi mila.")
        else:
            breakout_rows = []
            market = market_snapshot()
            for _, row in bo.iterrows():
                ticker_raw = row.get("_ticker_raw", row["Ticker"])
                result, status, _, _ = analyze_stock(ticker_raw, period="1y",
                    penny_threshold=penny_threshold, rvol_threshold=rvol_threshold, market=market)
                if status == "SUCCESS":
                    breakout_rows.append({
                        "Ticker": row["Ticker"], "Price": row["Price"],
                        "Change %": row["Change %"], "Trend": row["Trend"],
                        "Score": row["Score"], "Signal": row["Signal"],
                        "Status": row["Status"],
                        "Resistance": round(result["breakout"]["resistance"], 2),
                        "Dist %": round(result["breakout"]["distance_to_resistance"], 2)
                                    if result["breakout"]["distance_to_resistance"] is not None else "N/A",
                        "Momentum": result["momentum"]["label"], "RR": row["RR"],
                        "52W": "🚀 BREAKOUT" if result["breakout"]["is_52w_high_breakout"]
                               else "🔵 NEAR" if result["breakout"]["is_near_52w_high"] else "-",
                        "Why": row["Why"]})
            if breakout_rows:
                bo_df = pd.DataFrame(breakout_rows).sort_values("Score", ascending=False)
                st.dataframe(bo_df, use_container_width=True, hide_index=True)
            else: st.info("Koi detailed breakout data available nahi.")

# ============================================================
# NEXT SESSION TAB (BUTTON-ONLY — no auto scan)
# ============================================================

with tab_next:
    st.subheader("📅 Next Session Watchlist")
    st.caption("Next session ke top candidates — manual scan on demand")
    if st.button("🔄 Refresh Next Session", use_container_width=True):
        with st.spinner("Scanning candidates..."):
            combined = list(dict.fromkeys(PSX_LIQUID_UNIVERSE + PSX_SMALL_CAP_UNIVERSE))
            market = market_snapshot()
            st.session_state["next_session_df"], _ = run_screener(
                combined, period="1y", penny_threshold=penny_threshold,
                rvol_threshold=rvol_threshold, market=market)
    if "next_session_df" not in st.session_state:
        st.info("Click **Refresh Next Session** to scan candidates.")
    else:
        df_s = st.session_state["next_session_df"]
        top = df_s[df_s["Signal"].isin(["STRONG BUY", "BUY"])].dropna(subset=["Score"])
        top = top.sort_values("Score", ascending=False).head(10)
        if top.empty: st.info("Koi strong BUY candidate nahi mila.")
        else:
            rows = []
            market = market_snapshot()
            for _, row in top.iterrows():
                ticker_raw = row.get("_ticker_raw", row["Ticker"])
                if row["Score"] >= 75 and row["Trend"] in ("BULLISH", "STRONG BULLISH"): sq = "HIGH"
                elif row["Score"] >= 60: sq = "MEDIUM"
                else: sq = "LOW"
                result, status, _, _ = analyze_stock(ticker_raw, period="1y",
                    penny_threshold=penny_threshold, rvol_threshold=rvol_threshold, market=market)
                if status == "SUCCESS":
                    trade_type, pace = estimate_pace_to_target(result)
                    fresh, _, _ = get_freshness_status(result["data_date"])
                else: trade_type, pace, fresh = "N/A", "N/A", "UNAVAILABLE"
                rows.append({"Ticker": row["Ticker"], "Price": row["Price"],
                    "Trend": row["Trend"], "Score": row["Score"], "Signal": row["Signal"],
                    "Status": row["Status"], "RR": row["RR"], "Trade Type": trade_type,
                    "Est. Pace": pace, "Setup Quality": sq, "Data": fresh, "Why": row["Why"]})
            cap_df = pd.DataFrame(rows)
            st.dataframe(cap_df, use_container_width=True, hide_index=True)
            st.caption("⚠️ 'Est. Pace' ATR-based rough estimate. Guaranteed timeline nahi.")

# ============================================================
# WATCHLIST TAB
# ============================================================

with tab_watch:
    st.subheader("📋 Watchlist")
    st.markdown("**Edit Watchlist**")
    new_watchlist = st.text_area("Tickers (comma-separated)",
        value=st.session_state.watchlist, key="watchlist_editor",
        help="Jaise: SYS, OGDC, HBL, LUCK")
    wl1, wl2 = st.columns([1, 3])
    with wl1:
        if st.button("💾 Save & Update", use_container_width=True):
            st.session_state.watchlist = new_watchlist
            st.success("Watchlist saved!")
    with wl2: st.caption(f"Current: {st.session_state.watchlist}")
    st.divider()
    tickers = [t.strip() for t in st.session_state.watchlist.split(",") if t.strip()]
    if st.button("🔄 Refresh Analysis", use_container_width=True):
        with st.spinner("Analyzing watchlist..."):
            market = market_snapshot()
            watchlist_df, _ = run_screener(tickers, period="1y",
                penny_threshold=penny_threshold, rvol_threshold=rvol_threshold, market=market)
            st.session_state["watchlist_df"] = watchlist_df
    if "watchlist_df" in st.session_state:
        df_w = st.session_state["watchlist_df"]
        if not df_w.empty:
            cols = ["Ticker", "Price", "Change %", "Trend", "Score", "Signal",
                    "Status", "RR", "Cap Size", "Source", "Why"]
            cols = [c for c in cols if c in df_w.columns]
            st.dataframe(df_w[cols], use_container_width=True, hide_index=True)
        else: st.info("Watchlist empty ya koi data nahi mila")
    else: st.info("Click 'Refresh Analysis' to analyze")

# ============================================================
# PORTFOLIO TAB
# ============================================================

with tab_port:
    st.subheader("💼 Portfolio Tracker (max 5)")
    st.caption("Active Stop = max(Original Stop, Technical Stop) for long positions")
    st.caption("⚠️ Agar stop-loss data unavailable ho, decision 'WATCH' hoga")
    with st.form("add_holding"):
        c1, c2, c3, c4 = st.columns(4)
        h_ticker = c1.text_input("Ticker")
        h_price = c2.number_input("Buy Price (PKR)", min_value=0.0, step=0.5)
        h_shares = c3.number_input("Shares", min_value=0, step=1)
        h_stop = c4.number_input("Original Stop (PKR)", min_value=0.0, step=0.5,
                                 help="Aap ka original stop-loss level")
        submitted = st.form_submit_button("Add Holding")
        if submitted and h_ticker and h_price > 0 and h_shares > 0:
            if len(st.session_state.portfolio) >= 5: st.warning("Maximum 5 holdings")
            else:
                st.session_state.portfolio.append({
                    "ticker": h_ticker.strip().upper(), "buy_price": h_price,
                    "shares": h_shares,
                    "original_stop_loss": h_stop if h_stop > 0 else None})
                st.success(f"Added {h_ticker}")
    if st.session_state.portfolio:
        rows = []; total_invested = 0; total_current = 0
        market = market_snapshot()
        for h in st.session_state.portfolio:
            result, status, error, _ = analyze_stock(h["ticker"], period="1y",
                penny_threshold=penny_threshold, rvol_threshold=rvol_threshold, market=market)
            invested = h["buy_price"] * h["shares"]
            total_invested += invested
            if status == "SUCCESS":
                cur_price = result["last"]["Close"]
                cur_value = cur_price * h["shares"]
                total_current += cur_value
                pnl = cur_value - invested
                pnl_pct = pnl / invested * 100 if invested else 0
                decision, reason = portfolio_decision(h, result)
                technical_stop = result["risk"]["stop_loss"]
                original_stop = h.get("original_stop_loss")
                if original_stop is not None and technical_stop is not None:
                    active_stop = max(original_stop, technical_stop)
                elif original_stop is not None: active_stop = original_stop
                elif technical_stop is not None: active_stop = technical_stop
                else: active_stop = None
                rows.append({"Ticker": result["ticker_display"], "Buy": h["buy_price"],
                    "Shares": h["shares"], "Invested": round(invested, 2),
                    "Current": round(cur_price, 2), "Value": round(cur_value, 2),
                    "P/L": round(pnl, 2), "P/L %": round(pnl_pct, 2),
                    "Trend": result["trend"], "Signal": result["signal"]["signal"],
                    "Original Stop": round(original_stop, 2) if original_stop else "N/A",
                    "Technical Stop": round(technical_stop, 2) if technical_stop else "N/A",
                    "Active Stop": round(active_stop, 2) if active_stop else "N/A",
                    "Target 1": round(result["risk"]["target1"], 2) if result["risk"]["target1"] else "N/A",
                    "Decision": decision, "Reason": reason})
            else:
                rows.append({"Ticker": h["ticker"], "Buy": h["buy_price"],
                    "Shares": h["shares"], "Invested": round(invested, 2),
                    "Current": "N/A", "Value": "N/A", "P/L": "N/A", "P/L %": "N/A",
                    "Trend": None, "Signal": "ERROR",
                    "Original Stop": h.get("original_stop_loss", "N/A"),
                    "Technical Stop": None, "Active Stop": None, "Target 1": None,
                    "Decision": "WATCH", "Reason": error})
        # Valuation summary with PARTIAL detection
        successful_holdings = sum(1 for r in rows if r["Current"] != "N/A")
        total_holdings = len(rows)
        if total_holdings == 0: pass
        elif successful_holdings < total_holdings:
            st.warning(f"⚠️ Portfolio valuation incomplete — "
                f"{total_holdings - successful_holdings} of {total_holdings} prices unavailable.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Invested", f"PKR {round(total_invested, 2)}")
            c2.metric("Current", "PARTIAL / N/A")
            c3.metric("P/L", "N/A"); c4.metric("P/L %", "N/A")
        else:
            total_pnl = total_current - total_invested
            total_pnl_pct = total_pnl / total_invested * 100 if total_invested else 0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Invested", f"PKR {round(total_invested, 2)}")
            c2.metric("Current", f"PKR {round(total_current, 2)}")
            c3.metric("P/L", f"PKR {round(total_pnl, 2)}")
            c4.metric("P/L %", f"{round(total_pnl_pct, 2)}%")
        port_df = pd.DataFrame(rows)
        st.dataframe(port_df, use_container_width=True, hide_index=True)
        remove_idx = st.selectbox("Remove holding",
            options=["-"] + [h["ticker"] for h in st.session_state.portfolio])
        if remove_idx != "-" and st.button("Remove Selected"):
            st.session_state.portfolio = [h for h in st.session_state.portfolio
                                          if h["ticker"] != remove_idx]
            st.rerun()
    else: st.info("Koi holding nahi. Upar add karo (max 5).")

# ============================================================
# MARKET TAB (proxy button-only)
# ============================================================

with tab_market:
    st.subheader("📈 Market Overview")
    market = market_snapshot()
    if market["regime"] == "UNAVAILABLE":
        st.warning("🔴 KSE-100: DATA UNAVAILABLE")
        st.caption(f"Source: {market['source']}")
        st.caption("Koi provider se KSE-100 data nahi mila.")
    else:
        fresh, _, fresh_warn = get_freshness_status(market["last_date"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Regime", market["regime"])
        c2.metric("Trend", market["trend"])
        c3.metric("KSE-100 Level",
            round(market["last_close"], 2) if market["last_close"] else "N/A")
        st.caption(f"Source: {market['source']} | Last: "
            f"{market['last_date'].date() if market['last_date'] else 'N/A'} ({fresh})")
        if fresh == "STALE": st.warning(f"🔴 {fresh_warn}")
        st.info(f"**Reasoning:** {market['reasoning']}")
        idx_df, _ = fetch_market_index()
        if idx_df is not None and len(idx_df) > 30:
            idx_df["SMA20"] = sma(idx_df["Close"], 20)
            idx_df["SMA50"] = sma(idx_df["Close"], 50)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=idx_df.index, y=idx_df["Close"],
                line=dict(color="#3B82F6", width=2), name="KSE-100"))
            fig.add_trace(go.Scatter(x=idx_df.index, y=idx_df["SMA20"],
                line=dict(color="#F59E0B", width=1.5, dash="dot"), name="SMA20"))
            fig.add_trace(go.Scatter(x=idx_df.index, y=idx_df["SMA50"],
                line=dict(color="#8B5CF6", width=1.5, dash="dot"), name="SMA50"))
            fig.update_layout(template="plotly_white", height=400,
                margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader("📊 PSX Market Proxy")
    st.caption("⚠️ PROXY — NOT official KSE-100")
    if st.button("🔄 Refresh Market Proxy", use_container_width=False):
        with st.spinner("Scanning proxy basket..."):
            st.session_state["proxy_result"] = liquid_basket_trend()
    if "proxy_result" not in st.session_state:
        st.info("Click **Refresh Market Proxy** to scan the liquid basket.")
    else:
        proxy = st.session_state["proxy_result"]
        if proxy["change_pct"] is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Proxy Trend", proxy["trend"])
            c2.metric("Avg Change (5d)", f"{proxy['change_pct']}%")
            c3.metric("Stocks Contributing", proxy["stocks_contributing"])
            st.caption(f"Note: {proxy['note']}")
        else: st.info("Proxy unavailable")
    with st.expander("🔍 Provider Diagnostics"):
        diag = pd.DataFrame([{
            "Provider": k, "Available": "✅" if v["available"] else "❌",
            "Coverage": v["coverage"], "KSE-100": "✅" if v["kse100"] else "❌",
            "Last Success": v["last_success"].strftime("%d-%b %H:%M") if v["last_success"] else "Never",
            "Last Attempt": v["last_fetch_attempt"].strftime("%d-%b %H:%M") if v["last_fetch_attempt"] else "Never",
            "Error": (v["error"][:80] if v["error"] else "-")}
            for k, v in PROVIDER_STATUS.items()])
        st.dataframe(diag, use_container_width=True, hide_index=True)
        st.caption("⚠️ Diagnostics = actual network fetches only. Cache purani info de sakti hai.")

# ============================================================
# TOP PICKS TAB (button-only, 3 sections)
# ============================================================

with tab_top:
    st.subheader("🎯 Top Picks — Day / Swing / Hold")
    st.caption("⚠️ Sab estimates technical hain. Guaranteed returns nahi.")
    colA, colB = st.columns([2, 3])
    with colA:
        top_universe_choice = st.selectbox("Universe for Top Picks",
            ["Liquid PSX (~34)", "Small Cap (~25)", "Both (Liquid + Small)"],
            index=0, key="top_universe")
    with colB:
        st.caption("Button dabane pe scan hoga — auto nahi.")
    if st.button("🎯 Generate Top Picks", use_container_width=True):
        with st.spinner("Scanning for Top Picks..."):
            if top_universe_choice == "Liquid PSX (~34)": uni = PSX_LIQUID_UNIVERSE
            elif top_universe_choice == "Small Cap (~25)": uni = PSX_SMALL_CAP_UNIVERSE
            else: uni = list(dict.fromkeys(PSX_LIQUID_UNIVERSE + PSX_SMALL_CAP_UNIVERSE))
            market = market_snapshot()
            top_picks_df, _ = run_screener(uni, period="1y",
                penny_threshold=penny_threshold, rvol_threshold=rvol_threshold, market=market)
            st.session_state["top_picks_df"] = top_picks_df
    if "top_picks_df" not in st.session_state:
        st.info("Click **Generate Top Picks** to scan.")
    else:
        df_s = st.session_state["top_picks_df"]
        picks = categorize_top_picks(df_s, penny_threshold=penny_threshold,
            rvol_threshold=rvol_threshold)
        st.markdown("### ⚡ Day Trade Candidates")
        st.caption("High momentum + breakout structure — 1-2 sessions")
        if picks["day"].empty: st.info("Koi day-trade candidate nahi mila.")
        else:
            day_cols = ["Ticker", "Price", "Change %", "Score", "Signal",
                        "Status", "RR", "Hold Estimate", "Basis"]
            day_cols = [c for c in day_cols if c in picks["day"].columns]
            st.dataframe(picks["day"][day_cols], use_container_width=True, hide_index=True)
        st.divider()
        st.markdown("### 🌊 Swing Trade Candidates")
        st.caption("Clear trend + room to run + R:R ≥ 1.5 — 5-15 sessions")
        if picks["swing"].empty: st.info("Koi swing-trade candidate nahi mila.")
        else:
            swing_cols = ["Ticker", "Price", "Trend", "Score", "Signal",
                          "Status", "RR", "Hold Estimate", "Basis"]
            swing_cols = [c for c in swing_cols if c in picks["swing"].columns]
            st.dataframe(picks["swing"][swing_cols], use_container_width=True, hide_index=True)
        st.divider()
        st.markdown("### 💼 Hold Candidates")
        st.caption("Established strong bullish trend — positional horizon")
        if picks["hold"].empty: st.info("Koi hold candidate nahi mila.")
        else:
            hold_cols = ["Ticker", "Price", "Trend", "Score", "Signal",
                         "Status", "RR", "Hold Estimate", "Basis"]
            hold_cols = [c for c in hold_cols if c in picks["hold"].columns]
            st.dataframe(picks["hold"][hold_cols], use_container_width=True, hide_index=True)
        st.caption("**Hold Estimate** = ATR-based pace calculation. Technical estimate, guaranteed nahi.")

# ============================================================
# FOOTER
# ============================================================

st.sidebar.caption("---")
st.sidebar.caption("⚠️ **Disclaimer:** Signals analytical outputs hain historical "
    "price/volume data pe based. Guaranteed investment advice NAHI. Apna research zaroor karo.")
st.sidebar.caption("Data sourced from PSX Data Portal (dps.psx.com.pk) for personal, "
    "non-commercial use. Commercial redistribution requires PSX license.")

# ============================================================
# END
# ============================================================
