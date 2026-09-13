"""
PSX QUANT ENGINE - v8.0 (Final)
================================
PSX-focused quantitative decision-support terminal.

FEATURES:
- PSX official data portal integration (dps.psx.com.pk)
- Price reconciliation between PSX and yfinance
- Technical indicators: RSI, SMA/EMA, MACD, ATR, ADX, Bollinger
- Trend, breakout, pullback, momentum, projection engines
- Screener with simplified filters + penny toggle
- Watchlist (session-state based, inside tab)
- Portfolio tracker (max 5 holdings)
- Top Picks (Day Trade + Swing Trade with hold estimates)
- Light theme
- Chart projection arrows

DATA:
- Primary: PSX Data Portal (dps.psx.com.pk)
- Fallback: yfinance
- For personal, non-commercial use only

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
/* ============ LIGHT THEME - v8.0 ============ */
.stApp {
    background: linear-gradient(180deg, #FAFBFC 0%, #F4F6F8 100%);
    color: #1A1A2E;
}

/* Headings */
h1, h2, h3, h4 {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    font-weight: 600 !important;
    color: #16213E !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #64748B !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #0F172A !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}
section[data-testid="stSidebar"] * {
    color: #1E293B !important;
}

/* Captions */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] *,
.stCaption, .stCaption * {
    color: #64748B !important;
    font-size: 0.8rem !important;
}

/* Current price widget */
.current-price {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.6px;
}

/* Change colors */
.change-positive { color: #059669; font-weight: 700; }
.change-negative { color: #DC2626; font-weight: 700; }

/* Signal labels */
.signal-buy   { color: #059669; font-weight: 800; font-size: 1.2rem; }
.signal-wait  { color: #D97706; font-weight: 800; font-size: 1.2rem; }
.signal-avoid { color: #DC2626; font-weight: 800; font-size: 1.2rem; }

/* Expanders */
div[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
}
div[data-testid="stExpander"] summary {
    color: #1E293B !important;
    font-weight: 600 !important;
}

/* Buttons */
.stButton > button {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background-color: #F1F5F9 !important;
    border-color: #94A3B8 !important;
    color: #0F172A !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    border-bottom: 2px solid #E2E8F0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748B !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    border-radius: 8px 8px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #2563EB !important;
    background: #EFF6FF !important;
    border-bottom: 2px solid #2563EB !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    overflow: hidden;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stSelectbox div, .stMultiSelect div {
    background-color: #FFFFFF !important;
    border-color: #CBD5E1 !important;
    color: #1E293B !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
    border-left-width: 4px;
}

/* Divider */
hr {
    border-color: #E2E8F0 !important;
    opacity: 1 !important;
}

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
    if not t.endswith(".KA"):
        t = t + ".KA"
    return t


def normalize_ticker_display(raw: str) -> str:
    return raw.strip().upper().replace(".KA", "")


def trading_days_between(date1, date2):
    try:
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
# PSX OFFICIAL DATA PORTAL INTEGRATION
# ============================================================

PSX_BASE = "https://dps.psx.com.pk"
PSX_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36 PSXQuantEngine/8.0 (personal research)"
)

_LAST_PSX_REQUEST_TS = 0.0
_PSX_MIN_INTERVAL = 1.0


def _psx_get(url: str, timeout: int = 15) -> Optional[requests.Response]:
    """PSX pe rate-limited GET. 1 req/sec max."""
    global _LAST_PSX_REQUEST_TS
    elapsed = time.time() - _LAST_PSX_REQUEST_TS
    if elapsed < _PSX_MIN_INTERVAL:
        time.sleep(_PSX_MIN_INTERVAL - elapsed)
    try:
        resp = requests.get(
            url,
            headers={
                "User-Agent": PSX_USER_AGENT,
                "Accept": "text/html,application/json,*/*",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=timeout,
        )
        _LAST_PSX_REQUEST_TS = time.time()
        return resp
    except Exception:
        _LAST_PSX_REQUEST_TS = time.time()
        return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_psx_official_index_json(index_symbol: str = "KSE100") -> Tuple[Optional[pd.DataFrame], str, str]:
    """
    PSX EOD index fetch. Verified format: {"data": [[ts, close, vol, open], ...]}
    Daily data — 86400s gaps.
    """
    url = f"{PSX_BASE}/timeseries/eod/{index_symbol}"
    try:
        resp = _psx_get(url)
        if resp is None or resp.status_code != 200:
            return None, "EMPTY", f"HTTP {resp.status_code if resp else 'no-response'}"

        data = resp.json()
        if isinstance(data, dict):
            data = data.get("data", [])

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

        update_provider_status("psx_official", available=True, coverage=len(df),
                               kse100=(index_symbol == "KSE100"))
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
        if not tables:
            return None, "EMPTY", "HTML mein koi table nahi"

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
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", "").str.replace("%", "").str.strip(),
                    errors="coerce"
                )

        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        df = df[df["Symbol"].str.len() > 0]
        df = df.dropna(subset=["Current"])

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

        if df is None or df.empty:
            return None, "EMPTY", "No data from psxdata"

        col_map = {}
        for c in df.columns:
            cl = str(c).lower()
            if cl in ["open", "o"]: col_map[c] = "Open"
            elif cl in ["high", "h"]: col_map[c] = "High"
            elif cl in ["low", "l"]: col_map[c] = "Low"
            elif cl in ["close", "c", "price", "adj close"]: col_map[c] = "Close"
            elif cl in ["volume", "vol", "v"]: col_map[c] = "Volume"
            elif "date" in cl or "time" in cl: col_map[c] = "Date"
        if col_map:
            df = df.rename(columns=col_map)

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, "EXCEPTION", f"Missing columns: {missing}"

        df[required] = df[required].apply(pd.to_numeric, errors="coerce")
        df = df.dropna()

        if df.empty:
            return None, "EMPTY", "Empty after cleaning"

        if "Date" in df.columns:
            df.index = pd.to_datetime(df["Date"])
            df = df.drop(columns=["Date"])

        valid, msg = _validate_ohlcv(df)
        if not valid:
            return None, "EXCEPTION", msg

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
        if raw is None or raw.empty:
            return None, "EMPTY", "No data from yfinance"

        df = _flatten_columns(raw)
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, "EXCEPTION", f"Missing columns: {missing}"

        df = df[required].apply(pd.to_numeric, errors="coerce").dropna()
        if df.empty:
            return None, "EMPTY", "Empty after cleaning"

        valid, msg = _validate_ohlcv(df)
        if not valid:
            return None, "EXCEPTION", msg

        update_provider_status("yfinance", available=True, coverage=len(df))
        return df, "SUCCESS", None

    except Exception as e:
        update_provider_status("yfinance", available=True, error=str(e))
        return None, "EXCEPTION", f"yfinance error: {str(e)}"


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
        return False, "Contains NaN"
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
        return False, f"Insufficient history: {len(df)} < {MIN_HISTORY_DAYS}"
    return True, "Valid"


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], str, str, str]:
    """psxdata → yfinance → UNAVAILABLE."""
    provider_attempts = []

    df, status, error = fetch_psxdata_ohlcv(ticker, period)
    provider_attempts.append(f"psxdata: {status}")
    if status == "SUCCESS":
        return df, status, error, "psxdata"

    df, status, error = fetch_yfinance_ohlcv(ticker, period)
    provider_attempts.append(f"yfinance: {status}")
    if status == "SUCCESS":
        return df, status, error, "yfinance"

    return None, "UNAVAILABLE", " | ".join(provider_attempts), "UNAVAILABLE"


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_market_index():
    """KSE-100 master fetch — PSX official first, yfinance fallback."""
    df, status, err = fetch_psx_official_index_json("KSE100")
    if status == "SUCCESS" and df is not None and len(df) >= 40:
        last_close = float(df["Close"].iloc[-1])
        if KSE100_PLAUSIBLE_MIN <= last_close <= KSE100_PLAUSIBLE_MAX:
            return df, "psx_official (KSE100 EOD)"

    for cand in KSE100_CANDIDATES:
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
# RECONCILIATION
# ============================================================

def reconcile_psx_vs_yf(ticker: str, yf_df: pd.DataFrame) -> Dict[str, Any]:
    """PSX vs yfinance price comparison. 1% threshold."""
    out = {"mismatch": False, "diff_pct": None, "psx_price": None,
           "yf_price": None, "source_used": "yfinance"}

    yf_last = float(yf_df["Close"].iloc[-1]) if yf_df is not None and len(yf_df) else None
    out["yf_price"] = yf_last

    mw, status, _ = fetch_psx_official_market_watch()
    if status != "SUCCESS" or mw is None:
        return out

    sym = normalize_ticker_display(ticker)
    row = mw[mw["Symbol"] == sym]
    if row.empty or pd.isna(row.iloc[0].get("Current")):
        return out

    psx_last = float(row.iloc[0]["Current"])
    out["psx_price"] = psx_last

    if yf_last and psx_last and yf_last > 0:
        diff = abs(psx_last - yf_last) / yf_last
        out["diff_pct"] = round(diff * 100, 2)
        if diff > 0.01:
            out["mismatch"] = True
            out["source_used"] = "psx_official"
    return out

# ============================================================
# INDICATORS
# ============================================================

def sma(series, period):
    return series.rolling(period).mean()


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    """Wilder RSI - corrected. All edge cases handled."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    avg_gain = avg_gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = avg_loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.mask((avg_gain == 0) & (avg_loss == 0), 50)
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0)
    return result


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
    """Wilder ADX."""
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

    d["VOL_SMA20"] = d["Volume"].rolling(20).mean().shift(1)
    d["VOL_RATIO"] = d["Volume"] / d["VOL_SMA20"].replace(0, np.nan)

    d["RETURN_1D"] = d["Close"].pct_change()
    d["ROC_10"] = d["Close"].pct_change(10) * 100
    d["VOLATILITY_20"] = d["RETURN_1D"].rolling(20).std() * np.sqrt(252)

    d["52W_HIGH"] = d["High"].shift(1).rolling(252, min_periods=200).max()
    d["52W_LOW"] = d["Low"].shift(1).rolling(252, min_periods=200).min()

    return d


# ============================================================
# TREND / S&R / BREAKOUT / PULLBACK / MOMENTUM / PROJECTION
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
        hh = recent["High"].iloc[-1] > recent["High"].iloc[0]
        hl = recent["Low"].iloc[-1] > recent["Low"].iloc[0]
        if hh and hl:
            bullish += 1
            reasons.append("Higher highs and higher lows")
        elif (not hh) and (not hl):
            bearish += 1
            reasons.append("Lower highs and lower lows")

    if last["ADX14"] >= 25:
        reasons.append(f"ADX {round(last['ADX14'],1)} - trending")

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


def support_resistance(d):
    """Prior 20/60/120 session highs/lows. Current candle excluded."""
    prior = d.iloc[:-1] if len(d) > 1 else d
    r20 = prior.tail(20) if len(prior) >= 20 else prior
    r60 = prior.tail(60) if len(prior) >= 60 else prior
    r120 = prior.tail(120) if len(prior) >= 120 else prior

    primary_resistance = r20["High"].max()
    primary_support = r20["Low"].min()
    secondary_resistance = r60["High"].max()
    secondary_support = r60["Low"].min()

    if secondary_support == primary_support:
        secondary_support = r120["Low"].min()
    if secondary_resistance == primary_resistance:
        secondary_resistance = r120["High"].max()

    last = d.iloc[-1]
    return {
        "primary_support": primary_support,
        "primary_resistance": primary_resistance,
        "secondary_support": secondary_support,
        "secondary_resistance": secondary_resistance,
        "high_52w": last.get("52W_HIGH", np.nan),
        "low_52w": last.get("52W_LOW", np.nan),
        "secondary_support_is_distinct": secondary_support != primary_support,
        "secondary_resistance_is_distinct": secondary_resistance != primary_resistance,
    }


def breakout_engine(d, sr, vol_ratio_threshold=1.5):
    """No look-ahead. 52W breakouts separated into FRESH/CONTINUATION/NEAR."""
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
        status = "CONFIRMED BREAKOUT"
        note = "Closed above resistance with volume + momentum"
    elif now_above and volume_confirmed and momentum_positive and not fresh_cross:
        status = "EXTENDED BREAKOUT"
        note = "Above resistance - continuation"
    elif now_above and (not volume_confirmed or not momentum_positive):
        status = "BREAKOUT ATTEMPT"
        note = "Above resistance but weak confirmation"
    elif (not now_above) and distance_to_resistance is not None and 0 <= distance_to_resistance <= 3:
        status = "BREAKOUT READY"
        note = "Within 3% of resistance"
    elif prev["Close"] > baseline_resistance and price < baseline_resistance:
        status = "FAILED BREAKOUT"
        note = "Broke above but closed back below"
    else:
        status = "NO BREAKOUT"
        note = "Not near breakout level"

    is_near_52w_high = False
    is_52w_high_breakout = False
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

    return {
        "status": status,
        "note": note,
        "fresh_cross": fresh_cross,
        "resistance": resistance,
        "price": price,
        "volume_ratio": vol_ratio,
        "distance_to_resistance": distance_to_resistance,
        "is_near_52w_high": is_near_52w_high,
        "is_52w_high_breakout": is_52w_high_breakout,
    }


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
        return {"status": "BROKEN SUPPORT", "note": "Closed below primary support"}
    if (near_support or near_ema20) and cooling_rsi and bullish_candle:
        return {"status": "HEALTHY PULLBACK", "note": "At support/EMA20 with confirmation"}
    if near_support or near_ema20:
        return {"status": "PULLBACK WATCH", "note": "Approaching support/EMA20"}
    return {"status": "NO PULLBACK", "note": "Not near pullback zone"}


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

    if len(d) >= 2 and last["MACD_HIST"] > d["MACD_HIST"].iloc[-2]:
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
            signals.append("Bearish divergence")
            score -= 2
        price_low = window["Close"].min()
        rsi_low = window["RSI14"].min()
        if window["Close"].iloc[-1] <= price_low * 1.001 and window["RSI14"].iloc[-1] > rsi_low + 5:
            signals.append("Bullish divergence")
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
            upside_low = price + range_size * 0.5
            upside_high = price + range_size * 1.0
            next_res = sr["secondary_resistance"]
        else:
            upside_low = resistance
            upside_high = resistance + atr_val * 1.5
            next_res = sr["secondary_resistance"]
        return {
            "direction": "UP",
            "zone_low": upside_low,
            "zone_high": upside_high,
            "next_resistance": next_res,
            "label": f"Upside: {round(upside_low,2)} - {round(upside_high,2)}",
            "note": "Technical projection if uptrend continues"
        }
    elif is_bearish:
        if price <= support:
            range_size = resistance - support
            downside_low = price - range_size * 1.0
            downside_high = price - range_size * 0.5
            next_sup = sr["secondary_support"]
        else:
            downside_low = support - atr_val * 1.5
            downside_high = support
            next_sup = sr["secondary_support"]
        return {
            "direction": "DOWN",
            "zone_low": downside_low,
            "zone_high": downside_high,
            "next_support": next_sup,
            "label": f"Downside: {round(downside_low,2)} - {round(downside_high,2)}",
            "note": "Technical projection if downtrend continues"
        }
    else:
        return {
            "direction": "NEUTRAL",
            "zone_low": None,
            "zone_high": None,
            "label": "No clear direction",
            "note": "Neutral/range-bound structure"
        }


def detect_penny_setup(d, sr, threshold=PENNY_STOCK_THRESHOLD, rvol_threshold=2.0):
    last = d.iloc[-1]
    price = last["Close"]

    if price > threshold:
        return {"is_penny": False, "status": "NORMAL PRICE STOCK",
                "note": f"Price {price} > {threshold}"}

    vol_ratio = last["VOL_RATIO"] if not pd.isna(last["VOL_RATIO"]) else 0
    near_resistance = False
    if price > 0 and sr["primary_resistance"]:
        near_resistance = abs(price - sr["primary_resistance"]) / price < 0.05
    broke_resistance = price > sr["primary_resistance"]
    rvol_expansion = vol_ratio >= rvol_threshold
    momentum_positive = last["MACD_HIST"] > 0

    if broke_resistance and rvol_expansion and momentum_positive:
        status = "PENNY BREAKOUT"
        note = f"Low-priced breaking resistance with {round(vol_ratio,1)}x volume"
    elif near_resistance and rvol_expansion:
        status = "PENNY BREAKOUT READY"
        note = f"Near resistance with {round(vol_ratio,1)}x volume"
    elif rvol_expansion:
        status = "PENNY VOLUME SPIKE"
        note = f"Unusual volume ({round(vol_ratio,1)}x)"
    elif momentum_positive and near_resistance:
        status = "PENNY WATCH"
        note = "Momentum near resistance"
    else:
        status = "PENNY (NO SETUP)"
        note = "Low-priced but no unusual activity"

    return {"is_penny": True, "status": status, "note": note, "price": price,
            "vol_ratio": vol_ratio, "near_resistance": near_resistance,
            "broke_resistance": broke_resistance, "rvol_expansion": rvol_expansion}


def risk_engine(d, sr, breakout_status=""):
    """Risk calc with validation."""
    last = d.iloc[-1]
    price = last["Close"]
    atr_val = last["ATR14"] if not pd.isna(last["ATR14"]) else 0

    if price is None or price <= 0:
        return {"entry": None, "stop_loss": None, "risk_per_share": None,
                "target1": None, "target2": None, "rr1": None, "rr2": None,
                "conditional_entry": None, "conditional_entry_note": "Invalid price",
                "error": "Invalid entry price"}

    if "EXTENDED BREAKOUT" in breakout_status:
        tighter_stop = price - (2.5 * atr_val)
        stop_loss = max(tighter_stop, sr["primary_support"])
    else:
        swing_low = d.tail(10)["Low"].min()
        stop_loss = min(swing_low, sr["primary_support"]) - 0.3 * atr_val

    if stop_loss is None or stop_loss >= price:
        return {"entry": price, "stop_loss": None, "risk_per_share": None,
                "target1": None, "target2": None, "rr1": None, "rr2": None,
                "conditional_entry": None, "conditional_entry_note": "Invalid stop",
                "error": "Stop-loss must be below entry"}

    risk_per_share = price - stop_loss
    if risk_per_share <= 0:
        return {"entry": price, "stop_loss": stop_loss, "risk_per_share": None,
                "target1": None, "target2": None, "rr1": None, "rr2": None,
                "conditional_entry": None, "conditional_entry_note": "Non-positive risk",
                "error": "Risk per share must be positive"}

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
    resistance = sr["primary_resistance"]
    support = sr["primary_support"]
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
        signal = "WAIT"
        setup_quality = "R:R BELOW MINIMUM"
    elif adjusted_score >= 80:
        signal = "STRONG BUY"; setup_quality = "A+ SETUP"
    elif adjusted_score >= 65:
        signal = "BUY"; setup_quality = "A SETUP"
    elif adjusted_score >= 45:
        signal = "WAIT"; setup_quality = "B SETUP / WATCH"
    elif adjusted_score >= 25:
        signal = "REDUCE"; setup_quality = "WEAK"
    else:
        signal = "AVOID"; setup_quality = "POOR"

    if trend_ok:
        if components["trend"] >= 70:
            reasons.append(f"✅ Trend strong hai ({trend}) — upar ja raha hai")
        elif components["trend"] <= 35:
            reasons.append(f"❌ Trend kamzor hai ({trend}) — neeche ja raha hai")

        if components["momentum"] >= 70:
            reasons.append("✅ Momentum acha hai — buyers active")
        elif components["momentum"] <= 35:
            reasons.append("❌ Momentum kamzor — sellers havi hain")

        if "CONFIRMED" in breakout["status"]:
            reasons.append("✅ Resistance toot gaya — volume ke saath upar close")
        elif "BREAKOUT READY" in breakout["status"]:
            reasons.append("📌 Resistance ke qareeb hai — watch karo")

        if pullback["status"] == "HEALTHY PULLBACK":
            reasons.append("✅ Pullback hua hai support pe — achha entry point")
        elif pullback["status"] == "BROKEN SUPPORT":
            reasons.append("❌ Support toot gaya — structure kharab")

        if momentum["overbought"]:
            reasons.append("⚠️ RSI 70 se upar — overbought, correction aa sakta hai")
        if momentum["oversold"]:
            reasons.append("📌 RSI 30 se neeche — oversold, bounce aa sakta hai")

        if not rr_ok:
            rr_val = round(risk_data['rr1'], 2) if risk_data['rr1'] else 'N/A'
            reasons.append(f"❌ Risk:Reward kam hai ({rr_val}) — nuksan zyada mumkin")

        if market["regime"] == "BULLISH":
            reasons.append("✅ Overall market bullish hai — saath chal raha hai")
        elif market["regime"] == "BEARISH":
            reasons.append("❌ Overall market bearish hai — akele stock upar nahi rukega")

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
        regime = "UNAVAILABLE"; reasoning = "Insufficient SMA data"
    elif vol20 is not None and vol20 > 0.35:
        regime = "HIGH VOLATILITY"; reasoning = f"KSE-100 volatility {round(vol20*100,1)}%"
    elif last_close > sma20 > sma50:
        regime = "BULLISH"; reasoning = "KSE-100 > SMA20 > SMA50"
    elif last_close < sma20 < sma50:
        regime = "BEARISH"; reasoning = "KSE-100 < SMA20 < SMA50"
    else:
        regime = "NEUTRAL"; reasoning = "Mixed SMA alignment"

    return {"regime": regime, "trend": regime if regime in ("BULLISH","BEARISH") else "NEUTRAL",
            "source": source, "reasoning": reasoning, "last_date": last_date,
            "last_close": last_close, "sma20": sma20, "sma50": sma50, "sma200": sma200}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def liquid_basket_trend():
    changes = []
    successful = 0
    for ticker in PSX_LIQUID_UNIVERSE:
        try:
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
        return {"trend": "UNAVAILABLE", "change_pct": None,
                "stocks_contributing": successful,
                "note": "Insufficient data"}

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
# CLASSIFY STOCK
# ============================================================

def classify_stock(price: float, avg_volume: float) -> str:
    """Liquidity proxy — NOT official market cap."""
    if price > 200 and avg_volume > 100000: return "LARGE-LIKE (proxy)"
    elif price > 50 and avg_volume > 20000: return "MID-LIKE (proxy)"
    elif price > 20 and avg_volume > 5000: return "SMALL-LIKE (proxy)"
    elif price < 10 and avg_volume < 2000: return "MICRO (proxy)"
    elif price < 20: return "LOW-PRICE (proxy)"
    else: return "SMALL-LIKE (proxy)"


# ============================================================
# ANALYZE STOCK
# ============================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def analyze_stock(ticker: str, period: str = "1y",
                  penny_threshold: float = PENNY_STOCK_THRESHOLD,
                  rvol_threshold: float = 2.0):
    df, status, error, source = fetch_ohlcv(ticker, period=period)

    if status != "SUCCESS" or df is None or df.empty:
        return None, status, error, source

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


# ============================================================
# SCREENER
# ============================================================

def run_screener(universe: List[str], period: str = "1y",
                 penny_threshold: float = PENNY_STOCK_THRESHOLD,
                 rvol_threshold: float = 2.0):
    rows = []
    coverage = {"total": len(universe), "success": 0, "failed": 0, "analyzed": 0}
    total = len(universe)

    for idx, ticker in enumerate(universe):
        if idx % 10 == 0:
            st.caption(f"📊 Scanning {idx+1}/{total}...")

        result, status, error, source = analyze_stock(
            ticker, period=period,
            penny_threshold=penny_threshold,
            rvol_threshold=rvol_threshold
        )

        if status != "SUCCESS":
            coverage["failed"] += 1
            rows.append({
                "Ticker": ticker, "Price": None, "Change %": None,
                "Trend": None, "Score": None, "Signal": "ERROR",
                "Status": "DATA UNAVAILABLE", "Penny": None,
                "Cap Size": None, "Source": source, "Why": "Data unavailable",
                "Avg Volume": None, "RR": None, "_ticker_raw": ticker,
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
            pos = [r for r in result["signal"]["reasons"] if r.startswith("✅")]
            why_parts.extend(pos[:2])
        if result["risk"]["rr1"] and result["risk"]["rr1"] >= 1.5:
            why_parts.append(f"R:R 1:{round(result['risk']['rr1'],2)}")
        if result["penny"]["is_penny"] and result["penny"]["status"] != "PENNY (NO SETUP)":
            why_parts.insert(0, result["penny"]["note"])

        why_text = " + ".join(why_parts[:4]) if why_parts else "No clear setup"
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
            "Avg Volume": round(avg_volume, 0),
            "_ticker_raw": ticker,
        })

    return pd.DataFrame(rows), coverage


# ============================================================
# ESTIMATE PACE / HOLD PERIOD
# ============================================================

def estimate_pace_to_target(result: Dict) -> Tuple[str, str]:
    """Trade type + pace estimate based on ATR."""
    last = result["last"]
    risk = result["risk"]
    breakout = result["breakout"]
    pullback = result["pullback"]

    if "CONFIRMED BREAKOUT" in breakout["status"] and "EXTENDED" not in breakout["status"]:
        trade_type = "Day/Short-Term"
    elif pullback["status"] == "HEALTHY PULLBACK":
        trade_type = "Swing"
    else:
        trade_type = "Momentum"

    atr_val = last["ATR14"] if not pd.isna(last["ATR14"]) else 0
    entry = risk.get("entry")
    target1 = risk.get("target1")

    if not entry or not target1 or atr_val <= 0:
        return trade_type, "N/A"

    distance = abs(target1 - entry)
    est_sessions = max(1, round(distance / atr_val))
    return trade_type, f"~{est_sessions} sessions (technical estimate)"


def estimate_hold_period(result: Dict, category: str = "swing") -> str:
    """Hold estimate based on ATR pace."""
    if category == "day":
        return "1-2 sessions (intraday/short-term)"

    try:
        last = result["last"]
        risk = result["risk"]
        if risk.get("error") or not risk.get("target1") or not risk.get("entry"):
            return "5-15 sessions (estimate)"

        atr_val = last["ATR14"] if not pd.isna(last["ATR14"]) else 0
        if atr_val <= 0:
            return "N/A"

        distance = abs(risk["target1"] - risk["entry"])
        sessions = max(3, min(20, round(distance / (atr_val * 0.5))))
        return f"~{sessions} sessions (technical estimate)"
    except Exception:
        return "5-15 sessions (estimate)"


# ============================================================
# TOP PICKS - Day Trade vs Swing Trade
# ============================================================

def categorize_top_picks(screener_df: pd.DataFrame,
                        penny_threshold: float = PENNY_STOCK_THRESHOLD,
                        rvol_threshold: float = 2.0) -> Dict[str, pd.DataFrame]:
    """Day trade + Swing trade categorization with hold estimates."""
    if screener_df is None or screener_df.empty:
        return {"day": pd.DataFrame(), "swing": pd.DataFrame()}

    df = screener_df.copy()
    df = df[df["Signal"] != "ERROR"].copy()

    if df.empty:
        return {"day": pd.DataFrame(), "swing": pd.DataFrame()}

    # --- DAY TRADE ---
    day_mask = (
        df["Status"].astype(str).str.contains("BREAKOUT|READY|ATTEMPT", case=False, na=False) &
        (df["Score"].fillna(0) >= 55)
    )
    day_df = df[day_mask].copy()
    if not day_df.empty:
        day_df = day_df.sort_values("Score", ascending=False).head(10)
        day_df["Hold Estimate"] = "1-2 sessions"
        day_df["Basis"] = "Momentum + breakout structure"

    # --- SWING TRADE ---
    swing_mask = (
        df["Trend"].astype(str).str.contains("BULLISH", case=False, na=False) &
        (df["RR"].fillna(0) >= 1.5) &
        (df["Score"].fillna(0) >= 45)
    )
    swing_df = df[swing_mask].copy()

    if not swing_df.empty:
        swing_df = swing_df.sort_values("Score", ascending=False).head(10)
        hold_estimates = []
        for _, row in swing_df.iterrows():
            ticker_raw = row.get("_ticker_raw", row["Ticker"])
            result, status, _, _ = analyze_stock(
                ticker_raw, period="1y",
                penny_threshold=penny_threshold,
                rvol_threshold=rvol_threshold
            )
            if status == "SUCCESS":
                hold_estimates.append(estimate_hold_period(result, category="swing"))
            else:
                hold_estimates.append("5-15 sessions (estimate)")
        swing_df["Hold Estimate"] = hold_estimates
        swing_df["Basis"] = "Trend + R:R ≥ 1.5"

    return {"day": day_df, "swing": swing_df}


# ============================================================
# PORTFOLIO DECISION
# ============================================================

def portfolio_decision(holding: Dict, result: Dict) -> Tuple[str, str]:
    """Original + technical + active stop. None-safe."""
    if result is None:
        return "WATCH", "Data unavailable"

    signal = result["signal"]["signal"]
    trend = result["trend"]
    pullback = result["pullback"]["status"]
    breakout = result["breakout"]["status"]
    momentum = result["momentum"]
    risk = result["risk"]

    technical_stop = risk["stop_loss"]
    original_stop = holding.get("original_stop_loss")

    if original_stop is not None and technical_stop is not None:
        active_stop = max(original_stop, technical_stop)
    elif original_stop is not None:
        active_stop = original_stop
    elif technical_stop is not None:
        active_stop = technical_stop
    else:
        return "WATCH", "⚠️ Stop-loss data unavailable — cannot make exit decision"

    current_price = result["last"]["Close"]
    entry_price = holding["buy_price"]
    pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price else 0

    if current_price < active_stop:
        return "EXIT", f"❌ Price {current_price} < active stop {active_stop}"
    if pullback == "BROKEN SUPPORT":
        return "EXIT", "❌ Support broken"
    if trend in ("BEARISH", "STRONG BEARISH"):
        return "REDUCE", f"📉 Trend {trend}"

    if signal in ("STRONG BUY", "BUY") and trend in ("BULLISH", "STRONG BULLISH"):
        if "CONFIRMED" in breakout and pnl_pct > 5:
            return "TRAIL STOP", f"✅ Profit {pnl_pct:.1f}% with breakout — trail stop"
        elif "CONFIRMED" in breakout:
            return "ADD ON CONFIRMATION", "✅ Breakout — add if risk allows"
        elif pnl_pct > 10:
            return "TRAIL STOP", f"✅ Profit {pnl_pct:.1f}% — trail stop"
        else:
            return "HOLD", "✅ Trend + signal constructive"

    if signal in ("REDUCE", "AVOID"):
        if pnl_pct > 0:
            return "REDUCE", f"⚠️ Signal {signal} while in profit"
        else:
            return "EXIT", f"❌ Signal {signal} and losing"

    if pullback == "HEALTHY PULLBACK" and trend in ("BULLISH", "STRONG BULLISH"):
        if pnl_pct > 0:
            return "ADD ON CONFIRMATION", "✅ Healthy pullback"
        else:
            return "HOLD", "✅ Healthy pullback — maintain"

    if momentum["label"] in ("STRONG NEGATIVE MOMENTUM", "NEGATIVE MOMENTUM") \
            and trend not in ("BULLISH", "STRONG BULLISH"):
        if pnl_pct > 0:
            return "REDUCE", "⚠️ Negative momentum"
        else:
            return "EXIT", "❌ Negative momentum + losing"

    return "HOLD", f"📊 No clear signal — stop at {round(active_stop, 2)}"


# ============================================================
# CHART BUILDER (with Phase 3 projection arrow)
# ============================================================

def build_chart(result, show_bb=False, show_sma200=False,
                show_support_resistance=True, show_rsi=False, show_macd=False):
    d = result["df"].tail(150)
    risk = result["risk"]
    sr = result["sr"]
    trend = result["trend"]

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

    fig = make_subplots(
        rows=num_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=0.04,
        subplot_titles=subplot_titles,
    )

    row = 1
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="Price", increasing_line_color="#10B981", decreasing_line_color="#EF4444"
    ), row=row, col=1)

    trend_color = "#10B981" if trend in ("BULLISH", "STRONG BULLISH") else \
                  "#EF4444" if trend in ("BEARISH", "STRONG BEARISH") else "#F59E0B"
    trend_arrow = "↑" if trend in ("BULLISH", "STRONG BULLISH") else \
                  "↓" if trend in ("BEARISH", "STRONG BEARISH") else "→"
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=f"{trend_arrow} TREND: {trend}", showarrow=False,
        font=dict(color=trend_color, size=14, family="monospace"),
        bgcolor="rgba(255, 255, 255, 0.9)", bordercolor=trend_color,
        borderwidth=1, borderpad=4, opacity=0.95
    )

    fig.add_trace(go.Scatter(x=d.index, y=d["SMA20"],
                             line=dict(color="#3B82F6", width=1.2), name="SMA20"),
                  row=row, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA50"],
                             line=dict(color="#F59E0B", width=1.2), name="SMA50"),
                  row=row, col=1)

    if show_sma200 and result["has_sma200"]:
        fig.add_trace(go.Scatter(x=d.index, y=d["SMA200"],
                                 line=dict(color="#8B5CF6", width=1, dash="dot"),
                                 name="SMA200"), row=row, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_UPPER"],
                                 line=dict(color="#94A3B8", width=0.8, dash="dot"),
                                 name="BB Upper"), row=row, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_LOWER"],
                                 line=dict(color="#94A3B8", width=0.8, dash="dot"),
                                 name="BB Lower"), row=row, col=1)

    if show_support_resistance:
        fig.add_hline(y=sr["primary_resistance"], line_dash="dash",
                      line_color="#EF4444", annotation_text="Resistance",
                      row=row, col=1)
        fig.add_hline(y=sr["primary_support"], line_dash="dash",
                      line_color="#10B981", annotation_text="Support",
                      row=row, col=1)

    if risk.get("stop_loss") and risk["stop_loss"] > 0:
        fig.add_hline(y=risk["stop_loss"], line_dash="dot",
                      line_color="#F59E0B", annotation_text="Stop",
                      row=row, col=1)
    if risk.get("target1") and risk["target1"] > 0:
        fig.add_hline(y=risk["target1"], line_dash="dot",
                      line_color="#3B82F6", annotation_text="T1",
                      row=row, col=1)

    # ============================================================
    # PHASE 3: Future Projection Arrow
    # ============================================================
    try:
        if (risk.get("target1") and risk["target1"] > 0
                and len(d) >= 2 and risk.get("entry")):
            last_date = d.index[-1]
            future_dates = pd.bdate_range(start=last_date, periods=13, freq="B")[1:]

            current_price = float(d["Close"].iloc[-1])
            target_price = float(risk["target1"])

            n = len(future_dates)
            if n > 0:
                future_prices = np.linspace(current_price, target_price, n + 1)[1:]

                breakout_status = result.get("breakout", {}).get("status", "")
                if "CONFIRMED" in breakout_status:
                    basis = "Breakout measured-move"
                elif "EXTENDED" in breakout_status:
                    basis = "Trend continuation"
                else:
                    basis = "R:R Target (T1)"

                arrow_color = "#10B981" if target_price > current_price else "#EF4444"

                proj_x = [last_date] + list(future_dates)
                proj_y = [current_price] + list(future_prices)

                fig.add_trace(
                    go.Scatter(
                        x=proj_x, y=proj_y, mode="lines",
                        line=dict(color=arrow_color, width=2, dash="dash"),
                        name=f"Projection → {round(target_price, 2)}",
                        opacity=0.7,
                        hovertemplate=(
                            "Projection<br>Date: %{x|%d-%b}<br>"
                            "Price: %{y:.2f}<extra></extra>"
                        )
                    ),
                    row=row, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=[future_dates[-1]], y=[future_prices[-1]],
                        mode="markers+text",
                        marker=dict(color=arrow_color, size=10, symbol="triangle-right"),
                        text=[f"→ {round(target_price, 2)} ({basis})"],
                        textposition="middle right",
                        textfont=dict(color=arrow_color, size=11),
                        name="Projected", showlegend=False,
                        hovertemplate=(
                            f"<b>Projected: {round(target_price, 2)}</b><br>"
                            f"Basis: {basis}<br>"
                            f"<i>Technical estimate, not guaranteed</i><extra></extra>"
                        )
                    ),
                    row=row, col=1
                )
    except Exception:
        pass

    if show_rsi:
        row += 1
        fig.add_trace(go.Scatter(x=d.index, y=d["RSI14"],
                                 line=dict(color="#3B82F6", width=1.3), name="RSI"),
                      row=row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#EF4444", row=row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#10B981", row=row, col=1)
        fig.update_yaxes(range=[0, 100], row=row, col=1)

    if show_macd:
        row += 1
        fig.add_trace(go.Scatter(x=d.index, y=d["MACD"],
                                 line=dict(color="#3B82F6", width=1), name="MACD"),
                      row=row, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["MACD_SIGNAL"],
                                 line=dict(color="#F59E0B", width=1), name="Signal"),
                      row=row, col=1)
        hist_colors = np.where(d["MACD_HIST"] >= 0, "#10B981", "#EF4444")
        fig.add_trace(go.Bar(x=d.index, y=d["MACD_HIST"],
                             marker_color=hist_colors, name="Hist"),
                      row=row, col=1)

    fig.update_layout(
        height=700 if not show_rsi and not show_macd else 800,
        showlegend=True, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
    )
    return fig
