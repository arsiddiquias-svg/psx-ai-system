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
