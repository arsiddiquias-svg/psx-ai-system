"""
PSX QUANT ENGINE - v9.0.1
==========================
PSX-focused quantitative decision-support terminal.

CHANGELOG v9.0.1 (fixes found from live screenshots of v9.0):
- LIVE PRICE BUG: before the market opens PSX rolls LDCP to yesterday's close but leaves yesterday's
  CHANGE on the board, so LDCP+CHANGE double-counted the last move (PRL showed 83.52 instead of 80.39;
  Screener Price != Entry by exactly the day's change). Live price is now trusted only inside PSX
  session hours AND only when it is consistent with the last close; otherwise last PSX close is shown.
- Header change % now matches the displayed price; Trade Plan notes when live differs from plan base.
- PULLBACK: a pullback now needs price >= 3% off its 10-day high and <= 6% above EMA20 for the
  support case (v9.0's nearer pivot supports made almost every stock a "pullback").
- Chart: Target and Resistance lines that coincide are merged; zones > 30% from price are hidden.

CHANGELOG v9.0 (live-screenshot review, phase 2):
- STOP LOSS: user-controlled "Max Stop Loss %" slider (2-10%, default 5%). Stop = nearest
  real support inside that cap, else 1.5 ATR, else the cap. "Stop ATR" column + "tight stop"
  flag warn when the stop is tighter than one normal daily move (whipsaw risk).
- BUY ZONES: every plan now has "Buy on Dip" (downside zone) and "Buy on Breakout" (upside trigger)
  ranges with R:R computed AT the zone, so "WAIT" explains where entry becomes attractive.
- SUPPORT / RESISTANCE: swing-pivot clustering (touches counted) replaces raw 20-day high/low;
  chart draws shaded zones, fewer lines, weekend/holiday gaps removed, bars selector,
  vibrant candles. Legacy 20d range kept as range_high20 / range_low20.
- WAIT clarity: "WAIT kyun" reason on top of Why-this-signal (R:R gate explained with numbers).
- PULLBACKS: "Pullback Checks" (Trend / Zone / RSI / Candle) + Action column.
- PORTFOLIO: break-even %, technical projection, and a Recovery Planner per losing holding
  (levels to reclaim, trend-repair checklist, volatility-based time scale, rupee outcomes of
  exit/hold/reduce, averaging math, switch math vs. current screener candidates).
- TOP PICKS: a stock appears in one category only (Hold > Swing > Day).
- Dashboard: bearish stocks explain that this tool is long-only (no short setup).

CHANGELOG v8.9.2 (live-data review, all from real screenshots):
- STOP LOSS REDESIGN: stop was min(10d swing low, 20d support) - 0.3 ATR, i.e. the
  FARTHEST level, giving stops 25-36% below entry (PRL 106.95 -> 80.39) and R:R ~0.5
  everywhere, which is why almost every stock hit the R:R-gate. Now: nearest real
  support, kept between 1.0 and 2.5 ATR (1.5 ATR if support is closer than 1 ATR) and never more than 10% below entry.
  "Risk %" (entry->stop) is shown in every table and on the Dashboard.
- FIX: Top Picks "Day Trade" listed stocks with status "NO BREAKOUT" (substring match).
- FIX: Pullbacks tab listed "NO PULLBACK" rows (same substring bug); now only real
  pullbacks, plus an "approaching" list (bullish, <= 8% above EMA20).
- NEW: failed symbols in a scan now show the real reason (expander).
- NEW: Breakouts hides penny stocks by default (checkbox), shows Penny column.
- FIX: metric cards broke words mid-word ("MINIMU M").
- FIX: Portfolio table crashed Streamlit's Arrow converter when a stop column mixed numbers and "N/A".

CHANGELOG v8.9.1 (first live PSX diagnostic: token fix CONFIRMED on deployment):
- FIX: the real /trading-board/REG/main table has NO "CURRENT" column
  (SYMBOL, NAME, BID VOL., BID PRICE, OFFER VOL., OFFER PRICE, LDCP, CHANGE,
  VOLUME). Current is now derived (LDCP + CHANGE, mode auto-detected abs vs %).
  Before this the board was rejected ("Columns missing") -> no universe, no
  live price.
- FIX: live price is only labelled "live" when it really differs from the last
  PSX close; otherwise it is shown as the PSX close (no false freshness).
- FIX: labels said "Last yfinance close" / "using yfinance close" even when the
  data came from PSX official - now source-aware. Confidence no longer
  penalises official PSX data for lacking an intraday price.
- NEW: scan universe sorted by traded volume (most liquid first).
- NEW: KSE-100 chart marks the latest session (date + level).
- NEW: probe prints the raw table header + first row HTML for ground truth.

CHANGELOG v8.9  (root-cause fix for the PSX data outage):
- ROOT CAUSE: since 2026-09-24 PSX rejects data requests with HTTP 403 unless
  they carry an `X-Req-Id` header equal to the token PSX embeds in every HTML
  page (`window.__ps._k`; rotates ~every 5 min). Source: the open-source
  `psxdata` 1.2.0 package (constants.py / scrapers/token.py). The old
  /market-watch page now returns 404.
- NEW: token manager (fetch homepage -> extract token -> send X-Req-Id,
  refresh every 240s, one refresh+retry on 403).
- NEW: market snapshot / scan universe now come from
  /trading-board/REG/main (replaces the dead /market-watch page).
- NEW: per-stock OHLCV from PSX official (POST /historical, one request
  returns the full history) is now the PRIMARY source; yfinance is fallback.
- NEW: KSE-100 chain: /timeseries/eod -> POST /historical(KSE100) -> yfinance.
- FIX: Provider Status is now persisted across reruns (it used to reset when
  cached fetches were skipped, showing "Not available" while PSX was working).
- NEW: Diagnostic Probe tests with-token vs without-token to prove/disprove
  the token cause on the live deployment.
- PRESERVE: everything from v8.8.

DEPLOYMENT:
- Python 3.11 or 3.12 recommended
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import os
import re
import json
import time
import threading
import requests
from io import StringIO
from typing import Optional, Tuple, Dict, Any, List, Union

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PSX Quant Engine v9.0.1",
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
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color: #0F172A !important; font-family: 'Inter', 'Segoe UI', sans-serif !important; font-size: 1.5rem !important; font-weight: 700 !important; white-space: normal !important; overflow-wrap: break-word !important; word-break: normal !important; }
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
STOP_MIN_ATR = 1.0     # stop is never tighter than 1.0 ATR below entry (market noise)
STOP_MAX_ATR = 2.5     # ... and never further than 2.5 ATR
STOP_MAX_PCT = 0.10    # HARD ceiling: never more than 10% below entry (PSX daily limit is ~10%)
DEFAULT_STOP_PCT = 0.05  # default user cap (sidebar slider overrides)
DAY_TRADE_MIN_RR = 1.0
PENNY_STOCK_THRESHOLD = 50
MIN_HISTORY_DAYS = 60
CACHE_TTL = 300
CHART_DISPLAY_BARS = 504

WEIGHTS = {"trend": 0.25, "momentum": 0.20, "volume": 0.15, "setup": 0.20, "rr": 0.10, "sr": 0.10}

KSE100_PLAUSIBLE_MIN = 5000
KSE100_PLAUSIBLE_MAX = 1000000
KSE100_CANDIDATES = ["^KSE100", "KSE100.KA"]

SCREENER_COLUMNS = ["Ticker", "Price", "Price Src", "Change %", "Trend", "Score",
                    "Signal", "Gate", "Confidence", "Status", "Penny", "RR", "RR Flag",
                    "Entry", "Stop Loss", "Risk %", "Stop ATR", "Target 1",
                    "Buy on Dip", "RR @Dip", "Buy on Breakout",
                    "Cap Size", "Source", "Why", "Avg Volume",
                    "RVOL", "Pullback", "Pullback Checks", "RSI", "Momentum", "Resistance", "Dist %", "52W",
                    "EMA20", "Above EMA20 %", "_ticker_raw"]

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
# v8.7: Added "diagnostic" field to capture raw PSX error details.

@st.cache_resource(show_spinner=False)
def _psx_state() -> Dict[str, Any]:
    """Process-wide state that survives Streamlit reruns (session, token, provider status)."""
    return {
        "session": None, "token": None, "token_ts": 0.0, "token_fail_ts": 0.0,
        "token_err": None, "last_req": 0.0, "lock": threading.RLock(),
        "fail_streak": 0, "breaker_until": 0.0,
        "providers": {
            "psx_official": {"available": False, "last_success": None, "error": None,
                             "diagnostic": None, "coverage": 0, "kse100": False,
                             "last_fetch_attempt": None},
            "psxdata":      {"available": False, "last_success": None, "error": None,
                             "diagnostic": None, "coverage": 0, "kse100": False,
                             "last_fetch_attempt": None},
            "yfinance":     {"available": True,  "last_success": None, "error": None,
                             "diagnostic": None, "coverage": 0, "kse100": False,
                             "last_fetch_attempt": None},
        },
    }

# v8.9: shared dict -> status no longer resets when cached fetches are skipped.
PROVIDER_STATUS = _psx_state()["providers"]

def update_provider_status(provider: str, available: bool = None, error: str = None,
                           coverage: int = None, kse100: bool = None,
                           diagnostic: str = None):
    if provider in PROVIDER_STATUS:
        if available is not None: PROVIDER_STATUS[provider]["available"] = available
        if error is not None: PROVIDER_STATUS[provider]["error"] = error
        if coverage is not None: PROVIDER_STATUS[provider]["coverage"] = coverage
        if kse100 is not None: PROVIDER_STATUS[provider]["kse100"] = kse100
        if diagnostic is not None: PROVIDER_STATUS[provider]["diagnostic"] = diagnostic
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
        return "FRESH", trading_gap, f"~{trading_gap} business day(s) old (approx)"
    elif trading_gap <= 3:
        return "DELAYED", trading_gap, f"~{trading_gap} business days old (approx)"
    else:
        return "STALE", trading_gap, f"~{trading_gap} business days old — STALE (approx)"

# ============================================================
# CONFIDENCE CALCULATION
# ============================================================

def calculate_confidence(result: Dict, freshness_status: str) -> Dict[str, Any]:
    score = 0
    reasons = []
    if freshness_status == "FRESH": score += 3
    elif freshness_status == "DELAYED":
        score += 1; reasons.append("Data 2-3 business days old")
    else:
        score -= 3; reasons.append("Data STALE (>3 business days old)")
    if (result.get("display_price_source") == "psx_official_live"
            or str(result.get("data_source", "")).startswith("psx_official")):
        score += 2
    else:
        reasons.append("PSX data unavailable (using yfinance close)")
    trend = result.get("trend", "")
    if trend in ("BULLISH", "STRONG BULLISH"): score += 2
    elif trend in ("BEARISH", "STRONG BEARISH"):
        score -= 3; reasons.append(f"Bearish trend ({trend})")
    rr1 = result.get("risk", {}).get("rr1")
    if rr1 is not None:
        if rr1 >= 2.0: score += 2
        elif rr1 >= MIN_RR: score += 1
        elif rr1 >= 1.0:
            score -= 1; reasons.append(f"R:R {round(rr1,2)} below minimum")
        else:
            score -= 2; reasons.append(f"R:R {round(rr1,2)} poor")
    signal = result.get("signal", {}).get("signal", "")
    if signal in ("STRONG BUY", "BUY"): score += 2
    elif signal == "REDUCE":
        score -= 1; reasons.append("Signal REDUCE")
    elif signal == "AVOID":
        score -= 2; reasons.append("Signal AVOID")
    regime = result.get("market", {}).get("regime", "")
    if regime == "BULLISH": score += 1
    elif regime == "BEARISH":
        score -= 1; reasons.append("Market regime bearish")
    elif regime == "HIGH VOLATILITY":
        score -= 1; reasons.append("Market regime high volatility")
    if score >= 7: level = "HIGH"
    elif score >= 2: level = "MEDIUM"
    else: level = "LOW"
    return {"level": level, "score": score, "reasons": reasons}

# ============================================================
# PSX OFFICIAL DATA PORTAL
# ============================================================
# v8.7: Raw diagnostic capture on failure.
# ============================================================

PSX_BASE = os.environ.get("PSX_BASE_URL", "https://dps.psx.com.pk")  # env override = test hook
PSX_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36")

PSX_TOKEN_HEADER = "X-Req-Id"
PSX_TOKEN_TTL = 240        # seconds; PSX rotates the token roughly every 5 minutes
PSX_TOKEN_BACKOFF = 60     # seconds before retrying after a failed token fetch
_PSX_MIN_INTERVAL = 0.5    # ~2 requests / second
_PS_SCRIPT_RE = re.compile(r"window\.__ps\s*=\s*(\{.*?\})\s*(?:;|</script>)", re.DOTALL)
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,}$")

def extract_psx_token(html: str) -> Optional[str]:
    """Return the `window.__ps._k` token embedded in a PSX HTML page, or None."""
    m = _PS_SCRIPT_RE.search(html or "")
    if m is None:
        return None
    try:
        payload = json.loads(m.group(1))
    except ValueError:
        return None
    tok = payload.get("_k") if isinstance(payload, dict) else None
    if not isinstance(tok, str) or not _TOKEN_RE.fullmatch(tok):
        return None
    return tok

def _get_psx_session() -> requests.Session:
    stt = _psx_state()
    with stt["lock"]:
        if stt["session"] is None:
            s = requests.Session()
            s.headers.update({
                "User-Agent": PSX_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
                          "image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                # no "br": requests cannot decode brotli without an extra package
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            })
            stt["session"] = s
        return stt["session"]

def get_psx_token(force: bool = False) -> Optional[str]:
    """Fetch/cached PSX request token. Fails open (returns None, records the reason)."""
    stt = _psx_state()
    with stt["lock"]:
        now = time.time()
        if not force and stt["token"] and now - stt["token_ts"] < PSX_TOKEN_TTL:
            return stt["token"]
        if not force and stt["token_fail_ts"] and now - stt["token_fail_ts"] < PSX_TOKEN_BACKOFF:
            return None
        err = None
        try:
            r = _get_psx_session().get(PSX_BASE + "/", timeout=15)
            if r.status_code != 200:
                err = f"token page HTTP {r.status_code}"
            else:
                tok = extract_psx_token(r.text)
                if tok:
                    stt.update(token=tok, token_ts=time.time(), token_fail_ts=0.0, token_err=None)
                    return tok
                err = "window.__ps._k not found in homepage HTML"
        except Exception as e:
            err = f"token fetch failed: {type(e).__name__}: {str(e)[:100]}"
        stt.update(token=None, token_fail_ts=time.time(), token_err=err)
        return None

def _psx_request_headers(kind: str) -> Dict[str, Optional[str]]:
    """kind: 'json' | 'html_xhr' | 'page'. None value = drop the session default."""
    if kind == "json":
        return {"Accept": "application/json, text/plain, */*",
                "Referer": PSX_BASE + "/", "X-Requested-With": "XMLHttpRequest",
                "Sec-Fetch-Dest": "empty", "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin", "Sec-Fetch-User": None,
                "Upgrade-Insecure-Requests": None}
    if kind == "html_xhr":
        return {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Referer": PSX_BASE + "/", "X-Requested-With": "XMLHttpRequest",
                "Sec-Fetch-Dest": "empty", "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin", "Sec-Fetch-User": None,
                "Upgrade-Insecure-Requests": None}
    return {"Referer": PSX_BASE + "/", "Sec-Fetch-Site": "same-origin"}

def _psx_request(method: str, url: str, kind: str = "html_xhr", data: Optional[dict] = None,
                 timeout: int = 20, retries: int = 1,
                 use_breaker: bool = True) -> Optional[requests.Response]:
    """Rate-limited PSX request with X-Req-Id token, one token-refresh retry on 403
    (only if a token was actually sent), `retries` retries on network errors / 5xx, and a
    circuit breaker: after 6 consecutive failures PSX data calls are skipped for 120 s so
    scans fall back to yfinance quickly instead of hammering a failing endpoint."""
    stt = _psx_state()
    if use_breaker and kind != "page" and time.time() < stt["breaker_until"]:
        return None
    resp = None
    attempt = 0
    refreshed = False
    while attempt <= retries:
        headers = dict(_psx_request_headers(kind))
        sent_token = False
        if kind != "page":
            tok = get_psx_token()
            if tok:
                headers[PSX_TOKEN_HEADER] = tok
                sent_token = True
        with stt["lock"]:
            wait = stt["last_req"] + _PSX_MIN_INTERVAL - time.time()
            stt["last_req"] = max(time.time(), stt["last_req"] + _PSX_MIN_INTERVAL)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = _get_psx_session().request(method, url, data=data, headers=headers,
                                              timeout=timeout)
        except Exception:
            resp = None
        if resp is not None:
            if resp.status_code == 403 and kind != "page" and sent_token and not refreshed:
                refreshed = True
                get_psx_token(force=True)       # token rejected/expired -> refresh once
                continue                         # same attempt number
            if resp.status_code < 500:
                break
        attempt += 1
        if attempt <= retries:
            time.sleep(1.5)
    if kind != "page":
        failed = resp is None or resp.status_code in (401, 403) or resp.status_code >= 500
        with stt["lock"]:
            if failed:
                stt["fail_streak"] += 1
                if stt["fail_streak"] >= 6:
                    stt["breaker_until"] = time.time() + 120
                    stt["fail_streak"] = 0
            else:
                stt["fail_streak"] = 0
    return resp

def _psx_get(url: str, timeout: int = 15, api: bool = False, retries: int = 1,
             kind: Optional[str] = None, use_breaker: bool = True) -> Optional[requests.Response]:
    return _psx_request("GET", url, kind or ("json" if api else "html_xhr"),
                        timeout=timeout, retries=retries, use_breaker=use_breaker)

def _psx_post(url: str, data: dict, timeout: int = 25, retries: int = 1,
              use_breaker: bool = True) -> Optional[requests.Response]:
    return _psx_request("POST", url, "html_xhr", data=data, timeout=timeout,
                        retries=retries, use_breaker=use_breaker)

def _first_table(html: str) -> Optional[pd.DataFrame]:
    """Largest HTML table in a PSX response (header names kept as-is)."""
    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        return None
    if not tables:
        return None
    return max(tables, key=lambda t: t.shape[0] * t.shape[1])

def diagnostic_psx_probe() -> str:
    """Live probe. Reports token extraction, each endpoint WITH token, and a control
    request WITHOUT token (proves/disproves the X-Req-Id cause)."""
    L: List[str] = []
    L.append(f"**Timestamp:** {pkt_now().strftime('%Y-%m-%d %H:%M:%S PKT')}")
    L.append(f"**Base:** `{PSX_BASE}`")
    L.append("")
    # --- 1. homepage + token
    L.append("### 1) Homepage + request token")
    tok = None
    try:
        r = _get_psx_session().get(PSX_BASE + "/", timeout=15)
        L.append(f"- HTTP `{r.status_code}` · `{len(r.text)}` bytes · server `{r.headers.get('Server', 'n/a')}`")
        has_ps = "window.__ps" in r.text
        L.append(f"- `window.__ps` present in HTML: **{has_ps}**")
        tok = extract_psx_token(r.text)
        if tok:
            L.append(f"- Token extracted: ✅ length {len(tok)}, starts `{tok[:6]}…`")
            stt = _psx_state()
            with stt["lock"]:
                stt.update(token=tok, token_ts=time.time(), token_fail_ts=0.0, token_err=None)
        else:
            L.append("- Token extracted: ❌ NOT FOUND")
            if has_ps:
                i = r.text.index("window.__ps")
                L.append(f"- Context: `{r.text[i:i+160]!r}`")
            else:
                srcs = re.findall(r'<script[^>]+src="([^"]+)"', r.text)[:6]
                L.append(f"- Script tags (first 6): `{srcs}`")
    except Exception as e:
        L.append(f"- ❌ Exception: `{type(e).__name__}: {str(e)[:160]}`")
    L.append("")

    def _describe(resp, show_cols=False):
        if resp is None:
            L.append("- ❌ No response (timeout / DNS / connection refused)")
            return
        L.append(f"- HTTP `{resp.status_code}` · `{len(resp.content)}` bytes · "
                 f"type `{resp.headers.get('Content-Type', 'n/a')}`")
        if resp.status_code == 200:
            t = _first_table(resp.text)
            if t is not None:
                L.append(f"- Table rows: `{len(t)}` · columns: `{[str(c) for c in t.columns][:9]}`")
                if show_cols and len(t):
                    L.append(f"- First row: `{[str(x) for x in t.iloc[0].tolist()][:9]}`")
                    L.append(f"- Last row: `{[str(x) for x in t.iloc[-1].tolist()][:9]}`")
            else:
                L.append(f"- Body preview: `{resp.text[:160]!r}`")
        elif resp.status_code == 403:
            L.append("- 403: request rejected (token missing/expired/rejected)")
        elif resp.status_code == 404:
            L.append("- 404: path does not exist (page removed/renamed)")

    # --- 2. trading board WITH token
    L.append("### 2) Trading board `/trading-board/REG/main` (with token)")
    r_tb = None
    try:
        r_tb = _psx_get(f"{PSX_BASE}/trading-board/REG/main", timeout=20, kind="html_xhr", use_breaker=False)
        _describe(r_tb, show_cols=True)
        if r_tb is not None and r_tb.status_code == 200:
            flat = re.sub(r"\s+", " ", r_tb.text)
            ths = re.findall(r"<th[^>]*>(.*?)</th>", flat, flags=re.I)[:14]
            L.append(f"- Raw header cells: `{[re.sub('<[^>]+>', '', x).strip() for x in ths]}`")
            m1 = re.search(r"<tbody[^>]*>\s*(<tr.*?</tr>)", flat, flags=re.I)
            if m1:
                L.append(f"- Raw first row HTML: `{m1.group(1)[:700]}`")
            bdf, berr = _parse_trading_board(r_tb.text)
            if bdf is None:
                L.append(f"- Parsed for app: ❌ {berr}")
            else:
                L.append(f"- Parsed for app: ✅ {len(bdf)} symbols · Current source: "
                         f"`{bdf['Current_Src'].iloc[0]}` · top by volume: "
                         f"`{bdf['Symbol'].head(5).tolist()}`")
    except Exception as e:
        L.append(f"- ❌ Exception: `{type(e).__name__}: {str(e)[:160]}`")
    L.append("")
    # --- 3. control: same request WITHOUT token
    L.append("### 3) Control: same request WITHOUT token")
    r_ctl = None
    try:
        time.sleep(_PSX_MIN_INTERVAL)
        r_ctl = _get_psx_session().get(f"{PSX_BASE}/trading-board/REG/main", timeout=20,
                                       headers=_psx_request_headers("html_xhr"))
        _describe(r_ctl)
    except Exception as e:
        L.append(f"- ❌ Exception: `{type(e).__name__}: {str(e)[:160]}`")
    L.append("")
    # --- 4. historical
    L.append("### 4) Per-stock history `POST /historical` (symbol=OGDC)")
    try:
        r_h = _psx_post(f"{PSX_BASE}/historical", {"symbol": "OGDC"}, use_breaker=False)
        _describe(r_h, show_cols=True)
    except Exception as e:
        L.append(f"- ❌ Exception: `{type(e).__name__}: {str(e)[:160]}`")
    L.append("")
    # --- 5. legacy endpoints
    L.append("### 5) Legacy endpoints (old app used these)")
    for path, kind in (("/timeseries/eod/KSE100", "json"), ("/market-watch", "page")):
        try:
            r_o = _psx_get(f"{PSX_BASE}{path}", timeout=15, kind=kind, use_breaker=False)
            L.append(f"**`{path}`**")
            _describe(r_o)
        except Exception as e:
            L.append(f"- `{path}` ❌ `{type(e).__name__}: {str(e)[:120]}`")
    L.append("")
    stt_ = _psx_state()
    if r_tb is not None and r_tb.status_code == 200:
        with stt_["lock"]:
            stt_["breaker_until"] = 0.0; stt_["fail_streak"] = 0
    # --- verdict
    L.append("### Verdict")
    ok_with = r_tb is not None and r_tb.status_code == 200
    ok_without = r_ctl is not None and r_ctl.status_code == 200
    if ok_with and r_ctl is not None and r_ctl.status_code == 403:
        L.append("- ✅ **Token fix confirmed**: with token = 200, without token = 403.")
    elif ok_with and ok_without:
        L.append("- ℹ️ Works with and without token (PSX may have relaxed the token rule).")
    elif tok and not ok_with:
        L.append("- ⚠️ Token was extracted but the trading board still failed — "
                 "see status above (path or IP-bound token issue).")
    elif not tok:
        L.append("- ❌ No token could be extracted from the homepage — PSX may have changed "
                 "how the token is embedded. Send me this report.")
    return "\n".join(L)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_psx_official_index_json(index_symbol: str = "KSE100") -> Tuple[Optional[pd.DataFrame], str, str]:
    url = f"{PSX_BASE}/timeseries/eod/{index_symbol}"
    try:
        resp = _psx_get(url, api=True)
        if resp is None:
            msg = "Connection failed entirely (timeout/DNS/network — possibly IP blocked)"
            update_provider_status("psx_official", available=False, error=msg,
                                   diagnostic=f"No response from {url}")
            return None, "EMPTY", msg
        if resp.status_code != 200:
            msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
            update_provider_status("psx_official", available=False, error=msg,
                                   diagnostic=f"Status {resp.status_code} on {url}")
            return None, "EMPTY", msg
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
            msg = f"HTTP 200 but no parseable rows ({len(data) if isinstance(data, list) else 'non-list'} items)"
            update_provider_status("psx_official", available=False, error=msg,
                                   diagnostic=f"Response shape unexpected")
            return None, "EMPTY", msg
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
                               kse100=(index_symbol == "KSE100"), diagnostic=None)
        return df, "SUCCESS", None
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:150]}"
        update_provider_status("psx_official", available=False, error=msg,
                               diagnostic="Exception during parse")
        return None, "EXCEPTION", f"PSX EOD error: {msg}"

def _board_change_mode(ldcp: pd.Series, chg: pd.Series) -> str:
    """Is the board's CHANGE column absolute rupees ('abs') or percent ('pct')?
    PSX caps daily moves at ~7.5%, so an absolute change never exceeds ~8% of LDCP.
    Cheap stocks (LDCP <= 100) separate the two cases: a percent value like 2.3 is far
    larger than 8% of a PKR 10 price. Header says CHANGE (not CHANGE %) so 'abs' is the default."""
    m = (ldcp > 0) & chg.notna() & (chg != 0)
    within = chg.abs() <= 0.08 * ldcp + 0.011
    cheap = m & (ldcp <= 100)
    if int(cheap.sum()) >= 8:
        return "abs" if within[cheap].mean() >= 0.85 else "pct"
    if int(m.sum()) >= 5 and within[m].mean() < 0.9:
        return "pct"
    return "abs"

def _parse_trading_board(html: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    df = _first_table(html)
    if df is None or df.empty:
        return None, "HTTP 200 but no table parsed from trading-board HTML"
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
        elif "volume" in cl and "bid" not in cl and "offer" not in cl: rename_map[c] = "Volume"
    df = df.rename(columns=rename_map)
    if "Symbol" not in df.columns:
        return None, f"No Symbol column: {[str(c) for c in df.columns][:9]}"
    for col in ["LDCP", "Open", "High", "Low", "Current", "Change", "Change_Pct", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "")
                                    .str.replace("%", "").str.strip(), errors="coerce")
    if "Current" in df.columns:
        df["Current_Src"] = "board"
    elif "LDCP" in df.columns:
        # v8.9.1: the real board has LDCP + CHANGE but no CURRENT column
        if "Change" in df.columns:
            mode = _board_change_mode(df["LDCP"], df["Change"])
            if mode == "abs":
                df["Current"] = df["LDCP"] + df["Change"].fillna(0)
                df["Current_Src"] = "LDCP+CHANGE"
            else:
                df["Current"] = df["LDCP"] * (1 + df["Change"].fillna(0) / 100.0)
                df["Current_Src"] = "LDCP*(1+CHANGE%)"
        elif "Change_Pct" in df.columns:
            df["Current"] = df["LDCP"] * (1 + df["Change_Pct"].fillna(0) / 100.0)
            df["Current_Src"] = "LDCP*(1+CHANGE%)"
        else:
            df["Current"] = df["LDCP"]
            df["Current_Src"] = "LDCP"
    else:
        return None, f"Columns missing (no CURRENT/LDCP): {[str(c) for c in df.columns][:9]}"
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df = df[df["Symbol"].str.fullmatch(r"[A-Z0-9&-]{2,15}")]
    df = df[df["Current"].notna() & (df["Current"] > 0)]
    df = df.drop_duplicates(subset="Symbol", keep="first")
    if "Volume" in df.columns:    # most liquid first (stable; keeps order if all equal)
        df = df.sort_values("Volume", ascending=False, kind="mergesort", na_position="last")
    if df.empty:
        return None, "Trading board parsed but no valid symbol rows"
    return df.reset_index(drop=True), None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_psx_official_market_watch() -> Tuple[Optional[pd.DataFrame], str, str]:
    """Live snapshot of all symbols from /trading-board/REG/main
    (the old /market-watch page returns 404 since the PSX site change)."""
    url = f"{PSX_BASE}/trading-board/REG/main"
    try:
        resp = _psx_get(url, timeout=20, kind="html_xhr")
        if resp is None:
            msg = "Connection failed entirely (timeout/DNS/network)"
            update_provider_status("psx_official", available=False, error=msg,
                                   diagnostic=f"No response from {url}")
            return None, "EMPTY", msg
        if resp.status_code != 200:
            hint = " (token rejected?)" if resp.status_code == 403 else ""
            msg = f"HTTP {resp.status_code}{hint} on trading-board: {resp.text[:120]!r}"
            update_provider_status("psx_official", available=False, error=msg,
                                   diagnostic=f"Status {resp.status_code} on {url}")
            return None, "EMPTY", msg
        df, err = _parse_trading_board(resp.text)
        if df is None:
            update_provider_status("psx_official", available=False, error=err)
            return None, "EXCEPTION", err
        update_provider_status("psx_official", available=True, coverage=len(df), diagnostic=None)
        return df, "SUCCESS", None
    except Exception as e:
        msg = f"Trading-board error: {type(e).__name__}: {str(e)[:150]}"
        update_provider_status("psx_official", available=False, error=msg)
        return None, "EXCEPTION", msg

# ============================================================
# LIVE PSX PRICE
# ============================================================

def _price_src_label(result: Dict) -> str:
    if result.get("display_price_source") == "psx_official_live":
        return "📍 PSX live"
    if str(result.get("data_source", "")).startswith("psx_official"):
        return "📅 PSX EOD"
    return "📅 yf"

def _psx_session_open(now=None) -> bool:
    """Weekday 09:15-17:00 PKT (covers the Friday split session). Holidays are not detected."""
    now = now or pkt_now()
    if now.weekday() >= 5:
        return False
    hm = now.hour * 60 + now.minute
    return 9 * 60 + 15 <= hm <= 17 * 60

def get_live_psx_price(ticker: str, yf_reference_price: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Board price = LDCP + CHANGE. Outside the session PSX rolls LDCP to the last close but keeps the
    previous day's CHANGE, which would double-count it - so a board price is only returned when
    (a) the session is open, (b) the change is plausible (<= ~10.5% of LDCP) and (c) it differs from
    the last close. If it equals the last close there is nothing live to show (None)."""
    try:
        mw, status, _ = fetch_psx_official_market_watch()
        if status != "SUCCESS" or mw is None:
            return None
        sym = normalize_ticker_display(ticker).strip().upper()
        row = mw[mw["Symbol"].astype(str).str.strip().str.upper() == sym]
        if row.empty or pd.isna(row.iloc[0].get("Current")):
            return None
        psx_price = float(row.iloc[0]["Current"])
        ldcp = row.iloc[0].get("LDCP")
        if yf_reference_price is not None and yf_reference_price > 0:
            ratio = psx_price / yf_reference_price
            if ratio > 1.20 or ratio < 0.80:     # >20% off the last close = mis-parse, not a price
                return None
            tol = max(0.015, 0.0005 * yf_reference_price)
            if abs(psx_price - yf_reference_price) <= tol:
                return None                       # same as last close: nothing live
        if ldcp is not None and not pd.isna(ldcp) and float(ldcp) > 0:
            if abs(psx_price - float(ldcp)) / float(ldcp) > 0.105:
                return None                       # beyond PSX circuit limits: stale/garbled
        if not _psx_session_open():
            return None                           # pre-open / after-hours / weekend: CHANGE may be stale
        return {"price": psx_price, "date": pkt_now().date(), "source": "psx_official_live"}
    except Exception:
        return None

# ============================================================
# SCAN UNIVERSE
# ============================================================

# ex-dividend / non-convertible / rights lines (e.g. HUMNLXD, PASLNC, BAPLR1): no /historical table on PSX
_DERIVED_SYMBOL_RE = re.compile(r"^[A-Z]{3,}(XD|NC|NCXD)$|^[A-Z]{3,}R\d+$")

@st.cache_data(ttl=3600, show_spinner=False)
def get_scan_universe() -> Tuple[List[str], str]:
    mw, status, _ = fetch_psx_official_market_watch()
    if status == "SUCCESS" and mw is not None and len(mw) > 100:
        symbols = mw["Symbol"].dropna().unique().tolist()
        kept = [s for s in symbols if not _DERIVED_SYMBOL_RE.search(str(s).upper().replace(".KA", ""))]
        dropped = len(symbols) - len(kept)
        formatted = [s + ".KA" if not s.endswith(".KA") else s for s in kept]
        uniq = list(dict.fromkeys(formatted))
        return uniq, (f"PSX official trading board ({len(uniq)} symbols, by volume; "
                      f"{dropped} derived XD/NC/rights symbols skipped)")
    return PSX_FALLBACK_UNIVERSE, f"curated fallback ({len(PSX_FALLBACK_UNIVERSE)}) — PSX trading board unavailable"

# ============================================================
# OHLCV PROVIDERS
# ============================================================

def fetch_psxdata_ohlcv(ticker: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], str, str]:
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
    """Try multiple yfinance symbol formats. Returns 3-tuple: (df, status, error)."""
    base = ticker.strip().upper().replace(".KA", "").replace(".PK", "")
    if ticker.strip().upper().endswith(".KA"):
        candidates = [ticker.strip().upper(), base, f"{base}.PK"]
    elif ticker.strip().upper().endswith(".PK"):
        candidates = [ticker.strip().upper(), f"{base}.KA", base]
    else:
        candidates = [f"{base}.KA", base, f"{base}.PK"]

    last_error = "No candidates tried"
    for symbol in candidates:
        try:
            raw = yf.download(symbol, period=period, interval="1d",
                              auto_adjust=False, progress=False)
            if raw is None or raw.empty:
                last_error = f"{symbol}: EMPTY"
                continue
            df = _flatten_columns(raw)
            required = ["Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                last_error = f"{symbol}: missing {missing}"
                continue
            df = df[required].apply(pd.to_numeric, errors="coerce").dropna()
            if df.empty:
                last_error = f"{symbol}: empty after clean"
                continue
            valid, msg = _validate_ohlcv(df)
            if not valid:
                last_error = f"{symbol}: {msg}"
                continue
            update_provider_status("yfinance", available=True, coverage=len(df))
            return df, "SUCCESS", f"yfinance ({symbol})"
        except Exception as e:
            last_error = f"{symbol}: {str(e)[:60]}"
            continue

    update_provider_status("yfinance", available=False, error=last_error)
    return None, "EXCEPTION", f"yfinance failed: {last_error}"

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

def _parse_psx_history(html: str) -> Optional[pd.DataFrame]:
    """HTML table (DATE/OPEN/HIGH/LOW/CLOSE/VOLUME) -> clean ascending OHLCV frame."""
    t = _first_table(html)
    if t is None or t.empty:
        return None
    ren = {}
    for c in t.columns:
        cl = str(c).strip().lower()
        if cl == "date": ren[c] = "Date"
        elif cl == "open": ren[c] = "Open"
        elif cl == "high": ren[c] = "High"
        elif cl == "low": ren[c] = "Low"
        elif cl == "close": ren[c] = "Close"
        elif cl.startswith("vol"): ren[c] = "Volume"
    t = t.rename(columns=ren)
    need = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if any(c not in t.columns for c in need):
        return None
    t = t[need].copy()
    t["Date"] = pd.to_datetime(t["Date"], errors="coerce")
    for c in need[1:]:
        t[c] = pd.to_numeric(t[c].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    t = t.dropna(subset=["Date"] + need[1:5])
    # drop non-trading / corrupt rows instead of rejecting the whole history
    t = t[(t[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    t["Volume"] = t["Volume"].fillna(0).clip(lower=0)
    t = t[(t["High"] >= t["Low"]) & (t["High"] >= t["Open"]) & (t["High"] >= t["Close"]) &
          (t["Low"] <= t["Open"]) & (t["Low"] <= t["Close"])]
    t = t.drop_duplicates(subset="Date", keep="last").sort_values("Date").set_index("Date")
    t.index = pd.DatetimeIndex(t.index).tz_localize(None).normalize()
    return t[["Open", "High", "Low", "Close", "Volume"]]

def fetch_psx_official_ohlcv(ticker: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], str, str]:
    """v8.9: official PSX EOD history. POST /historical returns the FULL history in one call."""
    sym = normalize_ticker_display(ticker)
    url = f"{PSX_BASE}/historical"
    try:
        resp = _psx_post(url, {"symbol": sym})
        if resp is None:
            msg = "PSX /historical: no response"
            update_provider_status("psx_official", available=False, error=msg)
            return None, "EMPTY", msg
        if resp.status_code != 200:
            hint = " (token rejected?)" if resp.status_code == 403 else ""
            msg = f"PSX /historical HTTP {resp.status_code}{hint}"
            update_provider_status("psx_official", available=False, error=msg)
            return None, "EMPTY", msg
        df = _parse_psx_history(resp.text)
        if df is None or df.empty:
            return None, "EMPTY", f"PSX /historical: no table for {sym}"
        days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}.get(period, 365)
        df = df[df.index >= (df.index.max() - pd.Timedelta(days=days))]
        valid, vmsg = _validate_ohlcv(df)
        if not valid:
            return None, "EXCEPTION", f"PSX history invalid for {sym}: {vmsg}"
        update_provider_status("psx_official", available=True, coverage=len(df))
        return df, "SUCCESS", f"psx_official ({sym})"
    except Exception as e:
        msg = f"PSX /historical error: {type(e).__name__}: {str(e)[:120]}"
        update_provider_status("psx_official", available=False, error=msg)
        return None, "EXCEPTION", msg

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_ohlcv(ticker: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], str, str, str]:
    """Order: PSX official (/historical) -> yfinance."""
    provider_attempts = []
    df, status, error = fetch_psx_official_ohlcv(ticker, period)
    provider_attempts.append(f"psx_official: {status} {error or ''}".strip())
    if status == "SUCCESS": return df, status, error, "psx_official"
    df, status, error = fetch_yfinance_ohlcv(ticker, period)
    provider_attempts.append(f"yfinance: {status}")
    if status == "SUCCESS": return df, status, error, "yfinance"
    return None, "UNAVAILABLE", " | ".join(provider_attempts), "UNAVAILABLE"

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_market_index():
    df, status, err = fetch_psx_official_index_json("KSE100")
    if status == "SUCCESS" and df is not None and len(df) >= 40:
        last_close = float(df["Close"].iloc[-1])
        if KSE100_PLAUSIBLE_MIN <= last_close <= KSE100_PLAUSIBLE_MAX:
            return df, "psx_official (KSE100 EOD, verified)"
    # v8.9: second PSX route - same /historical endpoint stocks use, symbol=KSE100
    try:
        h = _parse_psx_history(_psx_post(f"{PSX_BASE}/historical", {"symbol": "KSE100"}).text)
        if h is not None and len(h) >= 40:
            h = h[h.index >= (h.index.max() - pd.Timedelta(days=1100))]
            lc = float(h["Close"].iloc[-1])
            if KSE100_PLAUSIBLE_MIN <= lc <= KSE100_PLAUSIBLE_MAX:
                update_provider_status("psx_official", available=True, kse100=True, coverage=len(h))
                return h, "psx_official (KSE100 /historical, verified)"
    except Exception:
        pass
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
    uni, src = get_scan_universe()
    return uni, src, None

# ============================================================
# RECONCILIATION
# ============================================================

def reconcile_psx_vs_yf(ticker: str, yf_df: pd.DataFrame) -> Dict[str, Any]:
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
    sym = normalize_ticker_display(ticker).strip().upper()
    row = mw[mw["Symbol"].astype(str).str.strip().str.upper() == sym]
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
            reasons.append("Trend Structure Proxy: aligned up")
        elif (not hh) and (not hl):
            bearish += 1
            reasons.append("Trend Structure Proxy: aligned down")
    if last["ADX14"] >= 25: reasons.append(f"ADX {round(last['ADX14'],1)} - trending")
    total = bullish + bearish
    score = (bullish / total * 100) if total > 0 else 50
    if score >= 80: trend = "STRONG BULLISH"
    elif score >= 60: trend = "BULLISH"
    elif score >= 40: trend = "NEUTRAL"
    elif score >= 20: trend = "BEARISH"
    else: trend = "STRONG BEARISH"
    return trend, reasons, score

def _sr_basic(d):
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
            "secondary_resistance_is_distinct": secondary_resistance != primary_resistance,
            "range_high20": primary_resistance, "range_low20": primary_support,
            "support_zones": [], "resistance_zones": []}

def _cluster_levels(points, tol):
    """points = [(bar_index, price)]. Greedy clustering on sorted prices (distance to cluster mean)."""
    clusters = []
    for idx, p in sorted(points, key=lambda t: t[1]):
        if clusters and p - float(np.mean(clusters[-1]["prices"])) <= tol:
            clusters[-1]["prices"].append(p); clusters[-1]["idx"].append(idx)
        else:
            clusters.append({"prices": [p], "idx": [idx]})
    return [{"level": float(np.median(c["prices"])), "low": float(min(c["prices"])),
             "high": float(max(c["prices"])), "touches": len(c["prices"]),
             "last_idx": int(max(c["idx"]))} for c in clusters]

def support_resistance(d):
    """v9.0: swing-pivot zones. primary_* = NEAREST zone below/above price (not the raw 20-day
    extreme); falls back to the 20-day range when no pivot qualifies. Zones carry touch counts."""
    base = _sr_basic(d)
    try:
        if len(d) < 40:
            return base
        price = float(d["Close"].iloc[-1])
        atr_val = float(d["ATR14"].iloc[-1]) if "ATR14" in d.columns and not pd.isna(d["ATR14"].iloc[-1]) else price * 0.02
        atr_val = max(atr_val, price * 0.004)
        x = d.tail(260); n = len(x); win = 4
        highs = x["High"].to_numpy(dtype=float); lows = x["Low"].to_numpy(dtype=float)
        pts = []
        for i in range(win, n - win):
            if highs[i] >= highs[i - win:i + win + 1].max(): pts.append((i, highs[i]))
            if lows[i] <= lows[i - win:i + win + 1].min(): pts.append((i, lows[i]))
        if len(pts) < 3:
            return base
        tol = max(0.012 * price, 0.6 * atr_val)
        pad = 0.25 * atr_val
        zones = _cluster_levels(pts, tol)
        for z in zones:
            z["low"] = min(z["low"], z["level"] - pad); z["high"] = max(z["high"], z["level"] + pad)
            z["note"] = f"{z['touches']} touch" + ("es" if z["touches"] != 1 else "")
        sup = sorted([z for z in zones if z["level"] < price * 0.998], key=lambda z: -z["level"])
        res = sorted([z for z in zones if z["level"] > price * 1.002], key=lambda z: z["level"])
        def _pick(lst):
            strong = [z for z in lst if z["touches"] >= 2]
            return (strong if strong else lst)[:2]
        sup = _pick(sup); res = _pick(res)
        lo20, hi20 = float(base["range_low20"]), float(base["range_high20"])
        # recent 20d extreme is always eligible if it is a nearer level than the pivot zone
        if lo20 < price * 0.998 and (not sup or lo20 > sup[0]["level"] * 1.005):
            sup = ([{"level": lo20, "low": lo20 - pad, "high": lo20 + pad, "touches": 1, "note": "20d low"}] + sup)[:2]
        if hi20 > price * 1.002 and (not res or hi20 < res[0]["level"] * 0.995):
            res = ([{"level": hi20, "low": hi20 - pad, "high": hi20 + pad, "touches": 1, "note": "20d high"}] + res)[:2]
        prior = d.iloc[:-1] if len(d) > 1 else d
        r120 = prior.tail(120)
        if sup:
            base["primary_support"] = sup[0]["level"]
            base["secondary_support"] = sup[1]["level"] if len(sup) > 1 else float(r120["Low"].min())
        if res:
            base["primary_resistance"] = res[0]["level"]
            base["secondary_resistance"] = res[1]["level"] if len(res) > 1 else float(r120["High"].max())
        if base["secondary_support"] >= base["primary_support"]:
            base["secondary_support"] = min(float(r120["Low"].min()), base["primary_support"] - 2 * atr_val)
        if base["secondary_resistance"] <= base["primary_resistance"]:
            base["secondary_resistance"] = max(float(r120["High"].max()), base["primary_resistance"] + 2 * atr_val)
        base["secondary_support_is_distinct"] = base["secondary_support"] != base["primary_support"]
        base["secondary_resistance_is_distinct"] = base["secondary_resistance"] != base["primary_resistance"]
        base["support_zones"] = sup; base["resistance_zones"] = res
    except Exception:
        return _sr_basic(d)
    return base

def breakout_engine(d, sr, vol_ratio_threshold=1.5):
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
    ema20_v = last["EMA20"]
    above_ema = (price / ema20_v - 1) if (not pd.isna(ema20_v) and ema20_v > 0) else None
    high10 = d["High"].tail(10).max()
    off_high = (1 - price / high10) if high10 > 0 else 0
    # v9.0.1: support case needs a real dip (>=3% off the 10d high) and price not far above EMA20
    near_support = (abs(price - support) / price < 0.03 and off_high >= 0.03
                    and (above_ema is None or above_ema <= 0.06)) if price > 0 else False
    near_ema20 = abs(price - ema20_v) / price < 0.02 if not pd.isna(ema20_v) and price > 0 else False
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
    broke_resistance = price > sr.get("range_high20", sr["primary_resistance"])
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

def compute_stop(d, sr, entry, atr_val, max_pct):
    """Stop = nearest real support (>= 1 ATR away) else 1.5 ATR, but never further than the user cap
    (and never beyond the 10% hard ceiling / 2.5 ATR). Returns (stop, basis)."""
    max_pct = max(0.01, min(float(max_pct), STOP_MAX_PCT))
    swing_low = d.tail(10)["Low"].min()
    buffer = 0.3 * atr_val
    levels = [swing_low - buffer, sr["primary_support"] - buffer]
    room = STOP_MIN_ATR * atr_val
    valid = [lv for lv in levels if lv < entry and lv <= entry - room]
    pct_floor = entry * (1 - max_pct)
    if atr_val > 0:
        structure = max(valid) if valid else entry - 1.5 * atr_val
        floor = max(entry - STOP_MAX_ATR * atr_val, pct_floor)
    else:
        structure = max(valid) if valid else entry * 0.95
        floor = pct_floor
    stop = max(structure, floor)
    if structure >= floor:
        basis = "support (nearest)" if valid else "1.5 ATR"
    elif atr_val > 0 and entry - STOP_MAX_ATR * atr_val >= pct_floor:
        basis = f"{STOP_MAX_ATR:g} ATR cap"
    else:
        basis = f"{max_pct * 100:g}% cap"
    return stop, basis

def _zone_rr(d, sr, zone_mid, target, atr_val, max_pct):
    if target is None or zone_mid is None or target <= zone_mid:
        return None
    stop_z, _ = compute_stop(d, sr, zone_mid, atr_val, max_pct)
    if stop_z >= zone_mid:
        return None
    return (target - zone_mid) / (zone_mid - stop_z)

def risk_engine(d, sr, breakout_status="", max_stop_pct=DEFAULT_STOP_PCT, trend=None):
    last = d.iloc[-1]; price = last["Close"]
    atr_val = last["ATR14"] if not pd.isna(last["ATR14"]) else 0
    if price is None or price <= 0:
        return {"entry": None, "stop_loss": None, "risk_per_share": None, "target1": None,
                "target2": None, "rr1": None, "rr2": None, "conditional_entry": None,
                "conditional_entry_note": "Invalid price", "error": "Invalid entry price"}
    stop_loss, stop_basis = compute_stop(d, sr, price, atr_val, max_stop_pct)
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
    # ---- v9.0: buy zones (long ideas only) ----
    bearish = trend in ("BEARISH", "STRONG BEARISH")
    dip_low = dip_high = brk_low = brk_high = rr_dip = rr_brk = None
    dip_note = ""
    if not bearish and atr_val > 0:
        ema20_val = last["EMA20"] if not pd.isna(last["EMA20"]) else None
        level20 = sr.get("range_high20")
        anchor = None
        if level20 is not None and price > level20 and price - level20 > 0.5 * atr_val:
            anchor, dip_note = float(level20), "breakout retest"
        elif ema20_val is not None and ema20_val < price - 0.25 * atr_val:
            anchor, dip_note = float(ema20_val), "EMA20 pullback"
        elif sr["primary_support"] < price - 0.25 * atr_val:
            anchor, dip_note = float(sr["primary_support"]), "support"
        if anchor is not None:
            lo = anchor - 0.25 * atr_val; hi = min(anchor + 0.5 * atr_val, float(price))
            if lo < hi:
                dip_low, dip_high = lo, hi
                rr_dip = _zone_rr(d, sr, (lo + hi) / 2, target1, atr_val, max_stop_pct)
        res1 = sr["primary_resistance"]
        if price < res1 and (res1 - price) / price <= 0.08:
            brk_low, brk_high = float(res1), float(res1 + 0.5 * atr_val)
            mid_b = (brk_low + brk_high) / 2
            t_b = max(sr["secondary_resistance"], mid_b + 2.5 * atr_val)
            rr_brk = _zone_rr(d, sr, mid_b, t_b, atr_val, max_stop_pct)
    stop_atr = risk_per_share / atr_val if atr_val > 0 else None
    return {"entry": price, "stop_loss": stop_loss, "risk_per_share": risk_per_share,
            "target1": target1, "target2": target2, "rr1": rr1, "rr2": rr2,
            "conditional_entry": conditional_entry,
            "conditional_entry_note": conditional_entry_note, "error": None,
            "risk_pct": round(risk_per_share / price * 100, 1) if price > 0 else None,
            "stop_basis": stop_basis, "stop_atr": stop_atr,
            "stop_noise": bool(stop_atr is not None and stop_atr < 1.0),
            "max_stop_pct": float(max_stop_pct),
            "dip_low": dip_low, "dip_high": dip_high, "dip_note": dip_note, "rr_dip": rr_dip,
            "brk_low": brk_low, "brk_high": brk_high, "rr_brk": rr_brk}

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
    gate = "none"
    if not trend_ok:
        gate = "trend"
        signal = "WAIT" if adjusted_score >= 45 else "AVOID"
        setup_quality = "TREND BEARISH - NO LONG SETUP"
        reasons.append(f"❌ Trend {trend} hai — long entry nahi")
    elif not rr_ok:
        gate = "rr"
        signal = "WAIT"
        setup_quality = "R:R BELOW MINIMUM"
    elif adjusted_score >= 80: signal = "STRONG BUY"; setup_quality = "A+ SETUP"
    elif adjusted_score >= 65: signal = "BUY"; setup_quality = "A SETUP"
    elif adjusted_score >= 45: signal = "WAIT"; setup_quality = "B SETUP / WATCH"
    elif adjusted_score >= 25: signal = "REDUCE"; setup_quality = "WEAK"
    else: signal = "AVOID"; setup_quality = "POOR"
    if trend_ok:
        if components["trend"] >= 70: reasons.append(f"✅ Trend strong hai ({trend})")
        elif components["trend"] <= 35: reasons.append(f"❌ Trend kamzor hai ({trend})")
        if components["momentum"] >= 70: reasons.append("✅ Momentum acha hai — buyers active")
        elif components["momentum"] <= 35: reasons.append("❌ Momentum kamzor — sellers havi hain")
        if "CONFIRMED" in breakout["status"]: reasons.append("✅ Resistance toot gaya — volume ke saath upar close")
        elif "BREAKOUT READY" in breakout["status"]: reasons.append("📌 Resistance ke qareeb hai — watch karo")
        if pullback["status"] == "HEALTHY PULLBACK": reasons.append("✅ Pullback hua hai support pe")
        elif pullback["status"] == "BROKEN SUPPORT": reasons.append("❌ Support toot gaya")
        if momentum["overbought"]: reasons.append("⚠️ RSI 70 se upar — overbought")
        if momentum["oversold"]: reasons.append("📌 RSI 30 se neeche — oversold")
        if not rr_ok:
            rr_val = round(risk_data['rr1'], 2) if risk_data['rr1'] else 'N/A'
            reasons.append(f"❌ Risk:Reward kam hai ({rr_val})")
            _dl, _dh, _rd = risk_data.get("dip_low"), risk_data.get("dip_high"), risk_data.get("rr_dip")
            if _dl is not None and _dh is not None:
                _rd_txt = f" (R:R ~{round(_rd, 2)})" if _rd else ""
                reasons.insert(0, f"⏳ WAIT kyun: abhi entry pe R:R {rr_val} < {MIN_RR}. "
                                  f"Dip zone {round(_dl, 2)}–{round(_dh, 2)} mein entry behtar{_rd_txt}")
            else:
                reasons.insert(0, f"⏳ WAIT kyun: abhi entry pe R:R {rr_val} < {MIN_RR} — "
                                  f"target, stop ke muqable door nahi")
        if market["regime"] == "BULLISH": reasons.append("✅ Overall market bullish")
        elif market["regime"] == "BEARISH": reasons.append("❌ Overall market bearish")
    return {"score": adjusted_score, "components": components, "signal": signal,
            "setup_quality": setup_quality, "reasons": reasons[:6], "gate": gate}

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
                  market: Optional[Dict] = None,
                  max_stop_pct: float = DEFAULT_STOP_PCT):
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
    risk = risk_engine(d, sr, breakout["status"], max_stop_pct=max_stop_pct, trend=trend)
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
    last_close = float(last["Close"])
    live = get_live_psx_price(ticker, yf_reference_price=last_close)
    # only call it "live" when it genuinely differs from the last close (no false freshness)
    if live is not None and abs(live["price"] - last_close) > max(0.015, 0.0005 * last_close):
        result["display_price"] = live["price"]
        result["display_price_source"] = "psx_official_live"
        result["display_price_date"] = live["date"]
    else:
        result["display_price"] = last_close
        result["display_price_source"] = "last_close"
        result["display_price_date"] = result["data_date"]
    fresh_status, _, _ = get_freshness_status(result["data_date"])
    result["confidence"] = calculate_confidence(result, fresh_status)
    return result, "SUCCESS", None, source

def _fmt_zone(lo, hi) -> Optional[str]:
    if lo is None or hi is None:
        return None
    return f"{lo:.2f} – {hi:.2f}"

def _empty_screener_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SCREENER_COLUMNS)

def run_screener(universe: List[str], period: str = "1y",
                 penny_threshold: float = PENNY_STOCK_THRESHOLD,
                 rvol_threshold: float = 2.0,
                 market: Optional[Dict] = None,
                 max_stop_pct: float = DEFAULT_STOP_PCT):
    rows = []
    coverage = {"total": len(universe), "success": 0, "failed": 0, "analyzed": 0}
    total = len(universe)
    if market is None:
        market = market_snapshot()
    for idx, ticker in enumerate(universe):
        if idx % 10 == 0:
            st.caption(f"📊 Scanning {idx+1}/{total}...")
        try:
            result, status, error, source = analyze_stock(
                ticker, period=period, penny_threshold=penny_threshold,
                rvol_threshold=rvol_threshold, market=market, max_stop_pct=max_stop_pct)
        except Exception as e:
            result, status, error, source = None, "EXCEPTION", str(e), "UNAVAILABLE"
        if status != "SUCCESS" or result is None:
            coverage["failed"] += 1
            rows.append({"Ticker": ticker, "Price": None, "Price Src": "N/A",
                "Change %": None, "Trend": None, "Score": None,
                "Signal": "ERROR", "Gate": "n/a", "Confidence": "N/A",
                "Status": "DATA UNAVAILABLE", "Penny": None, "RR": None, "RR Flag": "",
                "Entry": None, "Stop Loss": None, "Target 1": None,
                "Cap Size": None, "Source": source,
                "Why": f"Data unavailable: {str(error)[:170]}",
                "Avg Volume": None, "_ticker_raw": ticker})
            continue
        coverage["success"] += 1
        coverage["analyzed"] += 1
        try:
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
            gate = result["signal"].get("gate", "none")
            gate_label = {"trend": "trend-gate", "rr": "R:R-gate", "none": "-"}.get(gate, "-")
            rr_val = result["risk"]["rr1"]
            rr_badge = ""
            if rr_val is not None:
                if rr_val < DAY_TRADE_MIN_RR:
                    rr_badge = "⚠️ Low R:R"
                elif rr_val < MIN_RR:
                    rr_badge = "low-mid R:R"
            if result["risk"].get("stop_noise"):
                rr_badge = (rr_badge + " · " if rr_badge else "") + "🔔 tight stop"
            if result["trend"] in ("BEARISH", "STRONG BEARISH"):
                rr_badge = "🚫 bearish: long plan N/A"
            _rk = result["risk"]
            _rsi = float(last["RSI14"]) if not pd.isna(last["RSI14"]) else None
            _bull = result["trend"] in ("BULLISH", "STRONG BULLISH")
            _zone = result["pullback"]["status"] in ("HEALTHY PULLBACK", "PULLBACK WATCH")
            _rsi_ok = _rsi is not None and 35 <= _rsi <= 55
            _candle_ok = float(last["Close"]) > float(last["Open"])
            pb_checks = (f"{'✅' if _bull else '❌'} Trend · {'✅' if _zone else '❌'} Zone · "
                         f"{'✅' if _rsi_ok else '❌'} RSI · {'✅' if _candle_ok else '❌'} Candle")
            conf = result.get("confidence", {}).get("level", "N/A")
            rows.append({
                "Ticker": result["ticker_display"],
                "Price": round(result["display_price"], 2),
                "Price Src": _price_src_label(result),
                "Change %": round(change_pct, 2), "Trend": result["trend"],
                "Score": result["signal"]["score"], "Signal": result["signal"]["signal"],
                "Gate": gate_label, "Confidence": conf,
                "Status": result["breakout"]["status"],
                "Penny": result["penny"]["status"] if result["penny"]["is_penny"] else "N/A",
                "RR": round(rr_val, 2) if rr_val else None,
                "RR Flag": rr_badge,
                "Entry": round(result["risk"]["entry"], 2) if result["risk"].get("entry") else None,
                "Stop Loss": round(result["risk"]["stop_loss"], 2) if result["risk"].get("stop_loss") else None,
                "Target 1": round(result["risk"]["target1"], 2) if result["risk"].get("target1") else None,
                "Cap Size": result["cap_size"], "Source": result["data_source"],
                "Why": why_text, "Avg Volume": round(avg_volume, 0),
                "Risk %": result["risk"].get("risk_pct"),
                "Stop ATR": round(_rk["stop_atr"], 1) if _rk.get("stop_atr") is not None else None,
                "Buy on Dip": _fmt_zone(_rk.get("dip_low"), _rk.get("dip_high")),
                "RR @Dip": round(_rk["rr_dip"], 2) if _rk.get("rr_dip") else None,
                "Buy on Breakout": _fmt_zone(_rk.get("brk_low"), _rk.get("brk_high")),
                "Pullback Checks": pb_checks,
                "RSI": round(_rsi, 1) if _rsi is not None else None,
                "EMA20": round(float(last["EMA20"]), 2) if not pd.isna(last["EMA20"]) else None,
                "Above EMA20 %": round((float(last["Close"]) / float(last["EMA20"]) - 1) * 100, 1)
                                 if not pd.isna(last["EMA20"]) and last["EMA20"] > 0 else None,
                "RVOL": round(float(last["VOL_RATIO"]), 2) if not pd.isna(last["VOL_RATIO"]) else None,
                "Pullback": result["pullback"]["status"],
                "Momentum": result["momentum"]["label"],
                "Resistance": round(float(result["breakout"]["resistance"]), 2)
                              if result["breakout"].get("resistance") else None,
                "Dist %": round(float(result["breakout"]["distance_to_resistance"]), 2)
                          if result["breakout"].get("distance_to_resistance") is not None else None,
                "52W": "🚀 BREAKOUT" if result["breakout"].get("is_52w_high_breakout")
                       else "🔵 NEAR" if result["breakout"].get("is_near_52w_high") else "-",
                "_ticker_raw": ticker,
            })
        except Exception as e:
            coverage["failed"] += 1
            rows.append({"Ticker": ticker, "Price": None, "Price Src": "N/A",
                "Change %": None, "Trend": None, "Score": None,
                "Signal": "ERROR", "Gate": "n/a", "Confidence": "N/A",
                "Status": f"POST-PROCESS ERROR: {str(e)[:60]}", "Penny": None,
                "RR": None, "RR Flag": "",
                "Entry": None, "Stop Loss": None, "Target 1": None,
                "Cap Size": None, "Source": source, "Why": "Post-processing error",
                "Avg Volume": None, "_ticker_raw": ticker})
    if not rows:
        return _empty_screener_df(), coverage
    df = pd.DataFrame(rows)
    for col in SCREENER_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df, coverage

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
    """Each ticker appears in ONE category only: Hold > Swing > Day (v9.0)."""
    empty = pd.DataFrame()
    out = {"day": empty, "swing": empty, "hold": empty, "moved": []}
    if screener_df is None or screener_df.empty or "Signal" not in screener_df.columns:
        return out
    df = screener_df[screener_df["Signal"] != "ERROR"].copy()
    if df.empty:
        return out
    for col in ["Status", "Score", "Trend", "RR"]:
        if col not in df.columns:
            df[col] = None
    score = pd.to_numeric(df["Score"], errors="coerce").fillna(0)
    rr = pd.to_numeric(df["RR"], errors="coerce").fillna(0)
    trend_s = df["Trend"].astype(str)
    _st = df["Status"].astype(str)

    hold_mask = (trend_s.str.contains("STRONG BULLISH", case=False, na=False) & (score >= 60) & (rr >= 1.5))
    swing_mask = (trend_s.str.contains("BULLISH", case=False, na=False)
                  & ~trend_s.str.contains("STRONG BULLISH", case=False, na=False)
                  & (rr >= 1.5) & (score >= 45))
    day_mask = (_st.str.contains("BREAKOUT|READY|ATTEMPT", case=False, na=False)
                & ~_st.str.upper().str.startswith("NO BREAKOUT")
                & ~_st.str.contains("UNAVAILABLE|ERROR", case=False, na=False)
                & (score >= 55))

    hold_df = df[hold_mask].sort_values("Score", ascending=False).head(10).copy()
    if not hold_df.empty:
        hold_df["Hold Estimate"] = "15-40 sessions (technical estimate)"
        hold_df["Basis"] = "Strong trend + healthy R:R"
    taken = set(hold_df["Ticker"]) if not hold_df.empty else set()

    swing_df = df[swing_mask & ~df["Ticker"].isin(taken)].sort_values("Score", ascending=False).head(10).copy()
    if not swing_df.empty:
        swing_df["Hold Estimate"] = "5-15 sessions (technical estimate)"
        swing_df["Basis"] = "Trend + R:R >= 1.5"
    taken |= set(swing_df["Ticker"]) if not swing_df.empty else set()

    day_all = df[day_mask]
    moved = [t for t in day_all["Ticker"] if t in taken]
    day_df = day_all[~day_all["Ticker"].isin(taken)].sort_values("Score", ascending=False).head(10).copy()
    if not day_df.empty:
        day_df["Hold Estimate"] = "1-2 sessions"
        day_df["Basis"] = "Momentum + breakout structure"
    return {"day": day_df, "swing": swing_df, "hold": hold_df, "moved": moved}

# ============================================================
# RECOVERY PLANNER (v9.0) - facts and arithmetic for a losing holding, not advice
# ============================================================

def switch_candidates(df: Optional[pd.DataFrame], exclude: List[str], need_pct: float, n: int = 3) -> pd.DataFrame:
    """Best-scoring bullish candidates from the latest scan + how many successful T1 trades
    would be needed to climb back to break-even (pure arithmetic)."""
    cols = ["Ticker", "Signal", "Score", "Entry", "Stop Loss", "Risk %", "Target 1", "T1 gain %",
            "RR", "Wins to recover"]
    if df is None or df.empty or "Signal" not in df.columns:
        return pd.DataFrame(columns=cols)
    w = df[df["Signal"] != "ERROR"].copy()
    w = w[w["Trend"].isin(["BULLISH", "STRONG BULLISH"])]
    w["RR"] = pd.to_numeric(w["RR"], errors="coerce")
    w = w[(w["RR"] >= 1.0) & ~w["Ticker"].isin(exclude)]
    if w.empty:
        return pd.DataFrame(columns=cols)
    for c in ["Entry", "Target 1", "Stop Loss", "Score"]:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w.dropna(subset=["Entry", "Target 1"])
    w["T1 gain %"] = ((w["Target 1"] / w["Entry"] - 1) * 100).round(1)
    def _wins(g):
        if g is None or pd.isna(g) or g <= 0 or need_pct is None or need_pct <= 0:
            return None
        return int(np.ceil(np.log(1 + need_pct / 100) / np.log(1 + g / 100)))
    w["Wins to recover"] = w["T1 gain %"].map(_wins)
    w = w.sort_values("Score", ascending=False).head(n)
    return w[[c for c in cols if c in w.columns]]

def recovery_plan(holding: Dict, result: Dict, active_stop: Optional[float]) -> Optional[Dict]:
    price = float(result["display_price"]); buy = float(holding["buy_price"]); shares = int(holding["shares"])
    last = result["last"]; d = result["df"]; sr = result["sr"]
    invested = buy * shares; value = price * shares; pnl = value - invested
    if pnl >= 0 or price <= 0:
        return None
    loss = invested - value
    need_pct = (buy / price - 1) * 100
    atr = float(last["ATR14"]) if not pd.isna(last["ATR14"]) else None
    sigma = float(d["Close"].pct_change().tail(60).std()) * price
    dist = buy - price
    best_sessions = dist / atr if atr and atr > 0 else None
    scale_sessions = (dist / sigma) ** 2 if sigma and sigma > 0 else None
    def _date_after(nsess):
        if nsess is None or nsess > 1500:
            return None
        try:
            return pd.Timestamp(np.busday_offset(pkt_now().date(), int(round(nsess)), roll="forward"))
        except Exception:
            return None
    # levels to reclaim
    cands = [("EMA20", last["EMA20"]), ("SMA20", last["SMA20"]), ("SMA50", last["SMA50"]),
             ("Resistance 1", sr["primary_resistance"]), ("Resistance 2", sr["secondary_resistance"]),
             ("SMA200", last.get("SMA200", np.nan))]
    lv = []
    for name, v in cands:
        if v is None or pd.isna(v) or float(v) <= price * 1.003 or float(v) >= buy:
            continue
        lv.append((name, float(v)))
    lv.append(("Break-even (aap ki buy price)", buy))
    lv.sort(key=lambda t: t[1])
    merged = []
    for name, v in lv:
        if merged and abs(v / merged[-1][1] - 1) < 0.01:
            merged[-1] = (merged[-1][0] + " / " + name, merged[-1][1])
        else:
            merged.append((name, v))
    levels = [{"Level": n, "Price": round(v, 2), "Gain from here %": round((v / price - 1) * 100, 1),
               "Loss recovered %": round(min(100.0, (v - price) * shares / loss * 100), 0)} for n, v in merged]
    # trend repair checklist
    rsi_v = float(last["RSI14"]) if not pd.isna(last["RSI14"]) else np.nan
    macd_h = float(last["MACD_HIST"]) if not pd.isna(last["MACD_HIST"]) else np.nan
    rvol = float(last["VOL_RATIO"]) if not pd.isna(last["VOL_RATIO"]) else np.nan
    s20 = float(last["SMA20"]); s50 = float(last["SMA50"])
    checks = [("Price SMA20 ke upar", price > s20, f"{price:.2f} vs {s20:.2f}"),
              ("SMA20 > SMA50", s20 > s50, f"{s20:.2f} vs {s50:.2f}"),
              ("Price SMA50 ke upar", price > s50, f"{price:.2f} vs {s50:.2f}"),
              ("RSI > 50", (not pd.isna(rsi_v)) and rsi_v > 50, f"{rsi_v:.1f}"),
              ("MACD histogram > 0", (not pd.isna(macd_h)) and macd_h > 0, f"{macd_h:.3f}"),
              ("Volume support (RVOL >= 1.2)", (not pd.isna(rvol)) and rvol >= 1.2, f"{rvol:.2f}x")]
    further = (price - active_stop) * shares if active_stop is not None and active_stop < price else None
    return {"price": price, "buy": buy, "shares": shares, "invested": invested, "value": value,
            "pnl": pnl, "loss": loss, "need_pct": need_pct, "atr": atr,
            "best_sessions": best_sessions, "best_date": _date_after(best_sessions),
            "scale_sessions": scale_sessions, "scale_date": _date_after(scale_sessions),
            "levels": levels, "checks": checks, "further_to_stop": further,
            "active_stop": active_stop}

# ============================================================
# PORTFOLIO DECISION
# ============================================================

def portfolio_decision(holding: Dict, result: Dict) -> Tuple[str, str]:
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
    current_price = result["display_price"]
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
                show_support_resistance=True, show_rsi=False, show_macd=False,
                show_zones=True, bars=120):
    d = result["df"].tail(int(bars))
    risk = result["risk"]; sr = result["sr"]; trend = result["trend"]
    proj = result.get("projection", {})
    is_bear = trend in ("BEARISH", "STRONG BEARISH")
    price_now = float(d["Close"].iloc[-1])

    # ---- projection geometry first (so the y-range can include it) ----
    proj_dir = proj.get("direction"); proj_x = proj_y = None; target_price = None; basis = ""
    try:
        if (proj_dir in ("UP", "DOWN") and proj.get("zone_low") is not None
                and proj.get("zone_high") is not None and len(d) >= 2):
            future_dates = pd.bdate_range(start=d.index[-1], periods=11, freq="B")[1:]
            target_price = float(proj["zone_high"]) if proj_dir == "UP" else float(proj["zone_low"])
            fut = np.linspace(price_now, target_price, len(future_dates) + 1)[1:]
            proj_x = [d.index[-1]] + list(future_dates); proj_y = [price_now] + list(fut)
            basis = "Upside zone" if proj_dir == "UP" else "Downside zone"
    except Exception:
        proj_x = proj_y = None

    y_lo = float(d["Low"].min()); y_hi = float(d["High"].max())
    if target_price is not None:
        y_lo = min(y_lo, target_price); y_hi = max(y_hi, target_price)
    pad = (y_hi - y_lo) * 0.06 if y_hi > y_lo else price_now * 0.03

    num_rows = 2
    if show_rsi: num_rows += 1
    if show_macd: num_rows += 1
    row_heights = [0.62, 0.14]
    if show_rsi: row_heights.append(0.12)
    if show_macd: row_heights.append(0.12)
    titles = [" ", "Volume"] + (["RSI (14)"] if show_rsi else []) + (["MACD"] if show_macd else [])
    fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, row_heights=row_heights,
                        vertical_spacing=0.035, subplot_titles=titles)
    row = 1
    UP, DN = "#00B386", "#F23645"
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Price",
        increasing=dict(line=dict(color=UP, width=1.2), fillcolor=UP),
        decreasing=dict(line=dict(color=DN, width=1.2), fillcolor=DN)), row=row, col=1)
    trend_color = "#00B386" if trend in ("BULLISH", "STRONG BULLISH") else \
                  "#F23645" if is_bear else "#F59E0B"
    trend_arrow = "↑" if trend in ("BULLISH", "STRONG BULLISH") else "↓" if is_bear else "→"
    fig.add_annotation(x=0.01, y=0.99, xref="paper", yref="paper", xanchor="left", yanchor="top",
        text=f"{trend_arrow} {trend}", showarrow=False,
        font=dict(color=trend_color, size=13, family="monospace"),
        bgcolor="rgba(255,255,255,0.92)", bordercolor=trend_color, borderwidth=1, borderpad=3)
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA20"], line=dict(color="#2563EB", width=2), name="SMA20"), row=row, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA50"], line=dict(color="#F59E0B", width=2), name="SMA50"), row=row, col=1)
    if show_sma200 and result["has_sma200"]:
        fig.add_trace(go.Scatter(x=d.index, y=d["SMA200"], line=dict(color="#8B5CF6", width=1.6, dash="dot"),
                                 name="SMA200"), row=row, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_UPPER"], line=dict(color="#94A3B8", width=0.9, dash="dot"),
                                 name="BB Upper"), row=row, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_LOWER"], line=dict(color="#94A3B8", width=0.9, dash="dot"),
                                 name="BB Lower"), row=row, col=1)

    def _visible(lo, hi):
        if abs(((lo + hi) / 2) / price_now - 1) > 0.30:      # far zones only add clutter
            return False
        return not (hi < y_lo - pad or lo > y_hi + pad)

    if show_support_resistance:
        _rz = sr.get("resistance_zones", [])[:2]
        if len(_rz) == 2 and abs(_rz[1]["level"] - _rz[0]["level"]) / price_now < 0.02: _rz = _rz[:1]   # avoid label overlap
        for i, z in enumerate(_rz):
            if not _visible(z["low"], z["high"]): continue
            fig.add_hrect(y0=z["low"], y1=z["high"], fillcolor="#F23645", opacity=0.13 if i == 0 else 0.07,
                          line_width=0, row=row, col=1)
            fig.add_hline(y=z["level"], line_color="#D62839", line_width=1.5 if i == 0 else 1, line_dash="dash",
                          annotation_text=f"R{i+1}{'/T1' if (i == 0 and risk.get('target1') and abs(risk['target1'] / z['level'] - 1) < 0.005 and not is_bear) else ''} {z['level']:.2f} · {z.get('note','')}",
                          annotation_position="top right", annotation_font_size=10,
                          annotation_font_color="#D62839", row=row, col=1)
        _sz = sr.get("support_zones", [])[:2]
        if len(_sz) == 2 and abs(_sz[1]["level"] - _sz[0]["level"]) / price_now < 0.02: _sz = _sz[:1]
        for i, z in enumerate(_sz):
            if not _visible(z["low"], z["high"]): continue
            fig.add_hrect(y0=z["low"], y1=z["high"], fillcolor="#00B386", opacity=0.13 if i == 0 else 0.07,
                          line_width=0, row=row, col=1)
            fig.add_hline(y=z["level"], line_color="#00916C", line_width=1.5 if i == 0 else 1, line_dash="dash",
                          annotation_text=f"S{i+1} {z['level']:.2f} · {z.get('note','')}",
                          annotation_position="bottom right", annotation_font_size=10,
                          annotation_font_color="#00916C", row=row, col=1)
    if not is_bear and not risk.get("error"):
        if show_zones:
            if risk.get("dip_low") is not None and _visible(risk["dip_low"], risk["dip_high"]):
                fig.add_hrect(y0=risk["dip_low"], y1=risk["dip_high"], fillcolor="#2563EB", opacity=0.16,
                              line_width=0, row=row, col=1,
                              annotation_text="Buy dip", annotation_position="inside left",
                              annotation_font_size=10, annotation_font_color="#1D4ED8")
            if risk.get("brk_low") is not None and _visible(risk["brk_low"], risk["brk_high"]):
                fig.add_hrect(y0=risk["brk_low"], y1=risk["brk_high"], fillcolor="#7C3AED", opacity=0.16,
                              line_width=0, row=row, col=1,
                              annotation_text="Buy breakout", annotation_position="inside left",
                              annotation_font_size=10, annotation_font_color="#6D28D9")
        if risk.get("stop_loss") and _visible(risk["stop_loss"], risk["stop_loss"]):
            fig.add_hline(y=risk["stop_loss"], line_dash="dot", line_color="#F97316", line_width=1.6,
                          annotation_text=f"Stop {risk['stop_loss']:.2f} (-{risk.get('risk_pct','?')}%)",
                          annotation_position="bottom left", annotation_font_size=10,
                          annotation_font_color="#EA580C", row=row, col=1)
        _r1 = (sr.get("resistance_zones") or [{}])[0].get("level") if show_support_resistance else None
        _t1_dup = _r1 is not None and abs(risk["target1"] / _r1 - 1) < 0.005 if risk.get("target1") else False
        if risk.get("target1") and not _t1_dup and _visible(risk["target1"], risk["target1"]):
            fig.add_hline(y=risk["target1"], line_dash="dot", line_color="#2563EB", line_width=1.6,
                          annotation_text=f"T1 {risk['target1']:.2f}", annotation_position="top left",
                          annotation_font_size=10, annotation_font_color="#1D4ED8", row=row, col=1)
    if proj_x is not None:
        arrow_color = "#00B386" if proj_dir == "UP" else "#F23645"
        fig.add_trace(go.Scatter(x=proj_x, y=proj_y, mode="lines", line=dict(color=arrow_color, width=2.2, dash="dash"),
            name=f"Projection → {target_price:.2f}", opacity=0.85,
            hovertemplate="Projection<br>%{x|%d-%b}<br>%{y:.2f}<extra></extra>"), row=row, col=1)
        fig.add_trace(go.Scatter(x=[proj_x[-1]], y=[proj_y[-1]], mode="markers+text",
            marker=dict(color=arrow_color, size=11, symbol="triangle-right" if proj_dir == "UP" else "triangle-down"),
            text=[f"{target_price:.2f}"], textposition="top center", textfont=dict(color=arrow_color, size=11),
            showlegend=False,
            hovertemplate=f"<b>Projected {target_price:.2f}</b> ({basis})<br><i>Technical estimate, not guaranteed</i><extra></extra>"),
            row=row, col=1)
    fig.update_yaxes(range=[y_lo - pad, y_hi + pad], row=1, col=1)

    row += 1
    vol_colors = np.where(d["Close"] >= d["Open"], UP, DN)
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], marker_color=vol_colors, name="Volume", opacity=0.75), row=row, col=1)
    if "VOL_SMA20" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["VOL_SMA20"], line=dict(color="#6366F1", width=1.4),
                                 name="Vol avg20"), row=row, col=1)
    if show_rsi:
        row += 1
        fig.add_trace(go.Scatter(x=d.index, y=d["RSI14"], line=dict(color="#2563EB", width=1.6), name="RSI"), row=row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#F23645", row=row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#00B386", row=row, col=1)
        fig.update_yaxes(range=[0, 100], row=row, col=1)
    if show_macd:
        row += 1
        fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], line=dict(color="#2563EB", width=1.2), name="MACD"), row=row, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["MACD_SIGNAL"], line=dict(color="#F59E0B", width=1.2), name="Signal"), row=row, col=1)
        hist_colors = np.where(d["MACD_HIST"] >= 0, UP, DN)
        fig.add_trace(go.Bar(x=d.index, y=d["MACD_HIST"], marker_color=hist_colors, name="Hist"), row=row, col=1)

    # remove weekend + holiday gaps so candles are evenly spaced (less congested)
    try:
        idx = pd.DatetimeIndex(d.index).normalize()
        missing = pd.date_range(idx[0], idx[-1]).difference(idx)
        miss_wd = [m.strftime("%Y-%m-%d") for m in missing if m.weekday() < 5]
        rb = [dict(bounds=["sat", "mon"])]
        if miss_wd: rb.append(dict(values=miss_wd))
        fig.update_xaxes(rangebreaks=rb)
    except Exception:
        pass
    x_end = (proj_x[-1] if proj_x is not None else d.index[-1]) + pd.Timedelta(days=3)
    fig.update_xaxes(range=[d.index[0] - pd.Timedelta(days=1), x_end], showgrid=False)
    fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False)
    fig.update_layout(height=720 if not (show_rsi or show_macd) else 860,
        showlegend=True, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        template="plotly_white", plot_bgcolor="#FFFFFF", font=dict(size=12), bargap=0.15)
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
        "Stop Loss": "Yahan exit karo loss control ke liye (long setup).",
        "Target": "Technical price jahan tak ja sakta hai (long setup).",
        "ATR": "Rozana ka average movement.",
    }
    return explanations.get(indicator, "Technical indicator.")

def _style_portfolio_row(row):
    """Highlight STALE (pink) and DELAYED (yellow) rows."""
    styles = [""] * len(row)
    freshness = str(row.get("Freshness", "")).upper()
    if freshness == "STALE":
        styles = ["background-color: #FEE2E2; color: #7F1D1D;"] * len(row)
    elif freshness == "DELAYED":
        styles = ["background-color: #FEF3C7; color: #78350F;"] * len(row)
    return styles

def _render_data_quality_banner(result: Dict):
    """Top-of-Dashboard banner explaining data quality."""
    conf = result.get("confidence", {})
    level = conf.get("level", "N/A")
    reasons = conf.get("reasons", [])
    if level == "HIGH":
        st.success(f"✅ **Data Quality: HIGH** — Signal reliable, sab checks pass.")
    elif level == "MEDIUM":
        msg = f"⚠️ **Data Quality: MEDIUM** — Signal lekin warnings ke saath."
        if reasons:
            msg += " " + " · ".join(reasons[:3])
        st.warning(msg)
    elif level == "LOW":
        msg = f"❌ **Data Quality: LOW** — Signal unreliable, ihtiyaat karo."
        if reasons:
            msg += " " + " · ".join(reasons[:3])
        st.error(msg)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "<div style='font-family:monospace; color:#2563EB; font-size:22px; "
    "font-weight:bold; letter-spacing:1px;'>PSX QUANT ENGINE</div>"
    "<div style='color:#64748B; font-size:11px; margin-bottom:10px;'>"
    "Quantitative Decision Support · v9.0.1</div>", unsafe_allow_html=True)

_prov_slot = st.sidebar.container()   # v8.9: filled at the END of the script, after all fetches

def _render_provider_status():
    st.subheader("🔌 Provider Status")
    psx_avail = PROVIDER_STATUS.get("psx_official", {}).get("available", False)
    if psx_avail:
        kse100_ok = "✅ verified" if PROVIDER_STATUS["psx_official"].get("kse100") else "⚠️ unverified"
        st.success(f"✅ PSX Official: Working ({kse100_ok})")
    else:
        st.error("❌ PSX Official: NOT WORKING")
        psx_err = PROVIDER_STATUS.get("psx_official", {}).get("error")
        if psx_err:
            st.caption(f"**Error:** {psx_err[:250]}")
    yf_avail = PROVIDER_STATUS.get("yfinance", {}).get("available", True)
    if yf_avail:
        st.success("✅ yfinance: Working (fallback)")
    else:
        st.error("❌ yfinance: Failed")
        yf_err = PROVIDER_STATUS.get("yfinance", {}).get("error")
        if yf_err:
            st.caption(f"**Error:** {yf_err[:250]}")
    psxd_avail = PROVIDER_STATUS.get("psxdata", {}).get("available", False)
    if psxd_avail:
        st.info("ℹ️ psxdata: Working (experimental)")
    else:
        st.caption("psxdata: Not installed (optional)")

# v8.7 NEW: PSX Diagnostic button
with st.sidebar.expander("🩺 PSX Diagnostic Probe"):
    st.caption("PSX ki raw response check karo — timeout vs 403 vs 200+empty")
    if st.button("🔍 Run PSX Diagnostic", use_container_width=True, key="psx_diag_btn"):
        with st.spinner("Probing PSX endpoints..."):
            diag_text = diagnostic_psx_probe()
            st.session_state["psx_diag_result"] = diag_text
    if "psx_diag_result" in st.session_state:
        st.markdown(st.session_state["psx_diag_result"])

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
stop_cap_pct = st.sidebar.slider("Max Stop Loss (% of entry)",
    min_value=2.0, max_value=10.0, value=5.0, step=0.5,
    help="Stop-loss entry se is % se zyada door nahi hoga. Nearest support is cap ke andar ho to wahi stop banta hai, "
         "warna cap. Bohot tight (2-3%) rakhoge to roz ki normal hilna-julna (ATR) bhi stop hit kar sakta hai - "
         "table mein 'Stop ATR' aur 'tight stop' flag dekho. Badalne ke baad Screener dobara chalao.")
max_stop_pct = stop_cap_pct / 100.0

st.sidebar.divider()
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    fetch_ohlcv.clear()
    fetch_market_index.clear()
    market_snapshot.clear()
    analyze_stock.clear()
    fetch_psx_official_index_json.clear()
    fetch_psx_official_market_watch.clear()
    get_scan_universe.clear()
    for k in ["screener_df", "top_picks_df", "proxy_result", "psx_diag_result"]:
        st.session_state.pop(k, None)
    st.sidebar.success("Cache cleared!")

st.sidebar.caption("KSE-100: PSX Official → yfinance | Stock OHLCV: PSX /historical → yfinance")
st.sidebar.caption("Board: PSX trading-board (token) | Price: PSX live → last PSX close → yfinance")
st.sidebar.caption("Confidence: data quality + signal reliability marker")
st.sidebar.caption("Cache: 5min | Universe: 60min")
st.sidebar.caption(f"Checked: {pkt_now().strftime('%d-%b %H:%M')} PKT")

if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

# ============================================================
# MAIN TABS (7 tabs — Pullbacks added in v8.8)
# ============================================================

tab_dash, tab_screener, tab_breakouts, tab_pull, tab_port, tab_market, tab_top = st.tabs([
    "📊 Dashboard", "🔍 Screener", "🚀 Breakouts", "🔁 Pullbacks",
    "💼 Portfolio", "📈 Market", "🎯 Top Picks"])

# ============================================================
# DASHBOARD TAB
# ============================================================

with tab_dash:
    result, status, error, source = analyze_stock(
        sidebar_ticker, period=period,
        penny_threshold=penny_threshold, rvol_threshold=rvol_threshold, max_stop_pct=max_stop_pct)
    if status != "SUCCESS":
        st.error(f"❌ {sidebar_ticker} analyze nahi ho saka: {error}")
        with st.expander("🔍 Troubleshooting", expanded=True):
            st.write(f"**Status:** {status}")
            st.write(f"**Error:** {error}")
            st.write(f"**Source:** {source if source else 'N/A'}")
            st.markdown("**Try:**")
            st.write("1. Refresh Data button dabao (sidebar)")
            st.write("2. Ticker spelling check karo")
            st.write("3. PSX Diagnostic probe chalao (sidebar mein)")
    else:
        last = result["last"]; sig = result["signal"]
        trend = result["trend"]
        is_bearish = trend in ("BEARISH", "STRONG BEARISH")
        freshness_status, _, freshness_warning = get_freshness_status(result["data_date"])

        _render_data_quality_banner(result)
        show_stale_data_warning(freshness_status, freshness_warning)

        recon = result.get("reconciliation", {})
        if recon.get("mismatch"):
            st.warning(f"⚠️ Price mismatch — PSX: {recon['psx_price']} | "
                       f"yfinance: {recon['yf_price']} | farq {recon['diff_pct']}%")

        col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.2, 1.2, 1.2])
        with col1:
            disp_price = result["display_price"]
            st.markdown(f"<span class='current-price'>{round(disp_price, 2)}</span>",
                unsafe_allow_html=True)
            prev_close = result["df"]["Close"].iloc[-2] if len(result["df"]) >= 2 else last["Close"]
            if result["display_price_source"] == "psx_official_live":
                prev_close = last["Close"]                 # today's move is vs the last close
                change = disp_price - prev_close
            else:
                change = last["Close"] - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0
            cc = "change-positive" if change >= 0 else "change-negative"
            st.markdown(f"<span class='{cc}'>{'▲' if change >= 0 else '▼'} "
                f"{round(change, 2)} ({round(change_pct, 2)}%)</span>", unsafe_allow_html=True)
            st.caption(f"{result['ticker_display']} · {result['cap_size']}")
            st.caption(f"Data: {source} · {freshness_status} · {result['data_date'].date()}")
            if result["display_price_source"] == "psx_official_live":
                st.caption(f"📍 Live PSX price (aaj) · {result['display_price_date']}")
            elif str(result.get("data_source", "")).startswith("psx_official"):
                st.caption(f"📅 Last PSX close · {result['data_date'].date()}")
            else:
                st.caption(f"📅 Last yfinance close · {result['display_price_date']}")
        with col2:
            sc = get_signal_class(sig["signal"])
            st.markdown("**Signal**")
            st.markdown(f"<span class='{sc}'>{sig['signal']}</span>", unsafe_allow_html=True)
        with col3:
            st.metric("Score", f"{sig['score']}/100", help="Weighted technical score 0-100")
        with col4:
            st.metric("Trend", trend, help="Trend engine output")
        with col5:
            st.metric("Setup", sig["setup_quality"],
                help="Setup quality based on score + trend alignment (not statistical confidence)")

        conf = result.get("confidence", {})
        if conf.get("level"):
            with st.expander(f"🎯 Confidence: {conf['level']} (score {conf.get('score', 0)})", expanded=False):
                st.markdown("**Confidence = data quality + signal reliability**")
                if conf.get("reasons"):
                    st.markdown("**Warnings:**")
                    for r in conf["reasons"]:
                        st.write(f"- {r}")
                else:
                    st.write("Koi warning nahi — sab checks pass.")
                st.caption("HIGH = fresh data + PSX live + RR ≥ 1.5 + trend aligned. "
                           "MEDIUM = kuch warnings. LOW = stale data ya bearish trend.")

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
        risk = result["risk"]
        if is_bearish:
            st.warning("📉 Bearish structure — **long setup nahi**. Is tool mein sirf long (khareed) setups hain, "
                       "isliye 'AVOID' ka matlab hai: naya paisa mat lagao. Agar yeh stock aap ke paas hai to "
                       "Portfolio tab ka Recovery Planner dekho.")
            st.caption("ℹ️ Short setup yahan nahi diya jata: aam retail investor ke liye PSX par short selling "
                       "asaan/available option nahi hoti (futures ya broker ki facility par depend karta hai — "
                       "broker se confirm karo). Neeche downside projection sirf yeh batata hai ke support kahan "
                       "tak toot sakta hai.")
        elif risk.get("error"):
            st.warning(f"⚠️ Risk calculation error: {risk['error']}")
        else:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Entry", round(risk["entry"], 2))
            c2.metric("Stop Loss", round(risk["stop_loss"], 2))
            c3.metric("Target 1", round(risk["target1"], 2))
            c4.metric("Target 2", round(risk["target2"], 2))
            c5.metric("R:R", f"1:{round(risk['rr1'], 2) if risk['rr1'] else 'N/A'}")
            st.caption(f"🛑 Stop entry se {risk.get('risk_pct', '?')}% neeche · basis: {risk.get('stop_basis', '?')} · "
                       f"aap ka cap: {stop_cap_pct:g}% (sidebar) · "
                       f"{round(risk['stop_atr'], 1) if risk.get('stop_atr') is not None else '?'} ATR door")
            if risk.get("stop_noise"):
                st.warning("🔔 Tight stop: stop 1 ATR se kam door hai — roz ki normal hilna-julna hi isay hit kar sakta hai "
                           "(whipsaw). Cap barhao ya entry zone mein (neeche) lo.")
            z1, z2, z3 = st.columns(3)
            _dz = _fmt_zone(risk.get("dip_low"), risk.get("dip_high"))
            _bz = _fmt_zone(risk.get("brk_low"), risk.get("brk_high"))
            z1.metric("🔵 Buy on Dip (neeche aaye to)", _dz or "—",
                      help="Price is range mein wapas aaye to entry. Basis: " + (risk.get("dip_note") or "n/a"))
            z2.metric("🟣 Buy on Breakout (upar jaye to)", _bz or "—",
                      help="Resistance ke upar volume ke saath close ho to entry.")
            z3.metric("R:R inside zone",
                      " / ".join([f"dip {risk['rr_dip']:.2f}" if risk.get("rr_dip") else "dip —",
                                  f"brk {risk['rr_brk']:.2f}" if risk.get("rr_brk") else "brk —"]))
            _dp = result.get("display_price")
            if (result.get("display_price_source") == "psx_official_live" and _dp
                    and abs(_dp / risk["entry"] - 1) > 0.005):
                st.warning(f"ℹ️ Plan last close {risk['entry']:.2f} par bana hai, live price {_dp:.2f} "
                           f"({(_dp / risk['entry'] - 1) * 100:+.1f}%) hai — entry/stop/R:R live price ke hisaab se "
                           f"dobara dekho, zone se door ho to chase mat karo.")
            st.caption("Zone = range, ek point nahi. Stop/Target technical levels hain; zone ke andar R:R aksar abhi ke price se behtar hota hai.")
            if risk.get("conditional_entry") is not None:
                st.caption(f"💡 Better Entry: {risk['conditional_entry']} — {risk['conditional_entry_note']}")

        proj = result["projection"]
        if proj["direction"] != "NEUTRAL":
            dir_emoji = "📈" if proj["direction"] == "UP" else "📉"
            st.info(f"{dir_emoji} **Technical Projection ({proj['direction']}):** {proj['label']}")
            st.caption(proj["note"])

        if result["penny"]["is_penny"]:
            st.warning(f"🪙 {result['penny']['status']} - {result['penny']['note']}")

        st.subheader("📊 Chart")
        c1, c2, c3, c4, c5 = st.columns(5)
        show_bb = c1.checkbox("Bollinger Bands", value=False)
        show_sma200 = c2.checkbox("SMA200", value=False)
        show_sr = c3.checkbox("Support/Resistance", value=True)
        show_zones = c4.checkbox("Buy zones", value=True)
        show_rsi = c5.checkbox("RSI", value=False)
        e1, e2 = st.columns([1, 3])
        show_macd = e1.checkbox("MACD", value=False)
        chart_bars = e2.select_slider("Chart window (trading days)", options=[60, 90, 120, 180, 250], value=120)
        st.plotly_chart(build_chart(result, show_bb, show_sma200, show_sr, show_rsi, show_macd,
                                    show_zones, chart_bars), use_container_width=True)
        st.caption("📉 **Projection line** trend ki direction follow karti hai (technical estimate, guarantee nahi). "
                   "🟥/🟩 shaded band = resistance/support **zone** (kitni baar price wahan se palta). "
                   "🔵 Buy dip · 🟣 Buy breakout = entry ranges. Weekend/holiday gaps hata diye gaye hain.")

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
            if is_bearish:
                st.caption("ℹ️ Stop Loss / Target long setup ke liye hain — "
                           "bearish structure me relevant nahi.")
            else:
                st.markdown(f"**Stop Loss:** {get_indicator_explanation('Stop Loss')} "
                    f"({round(risk['stop_loss'],2) if risk['stop_loss'] else 'N/A'})")
                st.markdown(f"**Target:** {get_indicator_explanation('Target')} "
                    f"({round(risk['target1'],2) if risk['target1'] else 'N/A'})")
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
            _zrows = ([{"Type": f"Resistance {i+1}", "Level": round(z["level"], 2),
                        "Band": f"{z['low']:.2f} – {z['high']:.2f}", "Basis": z.get("note", "")}
                       for i, z in enumerate(sr.get("resistance_zones", []))]
                      + [{"Type": f"Support {i+1}", "Level": round(z["level"], 2),
                          "Band": f"{z['low']:.2f} – {z['high']:.2f}", "Basis": z.get("note", "")}
                         for i, z in enumerate(sr.get("support_zones", []))])
            if _zrows:
                st.dataframe(pd.DataFrame(_zrows), hide_index=True, use_container_width=True)
                st.caption("Zone = swing high/low ka cluster (touches = kitni baar price wahan se palta). "
                           "Primary = price ke sabse qareeb wala zone.")
            if not pd.isna(sr["high_52w"]):
                st.caption(f"52-Week High: {round(sr['high_52w'], 2)} | 52-Week Low: {round(sr['low_52w'], 2)}")
            else:
                st.caption("N/A — insufficient 52-week history (need ≥252 sessions)")

# ============================================================
# SCREENER TAB
# ============================================================

with tab_screener:
    st.subheader("🔍 PSX Opportunity Scanner")
    universe_option = st.selectbox("Universe",
        ["Dynamic (PSX official market-watch)", "Liquid PSX (~34)", "Small Cap (~25)", "Custom"],
        index=0)
    col1, col2 = st.columns(2)
    with col1: custom_syms = st.text_input("Add extra symbols (comma-separated)", "")
    with col2: scan_full = st.checkbox("Scan full dynamic universe (slow, 500+ symbols, 10+ min)",
        value=False, help="Default scans first 100 symbols for speed")
    st.caption(f"🛑 Stop cap abhi {stop_cap_pct:g}% hai (sidebar). Cap badlo to Screener dobara chalao, tab naye stops lagenge.")
    st.markdown("**Filters**")
    p1, p2, p3 = st.columns(3)
    with p1:
        signal_filter = st.multiselect("Signal",
            ["STRONG BUY", "BUY", "WAIT", "REDUCE", "AVOID"],
            default=["STRONG BUY", "BUY", "WAIT"],
            help="Bearish market mein sirf BUY filter rakho to screen khaali aayegi. "
                 "WAIT include karne se underlying candidates dikhte hain.")
    with p2:
        price_range = st.slider("Price Range (PKR)", min_value=0.0, max_value=5000.0,
            value=(0.0, 5000.0), step=10.0)
    with p3: min_score = st.slider("Min Score", 0, 100, 0)
    include_penny = st.checkbox(f"Include penny stocks (below PKR {int(penny_threshold)})",
        value=False, help="Penny stocks high volatility")
    if include_penny:
        st.caption("⚠️ Penny stocks: high volatility, liquidity risk, false breakouts")

    min_conf = st.multiselect("Min Confidence",
        ["HIGH", "MEDIUM", "LOW"],
        default=[], help="Sirf in confidence levels ke stocks dikhao")

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
        with st.spinner("Preparing universe..."):
            if universe_option == "Dynamic (PSX official market-watch)":
                universe, uni_source = get_scan_universe()
            elif universe_option == "Liquid PSX (~34)":
                universe, uni_source = PSX_LIQUID_UNIVERSE, f"Liquid PSX ({len(PSX_LIQUID_UNIVERSE)})"
            elif universe_option == "Small Cap (~25)":
                universe, uni_source = PSX_SMALL_CAP_UNIVERSE, f"Small Cap ({len(PSX_SMALL_CAP_UNIVERSE)})"
            else:
                universe = [normalize_ticker(sidebar_ticker)]
                if custom_syms:
                    universe += [normalize_ticker(t) for t in custom_syms.split(",") if t.strip()]
                universe = list(dict.fromkeys(universe))
                uni_source = f"Custom ({len(universe)})"
            if custom_syms and universe_option != "Custom":
                extra = [t.strip() + ".KA" if not t.strip().endswith(".KA") else t.strip()
                        for t in custom_syms.split(",") if t.strip()]
                universe = list(dict.fromkeys(universe + extra))
                uni_source += f" + {len(extra)} extra"
            if not scan_full and len(universe) > 100:
                total_count = len(universe)
                universe = universe[:100]
                st.caption(f"📊 Scanning first 100 of {total_count} ({uni_source}).")
            else:
                st.caption(f"📊 Scanning {len(universe)} symbols ({uni_source})")
        with st.spinner("Scanning PSX universe..."):
            market = market_snapshot()
            screener_df, coverage = run_screener(universe, period="1y",
                penny_threshold=penny_threshold, rvol_threshold=rvol_threshold, market=market,
                max_stop_pct=max_stop_pct)
            st.session_state["screener_df"] = screener_df
            st.session_state["screener_coverage"] = coverage
            st.session_state["screener_universe_source"] = uni_source
        if not screener_df.empty:
            missing = [c for c in ["RR", "Score", "Signal", "Price", "Status", "Confidence",
                                   "Entry", "Stop Loss", "Target 1"]
                       if c not in screener_df.columns]
            if missing:
                st.warning(f"⚠️ Screener output mein ye columns missing hain: {missing}")

    if "screener_coverage" in st.session_state:
        cov = st.session_state["screener_coverage"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", cov["total"]); c2.metric("Analyzed", cov["analyzed"])
        c3.metric("Success", cov["success"]); c4.metric("Failed", cov["failed"])
        if "screener_universe_source" in st.session_state:
            st.caption(f"Universe source: {st.session_state['screener_universe_source']}")
        if cov.get("failed", 0) > 0 and "screener_df" in st.session_state:
            _fd = st.session_state["screener_df"]
            if "Signal" in _fd.columns and "Why" in _fd.columns:
                _fd = _fd[_fd["Signal"] == "ERROR"]
                if not _fd.empty:
                    with st.expander(f"⚠️ Failed symbols ({len(_fd)}) — wajah"):
                        st.dataframe(_fd[["Ticker", "Why"]], use_container_width=True, hide_index=True)

    if "screener_df" in st.session_state:
        df_s = st.session_state["screener_df"]
        view = df_s.copy()

        def _has(col): return col in view.columns

        if signal_filter and _has("Signal"):
            view = view[view["Signal"].isin(signal_filter)]
        if min_score > 0 and _has("Score"):
            view = view[pd.to_numeric(view["Score"], errors="coerce").fillna(0) >= min_score]
        if _has("Price"):
            view = view[(pd.to_numeric(view["Price"], errors="coerce").fillna(0) >= price_range[0]) &
                        (pd.to_numeric(view["Price"], errors="coerce").fillna(0) <= price_range[1])]
        if not include_penny and _has("Penny"):
            penny_mask = view["Penny"].notna() & (view["Penny"] != "N/A")
            view = view[~penny_mask]
        if min_conf and _has("Confidence"):
            view = view[view["Confidence"].isin(min_conf)]
        if category_filter and _has("Cap Size"):
            view = view[view["Cap Size"].isin(category_filter)]
        if breakout_filter and _has("Status"):
            view = view[view["Status"].astype(str).apply(
                lambda s: any(bf in s for bf in breakout_filter))]
        if min_rr > 0 and _has("RR"):
            view = view[pd.to_numeric(view["RR"], errors="coerce").fillna(0) >= min_rr]
        if min_avg_volume > 0 and _has("Avg Volume"):
            view = view[pd.to_numeric(view["Avg Volume"], errors="coerce").fillna(0) >= min_avg_volume]
        if _has("Signal"):
            view = view[view["Signal"] != "ERROR"]

        sort_by = st.selectbox("Sort by", ["Score", "Confidence", "Change %", "Price"], index=0)
        if sort_by in view.columns:
            if sort_by == "Confidence":
                conf_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "N/A": 3}
                view = view.assign(_conf_sort=view["Confidence"].map(conf_order).fillna(4))
                view = view.sort_values("_conf_sort").drop(columns=["_conf_sort"])
            else:
                view = view.sort_values(sort_by, ascending=False, na_position="last")

        total_scanned = len(df_s)
        penny_count = len(view[view["Penny"].notna() & (view["Penny"] != "N/A")]) if "Penny" in view.columns else 0
        st.caption(f"📊 Showing **{len(view)}** of {total_scanned} scanned"
            + (f" ({penny_count} penny stocks)" if include_penny and penny_count > 0 else ""))
        st.caption("ℹ️ **Confidence** = data quality + signal reliability. "
                   "**Gate** = why signal became WAIT/REDUCE. "
                   "**Entry/Stop/Target** = actionable levels.")
        if not view.empty:
            display_cols = [c for c in view.columns if not c.startswith("_")]
            st.dataframe(view[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No stocks match filters")

# ============================================================
# BREAKOUTS TAB (v8.8: distinct - filter over Screener results, no re-analysis)
# ============================================================

BREAKOUT_COLS = ["Ticker", "Price", "Price Src", "Change %", "Trend", "Score", "Signal",
                 "Gate", "Confidence", "Status", "RVOL", "Resistance", "Dist %",
                 "Entry", "Stop Loss", "Risk %", "Stop ATR", "Target 1", "RR", "RR Flag",
                 "Buy on Dip", "RR @Dip", "Penny", "Momentum", "52W", "Why"]
PULLBACK_COLS = ["Ticker", "Price", "Price Src", "Change %", "Trend", "Score", "Signal",
                 "Pullback", "Pullback Checks", "EMA20", "Above EMA20 %", "RSI",
                 "Buy on Dip", "RR @Dip", "Entry", "Stop Loss", "Risk %", "Target 1", "RR", "RR Flag",
                 "RVOL", "Momentum", "Action"]

def _pullback_action(row) -> str:
    pb = str(row.get("Pullback", "")).upper()
    dip = row.get("Buy on Dip")
    zone = f" Dip zone: {dip}." if isinstance(dip, str) and dip else ""
    sl = row.get("Stop Loss")
    stop_txt = f" Stop {sl} ke neeche close = setup khatam." if sl is not None and not pd.isna(sl) else ""
    if pb == "HEALTHY PULLBACK":
        return "Confirmation mil gayi — entry dekh sakte ho." + zone + stop_txt
    if pb == "PULLBACK WATCH":
        return "Zone mein hai, confirmation baqi (RSI 35-55 + upar band hone wali candle)." + zone
    return f"Abhi dip nahi aaya — EMA20 ({row.get('EMA20')}) tak girne ka intezar." + zone

def _arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Columns mixing numbers and text (e.g. 'N/A') break st.dataframe's Arrow conversion."""
    df = df.copy()
    for c in df.columns:
        col = df[c]
        if col.dtype == object:
            vals = [v for v in col if v is not None and not (isinstance(v, float) and pd.isna(v))]
            has_str = any(isinstance(v, str) for v in vals)
            has_num = any(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in vals)
            if has_str and has_num:
                df[c] = col.map(lambda v: "N/A" if v is None or (isinstance(v, float) and pd.isna(v))
                                else (f"{v:.2f}" if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                                      else str(v)))
    return df

def _is_penny_row(df: pd.DataFrame) -> pd.Series:
    if "Penny" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["Penny"].notna() & (df["Penny"] != "N/A")

with tab_breakouts:
    st.subheader("🚀 Breakouts")
    st.caption("Sirf woh stocks jo resistance tod chuke hain ya tootne ke qareeb hain. "
               "Screener ke results se filter hota hai (naya scan nahi).")
    if "screener_df" not in st.session_state:
        st.info("Pehle Screener tab pe 'Run Screener' chalao.")
    else:
        df_s = st.session_state["screener_df"]
        needed = {"Status", "RVOL", "52W"}
        if df_s.empty or not needed.issubset(df_s.columns):
            st.info("Screener output purana/khali hai — dobara Run Screener chalao.")
        else:
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                bo_mode = st.radio("Breakout type",
                    ["Confirmed / 52W breakout", "Near breakout + volume", "Sab (confirmed + near)"],
                    horizontal=True, key="bo_mode")
            with c2:
                bo_rvol = st.number_input("Min RVOL", min_value=0.0, max_value=20.0,
                    value=2.0, step=0.5, key="bo_rvol",
                    help="Relative volume: aaj ka volume / 20-din average. 2.0 = double.")
            with c3:
                bo_penny = st.checkbox("Penny stocks bhi", value=False, key="bo_penny",
                    help="PKR 50 se sasti stocks mein false breakouts zyada hote hain.")
            work = df_s[df_s["Signal"] != "ERROR"].copy()
            if not bo_penny:
                work = work[~_is_penny_row(work)]
            status_s = work["Status"].astype(str)
            rvol_s = pd.to_numeric(work["RVOL"], errors="coerce").fillna(0)
            confirmed = status_s.str.contains("CONFIRMED", case=False, na=False) | \
                        work["52W"].astype(str).str.contains("BREAKOUT", na=False)
            near = status_s.str.contains("READY|ATTEMPT", case=False, na=False) & ~confirmed
            if bo_mode.startswith("Confirmed"):
                bo = work[confirmed & (rvol_s >= bo_rvol)]
            elif bo_mode.startswith("Near"):
                bo = work[near & (rvol_s >= bo_rvol)]
            else:
                bo = work[(confirmed | near) & (rvol_s >= bo_rvol)]
            if bo.empty:
                st.info(f"Koi candidate nahi mila (RVOL >= {bo_rvol}). "
                        "RVOL kam karke dekho ya Screener ka universe/filters check karo.")
            else:
                cols = [c for c in BREAKOUT_COLS if c in bo.columns]
                st.dataframe(_arrow_safe(bo.sort_values("RVOL", ascending=False)[cols]),
                             use_container_width=True, hide_index=True)
                st.caption("Volume ke bina breakout aksar jhootha hota hai (false breakout) — "
                           "RVOL, Risk % aur RR Flag dekho. Entry/Stop/Target technical levels hain.")

# ============================================================
# PULLBACKS TAB (swing entries near support / EMA20 in an uptrend)
# ============================================================

with tab_pull:
    st.subheader("🔁 Pullbacks (Swing Entries)")
    st.caption("Bullish trend wale stocks jo EMA20 ya support ke qareeb wapas aaye hain. "
               "Breakout ka ulta idea: tez bhaag chuke stock ko chase karne ke bajaye dip pe entry.")
    with st.expander("ℹ️ Pullback kya hota hai? (parhne ke liye)", expanded=False):
        st.markdown(
            "**Pullback** = uptrend ke beech chhota sa girna (dip). Strong stock seedha nahi chadhta — "
            "chadhta hai, thoda girta hai, phir chadhta hai. Is dip pe entry lene ka faida: **stop chhota** "
            "(support/EMA20 paas hota hai) aur R:R behtar.\n\n"
            "**System 4 cheezein check karta hai** (table ke 'Pullback Checks' column mein ✅/❌):\n"
            "1. **Trend** BULLISH ya STRONG BULLISH ho\n"
            "2. **Zone**: price EMA20 ya support ke 2-3% ke andar ho\n"
            "3. **RSI** thanda ho kar 35-55 mein ho (jo bhaag chuka ho, RSI 70+ hota hai)\n"
            "4. **Candle**: aaj ki candle upar band ho (khareedar wapas aaye)\n\n"
            "- 4 mein se Trend+Zone ✅ aur RSI/Candle ✅ = **HEALTHY PULLBACK** (confirmation ke saath)\n"
            "- Trend+Zone ✅ magar RSI/Candle ❌ = **PULLBACK WATCH** (dekho, abhi mat khareedo)\n"
            "- Zone ❌ = abhi pullback nahi; 'Approaching' list mein EMA20 se 2-8% upar wale stocks hain, unka dip ka intezar\n\n"
            "**Misaal (PRL):** pehle ~106.95 tak chadha, phir ~96.96 tak gira jo EMA20 (~96.5) ke qareeb hai — "
            "yani Zone ✅. Candle upar band na ho ya RSI 55 se upar ho to WATCH rehta hai.\n\n"
            "**'Buy on Dip'** column = woh range jahan entry behtar hai, **'RR @Dip'** = us range mein R:R. "
            "Stop ke neeche close ho to setup khatam.")
    if "screener_df" not in st.session_state:
        st.info("Pehle Screener tab pe 'Run Screener' chalao.")
    else:
        df_s = st.session_state["screener_df"]
        if df_s.empty or not {"Pullback", "Above EMA20 %"}.issubset(df_s.columns):
            st.info("Screener output purana/khali hai — dobara Run Screener chalao.")
        else:
            only_healthy = st.checkbox("Sirf HEALTHY PULLBACK (confirmation ke saath)", value=False,
                                       key="pb_healthy")
            work = df_s[df_s["Signal"] != "ERROR"].copy()
            work = work[~_is_penny_row(work)]
            pb_s = work["Pullback"].astype(str).str.strip().str.upper()
            bull = work["Trend"].isin(["BULLISH", "STRONG BULLISH"])
            wanted = ["HEALTHY PULLBACK"] if only_healthy else ["HEALTHY PULLBACK", "PULLBACK WATCH"]
            pb = work[pb_s.isin(wanted) & bull]
            st.markdown("**✅ Abhi pullback zone mein**")
            if pb.empty:
                st.info("Abhi koi stock pullback zone mein nahi hai. Bearish/sideways market mein yeh normal hai.")
            else:
                pb = pb.copy(); pb["Action"] = pb.apply(_pullback_action, axis=1)
                cols = [c for c in PULLBACK_COLS if c in pb.columns]
                st.dataframe(_arrow_safe(pb.sort_values("Score", ascending=False)[cols]),
                             use_container_width=True, hide_index=True)
            ab_x = pd.to_numeric(work["Above EMA20 %"], errors="coerce")
            appr = work[(pb_s == "NO PULLBACK") & bull & (ab_x > 2) & (ab_x <= 8)]
            st.markdown("**👀 Approaching — bullish, EMA20 se 2-8% upar (dip ka intezar)**")
            if appr.empty:
                st.caption("Koi stock is range mein nahi.")
            else:
                appr = appr.assign(_x=ab_x[appr.index]).sort_values("_x")
                appr["Action"] = appr.apply(_pullback_action, axis=1)
                cols = [c for c in PULLBACK_COLS if c in appr.columns]
                st.dataframe(_arrow_safe(appr[cols]), use_container_width=True, hide_index=True)
            st.caption("EMA20 = pullback ka nishana. Entry/Stop/Target technical levels hain, guarantee nahi.")


def _render_recovery(plan: Dict, h: Dict, res: Dict, src_df):
    t = res["ticker_display"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Loss (abhi)", f"PKR {plan['loss']:,.0f}")
    c2.metric("Break-even ke liye", f"+{plan['need_pct']:.1f}%",
              help="Abhi ke price se kitna badhna chahiye ke aap ki buy price wapas aa jaye")
    c3.metric("Break-even price", f"{plan['buy']:.2f}")
    c4.metric("Abhi price", f"{plan['price']:.2f}")
    proj = res.get("projection", {})
    if proj.get("direction") in ("UP", "DOWN"):
        st.info(f"{'📈' if proj['direction'] == 'UP' else '📉'} Technical projection: {proj['label']} — {proj['note']}")

    st.markdown("**1) Kaun se levels wapas lene hain** (neeche se upar)")
    st.dataframe(pd.DataFrame(plan["levels"]), hide_index=True, use_container_width=True)
    st.caption("'Loss recovered %' = us level tak pahunchne par aap ka kitna nuqsan wapas ho jata hai. "
               "Har level par seller aate hain — price wahan ruk sakta hai.")

    st.markdown("**2) Trend theek hone ki checklist**")
    n_ok = sum(1 for _, ok, _ in plan["checks"] if ok)
    for name, ok, val in plan["checks"]:
        st.write(f"{'✅' if ok else '❌'} {name} — {val}")
    st.caption(f"{n_ok}/{len(plan['checks'])} pass. Jab tak zyada points ❌ hain, system is stock ko recovery setup nahi maanta.")

    st.markdown("**3) Time ka andaza (volatility se, prediction nahi)**")
    parts = []
    if plan["best_sessions"]:
        d1 = plan["best_date"].strftime("%d-%b-%Y") if plan["best_date"] is not None else "bohot door"
        parts.append(f"- **Best case** (price roz ek ATR seedha upar): ~{plan['best_sessions']:.0f} sessions → ~{d1}")
    if plan["scale_sessions"]:
        d2 = plan["scale_date"].strftime("%d-%b-%Y") if plan["scale_date"] is not None else "multi-year (1500+ sessions)"
        parts.append(f"- **Volatility scale** (up-down chalte hue): ~{plan['scale_sessions']:.0f} sessions "
                     f"(~{plan['scale_sessions'] / 21:.0f} mahine) → ~{d2}")
    st.markdown("\n".join(parts) if parts else "Estimate ke liye data kam hai.")
    st.caption("⚠️ Yeh sirf scale batata hai. Bearish trend mein recovery is se bohot lambi ho sakti hai ya na bhi ho. "
               "Koi tarika exact date nahi bata sakta.")

    st.markdown("**4) Options — rupees mein**")
    opt = [{"Option": "Abhi poora exit", "Proceeds (PKR)": f"{plan['value']:,.0f}",
            "Realized P/L (PKR)": f"{plan['pnl']:,.0f}", "Baqi shares": 0},
           {"Option": "Aadha reduce", "Proceeds (PKR)": f"{plan['value'] / 2:,.0f}",
            "Realized P/L (PKR)": f"{plan['pnl'] / 2:,.0f}", "Baqi shares": plan["shares"] // 2}]
    if plan["further_to_stop"] is not None:
        opt.append({"Option": f"Hold, stop {plan['active_stop']:.2f} pe", "Proceeds (PKR)": "—",
                    "Realized P/L (PKR)": f"{-(plan['loss'] + plan['further_to_stop']):,.0f} (stop hit hone par)",
                    "Baqi shares": plan["shares"]})
    else:
        opt.append({"Option": "Hold", "Proceeds (PKR)": "—",
                    "Realized P/L (PKR)": "stop pehle hi toot chuka / data nahi", "Baqi shares": plan["shares"]})
    st.dataframe(pd.DataFrame(opt), hide_index=True, use_container_width=True)
    st.caption("Brokerage/tax shamil nahi. Faisla aap ka hai — system sirf numbers dikhata hai.")

    st.markdown("**5) Average karne ka hisaab**")
    add = st.number_input(f"Agar {t} mein aur lagao (PKR)", min_value=0, value=0, step=10000, key=f"avg_{t}")
    if add > 0:
        new_sh = add / plan["price"]; tot_sh = plan["shares"] + new_sh
        new_avg = (plan["invested"] + add) / tot_sh
        need2 = (new_avg / plan["price"] - 1) * 100
        st.write(f"Nayi average price **{new_avg:.2f}** → break-even ke liye **+{need2:.1f}%** "
                 f"(pehle +{plan['need_pct']:.1f}%). Lekin exposure PKR {add:,.0f} barh jata hai.")
        st.caption("Bearish trend mein average-down se risk barhta hai; checklist ke zyada points ✅ hone tak sirf hisaab dekho.")

    st.markdown("**6) Switch ka hisaab (latest scan ke bullish candidates)**")
    sw = switch_candidates(src_df, [t], plan["need_pct"])
    if sw.empty:
        st.caption("Pehle Screener ya Top Picks chalao — phir yahan candidates aur 'recover karne ke liye kitne successful trades' dikhenge.")
    else:
        st.dataframe(_arrow_safe(sw), hide_index=True, use_container_width=True)
        st.caption(f"'Wins to recover' = {plan['value']:,.0f} PKR ko {plan['invested']:,.0f} tak wapas lane ke liye "
                   f"candidate ke Target 1 jitne kitne kamyab trades chahiye (har trade apne stop/risk ke saath). "
                   f"Loss realize hota hai aur candidate ke bhi fail hone ka imkaan hai — yeh sirf arithmetic hai.")

# ============================================================
# PORTFOLIO TAB
# ============================================================

with tab_port:
    st.subheader("💼 Portfolio Tracker (max 5)")
    st.caption("Active Stop = max(Original Stop, Technical Stop) for long positions")
    st.caption("📍 = PSX live price · 📅 = yfinance close · "
               "Confidence = data quality + signal reliability")
    with st.form("add_holding"):
        c1, c2, c3, c4 = st.columns(4)
        h_ticker = c1.text_input("Ticker")
        h_price = c2.number_input("Buy Price (PKR)", min_value=0.0, step=0.5)
        h_shares = c3.number_input("Shares", min_value=0, step=1)
        h_stop = c4.number_input("Original Stop (PKR)", min_value=0.0, step=0.5,
                                 help="Aap ka original stop-loss level")
        submitted = st.form_submit_button("Add Holding")
        if submitted and h_ticker and h_price > 0 and h_shares > 0:
            if len(st.session_state.portfolio) >= 5:
                st.warning("Maximum 5 holdings")
            else:
                st.session_state.portfolio.append({
                    "ticker": h_ticker.strip().upper(), "buy_price": h_price,
                    "shares": h_shares,
                    "original_stop_loss": h_stop if h_stop > 0 else None})
                st.success(f"Added {h_ticker}")
    if st.session_state.portfolio:
        rows = []; total_invested = 0; total_current = 0
        plan_inputs = []
        market = market_snapshot()
        for h in st.session_state.portfolio:
            result, status, error, _ = analyze_stock(h["ticker"], period="1y",
                penny_threshold=penny_threshold, rvol_threshold=rvol_threshold, market=market,
                max_stop_pct=max_stop_pct)
            invested = h["buy_price"] * h["shares"]
            total_invested += invested
            if status == "SUCCESS" and result is not None:
                cur_price = result["display_price"]
                cur_value = cur_price * h["shares"]
                total_current += cur_value
                pnl = cur_value - invested
                pnl_pct = pnl / invested * 100 if invested else 0
                decision, reason = portfolio_decision(h, result)
                technical_stop = result["risk"]["stop_loss"]
                original_stop = h.get("original_stop_loss")
                if original_stop is not None and technical_stop is not None:
                    active_stop = max(original_stop, technical_stop)
                elif original_stop is not None:
                    active_stop = original_stop
                elif technical_stop is not None:
                    active_stop = technical_stop
                else:
                    active_stop = None
                plan_inputs.append((h, result, active_stop))
                fresh_status, _, _ = get_freshness_status(result["data_date"])
                data_date_str = result["data_date"].strftime("%Y-%m-%d") if result["data_date"] is not None else "N/A"
                price_src = _price_src_label(result)
                conf = result.get("confidence", {}).get("level", "N/A")
                rows.append({"Ticker": result["ticker_display"], "Buy": h["buy_price"],
                    "Shares": h["shares"], "Invested": round(invested, 2),
                    "Current": round(cur_price, 2), "Price Src": price_src,
                    "Value": round(cur_value, 2),
                    "P/L": round(pnl, 2), "P/L %": round(pnl_pct, 2),
                    "Trend": result["trend"], "Signal": result["signal"]["signal"],
                    "Confidence": conf,
                    "Original Stop": round(original_stop, 2) if original_stop else "N/A",
                    "Technical Stop": round(technical_stop, 2) if technical_stop else "N/A",
                    "Active Stop": round(active_stop, 2) if active_stop else "N/A",
                    "Target 1": round(result["risk"]["target1"], 2) if result["risk"]["target1"] else "N/A",
                    "Break-even +%": round((h["buy_price"] / cur_price - 1) * 100, 1) if cur_price < h["buy_price"] else 0.0,
                    "Projection": result["projection"]["label"],
                    "Data Date": data_date_str, "Freshness": fresh_status,
                    "Decision": decision, "Reason": reason})
            else:
                rows.append({"Ticker": h["ticker"], "Buy": h["buy_price"],
                    "Shares": h["shares"], "Invested": round(invested, 2),
                    "Current": "N/A", "Price Src": "❌ N/A",
                    "Value": "N/A", "P/L": "N/A", "P/L %": "N/A",
                    "Trend": None, "Signal": "ERROR",
                    "Confidence": "N/A",
                    "Original Stop": h.get("original_stop_loss", "N/A"),
                    "Technical Stop": None, "Active Stop": None, "Target 1": None,
                    "Data Date": "N/A", "Freshness": "UNAVAILABLE",
                    "Decision": "WATCH", "Reason": f"Data unavailable: {error}"})
        successful_holdings = sum(1 for r in rows if r["Current"] != "N/A")
        total_holdings = len(rows)
        if total_holdings == 0:
            pass
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
        port_df = _arrow_safe(pd.DataFrame(rows))
        try:
            styled = port_df.style.apply(_style_portfolio_row, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(port_df, use_container_width=True, hide_index=True)
        st.caption("**Freshness:** FRESH = 0-1 business days · DELAYED = 2-3 · "
                   "STALE = >3 (pink highlight). **Confidence:** ✅ HIGH / ⚠️ MEDIUM / ❌ LOW")
        if plan_inputs:
            st.markdown("### 🩺 Recovery Planner")
            st.caption("Har losing holding ke liye facts + hisaab: kaun se levels wapas lene hain, trend kab theek hoga, "
                       "time ka andaza, exit/hold ke rupee outcomes. Yeh advice nahi, numbers hain — faisla aap ka.")
            _src_df = st.session_state.get("screener_df")
            if _src_df is None or getattr(_src_df, "empty", True):
                _src_df = st.session_state.get("top_picks_df")
            for _h, _res, _astop in plan_inputs:
                _plan = recovery_plan(_h, _res, _astop)
                _tk = _res["ticker_display"]
                if _plan is None:
                    with st.expander(f"✅ {_tk} — profit mein"):
                        st.success("Profit mein hai. Recovery ki zaroorat nahi — Active Stop ko upar trail karne ka sochen.")
                else:
                    with st.expander(f"🩺 {_tk} — loss PKR {_plan['loss']:,.0f} · break-even +{_plan['need_pct']:.1f}% "
                                     f"· {_res['trend']} · {_res['signal']['signal']}"):
                        _render_recovery(_plan, _h, _res, _src_df)
        remove_idx = st.selectbox("Remove holding",
            options=["-"] + [h["ticker"] for h in st.session_state.portfolio])
        if remove_idx != "-" and st.button("Remove Selected"):
            st.session_state.portfolio = [h for h in st.session_state.portfolio
                                          if h["ticker"] != remove_idx]
            st.rerun()
    else:
        st.info("Koi holding nahi. Upar add karo (max 5).")

# ============================================================
# MARKET TAB
# ============================================================

with tab_market:
    st.subheader("📈 Market Overview")
    market = market_snapshot()
    if market["regime"] == "UNAVAILABLE":
        st.warning("🔴 KSE-100: DATA UNAVAILABLE")
        st.caption(f"Source: {market['source']}")
        psx_err = PROVIDER_STATUS.get("psx_official", {}).get("error")
        if psx_err:
            st.error(f"**PSX error:** {psx_err}")
            st.caption("Sidebar mein 'PSX Diagnostic Probe' chalao — exact cause batayega "
                       "(timeout / 403 / 200+empty).")
    else:
        fresh, _, fresh_warn = get_freshness_status(market["last_date"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Regime", market["regime"])
        c2.metric("Trend", market["trend"])
        c3.metric("KSE-100 Level",
            round(market["last_close"], 2) if market["last_close"] else "N/A")
        st.caption(f"Source: {market['source']} | Last: "
            f"{market['last_date'].date() if market['last_date'] else 'N/A'} ({fresh})")
        if fresh == "STALE":
            st.warning(f"🔴 {fresh_warn}")
        st.info(f"**Reasoning:** {market['reasoning']}")
        idx_df, _ = fetch_market_index()
        if idx_df is not None and len(idx_df) > 30:
            idx_df = idx_df.copy()
            idx_df["SMA20"] = sma(idx_df["Close"], 20)
            idx_df["SMA50"] = sma(idx_df["Close"], 50)
            idx_disp = idx_df.tail(CHART_DISPLAY_BARS)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=idx_disp.index, y=idx_disp["Close"],
                line=dict(color="#3B82F6", width=2), name="KSE-100"))
            fig.add_trace(go.Scatter(x=idx_disp.index, y=idx_disp["SMA20"],
                line=dict(color="#F59E0B", width=1.5, dash="dot"), name="SMA20"))
            fig.add_trace(go.Scatter(x=idx_disp.index, y=idx_disp["SMA50"],
                line=dict(color="#8B5CF6", width=1.5, dash="dot"), name="SMA50"))
            _lx, _ly = idx_disp.index[-1], float(idx_disp["Close"].iloc[-1])
            fig.add_trace(go.Scatter(x=[_lx], y=[_ly], mode="markers",
                marker=dict(color="#EF4444", size=9), name="Latest", showlegend=False))
            fig.add_annotation(x=_lx, y=_ly, text=f"{_lx:%d-%b-%Y}: {_ly:,.1f}",
                showarrow=True, arrowhead=2, ax=-70, ay=-40, bgcolor="#FFFFFF",
                bordercolor="#EF4444", font=dict(size=11))
            fig.update_layout(template="plotly_white", height=400,
                margin=dict(l=10, r=10, t=30, b=10),
                title=dict(text=f"KSE-100 — last {len(idx_disp)} sessions · latest {_lx:%d-%b-%Y}",
                           font=dict(size=13)))
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
        else:
            st.info("Proxy unavailable")
    with st.expander("🔍 Provider Diagnostics"):
        diag = pd.DataFrame([{
            "Provider": k, "Available": "✅" if v["available"] else "❌",
            "Coverage": v["coverage"], "KSE-100": "✅" if v["kse100"] else "❌",
            "Last Success": v["last_success"].strftime("%d-%b %H:%M") if v["last_success"] else "Never",
            "Last Attempt": v["last_fetch_attempt"].strftime("%d-%b %H:%M") if v["last_fetch_attempt"] else "Never",
            "Error": (v["error"][:100] if v["error"] else "-")}
            for k, v in PROVIDER_STATUS.items()])
        st.dataframe(diag, use_container_width=True, hide_index=True)
        st.caption("⚠️ Diagnostics = actual network fetches only.")

# ============================================================
# TOP PICKS TAB (Day/Swing/Hold with Entry/Stop/Target)
# ============================================================

with tab_top:
    st.subheader("🎯 Top Picks — Day / Swing / Hold")
    st.caption("⚠️ Sab estimates technical hain. Guaranteed returns nahi.")
    colA, colB = st.columns([2, 3])
    with colA:
        top_universe_choice = st.selectbox("Universe for Top Picks",
            ["Dynamic (PSX official market-watch)", "Liquid PSX (~34)",
             "Small Cap (~25)", "Both (Liquid + Small)"],
            index=0, key="top_universe")
    with colB:
        st.caption("Button dabane pe scan hoga — auto nahi.")
    if st.button("🎯 Generate Top Picks", use_container_width=True):
        with st.spinner("Scanning for Top Picks..."):
            if top_universe_choice == "Dynamic (PSX official market-watch)":
                uni, uni_source = get_scan_universe()
            elif top_universe_choice == "Liquid PSX (~34)":
                uni, uni_source = PSX_LIQUID_UNIVERSE, f"Liquid PSX ({len(PSX_LIQUID_UNIVERSE)})"
            elif top_universe_choice == "Small Cap (~25)":
                uni, uni_source = PSX_SMALL_CAP_UNIVERSE, f"Small Cap ({len(PSX_SMALL_CAP_UNIVERSE)})"
            else:
                uni = list(dict.fromkeys(PSX_LIQUID_UNIVERSE + PSX_SMALL_CAP_UNIVERSE))
                uni_source = f"Both ({len(uni)})"
            st.caption(f"Universe: {uni_source}")
            if len(uni) > 100:
                st.caption(f"📊 Scanning first 100 of {len(uni)} for speed.")
                uni = uni[:100]
            market = market_snapshot()
            top_picks_df, _ = run_screener(uni, period="1y",
                penny_threshold=penny_threshold, rvol_threshold=rvol_threshold, market=market,
                max_stop_pct=max_stop_pct)
            st.session_state["top_picks_df"] = top_picks_df
    if "top_picks_df" not in st.session_state:
        st.info("Click **Generate Top Picks** to scan.")
    else:
        df_s = st.session_state["top_picks_df"]
        if df_s.empty:
            st.info("Koi data nahi mila — dobara try karo.")
        else:
            picks = categorize_top_picks(df_s, penny_threshold=penny_threshold,
                rvol_threshold=rvol_threshold)
            if picks.get("moved"):
                st.caption("ℹ️ Har stock sirf ek category mein (Hold > Swing > Day). Day se hataye: "
                           + ", ".join(sorted(set(picks["moved"]))))
            st.markdown("### ⚡ Day Trade Candidates")
            st.caption("High momentum + breakout structure — 1-2 sessions. "
                       "⚠️ Low R:R = reward:risk < 1.0 (momentum play, not R:R play).")
            if picks["day"].empty:
                st.info("Koi day-trade candidate nahi mila.")
            else:
                day_cols = ["Ticker", "Price", "Price Src", "Change %", "Score", "Signal",
                            "Gate", "Confidence", "Status",
                            "Entry", "Stop Loss", "Risk %", "Target 1",
                            "Buy on Dip", "RR @Dip", "Buy on Breakout",
                            "RR", "RR Flag", "Hold Estimate", "Basis"]
                day_cols = [c for c in day_cols if c in picks["day"].columns]
                st.dataframe(_arrow_safe(picks["day"][day_cols]), use_container_width=True, hide_index=True)
            st.divider()
            st.markdown("### 🌊 Swing Trade Candidates")
            st.caption("Clear trend + room to run + R:R ≥ 1.5 — 5-15 sessions")
            if picks["swing"].empty:
                st.info("Koi swing-trade candidate nahi mila.")
            else:
                swing_cols = ["Ticker", "Price", "Price Src", "Trend", "Score", "Signal",
                              "Gate", "Confidence", "Status",
                              "Entry", "Stop Loss", "Risk %", "Target 1",
                              "Buy on Dip", "RR @Dip", "Buy on Breakout",
                              "RR", "Hold Estimate", "Basis"]
                swing_cols = [c for c in swing_cols if c in picks["swing"].columns]
                st.dataframe(_arrow_safe(picks["swing"][swing_cols]), use_container_width=True, hide_index=True)
            st.divider()
            st.markdown("### 💼 Hold Candidates")
            st.caption("Established strong bullish trend — positional horizon")
            if picks["hold"].empty:
                st.info("Koi hold candidate nahi mila.")
            else:
                hold_cols = ["Ticker", "Price", "Price Src", "Trend", "Score", "Signal",
                             "Gate", "Confidence", "Status",
                             "Entry", "Stop Loss", "Risk %", "Target 1",
                             "Buy on Dip", "RR @Dip", "Buy on Breakout",
                             "RR", "Hold Estimate", "Basis"]
                hold_cols = [c for c in hold_cols if c in picks["hold"].columns]
                st.dataframe(_arrow_safe(picks["hold"][hold_cols]), use_container_width=True, hide_index=True)
            st.caption("**Hold Estimate** = ATR-based pace calculation. Technical estimate, guaranteed nahi. "
                       "**Entry/Stop/Target 1** = long trade plan levels (skip if bearish).")

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

with _prov_slot:          # v8.9: provider status reflects THIS run's fetches
    _render_provider_status()
