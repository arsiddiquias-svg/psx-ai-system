"""
PSX QUANT ENGINE - Diagnostic Build (Phase 0)
Purpose: verify yfinance actually returns usable PSX data from a real,
open-internet environment (Streamlit Cloud), before building the full
signal/breakout/portfolio engine on top of it.

Architecture note:
- data provider logic lives in the DATA PROVIDER section below as
  isolated functions. The full Quant Engine (trend engine, breakout
  engine, signal engine, position sizing, portfolio tracker) will be
  added as separate function blocks in later phases WITHOUT touching
  this data layer, as long as fetch_ohlcv() keeps the same return
  contract: (dataframe_or_None, status_string, error_string_or_None)
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="PSX Quant Engine - Diagnostic", layout="wide")

DEFAULT_TEST_TICKERS = ["SYS", "OGDC", "HBL", "LUCK", "FFC"]


def normalize_ticker(raw):
    t = raw.strip().upper()
    if not t.endswith(".KA"):
        t = t + ".KA"
    return t


def fetch_ohlcv(ticker, period="1y"):
    symbol = normalize_ticker(ticker)
    try:
        data = yf.download(symbol, period=period, progress=False)
        if data is None or data.empty:
            return None, "EMPTY", "No data returned (ticker unavailable on this provider, or blocked/rate-limited)."
        return data, "SUCCESS", None
    except Exception as e:
        return None, "EXCEPTION", str(type(e).__name__) + ": " + str(e)


def test_single_ticker(raw_ticker, period="1y"):
    symbol = normalize_ticker(raw_ticker)
    data, status, error = fetch_ohlcv(raw_ticker, period=period)

    row = {
        "Ticker": symbol,
        "Status": "SUCCESS" if status == "SUCCESS" else "FAILED",
        "Rows": None,
        "First Date": None,
        "Last Date": None,
        "Latest Close": None,
        "Latest Volume": None,
        "Error": error if error else "",
    }

    if status == "SUCCESS":
        row["Rows"] = len(data)
        row["First Date"] = str(data.index[0].date())
        row["Last Date"] = str(data.index[-1].date())
        try:
            row["Latest Close"] = round(float(data["Close"].iloc[-1]), 2)
            row["Latest Volume"] = int(data["Volume"].iloc[-1])
        except Exception:
            pass
    return row


st.title("PSX Quant Engine")
st.caption("Phase 0 - Data Connectivity Diagnostic")

st.markdown(
    "Yeh sirf ek connectivity test hai. Iska result confirm karega ke yfinance "
    "PSX tickers ke liye deployed (open-internet) environment mein kaam karta hai ya nahi, "
    "us ke baad hi poora Quant Engine (signals, breakout, portfolio) build hoga."
)

col1, col2 = st.columns([3, 1])
with col1:
    custom_tickers = st.text_input(
        "Tickers (comma-separated, .KA optional)",
        value=", ".join(DEFAULT_TEST_TICKERS),
    )
with col2:
    period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1)

run_test = st.button("Test Data Connection", type="primary", use_container_width=True)

if run_test:
    tickers = [t.strip() for t in custom_tickers.split(",") if t.strip()]
    results = []
    progress = st.progress(0.0, text="Testing tickers...")

    for i, t in enumerate(tickers):
        results.append(test_single_ticker(t, period=period))
        progress.progress((i + 1) / len(tickers), text="Tested " + t)

    progress.empty()

    df = pd.DataFrame(results)
    success_count = (df["Status"] == "SUCCESS").sum()

    if success_count >= 3:
        st.success(str(success_count) + "/" + str(len(tickers)) + " tickers returned live data. yfinance is usable - Quant Engine can proceed.")
    elif success_count > 0:
        st.warning("Only " + str(success_count) + "/" + str(len(tickers)) + " tickers worked. Partial coverage - fallback provider strongly recommended.")
    else:
        st.error("0 tickers returned data. yfinance is not serving PSX data from this environment - need a fallback provider before building the full engine.")

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption("Test run at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " - period=" + period)

    with st.expander("Raw error detail (for debugging)"):
        for r in results:
            if r["Error"]:
                st.code(r["Ticker"] + ": " + r["Error"])

st.divider()
st.caption("Full PSX Quant Engine (signals, breakout/pullback detection, position sizing, portfolio tracker) will be added here once this diagnostic confirms reliable data access. Signals shown by this app, once built, are analytical outputs, not guaranteed investment advice.")
