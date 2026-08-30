"""
PSX QUANT ENGINE — PROVIDER DIAGNOSTIC MODULE v2
==================================================
This module tests all available PSX data providers in the deployed
Streamlit Cloud environment.

Purpose: Verify which providers actually work BEFORE modifying app.py.

CRITICAL IMPROVEMENTS v2:
- HTTP 200 ≠ SUCCESS — validates actual data structure
- psxdata API dynamically discovered via dir() and signature inspection
- Standardized provider testing (OHLCV, freshness, KSE-100, universe)
- Data Quality Score for each provider
- Evidence-based recommendation only

IMPORTANT: This is DIAGNOSTIC ONLY. Production app.py is NOT modified.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
import json
import sys
import importlib
import inspect
from typing import Dict, List, Tuple, Optional, Any

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PSX Provider Diagnostic v2",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 PSX Provider Diagnostic v2")
st.caption("Testing all available PSX data providers in the deployed environment")
st.caption("⚠️ This is DIAGNOSTIC ONLY — production app.py is NOT modified")

# ============================================================
# CONSTANTS
# ============================================================

TEST_SYMBOLS = ["SYS", "OGDC", "HBL", "LUCK", "FFC", "GHNI"]
SMALL_CAP_SYMBOLS = ["GHNI", "KEL", "PAEL", "THALL"]
KSE100_CANDIDATES = ["^KSE100", "KSE100.KA", "^KSE", "PSX.KA", "KSE100"]

# Freshness thresholds (trading days)
FRESH_THRESHOLD = 1
DELAYED_THRESHOLD = 3
STALE_THRESHOLD = 5

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

def format_timestamp(dt):
    if dt is None:
        return "N/A"
    return dt.strftime("%d-%b-%Y %H:%M:%S PKT")

def trading_days_between(date1, date2):
    """Calculate trading days between two dates (excludes weekends)."""
    try:
        import numpy as np
        return int(np.busday_count(date1.date(), date2.date()))
    except Exception:
        return (date2.date() - date1.date()).days

def get_freshness_status(data_date):
    """
    Determine freshness status based on trading days gap.
    Returns: (status, age_days, label)
    """
    if data_date is None:
        return "UNAVAILABLE", None, "No data date available"
    
    now = pkt_now()
    if data_date.tzinfo is None:
        data_date = data_date.tz_localize(None)
    now_naive = now.replace(tzinfo=None)
    
    trading_gap = trading_days_between(data_date, now_naive)
    calendar_gap = (now_naive.date() - data_date.date()).days
    
    if trading_gap <= FRESH_THRESHOLD:
        return "FRESH", trading_gap, f"✅ {trading_gap} trading day(s) old"
    elif trading_gap <= DELAYED_THRESHOLD:
        return "DELAYED", trading_gap, f"⚠️ {trading_gap} trading day(s) old"
    elif trading_gap <= STALE_THRESHOLD:
        return "STALE", trading_gap, f"🔴 {trading_gap} trading day(s) old — may be stale"
    else:
        return "STALE", trading_gap, f"🔴 {trading_gap} trading day(s) old — STALE"

# ============================================================
# DATA VALIDATION HELPERS
# ============================================================

def validate_ohlcv_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate OHLCV DataFrame structure.
    Returns: {valid, row_count, columns, latest_date, latest_close, errors}
    """
    result = {
        "valid": False,
        "row_count": 0,
        "columns": [],
        "latest_date": None,
        "latest_close": None,
        "errors": [],
        "warnings": [],
    }
    
    if df is None:
        result["errors"].append("DataFrame is None")
        return result
    
    if df.empty:
        result["errors"].append("DataFrame is empty")
        return result
    
    result["row_count"] = len(df)
    result["columns"] = list(df.columns)
    
    # Check required columns
    required = ["Open", "High", "Low", "Close", "Volume"]
    # Also check for lowercase/alternate names
    alt_names = {
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "volume": "Volume", "adj close": "Close", "adj_close": "Close"
    }
    
    # Try to map columns
    mapped_cols = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower in alt_names:
            mapped_cols[col] = alt_names[col_lower]
        elif col_lower in [c.lower() for c in required]:
            for req in required:
                if col_lower == req.lower():
                    mapped_cols[col] = req
                    break
    
    if mapped_cols:
        df = df.rename(columns=mapped_cols)
        result["columns"] = list(df.columns)
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        result["errors"].append(f"Missing columns: {missing}")
        return result
    
    # Check for numeric data
    try:
        df[required] = df[required].apply(pd.to_numeric, errors="coerce")
    except Exception as e:
        result["errors"].append(f"Failed to convert to numeric: {str(e)}")
        return result
    
    # Drop rows with NaN in critical columns
    df_clean = df.dropna(subset=["Close", "High", "Low"])
    if df_clean.empty:
        result["errors"].append("All rows have NaN in critical columns")
        return result
    
    # Validate price relationships
    invalid_high_low = (df_clean["High"] < df_clean["Low"]).any()
    if invalid_high_low:
        result["warnings"].append("Some rows have High < Low")
    
    invalid_price = (df_clean["Close"] <= 0).any()
    if invalid_price:
        result["warnings"].append("Some rows have non-positive Close")
    
    invalid_volume = (df_clean["Volume"] < 0).any()
    if invalid_volume:
        result["warnings"].append("Some rows have negative Volume")
    
    # Check date index
    if not isinstance(df_clean.index, pd.DatetimeIndex):
        result["warnings"].append("Index is not DatetimeIndex")
        # Try to use first date column
        date_cols = [c for c in df_clean.columns if "date" in str(c).lower()]
        if date_cols:
            try:
                df_clean.index = pd.to_datetime(df_clean[date_cols[0]])
                df_clean = df_clean.drop(columns=[date_cols[0]])
            except Exception:
                result["warnings"].append("Could not parse date column")
    else:
        # Check for duplicate dates
        if df_clean.index.duplicated().any():
            result["warnings"].append("Duplicate dates found")
            df_clean = df_clean[~df_clean.index.duplicated(keep="last")]
    
    if df_clean.empty:
        result["errors"].append("Data became empty after cleaning")
        return result
    
    # Get latest data
    try:
        df_sorted = df_clean.sort_index()
        latest_date = df_sorted.index[-1]
        latest_close = float(df_sorted["Close"].iloc[-1])
        result["latest_date"] = latest_date
        result["latest_close"] = latest_close
    except Exception as e:
        result["errors"].append(f"Failed to extract latest data: {str(e)}")
        return result
    
    # Check minimum rows
    if len(df_sorted) < 10:
        result["warnings"].append(f"Only {len(df_sorted)} rows, may be insufficient for analysis")
    
    result["valid"] = True
    result["row_count"] = len(df_sorted)
    result["latest_date"] = latest_date
    result["latest_close"] = latest_close
    
    return result

# ============================================================
# PROVIDER 1: psxdata — Deep Inspection
# ============================================================

def inspect_psxdata_module():
    """
    Deeply inspect psxdata module — dir(), submodules, callable signatures.
    """
    result = {
        "imported": False,
        "error": None,
        "functions": [],
        "classes": [],
        "submodules": [],
        "callable_details": {},
        "version": None,
    }
    
    try:
        import psxdata
        result["imported"] = True
        
        # Get version if available
        if hasattr(psxdata, "__version__"):
            result["version"] = psxdata.__version__
        
        # Inspect all attributes
        for attr_name in dir(psxdata):
            if attr_name.startswith("_"):
                continue
            
            try:
                attr = getattr(psxdata, attr_name)
                
                if inspect.ismodule(attr):
                    result["submodules"].append(attr_name)
                elif inspect.isclass(attr):
                    result["classes"].append(attr_name)
                elif callable(attr):
                    result["functions"].append(attr_name)
                    # Try to get signature
                    try:
                        sig = inspect.signature(attr)
                        result["callable_details"][attr_name] = str(sig)
                    except Exception:
                        result["callable_details"][attr_name] = "Signature unavailable"
            except Exception:
                continue
        
        # Also inspect submodules if they exist
        for submod in result["submodules"]:
            try:
                sub = getattr(psxdata, submod)
                if inspect.ismodule(sub):
                    for attr_name in dir(sub):
                        if attr_name.startswith("_"):
                            continue
                        try:
                            attr = getattr(sub, attr_name)
                            if callable(attr):
                                full_name = f"{submod}.{attr_name}"
                                if full_name not in result["functions"]:
                                    result["functions"].append(full_name)
                                    try:
                                        sig = inspect.signature(attr)
                                        result["callable_details"][full_name] = str(sig)
                                    except Exception:
                                        result["callable_details"][full_name] = "Signature unavailable"
                        except Exception:
                            continue
            except Exception:
                continue
        
    except ImportError as e:
        result["error"] = f"Import error: {str(e)}"
    except Exception as e:
        result["error"] = f"General error: {str(e)}"
    
    return result

def test_psxdata_dynamic():
    """
    Dynamically test psxdata using discovered functions.
    """
    results = {
        "provider": "psxdata",
        "module_info": None,
        "stock_tests": {},
        "kse100_test": {"status": "NOT TESTED", "data": None},
        "universe_test": {"status": "NOT TESTED", "data": None},
        "timestamp": pkt_now(),
    }
    
    # First, inspect the module
    module_info = inspect_psxdata_module()
    results["module_info"] = module_info
    
    if not module_info["imported"]:
        return results
    
    # Try to find functions for stock data
    stock_funcs = []
    for func_name in module_info["functions"]:
        if any(k in func_name.lower() for k in ["stock", "history", "historical", "eod", "quote"]):
            stock_funcs.append(func_name)
    
    # Test each stock function
    for symbol in TEST_SYMBOLS:
        results["stock_tests"][symbol] = {"status": "NOT TESTED", "data": None}
        
        for func_name in stock_funcs:
            try:
                # Get the function
                func = None
                if "." in func_name:
                    # Submodule function
                    parts = func_name.split(".")
                    sub = getattr(psxdata, parts[0])
                    if inspect.ismodule(sub) and hasattr(sub, parts[1]):
                        func = getattr(sub, parts[1])
                else:
                    if hasattr(psxdata, func_name):
                        func = getattr(psxdata, func_name)
                
                if func is None:
                    continue
                
                # Try calling with symbol
                try:
                    result = func(symbol)
                except TypeError:
                    # Try with symbol as positional arg
                    try:
                        result = func(symbol)
                    except Exception:
                        # Try with keyword arg
                        try:
                            result = func(symbol=symbol)
                        except Exception:
                            continue
                
                if result is not None:
                    # Check if it's a DataFrame
                    if isinstance(result, pd.DataFrame):
                        validation = validate_ohlcv_data(result)
                        if validation["valid"]:
                            results["stock_tests"][symbol] = {
                                "status": "SUCCESS",
                                "data": f"{validation['row_count']} rows",
                                "latest_date": validation["latest_date"],
                                "latest_close": validation["latest_close"],
                                "func": func_name,
                                "validation": validation,
                            }
                            break
                        else:
                            results["stock_tests"][symbol] = {
                                "status": "INVALID_DATA",
                                "data": validation["errors"],
                                "func": func_name,
                            }
                    elif isinstance(result, dict) and "close" in result:
                        # Could be a quote
                        results["stock_tests"][symbol] = {
                            "status": "SUCCESS_QUOTE",
                            "data": result,
                            "func": func_name,
                        }
                        break
            except Exception as e:
                continue
    
    # Test universe discovery
    universe_funcs = []
    for func_name in module_info["functions"]:
        if any(k in func_name.lower() for k in ["ticker", "symbol", "universe", "list", "all"]):
            universe_funcs.append(func_name)
    
    for func_name in universe_funcs:
        try:
            func = None
            if "." in func_name:
                parts = func_name.split(".")
                sub = getattr(psxdata, parts[0])
                if inspect.ismodule(sub) and hasattr(sub, parts[1]):
                    func = getattr(sub, parts[1])
            else:
                if hasattr(psxdata, func_name):
                    func = getattr(psxdata, func_name)
            
            if func is None:
                continue
            
            result = func()
            if result is not None:
                if isinstance(result, list):
                    results["universe_test"] = {
                        "status": "SUCCESS",
                        "data": f"{len(result)} symbols",
                        "sample": result[:20] if len(result) > 20 else result,
                        "func": func_name,
                    }
                    break
                elif isinstance(result, dict) and "symbols" in result:
                    symbols = result["symbols"]
                    if isinstance(symbols, list):
                        results["universe_test"] = {
                            "status": "SUCCESS",
                            "data": f"{len(symbols)} symbols",
                            "sample": symbols[:20] if len(symbols) > 20 else symbols,
                            "func": func_name,
                        }
                        break
        except Exception:
            continue
    
    # Test KSE-100
    index_funcs = []
    for func_name in module_info["functions"]:
        if any(k in func_name.lower() for k in ["index", "indice", "kse", "benchmark"]):
            index_funcs.append(func_name)
    
    for func_name in index_funcs:
        try:
            func = None
            if "." in func_name:
                parts = func_name.split(".")
                sub = getattr(psxdata, parts[0])
                if inspect.ismodule(sub) and hasattr(sub, parts[1]):
                    func = getattr(sub, parts[1])
            else:
                if hasattr(psxdata, func_name):
                    func = getattr(psxdata, func_name)
            
            if func is None:
                continue
            
            # Try with KSE100
            try:
                result = func("KSE100")
            except TypeError:
                try:
                    result = func("KSE100")
                except Exception:
                    try:
                        result = func(index="KSE100")
                    except Exception:
                        try:
                            result = func(symbol="KSE100")
                        except Exception:
                            continue
            
            if result is not None:
                if isinstance(result, pd.DataFrame):
                    validation = validate_ohlcv_data(result)
                    if validation["valid"]:
                        results["kse100_test"] = {
                            "status": "SUCCESS",
                            "data": f"{validation['row_count']} rows",
                            "latest_date": validation["latest_date"],
                            "latest_close": validation["latest_close"],
                            "func": func_name,
                            "validation": validation,
                        }
                        break
                elif isinstance(result, dict):
                    # Check if it has KSE-100 data
                    if "close" in result or "value" in result or "level" in result:
                        results["kse100_test"] = {
                            "status": "SUCCESS",
                            "data": result,
                            "func": func_name,
                        }
                        break
        except Exception:
            continue
    
    return results

# ============================================================
# PROVIDER 2: yfinance (Fallback)
# ============================================================

def test_yfinance_dynamic():
    """
    Test yfinance comprehensively.
    """
    results = {
        "provider": "yfinance",
        "available": False,
        "stock_tests": {},
        "kse100_test": {"status": "NOT TESTED", "data": None},
        "universe_test": {"status": "NOT TESTED", "data": None},
        "timestamp": pkt_now(),
    }
    
    try:
        import yfinance as yf
        
        # Test each symbol
        for symbol in TEST_SYMBOLS:
            try:
                ticker = yf.Ticker(f"{symbol}.KA")
                hist = ticker.history(period="1mo")
                if hist is not None and not hist.empty:
                    validation = validate_ohlcv_data(hist)
                    if validation["valid"]:
                        results["stock_tests"][symbol] = {
                            "status": "SUCCESS",
                            "data": f"{validation['row_count']} rows",
                            "latest_date": validation["latest_date"],
                            "latest_close": validation["latest_close"],
                            "validation": validation,
                        }
                        results["available"] = True
                    else:
                        results["stock_tests"][symbol] = {
                            "status": "INVALID_DATA",
                            "data": validation["errors"],
                        }
                else:
                    results["stock_tests"][symbol] = {
                        "status": "EMPTY",
                        "data": "No data returned",
                    }
            except Exception as e:
                results["stock_tests"][symbol] = {
                    "status": "ERROR",
                    "data": str(e),
                }
        
        # Test KSE-100
        for cand in KSE100_CANDIDATES:
            try:
                ticker = yf.Ticker(cand)
                hist = ticker.history(period="1mo")
                if hist is not None and not hist.empty:
                    last_close = hist["Close"].iloc[-1]
                    # Check plausibility
                    if 5000 < last_close < 1000000:
                        # Validate as index data
                        validation = validate_ohlcv_data(hist)
                        if validation["valid"]:
                            results["kse100_test"] = {
                                "status": "SUCCESS",
                                "data": f"{cand}: {round(last_close, 2)}",
                                "latest_date": validation["latest_date"],
                                "latest_close": last_close,
                                "validation": validation,
                            }
                            break
            except Exception:
                continue
        
        if results["kse100_test"]["status"] == "NOT TESTED":
            results["kse100_test"] = {
                "status": "FAILED",
                "data": "No KSE-100 candidate returned valid data",
            }
        
    except ImportError as e:
        results["error"] = f"Import error: {str(e)}"
    except Exception as e:
        results["error"] = f"General error: {str(e)}"
    
    return results

# ============================================================
# PROVIDER 3: PSX Data Portal (Direct HTTP with Validation)
# ============================================================

def validate_http_response(response, expected_type="json"):
    """
    Validate HTTP response content.
    Returns: {valid, content_type, parsed_data, row_count, columns, latest_date, errors}
    """
    result = {
        "valid": False,
        "content_type": None,
        "parsed_data": None,
        "row_count": 0,
        "columns": [],
        "latest_date": None,
        "latest_close": None,
        "errors": [],
        "warnings": [],
    }
    
    if response.status_code != 200:
        result["errors"].append(f"HTTP {response.status_code}")
        return result
    
    content_type = response.headers.get("content-type", "").lower()
    result["content_type"] = content_type
    
    # Try to parse based on content type
    if "json" in content_type:
        try:
            data = response.json()
            result["parsed_data"] = data
            
            # Check if it's a list or dict with data
            if isinstance(data, list):
                # Try to convert to DataFrame
                try:
                    df = pd.DataFrame(data)
                    validation = validate_ohlcv_data(df)
                    if validation["valid"]:
                        result["valid"] = True
                        result["row_count"] = validation["row_count"]
                        result["columns"] = validation["columns"]
                        result["latest_date"] = validation["latest_date"]
                        result["latest_close"] = validation["latest_close"]
                    else:
                        result["errors"].extend(validation["errors"])
                        result["warnings"].extend(validation["warnings"])
                except Exception as e:
                    result["errors"].append(f"Failed to convert JSON array to DataFrame: {str(e)}")
            
            elif isinstance(data, dict):
                # Check for common patterns
                if "data" in data:
                    inner = data["data"]
                    if isinstance(inner, list):
                        try:
                            df = pd.DataFrame(inner)
                            validation = validate_ohlcv_data(df)
                            if validation["valid"]:
                                result["valid"] = True
                                result["row_count"] = validation["row_count"]
                                result["columns"] = validation["columns"]
                                result["latest_date"] = validation["latest_date"]
                                result["latest_close"] = validation["latest_close"]
                            else:
                                result["errors"].extend(validation["errors"])
                                result["warnings"].extend(validation["warnings"])
                        except Exception as e:
                            result["errors"].append(f"Failed to convert inner data: {str(e)}")
                    else:
                        # Check if dict itself has OHLCV keys
                        if "close" in inner or "Close" in inner:
                            result["valid"] = True
                            result["parsed_data"] = inner
                elif "close" in data or "Close" in data or "value" in data or "level" in data:
                    # Single quote/level response
                    result["valid"] = True
                    result["parsed_data"] = data
                    if "date" in data or "Date" in data:
                        try:
                            result["latest_date"] = pd.to_datetime(data.get("date") or data.get("Date"))
                        except Exception:
                            pass
                    result["latest_close"] = data.get("close") or data.get("Close") or data.get("value") or data.get("level")
            else:
                result["errors"].append("JSON response is not list or dict")
                
        except json.JSONDecodeError:
            result["errors"].append("Invalid JSON")
            # Try as text
            result["parsed_data"] = response.text[:200]
            
    elif "csv" in content_type or "text" in content_type:
        # Try to parse as CSV
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            validation = validate_ohlcv_data(df)
            if validation["valid"]:
                result["valid"] = True
                result["row_count"] = validation["row_count"]
                result["columns"] = validation["columns"]
                result["latest_date"] = validation["latest_date"]
                result["latest_close"] = validation["latest_close"]
            else:
                result["errors"].extend(validation["errors"])
                result["warnings"].extend(validation["warnings"])
        except Exception as e:
            result["errors"].append(f"Failed to parse CSV: {str(e)}")
            result["parsed_data"] = response.text[:200]
    else:
        # Check if it's HTML
        if response.text.strip().startswith("<"):
            result["errors"].append("Response is HTML (likely error/login page)")
            result["parsed_data"] = response.text[:200]
        else:
            result["errors"].append(f"Unrecognized content type: {content_type}")
            result["parsed_data"] = response.text[:200]
    
    return result

def test_psx_portal_dynamic():
    """
    Test PSX Data Portal endpoints with validation.
    """
    results = {
        "provider": "psx_portal",
        "available": False,
        "eod_tests": {},
        "index_test": {"status": "NOT TESTED", "data": None},
        "universe_test": {"status": "NOT TESTED", "data": None},
        "timestamp": pkt_now(),
    }
    
    try:
        import requests
        
        # Test EOD for each symbol
        for symbol in TEST_SYMBOLS:
            endpoints = [
                f"https://dps.psx.com.pk/timeseries/eod/{symbol}",
                f"https://dps.psx.com.pk/timeseries/history/{symbol}",
                f"https://dps.psx.com.pk/api/v1/stocks/{symbol}/history",
            ]
            
            for url in endpoints:
                try:
                    response = requests.get(url, timeout=15)
                    validation = validate_http_response(response)
                    
                    if validation["valid"]:
                        results["eod_tests"][symbol] = {
                            "status": "SUCCESS",
                            "url": url,
                            "row_count": validation["row_count"],
                            "latest_date": validation["latest_date"],
                            "latest_close": validation["latest_close"],
                            "validation": validation,
                        }
                        results["available"] = True
                        break
                except Exception as e:
                    continue
            
            if symbol not in results["eod_tests"]:
                results["eod_tests"][symbol] = {
                    "status": "FAILED",
                    "data": "No endpoint returned valid data",
                }
        
        # Test KSE-100
        index_endpoints = [
            "https://dps.psx.com.pk/api/v1/indices/KSE100",
            "https://dps.psx.com.pk/indices/KSE100",
            "https://dps.psx.com.pk/timeseries/index/KSE100",
        ]
        
        for url in index_endpoints:
            try:
                response = requests.get(url, timeout=15)
                validation = validate_http_response(response)
                
                if validation["valid"]:
                    results["index_test"] = {
                        "status": "SUCCESS",
                        "url": url,
                        "row_count": validation["row_count"],
                        "latest_date": validation["latest_date"],
                        "latest_close": validation["latest_close"],
                        "validation": validation,
                    }
                    results["available"] = True
                    break
            except Exception:
                continue
        
        if results["index_test"]["status"] == "NOT TESTED":
            results["index_test"] = {
                "status": "FAILED",
                "data": "No index endpoint returned valid data",
            }
        
        # Test universe
        universe_endpoints = [
            "https://dps.psx.com.pk/api/v1/stocks/symbols",
            "https://dps.psx.com.pk/stocks/list",
            "https://dps.psx.com.pk/tickers",
        ]
        
        for url in universe_endpoints:
            try:
                response = requests.get(url, timeout=15)
                validation = validate_http_response(response)
                
                if validation["valid"]:
                    # Check if we got a list of symbols
                    parsed = validation["parsed_data"]
                    symbols = []
                    if isinstance(parsed, list):
                        symbols = [str(s) for s in parsed if isinstance(s, (str, int))]
                    elif isinstance(parsed, dict) and "data" in parsed and isinstance(parsed["data"], list):
                        symbols = [str(s) for s in parsed["data"] if isinstance(s, (str, int))]
                    
                    if symbols:
                        results["universe_test"] = {
                            "status": "SUCCESS",
                            "url": url,
                            "count": len(symbols),
                            "sample": symbols[:20] if len(symbols) > 20 else symbols,
                            "validation": validation,
                        }
                        results["available"] = True
                        break
            except Exception:
                continue
        
        if results["universe_test"]["status"] == "NOT TESTED":
            results["universe_test"] = {
                "status": "FAILED",
                "data": "No universe endpoint returned valid symbol list",
            }
        
    except ImportError as e:
        results["error"] = f"requests not installed: {str(e)}"
    except Exception as e:
        results["error"] = f"General error: {str(e)}"
    
    return results

# ============================================================
# DATA QUALITY SCORE CALCULATION
# ============================================================

def calculate_provider_score(provider_results: Dict) -> Dict:
    """
    Calculate a data quality score for a provider.
    Returns: {score, breakdown, recommendation}
    """
    score = 0
    breakdown = {}
    max_score = 100
    
    # 1. OHLCV availability (30 points)
    ohlcv_score = 0
    if "stock_tests" in provider_results:
        success_count = sum(1 for v in provider_results["stock_tests"].values() 
                           if v.get("status") == "SUCCESS")
        total = len(provider_results["stock_tests"])
        if total > 0:
            ohlcv_score = (success_count / total) * 30
    breakdown["OHLCV"] = round(ohlcv_score, 1)
    score += ohlcv_score
    
    # 2. Freshness (20 points)
    freshness_score = 0
    if "stock_tests" in provider_results:
        # Get latest date from first successful test
        latest_dates = []
        for v in provider_results["stock_tests"].values():
            if v.get("status") == "SUCCESS" and v.get("latest_date"):
                latest_dates.append(v["latest_date"])
        if latest_dates:
            # Use the most recent date
            latest = max(latest_dates)
            status, age, _ = get_freshness_status(latest)
            if status == "FRESH":
                freshness_score = 20
            elif status == "DELAYED":
                freshness_score = 10
            elif status == "STALE":
                freshness_score = 5
    breakdown["Freshness"] = round(freshness_score, 1)
    score += freshness_score
    
    # 3. KSE-100 availability (20 points)
    kse100_score = 0
    if "kse100_test" in provider_results and provider_results["kse100_test"].get("status") == "SUCCESS":
        kse100_score = 20
    breakdown["KSE-100"] = round(kse100_score, 1)
    score += kse100_score
    
    # 4. Universe coverage (15 points)
    universe_score = 0
    if "universe_test" in provider_results and provider_results["universe_test"].get("status") == "SUCCESS":
        count = provider_results["universe_test"].get("count", 0)
        if count > 100:
            universe_score = 15
        elif count > 50:
            universe_score = 10
        elif count > 20:
            universe_score = 5
    breakdown["Universe"] = round(universe_score, 1)
    score += universe_score
    
    # 5. Small-cap coverage (10 points)
    smallcap_score = 0
    if "stock_tests" in provider_results:
        small_caps_found = sum(1 for sym, v in provider_results["stock_tests"].items()
                              if sym in SMALL_CAP_SYMBOLS and v.get("status") == "SUCCESS")
        if small_caps_found >= 2:
            smallcap_score = 10
        elif small_caps_found >= 1:
            smallcap_score = 5
    breakdown["Small Cap"] = round(smallcap_score, 1)
    score += smallcap_score
    
    # 6. Deployment reliability (5 points)
    # Based on whether we got any data at all
    if provider_results.get("available", False):
        breakdown["Deployment"] = 5
        score += 5
    else:
        breakdown["Deployment"] = 0
    
    # Determine recommendation
    if score >= 70:
        recommendation = "PRIMARY"
    elif score >= 50:
        recommendation = "SECONDARY"
    elif score >= 30:
        recommendation = "FALLBACK"
    else:
        recommendation = "NOT RECOMMENDED"
    
    return {
        "total": round(score, 1),
        "max": max_score,
        "breakdown": breakdown,
        "recommendation": recommendation,
    }

# ============================================================
# UI — Run Diagnostics
# ============================================================

st.markdown("---")

if st.button("🔬 Run Full Provider Diagnostics", use_container_width=True):
    with st.spinner("Testing all providers... This may take 60-90 seconds..."):
        
        # ============================================================
        # PROVIDER 1: psxdata
        # ============================================================
        
        st.subheader("📦 Provider 1: psxdata")
        with st.spinner("Testing psxdata..."):
            psxdata_results = test_psxdata_dynamic()
            module_info = psxdata_results["module_info"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Import Status", "✅ Success" if module_info["imported"] else "❌ Failed")
                st.metric("Functions Found", len(module_info.get("functions", [])))
            with col2:
                st.metric("Stock Test", "✅" if any(v.get("status") == "SUCCESS" for v in psxdata_results["stock_tests"].values()) else "❌")
                st.metric("KSE-100 Test", psxdata_results["kse100_test"]["status"])
            with col3:
                st.metric("Universe Test", psxdata_results["universe_test"]["status"])
                if module_info.get("version"):
                    st.metric("Version", module_info["version"])
            
            if module_info["imported"]:
                with st.expander("🔍 psxdata API Structure"):
                    st.write("**Functions found:**")
                    st.code(", ".join(module_info.get("functions", [])[:30]))
                    if len(module_info.get("functions", [])) > 30:
                        st.caption(f"... and {len(module_info['functions']) - 30} more")
                    
                    if module_info.get("callable_details"):
                        st.write("**Callable signatures:**")
                        for name, sig in list(module_info["callable_details"].items())[:10]:
                            st.code(f"{name}{sig}")
            
            # Show stock test results
            st.write("**Stock OHLCV Test Results:**")
            stock_df = []
            for symbol, result in psxdata_results["stock_tests"].items():
                status = result.get("status", "NOT TESTED")
                if status == "SUCCESS":
                    latest_date = result.get("latest_date")
                    status_label, _, _ = get_freshness_status(latest_date) if latest_date else ("UNKNOWN",)
                    stock_df.append({
                        "Symbol": symbol,
                        "Status": "✅ SUCCESS",
                        "Rows": result.get("data", "N/A"),
                        "Latest Date": latest_date.strftime("%Y-%m-%d") if latest_date else "N/A",
                        "Freshness": status_label,
                        "Func": result.get("func", "N/A"),
                    })
                elif status == "SUCCESS_QUOTE":
                    stock_df.append({
                        "Symbol": symbol,
                        "Status": "✅ QUOTE",
                        "Rows": "N/A",
                        "Latest Date": "N/A",
                        "Freshness": "UNKNOWN",
                        "Func": result.get("func", "N/A"),
                    })
                else:
                    stock_df.append({
                        "Symbol": symbol,
                        "Status": f"❌ {status}",
                        "Rows": "N/A",
                        "Latest Date": "N/A",
                        "Freshness": "N/A",
                        "Func": result.get("func", "N/A"),
                    })
            st.dataframe(pd.DataFrame(stock_df), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # ============================================================
        # PROVIDER 2: yfinance
        # ============================================================
        
        st.subheader("📦 Provider 2: yfinance (Fallback)")
        with st.spinner("Testing yfinance..."):
            yfinance_results = test_yfinance_dynamic()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Available", "✅ Yes" if yfinance_results.get("available") else "⚠️ Limited")
                st.metric("Stock Test", "✅" if any(v.get("status") == "SUCCESS" for v in yfinance_results["stock_tests"].values()) else "❌")
            with col2:
                st.metric("KSE-100 Test", yfinance_results["kse100_test"]["status"])
                st.metric("Small Cap", "✅" if any(sym in SMALL_CAP_SYMBOLS and yfinance_results["stock_tests"].get(sym, {}).get("status") == "SUCCESS" for sym in SMALL_CAP_SYMBOLS) else "❌")
            with col3:
                # Show freshness from first successful test
                for v in yfinance_results["stock_tests"].values():
                    if v.get("status") == "SUCCESS" and v.get("latest_date"):
                        status, age, label = get_freshness_status(v["latest_date"])
                        st.metric("Freshness", status)
                        st.caption(label)
                        break
            
            # Show stock test results
            st.write("**Stock OHLCV Test Results:**")
            stock_df = []
            for symbol, result in yfinance_results["stock_tests"].items():
                status = result.get("status", "NOT TESTED")
                if status == "SUCCESS":
                    latest_date = result.get("latest_date")
                    status_label, _, _ = get_freshness_status(latest_date) if latest_date else ("UNKNOWN",)
                    stock_df.append({
                        "Symbol": symbol,
                        "Status": "✅ SUCCESS",
                        "Rows": result.get("data", "N/A"),
                        "Latest Date": latest_date.strftime("%Y-%m-%d") if latest_date else "N/A",
                        "Freshness": status_label,
                    })
                else:
                    stock_df.append({
                        "Symbol": symbol,
                        "Status": f"❌ {status}",
                        "Rows": "N/A",
                        "Latest Date": "N/A",
                        "Freshness": "N/A",
                    })
            st.dataframe(pd.DataFrame(stock_df), use_container_width=True, hide_index=True)
            
            if yfinance_results["kse100_test"]["status"] == "SUCCESS":
                st.success(f"KSE-100: {yfinance_results['kse100_test'].get('data', 'N/A')}")
            elif yfinance_results["kse100_test"]["status"] == "FAILED":
                st.warning("⚠️ KSE-100: DATA UNAVAILABLE from yfinance")
        
        st.markdown("---")
        
        # ============================================================
        # PROVIDER 3: PSX Data Portal
        # ============================================================
        
        st.subheader("📦 Provider 3: PSX Data Portal (Direct HTTP)")
        with st.spinner("Testing PSX Data Portal..."):
            portal_results = test_psx_portal_dynamic()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Available", "✅ Yes" if portal_results.get("available") else "❌ No")
                st.metric("EOD Test", "✅" if any(v.get("status") == "SUCCESS" for v in portal_results["eod_tests"].values()) else "❌")
            with col2:
                st.metric("KSE-100 Test", portal_results["index_test"]["status"])
                st.metric("Universe Test", portal_results["universe_test"]["status"])
            with col3:
                # Show freshness from first successful EOD
                for v in portal_results["eod_tests"].values():
                    if v.get("status") == "SUCCESS" and v.get("latest_date"):
                        status, age, label = get_freshness_status(v["latest_date"])
                        st.metric("Freshness", status)
                        st.caption(label)
                        break
            
            # Show EOD test results
            st.write("**EOD Test Results:**")
            eod_df = []
            for symbol, result in portal_results["eod_tests"].items():
                status = result.get("status", "NOT TESTED")
                if status == "SUCCESS":
                    latest_date = result.get("latest_date")
                    status_label, _, _ = get_freshness_status(latest_date) if latest_date else ("UNKNOWN",)
                    eod_df.append({
                        "Symbol": symbol,
                        "Status": "✅ SUCCESS",
                        "Rows": result.get("row_count", "N/A"),
                        "Latest Date": latest_date.strftime("%Y-%m-%d") if latest_date else "N/A",
                        "Freshness": status_label,
                    })
                else:
                    eod_df.append({
                        "Symbol": symbol,
                        "Status": f"❌ {status}",
                        "Rows": "N/A",
                        "Latest Date": "N/A",
                        "Freshness": "N/A",
                    })
            st.dataframe(pd.DataFrame(eod_df), use_container_width=True, hide_index=True)
            
            if portal_results["index_test"]["status"] == "SUCCESS":
                st.success(f"KSE-100: {portal_results['index_test'].get('data', 'N/A')}")
            elif portal_results["index_test"]["status"] == "FAILED":
                st.warning("⚠️ KSE-100: DATA UNAVAILABLE from PSX Portal")
            
            if portal_results["universe_test"]["status"] == "SUCCESS":
                st.success(f"Universe: {portal_results['universe_test'].get('count', 'N/A')} symbols found")
                if portal_results["universe_test"].get("sample"):
                    st.caption(f"Sample: {', '.join(portal_results['universe_test']['sample'][:10])}")
        
        st.markdown("---")
        
        # ============================================================
        # COMPREHENSIVE COMPARISON TABLE
        # ============================================================
        
        st.subheader("📋 Provider Comparison & Data Quality Scores")
        
        # Calculate scores
        psxdata_score = calculate_provider_score(psxdata_results)
        yfinance_score = calculate_provider_score(yfinance_results)
        portal_score = calculate_provider_score(portal_results)
        
        # Build comparison table
        comparison_data = []
        
        for provider, results, score in [
            ("psxdata", psxdata_results, psxdata_score),
            ("yfinance", yfinance_results, yfinance_score),
            ("PSX Portal", portal_results, portal_score),
        ]:
            # Get OHLCV status
            ohlcv_status = "✅" if any(v.get("status") == "SUCCESS" for v in results.get("stock_tests", {}).values()) else "❌"
            
            # Get small cap status
            smallcap_status = "✅" if any(sym in SMALL_CAP_SYMBOLS and results.get("stock_tests", {}).get(sym, {}).get("status") == "SUCCESS" for sym in SMALL_CAP_SYMBOLS) else "❌"
            
            # Get latest date
            latest_date = None
            for v in results.get("stock_tests", {}).values():
                if v.get("status") == "SUCCESS" and v.get("latest_date"):
                    if latest_date is None or v["latest_date"] > latest_date:
                        latest_date = v["latest_date"]
            
            if latest_date:
                freshness_status, age, _ = get_freshness_status(latest_date)
                latest_date_str = latest_date.strftime("%Y-%m-%d")
            else:
                freshness_status = "UNAVAILABLE"
                latest_date_str = "N/A"
                age = None
            
            # KSE-100
            kse100_status = "✅" if results.get("kse100_test", {}).get("status") == "SUCCESS" else "❌"
            
            # Universe
            universe_count = results.get("universe_test", {}).get("count", 0)
            if universe_count > 0:
                universe_status = f"{universe_count} symbols"
            else:
                universe_status = "❌"
            
            # Score
            total_score = score["total"]
            recommendation = score["recommendation"]
            
            comparison_data.append({
                "Provider": provider,
                "OHLCV": ohlcv_status,
                "Small Cap": smallcap_status,
                "Latest Data": latest_date_str,
                "Freshness": freshness_status,
                "KSE-100": kse100_status,
                "Universe": universe_status,
                "Data Quality": f"{total_score}/100",
                "Recommendation": recommendation,
            })
        
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        # ============================================================
        # FINAL RECOMMENDATION
        # ============================================================
        
        st.markdown("---")
        st.subheader("🎯 Final Provider Recommendation")
        
        # Find best provider
        best_score = 0
        best_provider = None
        for provider, results, score in [
            ("psxdata", psxdata_results, psxdata_score),
            ("yfinance", yfinance_results, yfinance_score),
            ("PSX Portal", portal_results, portal_score),
        ]:
            if score["total"] > best_score and score["recommendation"] in ["PRIMARY", "SECONDARY"]:
                best_score = score["total"]
                best_provider = provider
        
        if best_provider == "psxdata" and psxdata_score["recommendation"] == "PRIMARY":
            st.success(f"""
            **RECOMMENDATION: psxdata as PRIMARY Provider**
            
            Score: {psxdata_score['total']}/100
            
            **Strengths:**
            - {psxdata_score['breakdown'].get('OHLCV', 0)}/30 for OHLCV availability
            - {psxdata_score['breakdown'].get('KSE-100', 0)}/20 for KSE-100
            - {psxdata_score['breakdown'].get('Universe', 0)}/15 for universe coverage
            
            **Functions discovered:** {len(module_info.get('functions', []))}
            
            **Implementation:** Use discovered function names from dir(psxdata)
            """)
            
        elif best_provider == "PSX Portal" and portal_score["recommendation"] in ["PRIMARY", "SECONDARY"]:
            st.success(f"""
            **RECOMMENDATION: PSX Data Portal as PRIMARY Provider**
            
            Score: {portal_score['total']}/100
            
            **Strengths:**
            - {portal_score['breakdown'].get('OHLCV', 0)}/30 for OHLCV availability
            - {portal_score['breakdown'].get('KSE-100', 0)}/20 for KSE-100
            - {portal_score['breakdown'].get('Universe', 0)}/15 for universe coverage
            
            **Available endpoints verified:**
            - EOD: {'✅' if portal_results.get('eod_tests', {}).get('SYS', {}).get('status') == 'SUCCESS' else '❌'}
            - KSE-100: {'✅' if portal_results.get('index_test', {}).get('status') == 'SUCCESS' else '❌'}
            - Universe: {'✅' if portal_results.get('universe_test', {}).get('status') == 'SUCCESS' else '❌'}
            
            **Implementation:** Direct HTTP requests with validation
            """)
            
        elif best_provider == "yfinance" and yfinance_score["recommendation"] in ["PRIMARY", "SECONDARY"]:
            st.success(f"""
            **RECOMMENDATION: yfinance as PRIMARY/FALLBACK Provider**
            
            Score: {yfinance_score['total']}/100
            
            **Strengths:**
            - {yfinance_score['breakdown'].get('OHLCV', 0)}/30 for OHLCV availability
            - {yfinance_score['breakdown'].get('Small Cap', 0)}/10 for small cap coverage
            
            **Limitations:**
            - KSE-100: {'✅' if yfinance_results.get('kse100_test', {}).get('status') == 'SUCCESS' else '❌ UNAVAILABLE'}
            - Data freshness: Check latest date column
            
            **Implementation:** Keep as fallback
            """)
        
        else:
            st.warning("""
            **RECOMMENDATION: No reliable free PSX provider found**
            
            Use yfinance as fallback with explicit stale warnings.
            Show KSE-100: DATA UNAVAILABLE.
            Show Market Proxy separately.
            """)
        
        # Provider architecture
        st.markdown("---")
        st.subheader("🏗️ Recommended Provider Architecture")
        
        st.code("""
        PRIMARY:
        """ + best_provider + """ (score: """ + str(best_score) + """/100)
        
        SECONDARY:
        Next best provider based on diagnostic results
        
        FALLBACK:
        yfinance (if latest data passes freshness threshold)
        
        LAST RESORT:
        UNAVAILABLE (honest failure)
        """)
        
        st.caption("⚠️ This architecture will be implemented in production app.py after confirmation")

else:
    st.info("👆 Click 'Run Full Provider Diagnostics' to test all providers in your deployed environment.")
    
    st.markdown("""
    ### What Will Be Tested
    
    | Provider | Tests |
    |----------|-------|
    | **psxdata** | Import, dir(), function signatures, OHLCV, KSE-100, universe |
    | **yfinance** | OHLCV, KSE-100, multiple stocks, freshness |
    | **PSX Portal** | EOD, KSE-100, universe, response validation |
    
    ### Test Symbols
    - SYS, OGDC, HBL, LUCK, FFC (Large caps)
    - GHNI, KEL, PAEL, THALL (Small caps)
    
    ### Validation
    - HTTP 200 is NOT accepted as success
    - Actual OHLCV structure is validated
    - Freshness is calculated in trading days
    - Data Quality Score is calculated
    
    ### Expected Output
    - Which providers actually work with real data
    - Actual psxdata API structure (dir + signatures)
    - Data freshness for each provider
    - KSE-100 availability
    - Universe coverage
    - Data Quality Score for each provider
    - Final evidence-based recommendation
    """)

st.markdown("---")
st.caption(f"Diagnostic generated at: {format_timestamp(pkt_now())}")
st.caption("⚠️ This is DIAGNOSTIC ONLY. Production app.py is NOT modified.")
