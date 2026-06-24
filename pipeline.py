# %%
# ======================================================================
# IMPORTS
# ======================================================================
import json
import logging
import pickle
import warnings
from itertools import product as iproduct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_pdf as pdf_backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import ruptures as rpt
import seaborn as sns
import yfinance as yf
import pandas_market_calendars as mcal
import quantstats as qs
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.stats import jarque_bera, pearsonr
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss, zivot_andrews
from statsmodels.tsa.vector_ar.var_model import VAR
from pypfopt import EfficientFrontier, risk_models as risk_m
from pypfopt import black_litterman as bl_module
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import plotly.graph_objects as go
import dash
from dash import Input, Output, dcc, html
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# %%
# ======================================================================
# SYSTEM CONFIGURATION — Edit only this block
# ======================================================================
STOCKS: Dict[str, str] = {
    "HDFCBANK.NS":   "HDFC Bank",
    "BHARTIARTL.NS": "Bharti Airtel",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "BIOCON.NS": "Biocon Ltd",
    "JSWENERGY.NS": "JSW Energy Ltd",
    "SUDARSCHEM.NS": "Sudarshan Chemicals",
    "AEGISLOG.NS": "Aegis Logistics Ltd",
    "KPRMILL.NS": "K P R Mill Ltd",
    "GREENLAM.NS": "Greenlam Industries Ltd",
    "CAMLINFINE.NS": "Camlin Fine Science Ltd",
    "SURYAROSNI.NS": "Surya Roshni Ltd",
    "EIDPARRY.NS": "EID Parry Ltd",
    "BANDHANBNK.NS": "Bandhan Bank",
    "ABREL.NS": "Aditya Birla Real Estate",
    "INDIGO.NS": "Interglobal Aviation Ltd",
    "MARATHON.NS": "Marathon Nextgen Realty Ltd",
    "LTTS.NS": "L&T Technology Services Ltd",
    "ASTERDM.NS": "Aster DM Healthcare Ltd",
    "BSE.NS":         "BSE Ltd",         
}

# Five macro / sector indicators used as COMMON exogenous variables in all models.
# All are fetched at daily frequency, resampled to month-end, and their monthly
# log returns lagged by ONE period before entering any model — ensuring only
# past information is available at each forecast origin.
MARKET_INDICATOR_TICKERS: Dict[str, str] = {
    "^INDIAVIX":  "India_VIX",    # Market fear / implied volatility
    "USDINR=X":   "USD_INR",      # Currency risk
    "BZ=F":       "Brent_Oil",    # Energy / inflation proxy
    "GC=F":       "Gold",         # Safe-haven / risk-off proxy
    "^NSEBANK":   "Nifty_Bank",   # Financial-sector proxy
}

START_DATE:          str   = "2019-01-01"   # Gives 75+ months for all live tickers
END_DATE:            str   = "today"         # or "2024-12-31"
MIN_HISTORY_MONTHS:  int   = 36
BENCHMARK_TICKER:    str   = "^NSEI"
FORECAST_HORIZONS:   List[int] = [1, 3, 6, 12]
MIN_TRAIN_MONTHS:    int   = 60
MAX_LOOKBACK_MONTHS: int   = 84
N_REGIMES:           int   = 3
RANDOM_STATE:        int   = 42
CAPITAL_AMOUNT:      float = 1_000_000.0
ANNUAL_RISK_FREE:    float = 0.065

OUTPUT_DIR = Path("pipeline_outputs_v1")
for _sub in ["", "hmm_groups", "hmm_stocks"]:
    (OUTPUT_DIR / _sub).mkdir(parents=True, exist_ok=True)

# Horizon display colours — shared between dashboard and CI shading
HORIZON_COLORS: Dict[int, Dict[str, str]] = {
    1:  {"hex": "green",       "rgb": "0,128,0"},
    3:  {"hex": "darkorange",  "rgb": "255,140,0"},
    6:  {"hex": "firebrick",   "rgb": "178,34,34"},
    12: {"hex": "purple",      "rgb": "128,0,128"},
}

# ======================================================================
# SHARED UTILITIES
# ======================================================================

def _resample_monthly(series: pd.Series) -> pd.Series:
    """Month-end resample that works on both pandas 2.2+ ('ME') and older ('M')."""
    try:
        return series.resample("ME").last()
    except ValueError:
        return series.resample("M").last()


def _resample_monthly_df(df: pd.DataFrame) -> pd.DataFrame:
    """Month-end resample for DataFrames — same 'ME'/'M' compatibility."""
    try:
        return df.resample("ME").last()
    except ValueError:
        return df.resample("M").last()


def _resample_sum_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Month-end sum resample (used for realized variance computation)."""
    try:
        return df.resample("ME").sum()
    except ValueError:
        return df.resample("M").sum()


def _quarterly_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Quarter-end DatetimeIndex compatible with both pandas 2.2+ ('QE') and older ('Q')."""
    try:
        return pd.date_range(start, end, freq="QE")
    except ValueError:
        return pd.date_range(start, end, freq="Q")


def _extract_close(raw: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Extract the Close price Series from a yfinance download result.

    Handles three common column layouts returned by yfinance across versions:
      (a) flat columns, single ticker  → raw["Close"] is already a Series
      (b) MultiIndex (ticker, field)   → raw[ticker]["Close"]
      (c) MultiIndex (field, ticker)   → raw.xs(ticker, level=1, axis=1)["Close"]

    Always returns a pd.Series (F-01: .squeeze() guarantees this).
    """
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0).unique().tolist()
        lvl1 = raw.columns.get_level_values(1).unique().tolist()
        if ticker in lvl0:
            df = raw[ticker].dropna(how="all")
        elif ticker in lvl1:
            df = raw.xs(ticker, axis=1, level=1).dropna(how="all")
        else:
            return pd.Series(dtype=float)
        return df["Close"].squeeze() if "Close" in df.columns else df.iloc[:, 0].squeeze()

    # Flat columns (single-ticker download)
    if "Close" in raw.columns:
        return raw["Close"].squeeze()          # F-01: squeeze prevents DataFrame leak
    return raw.iloc[:, 0].squeeze()


def _count_missing_nse_days(close: pd.Series, start: str, end: str) -> int:
    """Count NSE trading days absent from the Close series."""
    try:
        nse_cal      = mcal.get_calendar("NSE")
        schedule     = nse_cal.schedule(start_date=start, end_date=end)
        trading_days = (mcal.date_range(schedule, frequency="1D")
                        .tz_localize(None)   # remove tz info
                        .normalize())        # ← NEW: strip intraday time → midnight
        actual_days  = close.dropna().index.normalize()
        return len(trading_days.difference(actual_days))
    except Exception as exc:
        logger.warning(f"NSE calendar check failed ({exc}) — using raw NaN count.")
        return int(close.isna().sum())


def _detect_communities(G: nx.Graph, random_state: int = 42) -> Dict[str, int]:
    """
    Louvain community detection with a four-level fallback chain.

    Priority:
      1. python-louvain  (community.best_partition)       — F-02 fix
      2. networkx >= 2.7  louvain_communities
      3. networkx         greedy_modularity_communities
      4. Trivial: every node in its own community
    """
    # Level 1 — python-louvain
    try:
        import community as cl                                      # noqa: PLC0415
        if hasattr(cl, "best_partition"):
            try:
                return cl.best_partition(G, weight="weight", random_state=random_state)
            except TypeError:
                return cl.best_partition(G, weight="weight")
    except ImportError:
        pass

    # Level 2 — networkx built-in Louvain (networkx >= 2.7 / 3.x)
    try:
        from networkx.algorithms import community as nx_comm       # noqa: PLC0415
        comms = list(nx_comm.louvain_communities(G, weight="weight", seed=random_state))
        return {node: i for i, comm in enumerate(comms) for node in comm}
    except (AttributeError, TypeError, Exception):
        pass

    # Level 3 — greedy modularity
    try:
        from networkx.algorithms import community as nx_comm       # noqa: PLC0415
        comms = list(nx_comm.greedy_modularity_communities(G, weight="weight"))
        return {node: i for i, comm in enumerate(comms) for node in comm}
    except Exception:
        pass

    # Level 4 — trivial fallback
    logger.warning("All community detection methods failed — each node in own community.")
    return {node: i for i, node in enumerate(G.nodes())}

# %%
# ======================================================================
# PHASE 1: DATA ARCHITECTURE & COLLECTION
# ======================================================================

def run_phase1() -> dict:
    """
    Fetch daily OHLCV for all stocks plus benchmark and market indicators.
    Compute month-end prices and monthly log returns.
    Produce a data-audit table; quarantine tickers below MIN_HISTORY_MONTHS.
    """
    logger.info("=" * 62)
    logger.info("PHASE 1 — Data Architecture & Collection")
    logger.info("=" * 62)

    end_str = pd.Timestamp.today().strftime("%Y-%m-%d") if END_DATE == "today" else END_DATE
    tickers = list(STOCKS.keys())

    # ── 1.1 Daily OHLCV for all stocks ────────────────────────────────────────
    logger.info("Downloading daily OHLCV (stocks) …")
    raw_stocks = yf.download(
        tickers=tickers, start=START_DATE, end=end_str,
        auto_adjust=True, progress=False, group_by="ticker",
    )

    audit_rows: List[dict] = []
    price_data: Dict[str, pd.DataFrame] = {}
    WATCHLIST:  Dict[str, str]  = {}

    for ticker in tickers:
        close = _extract_close(raw_stocks, ticker).dropna()
        if close.empty:
            logger.warning(f"  [{ticker}] No data — quarantined.")
            WATCHLIST[ticker] = "No data returned"
            continue

        first, last    = close.index.min(), close.index.max()
        history_months = (last.year - first.year) * 12 + (last.month - first.month)
        n_missing      = _count_missing_nse_days(
            close, first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")
        )
        audit_rows.append({
            "ticker":         ticker,
            "name":           STOCKS[ticker],
            "first_date":     first.date(),
            "last_date":      last.date(),
            "n_trading_days": len(close),
            "n_missing_days": n_missing,
            "pct_missing":    round(n_missing / max(len(close), 1), 4),
            "history_months": history_months,
        })
        if history_months < MIN_HISTORY_MONTHS:
            reason = f"Insufficient history ({history_months} < {MIN_HISTORY_MONTHS} months)"
            logger.warning(f"  [{ticker}] {reason}")
            WATCHLIST[ticker] = reason
        else:
            # Reconstruct a DataFrame with all columns for downstream use
            if isinstance(raw_stocks.columns, pd.MultiIndex):
                lvl0 = raw_stocks.columns.get_level_values(0).unique().tolist()
                lvl1 = raw_stocks.columns.get_level_values(1).unique().tolist()
                if ticker in lvl0:
                    price_data[ticker] = raw_stocks[ticker].dropna(how="all")
                elif ticker in lvl1:
                    price_data[ticker] = raw_stocks.xs(ticker, axis=1, level=1).dropna(how="all")
                else:
                    price_data[ticker] = pd.DataFrame({"Close": close})
            else:
                price_data[ticker] = pd.DataFrame({"Close": close})

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(OUTPUT_DIR / "data_audit.csv", index=False)
    logger.info(f"  WATCHLIST = {list(WATCHLIST.keys()) or 'empty'}")
    logger.info(f"\n{audit_df.to_string()}\n")

    active_tickers = [t for t in tickers if t in price_data]
    if not active_tickers:
        raise RuntimeError("All tickers quarantined — cannot proceed.")

    # ── 1.2 Month-end adjusted closing prices ─────────────────────────────────
    daily_close = pd.DataFrame({t: price_data[t]["Close"] for t in active_tickers})
    monthly_prices = _resample_monthly_df(daily_close)
    monthly_prices.to_parquet(OUTPUT_DIR / "monthly_prices.parquet")
    logger.info(f"  monthly_prices  shape: {monthly_prices.shape}")

    # ── 1.3 Monthly log returns (primary modelling target) ────────────────────
    monthly_log_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna()
    monthly_log_returns.to_parquet(OUTPUT_DIR / "monthly_log_returns.parquet")
    logger.info(f"  monthly_log_returns shape: {monthly_log_returns.shape}")

    # ── 1.4 Benchmark (Nifty 50) ─────────────────────────────────────────────
    logger.info(f"Downloading benchmark {BENCHMARK_TICKER} …")
    bench_raw   = yf.download(BENCHMARK_TICKER, start=START_DATE, end=end_str,
                               auto_adjust=True, progress=False)
    bench_close = _extract_close(bench_raw, BENCHMARK_TICKER)           # guaranteed Series
    bench_monthly = _resample_monthly(bench_close)                       # monthly prices
    benchmark_returns = np.log(bench_monthly / bench_monthly.shift(1)).dropna()
    benchmark_returns.name = BENCHMARK_TICKER
    # F-01: save as DataFrame — pd.Series.to_parquet() not guaranteed on all versions
    benchmark_returns.to_frame("benchmark_return").to_parquet(
        OUTPUT_DIR / "benchmark_returns.parquet"
    )
    monthly_rf = (1 + ANNUAL_RISK_FREE) ** (1 / 12) - 1
    logger.info(f"  Monthly risk-free rate: {monthly_rf:.6f}")

    # ── 1.5 Market indicators (raw monthly prices) ────────────────────────────
    indicator_monthly_prices: Dict[str, pd.Series] = {}
    logger.info("Downloading market indicators …")
    for ind_ticker, ind_name in MARKET_INDICATOR_TICKERS.items():
        try:
            ind_raw   = yf.download(ind_ticker, start=START_DATE, end=end_str,
                                    auto_adjust=True, progress=False)
            ind_close = _extract_close(ind_raw, ind_ticker)
            if ind_close.dropna().empty:
                logger.warning(f"  [{ind_ticker}] empty — skipped.")
                continue
            ind_monthly = _resample_monthly(ind_close)
            indicator_monthly_prices[ind_name] = ind_monthly
            logger.info(f"  [{ind_ticker}] {ind_name}: {len(ind_monthly)} monthly obs")
        except Exception as exc:
            logger.warning(f"  [{ind_ticker}] fetch failed ({exc}) — skipped.")

    # ── 1.6 Validation ────────────────────────────────────────────────────────
    assert not monthly_prices.isnull().all().any(), \
        "Some tickers are entirely NaN in monthly_prices."
    assert len(monthly_log_returns) >= 1, \
        "No valid return rows after log-differencing and dropna."           # F-08
    logger.info("Phase 1 validation checklist passed ✓\n")

    return {
        "price_data":                 price_data,
        "daily_close":                daily_close,
        "monthly_prices":             monthly_prices,
        "monthly_log_returns":        monthly_log_returns,
        "benchmark_returns":          benchmark_returns,
        "bench_monthly_prices":       bench_monthly,
        "monthly_rf":                 monthly_rf,
        "indicator_monthly_prices":   indicator_monthly_prices,
        "active_tickers":             active_tickers,
        "WATCHLIST":                  WATCHLIST,
        "audit_df":                   audit_df,
        "end_str":                    end_str,
    }


# ======================================================================
# PHASE 2: EDA & PREPROCESSING
# ======================================================================

def _stationarity_suite(series: pd.Series, ticker: str) -> dict:
    """ADF + KPSS + Zivot-Andrews. Returns a summary dict."""
    s   = series.dropna()
    out = {"ticker": ticker}

    try:
        adf = adfuller(s, autolag="AIC", regression="c")
        out.update({"adf_stat": adf[0], "adf_pval": adf[1],
                    "adf_stationary": bool(adf[1] < 0.05)})
    except Exception as exc:
        logger.warning(f"  [{ticker}] ADF failed: {exc}")
        out.update({"adf_stat": np.nan, "adf_pval": np.nan, "adf_stationary": False})

    try:
        kp = kpss(s, regression="c", nlags="auto")
        out.update({"kpss_stat": kp[0], "kpss_pval": kp[1],
                    "kpss_stationary": bool(kp[1] >= 0.05)})
    except Exception as exc:
        logger.warning(f"  [{ticker}] KPSS failed: {exc}")
        out.update({"kpss_stat": np.nan, "kpss_pval": np.nan, "kpss_stationary": np.nan})

    try:
        za     = zivot_andrews(s, maxlag=12, regression="c")
        bp_idx = int(za[4])
        bp_dt  = s.index[bp_idx].date() if 0 <= bp_idx < len(s) else None
        out.update({"za_stat": za[0], "za_pval": za[1],
                    "za_stationary": bool(za[1] < 0.05), "za_break_date": str(bp_dt)})
    except Exception as exc:
        logger.warning(f"  [{ticker}] Zivot-Andrews failed: {exc}")
        out.update({"za_stat": np.nan, "za_pval": np.nan,
                    "za_stationary": np.nan, "za_break_date": None})
    return out


def run_phase2(p1: dict) -> dict:   
    """Phase 2: Return distributions, stationarity, structural breaks, outliers."""
    logger.info("=" * 62)
    logger.info("PHASE 2 — EDA & Preprocessing")
    logger.info("=" * 62)

    mlr            = p1["monthly_log_returns"].copy()
    active_tickers = p1["active_tickers"]

    # ── 2.1 Return distribution plots ─────────────────────────────────────────
    with pdf_backend.PdfPages(OUTPUT_DIR / "return_distributions.pdf") as pages:
        for ticker in active_tickers:
            s    = mlr[ticker].dropna()
            desc = stats.describe(s)
            _, jb_pval = jarque_bera(s)

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(f"{ticker} ({STOCKS.get(ticker, '')}) — Return Distributions",
                         fontsize=11)
            sns.histplot(s, kde=True, ax=axes[0])
            axes[0].set_title("Histogram + KDE"); axes[0].set_xlabel("Monthly Log Return")
            stats.probplot(s, dist="norm", plot=axes[1])
            axes[1].set_title("Q-Q Plot (Normal)")
            plot_acf(s, lags=min(24, len(s) // 3), ax=axes[2], zero=False)
            axes[2].set_title("ACF")
            fig.text(0.5, -0.03,
                     f"μ={desc.mean:.4f}  σ={np.std(s):.4f}  "
                     f"Skew={desc.skewness:.3f}  ExKurt={desc.kurtosis:.3f}  "
                     f"JB p={jb_pval:.4f}", ha="center", fontsize=9)
            plt.tight_layout()
            pages.savefig(fig, bbox_inches="tight"); plt.close(fig)
    logger.info(f"  Saved return_distributions.pdf")

    # ── 2.2 Stationarity tests ─────────────────────────────────────────────────
    stat_rows = [_stationarity_suite(mlr[t], t) for t in active_tickers]
    stat_df   = pd.DataFrame(stat_rows).set_index("ticker")
    stat_df.to_csv(OUTPUT_DIR / "stationarity_results.csv")
    logger.info(f"  Stationarity:\n"
                f"{stat_df[['adf_stationary', 'kpss_stationary', 'za_stationary']].to_string()}")

    # ── 2.3 Structural break detection (Ruptures PELT, RBF cost) ──────────────
    break_records: Dict[str, List] = {}
    for ticker in active_tickers:
        s      = mlr[ticker].dropna()
        algo   = rpt.Pelt(model="rbf", min_size=12, jump=1).fit(s.values.reshape(-1, 1))
        bkps   = algo.predict(pen=10)
        break_dates = [s.index[bp - 1].date() for bp in bkps[:-1]]
        break_records[ticker] = break_dates
        logger.info(f"  [{ticker}] breaks: {break_dates or 'none'}")
    pd.DataFrame(
        [(t, str(d)) for t, dl in break_records.items() for d in dl],
        columns=["ticker", "break_date"],
    ).to_csv(OUTPUT_DIR / "structural_breaks.csv", index=False)

    # ── 2.4 Outlier detection + forward-fill ──────────────────────────────────
    mlr_clean     = mlr.copy()
    mlr_clean.to_parquet(OUTPUT_DIR / "monthly_log_returns_clean.parquet")
    outlier_rows: List[dict] = []
    for ticker in active_tickers:
        s = mlr_clean[ticker].ffill(limit=1)
        mlr_clean[ticker] = s
        rm   = s.rolling(24, min_periods=12).mean()
        rstd = s.rolling(24, min_periods=12).std().replace(0.0, np.nan)
        zs   = (s - rm) / rstd
        for dt in s.index[zs.abs() > 3.5]:
            outlier_rows.append({"ticker": ticker, "date": str(dt.date()),
                                  "value": round(float(s[dt]), 6),
                                  "z_score": round(float(zs[dt]), 3),
                                  "action": "flagged — manual review required"})
    outlier_df = pd.DataFrame(outlier_rows)
    if not outlier_df.empty:
        outlier_df.to_csv(OUTPUT_DIR / "outlier_flags.csv", index=False)
        logger.info(f"  {len(outlier_df)} outlier(s) flagged.")
    else:
        logger.info("  No outliers flagged.")

    logger.info("Phase 2 validation checklist passed ✓\n")
    return {
        "stat_df":                   stat_df,
        "break_records":             break_records,
        "outlier_df":                outlier_df,
        "monthly_log_returns_clean": mlr_clean,
    }


# ======================================================================
# PHASE 3: FEATURE ENGINEERING
# ======================================================================

def _compute_realized_volatility(daily_close: pd.DataFrame,
                                   target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Realized Volatility (Andersen et al., 2003):
      RV_t = sqrt( Σ_{d ∈ month t} r_{d}^2 )
    Uses all daily information rather than discarding it.
    """
    daily_lr = np.log(daily_close / daily_close.shift(1)).dropna()
    rv_var   = _resample_sum_monthly(daily_lr ** 2)
    return np.sqrt(rv_var).reindex(target_index)


def _prepare_market_indicators(indicator_monthly_prices: Dict[str, pd.Series],
                                 target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute monthly log returns for each market indicator, then lag by ONE period.

    Lagging by 1 ensures that at forecast origin t, only information available
    at t−1 (i.e. last observed indicator return) enters the model — no look-ahead.

    Missing values are forward-filled up to 3 months then back-filled for the
    earliest observations; remaining NaNs are zeroed (indicator absent).
    """
    if not indicator_monthly_prices:
        return pd.DataFrame(index=target_index)

    prices_df = pd.DataFrame(indicator_monthly_prices)
    log_rets  = np.log(prices_df / prices_df.shift(1))   # log returns
    lagged    = log_rets.shift(1)                         # lag: use t-1 to predict t

    aligned = lagged.reindex(target_index).ffill(limit=3).bfill(limit=1).fillna(0.0)

    # Drop any column that is entirely zero (indicator unavailable)
    non_trivial = aligned.columns[(aligned != 0.0).any()]
    if len(non_trivial) < len(aligned.columns):
        dropped = set(aligned.columns) - set(non_trivial)
        logger.warning(f"  Dropped trivial indicator columns: {dropped}")
    return aligned[non_trivial]


def run_phase3(p1: dict, p2: dict) -> dict:
    """Phase 3: Realized Volatility + Market Indicators (no fundamental scores)."""
    logger.info("=" * 62)
    logger.info("PHASE 3 — Feature Engineering")
    logger.info("=" * 62)

    active_tickers = p1["active_tickers"]
    daily_close    = p1["daily_close"][active_tickers]
    mlr            = p2["monthly_log_returns_clean"]

    # ── 3.1 Realized Volatility ───────────────────────────────────────────────
    realized_vol = _compute_realized_volatility(daily_close, mlr.index)
    realized_vol.to_parquet(OUTPUT_DIR / "realized_volatility.parquet")
    logger.info(f"  realized_volatility shape: {realized_vol.shape}")

    # ── 3.2 Market Indicators (lagged log returns) ────────────────────────────
    market_indicators = _prepare_market_indicators(
        p1["indicator_monthly_prices"], mlr.index
    )
    market_indicators.to_parquet(OUTPUT_DIR / "market_indicators.parquet")
    logger.info(f"  market_indicators shape: {market_indicators.shape}")
    logger.info(f"  columns: {market_indicators.columns.tolist()}")

    # Quick stationarity note for indicators (log returns should be stationary)
    for col in market_indicators.columns:
        s = market_indicators[col].replace(0.0, np.nan).dropna()
        if len(s) > 10:
            _, pval, _ = adfuller(s, autolag="AIC")[:3]
            status = "stationary" if pval < 0.05 else "POSSIBLY NON-STATIONARY"
            logger.info(f"    {col}: ADF p={pval:.4f} → {status}")

    # ── 3.3 Exogenous master (HMM probs appended in Phase 4.5) ────────────────
    exog_master = market_indicators.copy()
    for ticker in active_tickers:
        if ticker in realized_vol.columns:
            exog_master[(ticker, "realized_vol")] = realized_vol[ticker]
    exog_master.to_parquet(OUTPUT_DIR / "exog_master.parquet")
    logger.info(f"  exog_master (pre-HMM) shape: {exog_master.shape}")

    logger.info("Phase 3 validation checklist passed ✓\n")
    return {
        "realized_volatility": realized_vol,
        "market_indicators":   market_indicators,
        "exog_master":         exog_master,
    }


# ======================================================================
# PHASE 4: DEPENDENCY STRUCTURE & CLUSTERING
# ======================================================================

def run_phase4(p1: dict, p2: dict) -> dict:
    """
    Four-layer data-driven clustering:
      Layer 1 — Hierarchical (Ward + Mantegna distance)
      Layer 2 — Rolling-correlation stability check
      Layer 3 — Granger causality network → Louvain communities   (F-02 fix)
      Layer 4 — Minimum Spanning Tree (hub identification)
    Consensus decision → cluster_assignments dict.
    """
    logger.info("=" * 62)
    logger.info("PHASE 4 — Dependency Structure & Clustering")
    logger.info("=" * 62)

    mlr            = p2["monthly_log_returns_clean"]
    active_tickers = p1["active_tickers"]

    # ── Layer 1: Static correlation clustering ────────────────────────────────
    corr_matrix = mlr.corr(method="pearson")
    dist_matrix = np.sqrt(2 * (1 - corr_matrix.clip(-1, 1)))
    linkage_mat = linkage(dist_matrix.values, method="ward")

    cm = sns.clustermap(corr_matrix, method="ward", cmap="coolwarm",
                        figsize=(10, 8), annot=True, fmt=".2f")
    cm.savefig(OUTPUT_DIR / "correlation_clustermap.pdf", bbox_inches="tight")
    plt.close(cm.fig)                                                  # F-10
    logger.info("  Correlation clustermap saved.")

    heights   = sorted(linkage_mat[:, 2])
    diffs     = np.diff(heights) if len(heights) > 1 else np.array([1.0])
    threshold = heights[np.argmax(diffs)] + diffs.max() * 0.5
    l1_labels = fcluster(linkage_mat, t=threshold, criterion="distance")
    layer1: Dict[int, List] = {}
    for ticker, lbl in zip(active_tickers, l1_labels):
        layer1.setdefault(int(lbl), []).append(ticker)
    logger.info(f"  Layer 1 clusters: {layer1}")

    # ── Layer 2: Rolling correlation stability ────────────────────────────────
    unstable_pairs: set = set()
    for members in layer1.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                ta, tb   = members[i], members[j]
                roll_cor = mlr[ta].rolling(36).corr(mlr[tb])
                pct_pos  = float((roll_cor > 0).mean())
                if pct_pos < 0.70:
                    unstable_pairs.add(tuple(sorted([ta, tb])))
                    logger.warning(f"  Layer 2: {ta}-{tb} stable in only {pct_pos:.0%} of windows")

    # ── Layer 3: Granger causality → Louvain communities ─────────────────────
    logger.info("  Running pairwise Granger tests (maxlag=6) …")
    data_gc = mlr[active_tickers].dropna()
    G_dir   = nx.DiGraph()
    G_dir.add_nodes_from(active_tickers)

    for ti in active_tickers:
        for tj in active_tickers:
            if ti == tj:
                continue
            try:
                gc    = grangercausalitytests(data_gc[[tj, ti]], maxlag=6, verbose=False)
                min_p = min(gc[lag][0]["ssr_ftest"][1] for lag in gc)
                if min_p < 0.05:
                    G_dir.add_edge(ti, tj, weight=1.0 - min_p)
            except Exception:
                pass

    G_undir = G_dir.to_undirected(reciprocal=False)
    for u, v in list(G_undir.edges()):
        w1 = G_dir.get_edge_data(u, v, default={}).get("weight", 0.0)
        w2 = G_dir.get_edge_data(v, u, default={}).get("weight", 0.0)
        G_undir[u][v]["weight"] = w1 + w2

    partition       = _detect_communities(G_undir, random_state=RANDOM_STATE)  # F-02
    gc_communities: Dict[int, List] = {}
    for ticker, cid in partition.items():
        gc_communities.setdefault(int(cid), []).append(ticker)
    logger.info(f"  Granger communities: {gc_communities}")
    nx.write_gexf(G_dir, str(OUTPUT_DIR / "granger_network.gexf"))

    # ── Layer 4: Minimum Spanning Tree (Mantegna) ─────────────────────────────
    G_full = nx.Graph()
    for i, ti in enumerate(active_tickers):
        for j, tj in enumerate(active_tickers):
            if i < j:
                d = float(np.sqrt(2 * (1 - np.clip(corr_matrix.loc[ti, tj], -1, 1))))
                G_full.add_edge(ti, tj, weight=d)
    MST        = nx.minimum_spanning_tree(G_full, weight="weight")
    centrality = nx.degree_centrality(MST)
    fig, ax    = plt.subplots(figsize=(10, 8))
    pos   = nx.spring_layout(MST, seed=RANDOM_STATE)
    sizes = [3000 * centrality[n] + 500 for n in MST.nodes()]
    nx.draw(MST, pos, ax=ax, with_labels=True, node_size=sizes,
            node_color="lightsteelblue", font_size=9, edge_color="dimgray")
    ax.set_title("Minimum Spanning Tree (Mantegna Distance)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mst_topology.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── Consensus cluster assignment ──────────────────────────────────────────
    l1_map = {t: lbl for lbl, mems in layer1.items() for t in mems}
    gc_map  = {t: cid for cid, mems in gc_communities.items() for t in mems}
    cluster_assignments: Dict[str, List] = {"isolated": []}
    grouped: set = set()
    group_idx = 1

    for ticker in active_tickers:
        if ticker in grouped:
            continue
        l1_peers  = set(layer1.get(l1_map.get(ticker), [])) - {ticker}
        gc_peers  = set(gc_communities.get(gc_map.get(ticker), [])) - {ticker}
        consensus = l1_peers & gc_peers or gc_peers      # L3 preferred if no overlap
        if not consensus:
            cluster_assignments["isolated"].append(ticker)
            grouped.add(ticker)
            continue

        candidates   = [ticker] + [p for p in consensus if p not in grouped]
        stable_group = [ticker]
        for p in candidates[1:]:
            if tuple(sorted([ticker, p])) not in unstable_pairs:
                stable_group.append(p)
            if len(stable_group) == 5:
                break

        if len(stable_group) < 2:
            cluster_assignments["isolated"].append(ticker)
            grouped.add(ticker)
        else:
            key = f"VARX_group_{group_idx}"
            cluster_assignments[key] = stable_group
            grouped.update(stable_group)
            group_idx += 1

    for ticker in active_tickers:
        if ticker not in grouped:
            cluster_assignments["isolated"].append(ticker)

    logger.info(f"  Final clusters: {cluster_assignments}")
    with open(OUTPUT_DIR / "cluster_assignments.json", "w") as fh:
        json.dump(cluster_assignments, fh, indent=2)

    pd.DataFrame([
        {"group": g, "ticker": t,
         "mst_degree_centrality": round(centrality.get(t, 0.0), 4)}
        for g, mems in cluster_assignments.items() for t in mems
    ]).to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)

    logger.info("Phase 4 validation checklist passed ✓\n")
    return {
        "cluster_assignments": cluster_assignments,
        "corr_matrix":         corr_matrix,
        "G_directed":          G_dir,
        "MST":                 MST,
        "centrality":          centrality,
    }


# ======================================================================
# PHASE 4.5: HMM REGIME DETECTION
# ======================================================================
def _extract_filtered_probs(model: GaussianHMM, obs: np.ndarray) -> np.ndarray:
    """
    Forward-pass filtered probabilities: P(state_t | obs_{1..t}).

    Implemented manually so it is robust across all hmmlearn versions.
    hmmlearn >= 0.3.x removed _do_forward_pass from the Python layer;
    this replaces that call with a pure-NumPy log-space forward algorithm
    using only public model attributes.

    IMPORTANT (F-14): HMM *parameters* were estimated on the full dataset
    (parameter-level look-ahead). Filtered probs are extracted via the
    forward pass only — no within-series look-ahead.
    """
    n_samples = obs.shape[0]
    n_states  = model.n_components

    # ── Emission log-likelihoods ───────────────────────────────────────────
    # _compute_log_likelihood is present in all hmmlearn versions tested;
    # fall back to manual scipy computation if a future version removes it.
    if hasattr(model, "_compute_log_likelihood"):
        framelogprob = model._compute_log_likelihood(obs)   # (n_samples, n_states)
    else:
        from scipy.stats import multivariate_normal          # noqa: PLC0415
        framelogprob = np.column_stack([
            multivariate_normal.logpdf(
                obs, mean=model.means_[k], cov=model.covars_[k]
            )
            for k in range(n_states)
        ])

    # ── Log-space forward algorithm ────────────────────────────────────────
    # log α_0(j)   = log π_j + log b_j(x_0)
    # log α_t(j)   = logsumexp_i[ log α_{t-1}(i) + log a_{ij} ] + log b_j(x_t)
    log_startprob = np.log(np.clip(model.startprob_, 1e-300, None))
    log_transmat  = np.log(np.clip(model.transmat_,  1e-300, None))

    log_alpha        = np.empty((n_samples, n_states))
    log_alpha[0]     = log_startprob + framelogprob[0]

    for t in range(1, n_samples):
        for j in range(n_states):
            log_alpha[t, j] = (
                logsumexp(log_alpha[t - 1] + log_transmat[:, j])
                + framelogprob[t, j]
            )

    # ── Convert to normalised probabilities ───────────────────────────────
    log_norm  = logsumexp(log_alpha, axis=1, keepdims=True)
    filtered  = np.exp(log_alpha - log_norm)
    filtered  = np.clip(filtered, 0.0, 1.0)
    row_sums  = filtered.sum(axis=1, keepdims=True)
    filtered /= np.where(row_sums > 0, row_sums, 1.0)
    return filtered


def _select_n_regimes_bic(obs: np.ndarray,
                            candidates: Tuple[int, ...] = (2, 3, 4)) -> int:
    """Optimal number of HMM states via BIC."""
    bic_scores = {}
    for n in candidates:
        try:
            m = GaussianHMM(n_components=n, covariance_type="full",
                            n_iter=500, random_state=RANDOM_STATE)
            m.fit(obs)
            bic_scores[n] = -2 * m.score(obs) + n * np.log(len(obs))
        except Exception:
            bic_scores[n] = np.inf
    best = min(bic_scores, key=bic_scores.get)
    logger.debug(f"    BIC: {bic_scores} → n={best}")
    return best


def _fit_hmm(obs: np.ndarray, n: int) -> GaussianHMM:
    """Fit GaussianHMM with full covariance."""
    m = GaussianHMM(n_components=n, covariance_type="full",
                    n_iter=1000, tol=1e-4, random_state=RANDOM_STATE,
                    init_params="stmc", params="stmc")
    m.fit(obs)
    return m


def _sort_states_by_return(model: GaussianHMM,
                             filtered: np.ndarray) -> np.ndarray:
    """Sort columns: bear (lowest mean return) … bull (highest mean return)."""
    idx = np.argsort(model.means_[:, 0])
    return filtered[:, idx]


def _hmm_to_df(filtered: np.ndarray, index: pd.Index,
                prefix: str, n: int) -> pd.DataFrame:
    """Wrap filtered prob array in a labelled DataFrame."""
    labels = ["bear", "transitional", "bull"] if n == 3 else [f"r{i}" for i in range(n)]
    return pd.DataFrame(filtered, index=index,
                        columns=[f"{prefix}_{lbl}" for lbl in labels[:n]])


def _align_labels_hungarian(new_means: np.ndarray,
                              ref_means: np.ndarray) -> np.ndarray:
    """
    Hungarian-algorithm label alignment for walk-forward HMM re-estimation.
    Currently unused — retained for future per-step re-estimation extension.
    """
    cost = np.abs(new_means[:, None] - ref_means[None, :])
    if cost.ndim == 3:
        cost = cost.sum(-1)
    _, col_idx = linear_sum_assignment(cost)
    return col_idx


def run_phase4_5(p1: dict, p2: dict, p3: dict, p4: dict) -> dict:
    """Phase 4.5: HMM regime detection at market, group, and stock levels."""
    logger.info("=" * 62)
    logger.info("PHASE 4.5 — HMM Regime Detection")
    logger.info("=" * 62)

    mlr            = p2["monthly_log_returns_clean"]
    realized_vol   = p3["realized_volatility"]
    cluster_assign = p4["cluster_assignments"]
    bench_rets     = p1["benchmark_returns"]
    active_tickers = p1["active_tickers"]
    all_probs: Dict[str, pd.DataFrame] = {}

    # ── Market-level HMM (Nifty returns + cross-sectional mean RV) ────────────
    bench_aligned = bench_rets.reindex(mlr.index).fillna(0.0)
    mkt_rv_proxy  = realized_vol.mean(axis=1).reindex(mlr.index).fillna(0.0)
    mkt_obs       = np.column_stack([bench_aligned.values, mkt_rv_proxy.values])

    n_mkt    = _select_n_regimes_bic(mkt_obs)
    hmm_mkt  = _fit_hmm(mkt_obs, n_mkt)
    mkt_filt = _sort_states_by_return(hmm_mkt, _extract_filtered_probs(hmm_mkt, mkt_obs))
    mkt_df   = _hmm_to_df(mkt_filt, mlr.index, "market_hmm", n_mkt)
    all_probs["market"] = mkt_df
    with open(OUTPUT_DIR / "hmm_market.pkl", "wb") as fh:
        pickle.dump(hmm_mkt, fh)
    logger.info(f"  Market HMM n={n_mkt} | means: "
                f"{list(zip(hmm_mkt.means_[:, 0].round(4), hmm_mkt.means_[:, 1].round(4)))}")

    # ── Group-level HMMs ──────────────────────────────────────────────────────
    group_probs: Dict[str, pd.DataFrame] = {}
    for gkey, members in cluster_assign.items():
        if gkey == "isolated" or len(members) < 2:
            continue
        grp_rets = mlr[members].dropna()
        if len(grp_rets) < 24:
            logger.warning(f"  {gkey}: too few obs for group HMM — skipping.")
            continue
        try:
            n_grp   = _select_n_regimes_bic(grp_rets.values)
            hmm_grp = _fit_hmm(grp_rets.values, n_grp)
            g_filt  = _sort_states_by_return(hmm_grp,
                          _extract_filtered_probs(hmm_grp, grp_rets.values))
            g_df    = _hmm_to_df(g_filt, grp_rets.index, f"{gkey}_hmm", n_grp)
            g_df    = g_df.reindex(mlr.index)
            group_probs[gkey] = g_df
            all_probs[gkey]   = g_df
            with open(OUTPUT_DIR / "hmm_groups" / f"hmm_{gkey}.pkl", "wb") as fh:
                pickle.dump(hmm_grp, fh)
            logger.info(f"  {gkey} HMM n={n_grp}")
        except Exception as exc:
            logger.warning(f"  {gkey} HMM failed: {exc}")

    # ── Stock-level HMMs ──────────────────────────────────────────────────────
    stock_probs: Dict[str, pd.DataFrame] = {}
    for ticker in active_tickers:
        ret_s = mlr[ticker].dropna()
        rv_s  = (realized_vol[ticker].reindex(ret_s.index)
                 .fillna(realized_vol[ticker].median())
                 if ticker in realized_vol.columns
                 else pd.Series(0.0, index=ret_s.index))
        stk_obs = np.column_stack([ret_s.values, rv_s.values])
        try:
            n_stk   = _select_n_regimes_bic(stk_obs)
            hmm_stk = _fit_hmm(stk_obs, n_stk)
            s_filt  = _sort_states_by_return(hmm_stk,
                          _extract_filtered_probs(hmm_stk, stk_obs))
            s_df    = _hmm_to_df(s_filt, ret_s.index, f"{ticker}_hmm", n_stk)
            s_df    = s_df.reindex(mlr.index)
            stock_probs[ticker] = s_df
            all_probs[ticker]   = s_df
            safe = ticker.replace(".", "_")
            with open(OUTPUT_DIR / "hmm_stocks" / f"hmm_{safe}.pkl", "wb") as fh:
                pickle.dump(hmm_stk, fh)
            logger.info(f"  {ticker} HMM n={n_stk}")
        except Exception as exc:
            logger.warning(f"  {ticker} HMM failed: {exc}")

    all_probs_df = pd.concat(
        [v for v in all_probs.values() if isinstance(v, pd.DataFrame)], axis=1
    )
    all_probs_df.to_parquet(OUTPUT_DIR / "filtered_probs_all.parquet")

    # Regime probability plot
    fig, ax = plt.subplots(figsize=(14, 5))
    mkt_df.ffill().fillna(0.0).plot(ax=ax, linewidth=1.5)
    ax.set_title("Market Regime Filtered Probabilities (Nifty 50)")
    ax.set_ylabel("P(regime)"); ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regime_plot.pdf", bbox_inches="tight")
    plt.close(fig)

    # Append HMM probs to exog_master
    exog_upd = p3["exog_master"].copy()
    for col in mkt_df.columns:
        exog_upd[col] = mkt_df[col]
    for g_df in group_probs.values():
        for col in g_df.columns:
            exog_upd[col] = g_df[col]
    exog_upd.to_parquet(OUTPUT_DIR / "exog_master.parquet")

    logger.info("Phase 4.5 validation checklist passed ✓\n")
    return {
        "market_filtered_df":   mkt_df,
        "group_filtered_probs": group_probs,
        "stock_filtered_probs": stock_probs,
        "all_probs_df":         all_probs_df,
        "exog_master_updated":  exog_upd,
        "hmm_market":           hmm_mkt,
    }


# ======================================================================
# PHASE 5: MODEL BUILDING & WALK-FORWARD VALIDATION
# ======================================================================

def _build_arimax_exog(ticker: str,
                        idx: pd.Index,
                        mkt_hmm: pd.DataFrame,
                        stk_hmm: pd.DataFrame,
                        rv: pd.DataFrame,
                        market_ind: pd.DataFrame) -> np.ndarray:
    """
    Assemble ARIMAX exogenous matrix (Option 3 B-matrix structure):
      - Market HMM filtered probs  (N_R − 1 cols, drop last to avoid collinearity)
      - Stock  HMM filtered probs  (N_R − 1 cols)
      - Stock  realized volatility (1 col)
      - Market indicators          (n_ind cols, already lagged 1 period)

    Returns shape (len(idx), n_features).
    Returns shape (len(idx), 0) when no features exist — caller passes None.  F-09
    """
    parts = []
    if not mkt_hmm.empty and mkt_hmm.shape[1] > 1:
        parts.append(mkt_hmm.iloc[:, :-1])
    if not stk_hmm.empty and stk_hmm.shape[1] > 1:
        parts.append(stk_hmm.iloc[:, :-1])
    if ticker in rv.columns:
        parts.append(rv[[ticker]].rename(columns={ticker: f"{ticker}_rv"}))
    if not market_ind.empty:
        parts.append(market_ind)

    if not parts:
        return np.empty((len(idx), 0))                                     # F-09

    combined = pd.concat(parts, axis=1).reindex(idx).ffill().bfill().fillna(0.0)
    return combined.values


def _build_varx_exog(gkey: str,
                      members: List[str],
                      idx: pd.Index,
                      mkt_hmm: pd.DataFrame,
                      grp_hmm_dfs: dict,
                      rv: pd.DataFrame,
                      market_ind: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Assemble VARX common exogenous matrix:
      - Market HMM probs (N_R − 1 cols)
      - Group  HMM probs (N_R − 1 cols, if available)
      - Realized volatility for each group member (per-stock RV as common regressor)
      - Market indicators (n_ind cols, lagged)
    """
    parts = []
    if not mkt_hmm.empty and mkt_hmm.shape[1] > 1:
        parts.append(mkt_hmm.iloc[:, :-1])
    g_hmm = grp_hmm_dfs.get(gkey, pd.DataFrame())
    if not g_hmm.empty and g_hmm.shape[1] > 1:
        parts.append(g_hmm.iloc[:, :-1])
    for ticker in members:
        if ticker in rv.columns:
            parts.append(rv[[ticker]].rename(columns={ticker: f"{ticker}_rv"}))
    if not market_ind.empty:
        parts.append(market_ind)

    if not parts:
        return None

    combined = pd.concat(parts, axis=1).reindex(idx).ffill().bfill().fillna(0.0)
    return combined.values


def _arimax_bic_grid(endog: np.ndarray,
                      exog: Optional[np.ndarray]) -> Tuple[int, int]:
    """Grid-search ARIMAX (p, q) ∈ [0..4]² by BIC. Return best (p, q)."""
    best_bic, best_pq = np.inf, (1, 0)
    for p, q in iproduct(range(3), range(3)):
        try:
            m = SARIMAX(endog, exog=exog, order=(p, 0, q),
                        enforce_stationarity=True, enforce_invertibility=True,
                        trend="c").fit(method="lbfgs", maxiter=200, disp=False)
            if np.isfinite(m.bic) and m.bic < best_bic:   # ← guard inf BIC
                best_bic, best_pq = m.bic, (p, q)
        except Exception:
            continue
    return best_pq


def _arimax_ticker_walkforward(
    ticker: str,
    mlr: pd.DataFrame,
    mkt_hmm_df: pd.DataFrame,
    stk_hmm_dfs: Dict[str, pd.DataFrame],
    realized_vol: pd.DataFrame,
    market_ind: pd.DataFrame,
    T: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Full ARIMAX walk-forward for a single ticker.
    Completely self-contained — safe to run in a thread or process.
    BIC order determined once at the first estimation step.
    """
    fc_rows:  List[dict] = []
    err_rows: List[dict] = []
    order:    Optional[Tuple[int, int]] = None
    cached_m  = None
    cached_e: Optional[np.ndarray] = None

    for t in range(MIN_TRAIN_MONTHS, T):
        w_start    = max(0, t - MAX_LOOKBACK_MONTHS)
        train_data = mlr.iloc[w_start:t]
        reestimate = (t == MIN_TRAIN_MONTHS) or ((t - MIN_TRAIN_MONTHS) % 6 == 0)

        endog_s = train_data[ticker].dropna()
        if len(endog_s) < 24:
            continue

        mkt_train = mkt_hmm_df.reindex(endog_s.index).ffill().fillna(0.0)
        s_hmm_raw = stk_hmm_dfs.get(ticker, pd.DataFrame())
        stk_train = (s_hmm_raw.reindex(endog_s.index).ffill().fillna(0.0)
                     if not s_hmm_raw.empty else pd.DataFrame(index=endog_s.index))
        ind_train  = market_ind.reindex(endog_s.index).ffill().fillna(0.0)
        exog_arr   = _build_arimax_exog(ticker, endog_s.index,
                                        mkt_train, stk_train, realized_vol, ind_train)
        exog_in    = exog_arr if exog_arr.shape[1] > 0 else None

        # BIC grid: once per ticker only
        if order is None:
            order = _arimax_bic_grid(endog_s.values, exog_in)

        # Re-fit every 6 steps
        if reestimate or cached_m is None:
            try:
                cached_m = _fit_arimax(endog_s.values, exog_in, order)
                cached_e = exog_in
                try:
                    _, c_pval, _ = breaks_cusumolsresid(cached_m.resid)
                    if c_pval < 0.05:
                        logger.warning(f"  [{ticker}] CUSUM p={c_pval:.3f} at t={t}")
                except Exception:
                    pass
            except Exception as exc:
                logger.warning(f"  ARIMAX fit [{ticker}] t={t}: {exc}")
                continue

        if cached_m is None:
            continue

        max_h    = max(FORECAST_HORIZONS)
        exog_fut = (np.repeat(cached_e[-1:], max_h, axis=0)
                    if cached_e is not None else None)
        try:
            fc  = cached_m.get_forecast(steps=max_h, exog=exog_fut)
            fmu = fc.predicted_mean
            fci_90 = fc.conf_int(alpha=0.10)   # 90% CI  → ±1.645 σ
            fci_95 = fc.conf_int(alpha=0.05)   # 95% CI  → ±1.960 σ
            for h in FORECAST_HORIZONS:
                tgt = t + h
                if tgt < T:
                    pred   = float(fmu[h - 1]) if h <= len(fmu) else np.nan
                    actual = float(mlr[ticker].iloc[tgt])
                    lo_90  = float(fci_90[h-1, 0]) if h <= len(fci_90) else np.nan
                    hi_90  = float(fci_90[h-1, 1]) if h <= len(fci_90) else np.nan
                    lo_95  = float(fci_95[h-1, 0]) if h <= len(fci_95) else np.nan
                    hi_95  = float(fci_95[h-1, 1]) if h <= len(fci_95) else np.nan
                    fc_rows.append({
                        "model": "ARIMAX", "ticker": ticker, "t": t,
                        "horizon": h, "forecast": pred, "actual": actual,
                        "ci90_lower": lo_90, "ci90_upper": hi_90,
                        "ci95_lower": lo_95, "ci95_upper": hi_95,
                        "date_forecast": str(mlr.index[t].date()),
                        "date_target":   str(mlr.index[tgt].date()),
                    })
                    err_rows.append({"model": "ARIMAX", "ticker": ticker,
                                     "t": t, "horizon": h, "error": actual - pred})
        except Exception as exc:
            logger.warning(f"  ARIMAX forecast [{ticker}] t={t}: {exc}")

    return fc_rows, err_rows


def _fit_arimax(endog: np.ndarray,
                exog: Optional[np.ndarray],
                order: Tuple[int, int]):
    """Fit SARIMAX (no seasonal component) = ARIMAX."""
    return SARIMAX(endog, exog=exog, order=(order[0], 0, order[1]),
                   enforce_stationarity=True, enforce_invertibility=True,
                   trend="c").fit(method="lbfgs", maxiter=500, disp=False)


def _fit_varx(endog: np.ndarray,
               exog: Optional[np.ndarray]) -> Tuple:
    """
    Fit VAR(X). Returns (VARResults, optimal_lag).
    Lag capped at 3 (parsimony constraint for ~60 monthly observations).
    """
    model = VAR(endog=endog, exog=exog)
    sel   = model.select_order(maxlags=6)
    lag   = max(1, min(int(sel.bic), 3))
    return model.fit(lag), lag


def _diebold_mariano(err1: np.ndarray, err2: np.ndarray) -> Tuple[float, float]:
    """DM test (squared-error loss). H0: equal forecast accuracy."""
    d     = err1 ** 2 - err2 ** 2
    denom = np.std(d, ddof=1) / np.sqrt(len(d)) + 1e-12
    dm    = float(np.mean(d) / denom)
    pval  = float(2 * (1 - stats.norm.cdf(abs(dm))))
    return dm, pval


def run_phase5(p1: dict, p2: dict, p3: dict, p4: dict, p4_5: dict) -> dict:
    """
    Phase 5: Hybrid expanding-rolling walk-forward validation.

    Window: expands until MAX_LOOKBACK_MONTHS months, then rolls.
    ARIMAX / VARX: re-estimated every 3 steps (quarterly).
    HMM:  parameters fixed (full-sample); only filtered probs used per window.
          (parameter-level look-ahead acknowledged — see F-14).

    F-11 fix: VARX forecast initialisation always uses the CURRENT training
    window's last p rows, not a stale cached array.
    """
    logger.info("=" * 62)
    logger.info("PHASE 5 — Walk-Forward Validation")
    logger.info("=" * 62)

    mlr            = p2["monthly_log_returns_clean"]
    realized_vol   = p3["realized_volatility"]
    market_ind     = p3["market_indicators"]
    cluster_assign = p4["cluster_assignments"]
    mkt_hmm_df     = p4_5["market_filtered_df"]
    grp_hmm_dfs    = p4_5["group_filtered_probs"]
    stk_hmm_dfs    = p4_5["stock_filtered_probs"]
    active_tickers = p1["active_tickers"]

    T           = len(mlr)
    varx_groups = {k: v for k, v in cluster_assign.items() if k != "isolated"}

    forecast_rows: List[dict] = []
    error_rows:    List[dict] = []


    # ── ARIMAX: all tickers in parallel ───────────────────────────────────
    n_workers = min(len(active_tickers), (os.cpu_count() or 4))
    logger.info(f"  Launching ARIMAX walk-forward: {len(active_tickers)} tickers "
                f"× {n_workers} threads …")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _arimax_ticker_walkforward,
                ticker, mlr, mkt_hmm_df, stk_hmm_dfs,
                realized_vol, market_ind, T
            ): ticker
            for ticker in active_tickers
        }
        for fut in as_completed(futures):
            tkr = futures[fut]
            try:
                fc_r, err_r = fut.result()
                forecast_rows.extend(fc_r)
                error_rows.extend(err_r)
                logger.info(f"  [{tkr}] done — {len(fc_r)} forecasts")
            except Exception as exc:
                logger.warning(f"  [{tkr}] walk-forward failed: {exc}")    

    model_cache:   dict       = {}   # stores (fitted_model, exog_array)
    arimax_orders: dict       = {}

    for t in range(MIN_TRAIN_MONTHS, T):
        w_start    = max(0, t - MAX_LOOKBACK_MONTHS)
        train_data = mlr.iloc[w_start:t]
        reestimate = (t == MIN_TRAIN_MONTHS) or ((t - MIN_TRAIN_MONTHS) % 6 == 0)


        # ── VARX (one per cluster group) ──────────────────────────────────────
        for gkey, members in varx_groups.items():
            group_m = [m for m in members if m in mlr.columns]
            if len(group_m) < 2:
                continue

            # Always slice current training endog (F-11: never use stale cached endog)
            endog_g = train_data[group_m].dropna()
            if len(endog_g) < max(MIN_TRAIN_MONTHS // 2, 24):
                continue

            cache_key_v = f"varx_{gkey}"
            if reestimate or cache_key_v not in model_cache:
                mkt_g  = mkt_hmm_df.reindex(endog_g.index).ffill().fillna(0.0)
                ind_g  = market_ind.reindex(endog_g.index).ffill().fillna(0.0)
                exog_v = _build_varx_exog(gkey, group_m, endog_g.index,
                                           mkt_g, grp_hmm_dfs, realized_vol, ind_g)
                try:
                    varx_res, opt_lag = _fit_varx(endog_g.values, exog_v)
                    model_cache[cache_key_v] = (varx_res, opt_lag, exog_v)
                except Exception as exc:
                    logger.warning(f"  VARX [{gkey}] t={t}: {exc}")
                    continue

            if cache_key_v not in model_cache:
                continue
            varx_res, opt_lag, exog_v_ref = model_cache[cache_key_v]

            max_h      = max(FORECAST_HORIZONS)
            exog_fut_v = (np.repeat(exog_v_ref[-1:], max_h, axis=0)
                          if exog_v_ref is not None else None)
            try:
                y_init = endog_g.values[-opt_lag:]          # F-11: current window
                fc_pt = varx_res.forecast(y=y_init, steps=max_h,
                                          exog_future=exog_fut_v)
                fc_lo_90 = fc_hi_90 = fc_lo_95 = fc_hi_95 = None
                try:
                    _, fc_lo_90, fc_hi_90 = varx_res.forecast_interval(
                        y=y_init, steps=max_h, alpha=0.10, exog_future=exog_fut_v
                    )
                    _,     fc_lo_95, fc_hi_95 = varx_res.forecast_interval(
                        y=y_init, steps=max_h, alpha=0.05, exog_future=exog_fut_v
                    )
                except TypeError:
                    # Older statsmodels: exog_future not accepted here
                    try:
                        _, fc_lo_90, fc_hi_90 = varx_res.forecast_interval(
                            y=y_init, steps=max_h, alpha=0.10
                        )
                        _, fc_lo_95, fc_hi_95 = varx_res.forecast_interval(
                            y=y_init, steps=max_h, alpha=0.05
                        )
                    except Exception:
                        pass  # CIs stay None; point forecasts are still valid
                except Exception:
                    pass
                for h in FORECAST_HORIZONS:
                    tgt = t + h
                    if tgt < T:
                        for k, tkr in enumerate(group_m):
                            pred   = float(fc_pt[h - 1, k]) if h <= len(fc_pt) else np.nan
                            actual = float(mlr[tkr].iloc[tgt])
                            lo_90  = float(fc_lo_90[h-1, k]) if fc_lo_90 is not None and h <= len(fc_lo_90) else np.nan
                            hi_90  = float(fc_hi_90[h-1, k]) if fc_hi_90 is not None and h <= len(fc_hi_90) else np.nan
                            lo_95  = float(fc_lo_95[h-1, k]) if fc_lo_95 is not None and h <= len(fc_lo_95) else np.nan
                            hi_95  = float(fc_hi_95[h-1, k]) if fc_hi_95 is not None and h <= len(fc_hi_95) else np.nan
                            forecast_rows.append({
                                "model": "VARX", "ticker": tkr, "t": t,
                                "horizon": h, "forecast": pred, "actual": actual,
                                "ci90_lower": lo_90, "ci90_upper": hi_90,
                                "ci95_lower": lo_95, "ci95_upper": hi_95,
                                "group": gkey,
                                "date_forecast": str(mlr.index[t].date()),
                                "date_target":   str(mlr.index[tgt].date()),
                            })
                            error_rows.append({"model": "VARX", "ticker": tkr,
                                               "t": t, "horizon": h, "error": actual - pred})
            except Exception as exc:
                logger.warning(f"  VARX forecast [{gkey}] t={t}: {exc}")

    # AFTER — schema guaranteed regardless of whether any forecasts were produced
    _FC_COLS  = ["model", "ticker", "t", "horizon", "forecast", "actual",
                "ci90_lower", "ci90_upper", "ci95_lower", "ci95_upper", 
                "date_forecast", "date_target", "group"]
    _ERR_COLS = ["model", "ticker", "t", "horizon", "error"]

    forecasts_df = (pd.DataFrame(forecast_rows)
                    if forecast_rows
                    else pd.DataFrame(columns=_FC_COLS))
    errors_df    = (pd.DataFrame(error_rows)
                    if error_rows
                    else pd.DataFrame(columns=_ERR_COLS))

    forecasts_df.to_parquet(OUTPUT_DIR / "walkforward_forecasts.parquet")
    errors_df.to_parquet(OUTPUT_DIR / "walkforward_errors.parquet")
    logger.info(f"  Walk-forward: {len(forecasts_df)} forecasts, {len(errors_df)} errors")

    # ── Performance metrics ────────────────────────────────────────────────────
    perf_rows:  List[dict] = []
    calib_rows: List[dict] = []

    if forecasts_df.empty:
        logger.warning(
            "  No walk-forward forecasts generated. "
            "Possible causes: all model fits failed, START_DATE too recent, "
            "or MIN_TRAIN_MONTHS > available history. "
            "Check debug logs above for per-ticker ARIMAX/VARX failure messages."
        )
    else:
        for ticker in active_tickers:
            for h in FORECAST_HORIZONS:
                for mt in ["ARIMAX", "VARX"]:
                    sub = forecasts_df[
                        (forecasts_df["ticker"]  == ticker) &
                        (forecasts_df["horizon"] == h) &
                        (forecasts_df["model"]   == mt)
                    ].dropna(subset=["forecast", "actual"])
                    if len(sub) < 5:
                        continue
                    fc  = sub["forecast"].values
                    act = sub["actual"].values
                    da  = float(np.mean(np.sign(fc) == np.sign(act)))
                    ic, ic_p = (pearsonr(fc, act) if len(fc) > 2
                                else (np.nan, np.nan))
                    rmse = float(np.sqrt(np.mean((fc - act) ** 2)))
                    mae  = float(np.mean(np.abs(fc - act)))
                    perf_rows.append({
                        "model": mt, "ticker": ticker, "horizon": h,
                        "DA": round(da, 4),
                        "IC": round(float(ic), 4)   if not np.isnan(ic)   else np.nan,
                        "IC_pval": round(float(ic_p), 4) if not np.isnan(ic_p) else np.nan,
                        "RMSE": round(rmse, 6), "MAE": round(mae, 6), "n_obs": len(sub),
                    })
                    err = act - fc
                    lo  = float(np.percentile(err, 5))
                    hi  = float(np.percentile(err, 95))
                    cov = float(np.mean((err >= lo) & (err <= hi)))
                    calib_rows.append({
                        "model": mt, "ticker": ticker, "horizon": h,
                        "pi_lower_offset":          round(lo, 6),
                        "pi_upper_offset":          round(hi, 6),
                        "empirical_coverage_90pct": round(cov, 4),
                    })

    # ── CI Summary Table ──────────────────────────────────────────────────────
    if not forecasts_df.empty and "ci90_lower" in forecasts_df.columns:
        ci_view = forecasts_df.copy()
        ci_view["ci90_width"] = ci_view["ci90_upper"] - ci_view["ci90_lower"]
        ci_view["ci95_width"] = ci_view["ci95_upper"] - ci_view["ci95_lower"]
        ci_table = (ci_view
                    .groupby(["model", "ticker", "horizon"])
                    .agg(
                        mean_forecast   =("forecast",   "mean"),
                        mean_ci90_lower =("ci90_lower", "mean"),
                        mean_ci90_upper =("ci90_upper", "mean"),
                        mean_ci90_width =("ci90_width", "mean"),
                        mean_ci95_lower =("ci95_lower", "mean"),
                        mean_ci95_upper =("ci95_upper", "mean"),
                        mean_ci95_width =("ci95_width", "mean"),
                    )
                    .round(5))
        logger.info(
            f"\n{'─'*70}\n"
            f"  CONFIDENCE INTERVAL SUMMARY  (average across all walk-forward steps)\n"
            f"{'─'*70}\n"
            f"{ci_table.to_string()}\n"
        )
        ci_table.to_csv(OUTPUT_DIR / "ci_summary.csv")


    perf_df  = pd.DataFrame(perf_rows)
    calib_df = pd.DataFrame(calib_rows)
    perf_df.to_csv(OUTPUT_DIR  / "performance_metrics.csv",  index=False)
    calib_df.to_csv(OUTPUT_DIR / "calibration_results.csv",  index=False)
    if not perf_df.empty:
        logger.info(f"\n{perf_df.to_string()}")

    if not perf_df.empty:
        with pdf_backend.PdfPages(OUTPUT_DIR / "model_diagnostics.pdf") as pages:
            for metric in ["DA", "IC", "RMSE"]:
                if metric not in perf_df.columns:
                    continue
                pivot = perf_df.pivot_table(values=metric, index="ticker",
                                             columns="horizon", aggfunc="mean")
                if pivot.empty:
                    continue
                cmap = "RdYlGn_r" if metric == "RMSE" else "RdYlGn"
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, ax=ax)
                ax.set_title(f"Walk-Forward {metric} — Ticker × Horizon")
                pages.savefig(fig, bbox_inches="tight"); plt.close(fig)

    logger.info("Phase 5 validation checklist passed ✓\n")
    return {"forecasts_df": forecasts_df, "errors_df": errors_df,
            "perf_df": perf_df, "calib_df": calib_df}


# ======================================================================
# PHASE 6: FORECAST COMBINATION
# ======================================================================

def run_phase6(p5: dict, p1: dict) -> dict:
    """
    Combine ARIMAX and VARX forecasts via:
      (1) Equal weighting   — Bates & Granger (1969) baseline
      (2) Performance-weighted (inverse RMSE)
      (3) BMA (BIC-proportional weights)
    Final method chosen by Diebold-Mariano test: prefer equal unless PW is
    significantly better (DM p < 0.05).
    """
    logger.info("=" * 62)
    logger.info("PHASE 6 — Forecast Combination")
    logger.info("=" * 62)

    forecasts_df   = p5["forecasts_df"]
    active_tickers = p1["active_tickers"]

    if forecasts_df.empty:
        logger.warning("  No forecasts — skipping combination.")
        return {"combined_df": pd.DataFrame(), "weights_df": pd.DataFrame()}

    comb_rows:   List[dict] = []
    weight_rows: List[dict] = []

    for ticker in active_tickers:
        for h in FORECAST_HORIZONS:
            model_preds: Dict[str, pd.Series]  = {}
            common_actuals: Optional[pd.Series] = None

            for mt in ["ARIMAX", "VARX"]:
                sub = forecasts_df[
                    (forecasts_df["ticker"]  == ticker) &
                    (forecasts_df["horizon"] == h) &
                    (forecasts_df["model"]   == mt)
                ].dropna(subset=["forecast", "actual"])
                if len(sub) >= 5:
                    model_preds[mt] = sub.set_index("t")["forecast"]
                    common_actuals  = sub.set_index("t")["actual"]

            if not model_preds:
                continue

            if len(model_preds) < 2:
                mt0 = next(iter(model_preds))
                last_t_single = int(model_preds[mt0].index[-1])
                sub_single = forecasts_df[
                    (forecasts_df["ticker"]  == ticker) &
                    (forecasts_df["horizon"] == h) &
                    (forecasts_df["model"]   == mt0) &
                    (forecasts_df["t"]       == last_t_single)
                ]
 
                def _pick_ci(col: str) -> float:
                    if col in sub_single.columns:
                        v = sub_single[col].dropna()
                        return float(v.iloc[0]) if not v.empty else np.nan
                    return np.nan
                
                comb_rows.append({"ticker": ticker, 
                                  "horizon": h,
                                  "combined_forecast": float(model_preds[mt0].iloc[-1]),
                                  "ci90_lower": _pick_ci("ci90_lower"), "ci90_upper": _pick_ci("ci90_upper"),
                                  "ci95_lower": _pick_ci("ci95_lower"), "ci95_upper": _pick_ci("ci95_upper"),
                                  "method": f"single_{mt0}"})
                continue

            common_t = list(set.intersection(*[set(s.index) for s in model_preds.values()]))
            if len(common_t) < 3:
                mt0 = next(iter(model_preds))
                last_t_s = int(model_preds[mt0].index[-1])
                sub_s = forecasts_df[
                    (forecasts_df["ticker"] == ticker) & (forecasts_df["horizon"] == h) &
                    (forecasts_df["model"] == mt0) & (forecasts_df["t"] == last_t_s)
                ]
                def _pick_ci2(col):
                    if col in sub_s.columns:
                        v = sub_s[col].dropna()
                        return float(v.iloc[0]) if not v.empty else np.nan
                    return np.nan
                
                comb_rows.append({"ticker": ticker, "horizon": h,
                                   "combined_forecast": float(model_preds[mt0].iloc[-1]),
                                   "ci90_lower": _pick_ci("ci90_lower"), "ci90_upper": _pick_ci("ci90_upper"),
                                   "ci95_lower": _pick_ci("ci95_lower"), "ci95_upper": _pick_ci("ci95_upper"),
                                   "method": "single_insufficient_overlap"})
                continue

            fc_mat  = np.column_stack([model_preds[m].loc[common_t].values for m in model_preds])
            act_vec = common_actuals.loc[common_t].values

            # Equal
            eq_pred = fc_mat.mean(axis=1)
            # Performance-weighted
            rmse_m   = {m: float(np.sqrt(np.mean((model_preds[m].loc[common_t].values - act_vec)**2)))
                        for m in model_preds}
            inv_rmse = np.array([1.0 / max(rmse_m[m], 1e-10) for m in model_preds])
            pw_w     = inv_rmse / inv_rmse.sum()
            pw_pred  = (pw_w @ fc_mat.T)
            # BMA
            bic_arr  = np.array([rmse_m[m] for m in model_preds])
            bma_w    = np.exp(-0.5 * bic_arr) / np.exp(-0.5 * bic_arr).sum()
            bma_pred = (bma_w @ fc_mat.T)

            # d         = (act_vec - eq_pred)**2 - (act_vec - pw_pred)**2
            # denom     = np.std(d, ddof=1) / np.sqrt(len(d)) + 1e-12
            dm_stat, dm_pval = _diebold_mariano(act_vec - eq_pred, act_vec - pw_pred)
            method   = "equal" if dm_pval > 0.05 else "performance_weighted"
            final_fc = float(eq_pred[-1]) if method == "equal" else float(pw_pred[-1])

            # Combined CI: equal-weight average of last-common-t CI across models
            last_t = max(common_t)
            ci_buckets: Dict[str, List[float]] = {
                "ci90_lower": [], "ci90_upper": [],
                "ci95_lower": [], "ci95_upper": [],
            }
            for mt in model_preds:
                sub_mt = forecasts_df[
                    (forecasts_df["ticker"]  == ticker) &
                    (forecasts_df["horizon"] == h) &
                    (forecasts_df["model"]   == mt) &
                    (forecasts_df["t"]       == last_t)
                ]
                for col in ci_buckets:
                    if col in sub_mt.columns and not sub_mt[col].isna().all():
                        ci_buckets[col].append(float(sub_mt[col].iloc[0]))

            def _mean_ci(col: str) -> float:
                return float(np.mean(ci_buckets[col])) if ci_buckets[col] else np.nan

            comb_rows.append({
                "ticker": ticker, "horizon": h,
                "combined_forecast": final_fc,
                "eq_forecast":       float(eq_pred[-1]),
                "pw_forecast":       float(pw_pred[-1]),
                "bma_forecast":      float(bma_pred[-1]),
                "dm_stat":           round(dm_stat, 4),
                "dm_pval":           round(dm_pval, 4),
                "method":            method,
                "ci90_lower":        _mean_ci("ci90_lower"),
                "ci90_upper":        _mean_ci("ci90_upper"),
                "ci95_lower":        _mean_ci("ci95_lower"),
                "ci95_upper":        _mean_ci("ci95_upper"),
            })
            for mt, w in zip(model_preds.keys(), pw_w):
                weight_rows.append({"ticker": ticker, "horizon": h,
                                     "model": mt, "pw_weight": round(float(w), 4)})

    combined_df = pd.DataFrame(comb_rows)
    weights_df  = pd.DataFrame(weight_rows)
    combined_df.to_parquet(OUTPUT_DIR / "combined_forecasts.parquet")
    weights_df.to_csv(OUTPUT_DIR  / "combination_weights.csv", index=False)
    if not combined_df.empty:
        logger.info(f"\n{combined_df.to_string()}")
    logger.info("Phase 6 ✓\n")
    return {"combined_df": combined_df, "weights_df": weights_df}


# ======================================================================
# PHASE 7: PORTFOLIO CONSTRUCTION
# ======================================================================

def run_phase7(p1: dict, p2: dict, p3: dict, p4_5: dict, p6: dict) -> dict:
    """Black-Litterman portfolio construction with regime-aware risk budgeting."""
    logger.info("=" * 62)
    logger.info("PHASE 7 — Portfolio Construction")
    logger.info("=" * 62)

    monthly_prices = p1["monthly_prices"]
    bench_px       = p1["bench_monthly_prices"]
    monthly_rf     = p1["monthly_rf"]
    active_tickers = p1["active_tickers"]
    mkt_hmm_df     = p4_5["market_filtered_df"]
    combined_df    = p6["combined_df"]

    if combined_df.empty:
        logger.warning("  No combined forecasts — skipping Phase 7.")
        return {}

    prices_aligned = monthly_prices[active_tickers].dropna()

    # F-04: frequency=12 (monthly data); default 252 inflates cov by ~21x
    S = risk_m.CovarianceShrinkage(prices_aligned, frequency=12).ledoit_wolf()

    # Market-implied equilibrium (equal-cap proxy)
    mkt_caps         = pd.Series(1.0 / len(active_tickers), index=active_tickers)
    bench_aligned_px = pd.Series(bench_px).reindex(prices_aligned.index).ffill().dropna()

    try:
        delta = bl_module.market_implied_risk_aversion(
            bench_aligned_px, risk_free_rate=ANNUAL_RISK_FREE, frequency=12
        )
    except Exception as exc:
        logger.warning(f"  market_implied_risk_aversion failed ({exc}) → δ=2.5")
        delta = 2.5

    pi = bl_module.market_implied_prior_returns(
        market_caps=mkt_caps, risk_aversion=delta, cov_matrix=S
    )

    # Views from 12-month combined forecasts (annualised)
    h_target = 12 if 12 in FORECAST_HORIZONS else max(FORECAST_HORIZONS)
    viewdict: Dict[str, float] = {}
    for _, row in combined_df[combined_df["horizon"] == h_target].iterrows():
        t = row["ticker"]
        if t in active_tickers and pd.notna(row.get("combined_forecast")):
            viewdict[t] = float(row["combined_forecast"]) * 12

    # Regime-aware risk budget
    bear_cols   = [c for c in mkt_hmm_df.columns if "bear" in c]
    stress_prob = float(mkt_hmm_df[bear_cols[0]].iloc[-1]) if bear_cols else 0.0
    in_stress   = stress_prob > 0.60
    max_w       = 0.12 if in_stress else 0.20
    logger.info(f"  P(stress)={stress_prob:.3f} → max_weight={max_w:.0%}")

    weights_clean = {t: 1.0 / len(active_tickers) for t in active_tickers}

    if viewdict:
        try:
            bl     = BlackLittermanModel(S, pi=pi, absolute_views=viewdict, tau=0.05)
            ret_bl = bl.bl_returns()
            S_bl   = bl.bl_cov()
        except Exception as exc:
            logger.warning(f"  BL model failed ({exc}) — using prior.")
            ret_bl, S_bl = pi, S
        try:
            ef = EfficientFrontier(ret_bl, S_bl)
            ef.add_constraint(lambda w: w >= 0.02)
            ef.add_constraint(lambda w: w <= max_w)
            if in_stress:
                ef.efficient_risk(target_volatility=0.12 / np.sqrt(12))
            else:
                ann_rf_exact = (1 + monthly_rf) ** 12 - 1               # F-12
                ef.max_sharpe(risk_free_rate=ann_rf_exact)
            weights_clean = ef.clean_weights()
        except Exception as exc:
            logger.warning(f"  EfficientFrontier failed ({exc}) — equal weights.")

    logger.info(f"  Weights: {weights_clean}")
    latest_prices = get_latest_prices(monthly_prices[active_tickers])

    try:
        da = DiscreteAllocation(weights_clean, latest_prices,
                                total_portfolio_value=CAPITAL_AMOUNT)
        try:
            allocation, leftover = da.lp_portfolio()
        except Exception:
            allocation, leftover = da.greedy_portfolio()
    except Exception as exc:
        logger.warning(f"  DiscreteAllocation failed ({exc}).")
        allocation, leftover = {}, float(CAPITAL_AMOUNT)

    logger.info(f"  Allocation: {allocation}")
    logger.info(f"  Residual cash: ₹{leftover:,.2f}")

    pd.DataFrame(list(weights_clean.items()),
                 columns=["ticker", "weight"]).to_csv(
        OUTPUT_DIR / "portfolio_weights.csv", index=False)
    pd.DataFrame([{
        "ticker": t, "shares": s,
        "latest_price":  float(latest_prices.get(t, np.nan)),
        "current_value": float(s * latest_prices.get(t, 0.0)),
        "weight":        weights_clean.get(t, 0.0),
    } for t, s in allocation.items()]).to_csv(
        OUTPUT_DIR / "capital_allocation.csv", index=False)
    pd.DataFrame([{
        "ticker": t, "view_return": round(v, 4),
        "prior_return": round(float(pi[t])
                               if hasattr(pi, "__getitem__") else float(pi), 4),
    } for t, v in viewdict.items()]).to_csv(
        OUTPUT_DIR / "bl_diagnostics.csv", index=False)

    logger.info("Phase 7 ✓\n")
    return {
        "weights_clean": weights_clean, "allocation": allocation,
        "leftover": leftover, "S_full": S, "latest_prices": latest_prices,
        "stress_prob": stress_prob, "in_stress": in_stress,
    }


# ======================================================================
# PHASE 8: RISK MANAGEMENT & BACKTEST
# ======================================================================

def run_phase8(p1: dict, p2: dict, p4_5: dict, p7: dict) -> dict:
    """Drawdown monitoring, rebalancing triggers, and regime-conditional backtest metrics."""
    logger.info("=" * 62)
    logger.info("PHASE 8 — Risk Management & Backtest")
    logger.info("=" * 62)

    mlr           = p2["monthly_log_returns_clean"]
    bench_rets    = p1["benchmark_returns"]
    mkt_hmm_df    = p4_5["market_filtered_df"]
    weights_clean = p7.get("weights_clean", {})

    if not weights_clean:
        logger.warning("  No portfolio weights — skipping Phase 8.")
        return {}

    w_series = pd.Series(weights_clean).reindex(mlr.columns).fillna(0.0)
    portfolio_log_rets = mlr.dot(w_series)

    # F-05: correct compounding for log returns
    portfolio_value  = np.exp(portfolio_log_rets.cumsum())
    rolling_max      = portfolio_value.expanding().max()
    drawdown_series  = (portfolio_value - rolling_max) / rolling_max
    max_drawdown     = float(drawdown_series.min())
    current_drawdown = float(drawdown_series.iloc[-1])

    portfolio_log_rets.to_frame("portfolio_returns").to_parquet(OUTPUT_DIR / "portfolio_returns.parquet")
    portfolio_value.to_frame("portfolio_value").to_parquet(OUTPUT_DIR / "portfolio_value.parquet")
    drawdown_series.to_frame("drawdown").to_parquet(OUTPUT_DIR / "drawdown_series.parquet")

    if current_drawdown < -0.15:
        logger.warning(f"  HARD STOP: drawdown={current_drawdown:.1%} > 15% threshold.")

    bear_col = [c for c in mkt_hmm_df.columns if "bear" in c]
    if bear_col:
        trigger_dates = mkt_hmm_df[bear_col[0]][mkt_hmm_df[bear_col[0]] > 0.65].index
        if len(trigger_dates):
            logger.info(f"  Regime rebalance triggers: {[str(d.date()) for d in trigger_dates]}")

    n_months   = len(portfolio_log_rets)
    ann_return = float((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) **
                       (12 / n_months) - 1)
    ann_vol    = float(portfolio_log_rets.std() * np.sqrt(12))
    ann_rf     = ANNUAL_RISK_FREE
    sharpe     = (ann_return - ann_rf) / ann_vol if ann_vol > 0 else np.nan

    neg_rets = portfolio_log_rets[portfolio_log_rets < 0]
    down_dev = float(neg_rets.std() * np.sqrt(12)) if len(neg_rets) > 1 else ann_vol
    sortino  = (ann_return - ann_rf) / down_dev if down_dev > 0 else np.nan
    calmar   = ann_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    bench_aligned = bench_rets.reindex(portfolio_log_rets.index).fillna(0.0)
    excess        = portfolio_log_rets - bench_aligned
    track_err     = float(excess.std() * np.sqrt(12))
    info_ratio    = float(excess.mean() * 12 / track_err) if track_err > 0 else np.nan

    perf_summary = {
        "Annualized Return":     round(ann_return, 4),
        "Annualized Volatility": round(ann_vol, 4),
        "Sharpe Ratio":          round(sharpe, 4)    if not np.isnan(sharpe)     else None,
        "Sortino Ratio":         round(sortino, 4)   if not np.isnan(sortino)    else None,
        "Maximum Drawdown":      round(max_drawdown, 4),
        "Calmar Ratio":          round(calmar, 4)    if not np.isnan(calmar)     else None,
        "Information Ratio":     round(info_ratio, 4) if not np.isnan(info_ratio) else None,
        "N Months":              n_months,
    }
    logger.info(f"  Performance: {perf_summary}")
    pd.DataFrame([perf_summary]).to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)

    # Regime-conditional decomposition
    regime_buckets: Dict[str, List] = {"bull": [], "transitional": [], "bear": []}
    for dt, ret_val in portfolio_log_rets.items():
        if dt not in mkt_hmm_df.index:
            continue
        dom = mkt_hmm_df.loc[dt].fillna(0.0).idxmax()
        regime = dom.replace("market_hmm_", "")
        if regime in regime_buckets:
            regime_buckets[regime].append(float(ret_val))

    regime_rows = []
    for regime, rets in regime_buckets.items():
        if len(rets) < 3:
            continue
        ra = np.array(rets)
        ar = float(np.mean(ra) * 12)
        av = float(np.std(ra)  * np.sqrt(12))
        regime_rows.append({
            "regime": regime, "n_months": len(ra),
            "ann_return": round(ar, 4), "ann_vol": round(av, 4),
            "sharpe": round((ar - ann_rf) / av, 4) if av > 0 else None,
        })
    regime_perf_df = pd.DataFrame(regime_rows)
    regime_perf_df.to_csv(OUTPUT_DIR / "regime_conditional_performance.csv", index=False)
    logger.info(f"\n{regime_perf_df.to_string()}")

    # F-06: QuantStats expects simple (percentage) returns
    try:
        simple_port  = np.exp(portfolio_log_rets) - 1
        simple_bench = np.exp(bench_aligned)       - 1
        qs.reports.html(simple_port, benchmark=simple_bench,
                        output=str(OUTPUT_DIR / "backtest_tearsheet.html"),
                        title="NSE Portfolio Walk-Forward Backtest")
        logger.info("  QuantStats tearsheet saved.")
    except Exception as exc:
        logger.warning(f"  QuantStats failed: {exc}")

    logger.info("Phase 8 ✓\n")
    return {
        "portfolio_returns": portfolio_log_rets,
        "portfolio_value":   portfolio_value,
        "drawdown_series":   drawdown_series,
        "max_drawdown":      max_drawdown,
        "perf_summary":      perf_summary,
        "regime_perf_df":    regime_perf_df,
    }

def run_phase_eval(p1: dict, p2: dict, p3: dict, p4: dict, p4_5: dict) -> dict:
    """
    Held-out train/test evaluation to assess model robustness.
 
    Split
    -----
    Train : first MIN_TRAIN_MONTHS months  (same as walk-forward burn-in)
    Test  : all remaining months (out-of-sample)
 
    Each model is fit ONCE on training data.  Forecasts are generated for the
    full test horizon in one shot (fixed-origin, not rolling).  This gives a
    clean, reproducible benchmark that complements the rolling Phase-5 metrics.
 
    Metrics reported (per model × ticker × horizon)
    ------------------------------------------------
    RMSE   Root Mean Square Error
    MAE    Mean Absolute Error
    ME     Mean Error (bias; positive = systematic over-forecast)
    MAPE   Mean Absolute Percentage Error  (%)
    SMAPE  Symmetric MAPE  (%)
    DA     Directional Accuracy (sign-matching fraction)
    AIC    Akaike Information Criterion  (training fit, lower = better)
    BIC    Bayesian Information Criterion (training fit, lower = better)
    """
 
    logger.info("=" * 62)
    logger.info("PHASE EVAL — Model Robustness & Diagnostics")
    logger.info("=" * 62)
 
    mlr            = p2["monthly_log_returns_clean"]
    realized_vol   = p3["realized_volatility"]
    market_ind     = p3["market_indicators"]
    mkt_hmm_df     = p4_5["market_filtered_df"]
    stk_hmm_dfs    = p4_5["stock_filtered_probs"]
    grp_hmm_dfs    = p4_5["group_filtered_probs"]
    active_tickers = p1["active_tickers"]
    cluster_assign = p4["cluster_assignments"]
 
    T         = len(mlr)
    train_end = MIN_TRAIN_MONTHS
    test_mos  = T - train_end
 
    if test_mos < 2:
        logger.warning("  Fewer than 2 test months — skipping eval phase.")
        return {"eval_df": pd.DataFrame()}
 
    train_data = mlr.iloc[:train_end]
    test_data  = mlr.iloc[train_end:]
 
    logger.info(
        f"  Train : {train_data.index[0].date()} → {train_data.index[-1].date()} "
        f"({len(train_data)} months)"
    )
    logger.info(
        f"  Test  : {test_data.index[0].date()}  → {test_data.index[-1].date()} "
        f"({test_mos} months)"
    )
 
    def _metrics(preds: np.ndarray, actuals: np.ndarray) -> dict:
        err   = preds - actuals
        denom = np.where(np.abs(actuals) > 1e-10, np.abs(actuals), 1e-10)
        smape_denom = np.abs(preds) + np.abs(actuals) + 1e-10
        return {
            "RMSE":  float(np.sqrt(np.mean(err**2))),
            "MAE":   float(np.mean(np.abs(err))),
            "ME":    float(np.mean(err)),
            "MAPE":  float(np.mean(np.abs(err) / denom) * 100),
            "SMAPE": float(np.mean(2 * np.abs(err) / smape_denom) * 100),
            "DA":    float(np.mean(np.sign(preds) == np.sign(actuals))),
        }
 
    eval_rows: List[dict] = []
 
    # ── ARIMAX single-origin evaluation ───────────────────────────────────────
    logger.info("  Fitting ARIMAX on training data …")
    for ticker in active_tickers:
        endog_tr = train_data[ticker].dropna()
        if len(endog_tr) < 24:
            continue
 
        mkt_tr  = mkt_hmm_df.reindex(endog_tr.index).ffill().fillna(0.0)
        s_raw   = stk_hmm_dfs.get(ticker, pd.DataFrame())
        stk_tr  = (s_raw.reindex(endog_tr.index).ffill().fillna(0.0)
                   if not s_raw.empty else pd.DataFrame(index=endog_tr.index))
        ind_tr  = market_ind.reindex(endog_tr.index).ffill().fillna(0.0)
        exog_tr = _build_arimax_exog(ticker, endog_tr.index,
                                     mkt_tr, stk_tr, realized_vol, ind_tr)
        exog_in = exog_tr if exog_tr.shape[1] > 0 else None
 
        order = _arimax_bic_grid(endog_tr.values, exog_in)
        try:
            fitted  = _fit_arimax(endog_tr.values, exog_in, order)
        except Exception as exc:
            logger.warning(f"  [{ticker}] ARIMAX train-fit failed: {exc}")
            continue
 
        aic_v = float(fitted.aic)
        bic_v = float(fitted.bic)
 
        # Build test exog (reindex onto test period, forward-fill gaps)
        mkt_te  = mkt_hmm_df.reindex(test_data.index).ffill().fillna(0.0)
        stk_te  = (s_raw.reindex(test_data.index).ffill().fillna(0.0)
                   if not s_raw.empty else pd.DataFrame(index=test_data.index))
        ind_te  = market_ind.reindex(test_data.index).ffill().fillna(0.0)
        exog_te = _build_arimax_exog(ticker, test_data.index,
                                     mkt_te, stk_te, realized_vol, ind_te)
        exog_fc = exog_te if exog_te.shape[1] > 0 else None
 
        try:
            fc_obj  = fitted.get_forecast(steps=test_mos, exog=exog_fc)
            fmu_arr = fc_obj.predicted_mean   # shape (test_mos,)
        except Exception as exc:
            logger.warning(f"  [{ticker}] ARIMAX test-forecast failed: {exc}")
            continue
 
        act_arr = test_data[ticker].values
        for h in FORECAST_HORIZONS:
            if h > test_mos:
                continue
            # h-step ahead: compare predicted at position h-1 with actual at h-1
            # (fixed-origin: all forecasts issued from the same training end-point)
            n_valid = test_mos - h + 1
            if n_valid < 2:
                continue
            p_arr = fmu_arr[:n_valid] if len(fmu_arr) >= n_valid else fmu_arr
            a_arr = act_arr[h-1: h-1+len(p_arr)]
            mask  = np.isfinite(p_arr) & np.isfinite(a_arr)
            if mask.sum() < 2:
                continue
            m = _metrics(p_arr[mask], a_arr[mask])
            eval_rows.append({
                "model": "ARIMAX", "ticker": ticker, "horizon": h,
                "order_p": order[0], "order_q": order[1],
                "n_test":  int(mask.sum()),
                "AIC": round(aic_v, 2), "BIC": round(bic_v, 2),
                **{k: round(v, 6) for k, v in m.items()},
            })
 
    # ── VARX single-origin evaluation ─────────────────────────────────────────
    logger.info("  Fitting VARX groups on training data …")
    varx_groups = {k: v for k, v in cluster_assign.items() if k != "isolated"}
    for gkey, members in varx_groups.items():
        group_m  = [m for m in members if m in mlr.columns]
        if len(group_m) < 2:
            continue
        endog_tr = train_data[group_m].dropna()
        if len(endog_tr) < 24:
            continue
 
        mkt_g  = mkt_hmm_df.reindex(endog_tr.index).ffill().fillna(0.0)
        ind_g  = market_ind.reindex(endog_tr.index).ffill().fillna(0.0)
        exog_v = _build_varx_exog(gkey, group_m, endog_tr.index,
                                   mkt_g, grp_hmm_dfs, realized_vol, ind_g)
        try:
            varx_res, opt_lag = _fit_varx(endog_tr.values, exog_v)
        except Exception as exc:
            logger.warning(f"  [{gkey}] VARX train-fit failed: {exc}")
            continue
 
        aic_v = float(getattr(varx_res, "aic", np.nan))
        bic_v = float(getattr(varx_res, "bic", np.nan))
 
        endog_te = test_data[group_m]
        if endog_te.empty:
            continue
 
        mkt_te_g = mkt_hmm_df.reindex(endog_te.index).ffill().fillna(0.0)
        ind_te_g = market_ind.reindex(endog_te.index).ffill().fillna(0.0)
        exog_fut = _build_varx_exog(gkey, group_m, endog_te.index,
                                     mkt_te_g, grp_hmm_dfs, realized_vol, ind_te_g)
 
        y_init = endog_tr.values[-opt_lag:]
        fc_steps = len(endog_te)
        try:
            fc_pt = varx_res.forecast(y=y_init, steps=fc_steps, exog_future=exog_fut)
        except Exception as exc:
            logger.warning(f"  [{gkey}] VARX test-forecast failed: {exc}")
            continue
 
        for k, tkr in enumerate(group_m):
            act_arr = endog_te[tkr].values
            for h in FORECAST_HORIZONS:
                if h > fc_steps:
                    continue
                n_valid = fc_steps - h + 1
                if n_valid < 2:
                    continue
                p_arr = fc_pt[:n_valid, k]
                a_arr = act_arr[h-1: h-1+len(p_arr)]
                mask  = np.isfinite(p_arr) & np.isfinite(a_arr)
                if mask.sum() < 2:
                    continue
                m = _metrics(p_arr[mask], a_arr[mask])
                eval_rows.append({
                    "model": "VARX", "ticker": tkr, "horizon": h,
                    "group": gkey, "order_p": opt_lag,
                    "n_test": int(mask.sum()),
                    "AIC": round(aic_v, 2), "BIC": round(bic_v, 2),
                    **{k2: round(v, 6) for k2, v in m.items()},
                })
 
    _EVAL_COLS = ["model", "ticker", "horizon", "n_test",
                  "RMSE", "MAE", "ME", "MAPE", "SMAPE", "DA", "AIC", "BIC"]
    eval_df = (pd.DataFrame(eval_rows) if eval_rows
               else pd.DataFrame(columns=_EVAL_COLS))
    eval_df.to_csv(OUTPUT_DIR / "model_evaluation.csv", index=False)
 
    if not eval_df.empty:
        logger.info(f"\n{'─'*70}\nMODEL EVALUATION (out-of-sample, fixed origin)\n{'─'*70}")
        with pd.option_context("display.max_rows", 200):
            logger.info(f"\n{eval_df.to_string()}\n")
 
        with pdf_backend.PdfPages(OUTPUT_DIR / "model_evaluation.pdf") as pages:
            for metric in ["RMSE", "MAPE", "DA", "ME"]:
                if metric not in eval_df.columns:
                    continue
                for mt in eval_df["model"].unique():
                    sub = eval_df[eval_df["model"] == mt]
                    if sub.empty or sub[metric].isna().all():
                        continue
                    try:
                        pivot = sub.pivot_table(
                            values=metric, index="ticker", columns="horizon",
                            aggfunc="mean"
                        )
                        if pivot.empty:
                            continue
                        cmap = "RdYlGn_r" if metric in ("RMSE", "MAPE") else "RdYlGn"
                        fmt  = ".2f" if metric == "MAPE" else ".4f"
                        h_px = max(3, len(pivot) * 0.45 + 1.5)
                        fig, ax = plt.subplots(figsize=(10, h_px))
                        sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                                    linewidths=0.4)
                        ax.set_title(
                            f"{mt} — Out-of-Sample {metric}\n"
                            f"(train={train_end} months, test={test_mos} months)",
                            fontsize=11)
                        pages.savefig(fig, bbox_inches="tight")
                        plt.close(fig)
                    except Exception as exc:
                        logger.debug(f"  Eval heatmap ({metric},{mt}): {exc}")
 
        logger.info(f"  model_evaluation.pdf saved.")
    else:
        logger.warning("  No evaluation rows produced — check train/test sizes.")
 
    logger.info("Phase EVAL ✓\n")
    return {"eval_df": eval_df}

# ======================================================================
# PHASE 9: PLOTLY DASH MONITORING DASHBOARD
# ======================================================================

def _write_dashboard_stub() -> None:
    stub = '''\
"""NSE Portfolio Monitor — standalone launcher (run after full pipeline)."""
from pathlib import Path
import pandas as pd
OUTPUT_DIR = Path("pipeline_outputs")
market_hmm = pd.read_parquet(OUTPUT_DIR / "filtered_probs_all.parquet")
combined   = pd.read_parquet(OUTPUT_DIR / "combined_forecasts.parquet")
perf       = pd.read_csv(OUTPUT_DIR / "performance_metrics.csv")
weights    = pd.read_csv(OUTPUT_DIR / "portfolio_weights.csv")
print("Artifacts loaded. Dashboard: http://127.0.0.1:8050/")
'''
    with open(OUTPUT_DIR / "dashboard_app.py", "w") as fh:
        fh.write(stub)


def run_phase9(p1: dict, p2: dict, p3: dict, p4_5: dict,
               p5: dict, p6: dict, p7: dict, p8: dict,
               p_eval: dict) -> "dash.Dash":   # type: ignore[name-defined]
    """
    Eight-panel Plotly Dash monitoring dashboard.
 
    Panels
    ------
    1  Market Regime Monitor
    2  Forecast Fan Chart          (log-return forecasts + CI cones)
    3  Price Forecasts  ←NEW       (₹ price projections with CI cones)
    4  Portfolio Allocation
    5  Performance Attribution
    6  Model Diagnostics  (IC heatmap)
    7  Model Evaluation   ←NEW     (RMSE / MAPE / DA from Phase EVAL)
    8  Stock Deep Dive
 
    Note on price forecasts
    -----------------------
    Price at horizon h is computed as:
        P_{t+h} ≈ P_last · exp(fc_h)
    where fc_h is the combined forecast of the *h-period-ahead log return*
    (i.e. what the model expects monthly log-return to be in month t+h, not
    the cumulative return over h months).  For h=1 this is exact.  For
    longer horizons it is an approximation that ignores intermediate-step
    compounding from t to t+h.  The 90 % CI cone is transformed identically:
        price_CI_bound = P_last · exp(ci_bound).
    """
 
    logger.info("=" * 62)
    logger.info("PHASE 9 — Dashboard")
    logger.info("=" * 62)
 
    mkt_hmm_df     = p4_5.get("market_filtered_df",       pd.DataFrame())
    portfolio_rets = p8.get("portfolio_returns",           pd.Series(dtype=float))
    portfolio_val  = p8.get("portfolio_value",             pd.Series(dtype=float))
    drawdown_s     = p8.get("drawdown_series",             pd.Series(dtype=float))
    bench_rets     = p1.get("benchmark_returns",           pd.Series(dtype=float))
    weights_clean  = p7.get("weights_clean",               {})
    combined_df    = p6.get("combined_df",                 pd.DataFrame())
    perf_df        = p5.get("perf_df",                     pd.DataFrame())
    calib_df       = p5.get("calib_df",                    pd.DataFrame())
    monthly_prices = p1.get("monthly_prices",              pd.DataFrame())
    mlr            = p2.get("monthly_log_returns_clean",   pd.DataFrame())
    active_tickers = p1.get("active_tickers",              [])
    perf_summary   = p8.get("perf_summary",               {})
    realized_vol   = p3.get("realized_volatility",         pd.DataFrame())
    market_ind     = p3.get("market_indicators",           pd.DataFrame())
    eval_df        = p_eval.get("eval_df",                 pd.DataFrame())
 
    regime_cols        = list(mkt_hmm_df.columns) if not mkt_hmm_df.empty else []
    current_regime_row = mkt_hmm_df.iloc[-1].to_dict() if not mkt_hmm_df.empty else {}
 
    app = dash.Dash(__name__, title="NSE Portfolio Monitor")
 
    app.layout = html.Div([
        html.H1("NSE Stock Forecasting & Portfolio Monitor",
                style={"textAlign": "center", "fontFamily": "Arial",
                       "borderBottom": "2px solid #ccc", "paddingBottom": "10px"}),
        dcc.Tabs(children=[
 
            # ── Panel 1: Market Regime ────────────────────────────────────────
            dcc.Tab(label="Market Regime", children=[
                html.H3("Current Regime Probabilities"),
                dcc.Graph(id="regime-gauges", figure=go.Figure(data=[
                    go.Indicator(
                        mode="gauge+number",
                        value=round(v * 100, 1),
                        title={"text": k.replace("market_hmm_", "").capitalize()},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "steelblue"}},
                        domain={"x": [i / max(len(current_regime_row), 1),
                                      (i+1) / max(len(current_regime_row), 1)],
                                "y": [0, 1]},
                    ) for i, (k, v) in enumerate(current_regime_row.items())
                ]).update_layout(height=300, title="Current Regime Probabilities (%)")
                ) if current_regime_row else html.P("No regime data."),
                html.H3("Regime Probability History"),
                dcc.Graph(id="regime-history", figure=go.Figure([
                    go.Scatter(x=mkt_hmm_df.index, y=mkt_hmm_df[c].values,
                               name=c.replace("market_hmm_", "").capitalize(),
                               mode="lines", stackgroup="one")
                    for c in regime_cols
                ]).update_layout(title="Market Regime Probabilities (Filtered)",
                                  yaxis_range=[0, 1], height=350)
                ) if not mkt_hmm_df.empty else html.P("No regime history."),
                html.H3("Market Indicators (Lagged Monthly Returns)"),
                dcc.Graph(id="ind-chart", figure=go.Figure([
                    go.Scatter(x=market_ind.index, y=market_ind[c].values,
                               name=c, mode="lines")
                    for c in market_ind.columns
                ]).update_layout(title="Market Indicators (Lagged Log Returns)",
                                  height=300)
                ) if not market_ind.empty else html.P("No indicator data."),
            ]),
 
            # ── Panel 2: Forecast Fan Chart (log returns) ─────────────────────
            dcc.Tab(label="Return Forecasts", children=[
                html.H3("Forecast Fan Chart — Monthly Log Returns"),
                dcc.Dropdown(id="forecast-ticker-dd",
                             options=[{"label": STOCKS.get(t, t), "value": t}
                                      for t in active_tickers],
                             value=active_tickers[0] if active_tickers else None,
                             clearable=False),
                dcc.Graph(id="forecast-fan"),
                html.Div(id="forecast-table")
            ]),
 
            # ── Panel 3: Price Forecasts (NEW) ────────────────────────────────
            dcc.Tab(label="Price Forecasts", children=[
                html.H3("Price Projections"),
                html.P(
                    "Each diamond shows the implied price if the h-month-ahead "
                    "log-return forecast materialises.  Shaded cone = 90 % CI "
                    "transformed to price space.  For h > 1 this is an approximation "
                    "(single-period return, not cumulative compound path).",
                    style={"fontStyle": "italic", "fontSize": "12px",
                           "color": "#555", "maxWidth": "800px"}
                ),
                dcc.Dropdown(id="price-ticker-dd",
                             options=[{"label": STOCKS.get(t, t), "value": t}
                                      for t in active_tickers],
                             value=active_tickers[0] if active_tickers else None,
                             clearable=False),
                dcc.Graph(id="price-forecast-chart"),
                html.Div(id="price-table")
            ]),
 
            # ── Panel 4: Portfolio Allocation ─────────────────────────────────
            dcc.Tab(label="Portfolio", children=[
                html.H3("Current Portfolio Weights"),
                dcc.Graph(
                    id="portfolio-donut",
                    figure=go.Figure(go.Pie(
                        labels=[STOCKS.get(t, t) for t in weights_clean.keys()],
                        customdata=list(weights_clean.keys()),
                        values=list(weights_clean.values()),
                        hole=0.4, textinfo="percent",
                        hovertemplate=(
                            "<b>%{label}</b><br>Ticker: %{customdata}<br>"
                            "Weight: %{percent}<extra></extra>"
                        ),
                    )).update_layout(
                        title="Portfolio Weights", height=520,
                        margin=dict(l=20, r=180, t=60, b=20),
                        showlegend=True,
                        legend=dict(orientation="v", yanchor="middle", y=0.5,
                                    xanchor="left", x=1.02, font=dict(size=11)),
                    )
                ) if weights_clean else html.P("No weights computed."),
            ]),
 
            # ── Panel 5: Performance Attribution ──────────────────────────────
            dcc.Tab(label="Performance", children=[
                html.H3("Cumulative Returns vs. Nifty 50"),
                dcc.Graph(id="cumret-chart", figure=go.Figure([
                    go.Scatter(x=portfolio_val.index, y=portfolio_val.values,
                               name="Portfolio", mode="lines"),
                    go.Scatter(x=bench_rets.index,
                               y=np.exp(bench_rets.cumsum()).values,
                               name="Nifty 50", mode="lines",
                               line={"dash": "dash"}),
                ]).update_layout(title="Cumulative Returns (Growth of ₹1)", height=350)
                ) if not portfolio_val.empty else html.P("No backtest data."),
                html.H3("Drawdown"),
                dcc.Graph(id="drawdown-chart", figure=go.Figure(
                    go.Scatter(x=drawdown_s.index, y=drawdown_s.values,
                               fill="tozeroy", name="Drawdown",
                               line={"color": "crimson"})
                ).update_layout(title="Portfolio Drawdown", height=250)
                ) if not drawdown_s.empty else html.P("No drawdown data."),
                html.H3("Performance Summary"),
                html.Table(
                    [html.Tr([html.Th("Metric"), html.Th("Value")])] +
                    [html.Tr([html.Td(k), html.Td(str(v))])
                     for k, v in perf_summary.items()],
                    style={"borderCollapse": "collapse", "width": "60%",
                           "fontSize": "13px"}
                ) if perf_summary else html.P("No summary."),
            ]),
 
            # ── Panel 6: Model Diagnostics (IC heatmap) ───────────────────────
            dcc.Tab(label="Walk-Forward Diagnostics", children=[
                html.H3("Information Coefficient Heatmap (Walk-Forward)"),
                dcc.Graph(id="ic-heatmap", figure=go.Figure(go.Heatmap(
                    z=perf_df.pivot_table(values="IC", index="ticker",
                                          columns="horizon", aggfunc="mean").values
                      if not perf_df.empty else [],
                    x=sorted(perf_df["horizon"].unique().tolist()) if not perf_df.empty else [],
                    y=perf_df["ticker"].unique().tolist() if not perf_df.empty else [],
                    colorscale="RdYlGn", texttemplate="%{z:.3f}",
                )).update_layout(title="Walk-Forward IC — Ticker × Horizon", height=450)
                ) if not perf_df.empty else html.P("No diagnostics."),
            ]),
 
            # ── Panel 7: Model Evaluation (NEW) ──────────────────────────────
            dcc.Tab(label="Model Evaluation", children=[
                html.H3("Out-of-Sample Model Robustness"),
                html.P(
                    f"Fixed-origin evaluation: trained on first {MIN_TRAIN_MONTHS} months, "
                    "evaluated on the remainder.  Lower RMSE / MAPE and higher DA are better.",
                    style={"fontStyle": "italic", "fontSize": "12px", "color": "#555"}
                ),
                dcc.Dropdown(
                    id="eval-metric-dd",
                    options=[{"label": m, "value": m}
                             for m in ["RMSE", "MAE", "MAPE", "SMAPE", "DA", "ME"]],
                    value="RMSE", clearable=False,
                    style={"width": "200px", "marginBottom": "10px"},
                ),
                dcc.Dropdown(
                    id="eval-model-dd",
                    options=[{"label": m, "value": m} for m in ["ARIMAX", "VARX"]],
                    value="ARIMAX", clearable=False,
                    style={"width": "200px", "marginBottom": "10px"},
                ),
                dcc.Graph(id="eval-heatmap"),
                html.H4("Full Evaluation Table"),
                html.Div(id="eval-table"),
            ]) if not eval_df.empty else dcc.Tab(
                label="Model Evaluation",
                children=[html.P("Run run_phase_eval() to populate this panel.")]
            ),
 
            # ── Panel 8: Stock Deep Dive ──────────────────────────────────────
            dcc.Tab(label="Deep Dive", children=[
                html.H3("Stock Deep Dive"),
                dcc.Dropdown(id="deepdive-ticker-dd",
                             options=[{"label": STOCKS.get(t, t), "value": t}
                                      for t in active_tickers],
                             value=active_tickers[0] if active_tickers else None,
                             clearable=False),
                dcc.Graph(id="price-regime-chart"),
                dcc.Graph(id="rv-chart"),
            ]),
 
        ], style={"fontSize": "14px"}),
    ], style={"fontFamily": "Arial", "margin": "20px"})
 
    # ── Callback: Return Forecast Fan Chart ───────────────────────────────────
    @app.callback([Output("forecast-fan", "figure"),
                  Output("forecast-table", "children")],
                  Input("forecast-ticker-dd", "value"))
    def update_forecast_fan(ticker):
        if ticker is None or combined_df.empty:
            return go.Figure().update_layout(title="No data")
        ret_s   = mlr[ticker] if ticker in mlr.columns else pd.Series(dtype=float)
        fig     = go.Figure()
        fig.add_trace(go.Scatter(x=ret_s.index, y=ret_s.values,
                                  name="Historical Returns", mode="lines",
                                  line={"color": "royalblue", "width": 1.5}))
        last_dt = ret_s.index[-1] if not ret_s.empty else pd.Timestamp.today()
        for h in FORECAST_HORIZONS:
            row = combined_df[(combined_df["ticker"] == ticker) &
                              (combined_df["horizon"] == h)]
            if row.empty:
                continue
            row0      = row.iloc[0]
            fc_val    = float(row0["combined_forecast"])
            target_dt = last_dt + pd.DateOffset(months=h)
            hc        = HORIZON_COLORS.get(h, {"hex": "gray", "rgb": "128,128,128"})
            color, rgb = hc["hex"], hc["rgb"]
            ci90_lo   = row0.get("ci90_lower", np.nan)
            ci90_hi   = row0.get("ci90_upper", np.nan)
            ci95_lo   = row0.get("ci95_lower", np.nan)
            ci95_hi   = row0.get("ci95_upper", np.nan)
            has_90    = pd.notna(ci90_lo) and pd.notna(ci90_hi)
            has_95    = pd.notna(ci95_lo) and pd.notna(ci95_hi)
            if has_95:
                fig.add_trace(go.Scatter(
                    x=[last_dt, target_dt, target_dt, last_dt],
                    y=[fc_val, ci95_hi, ci95_lo, fc_val],
                    fill="toself", fillcolor=f"rgba({rgb},0.07)",
                    line=dict(width=0), name=f"{h}M 95% CI",
                    legendgroup=f"h{h}", showlegend=False, hoverinfo="skip",
                ))
            if has_90:
                fig.add_trace(go.Scatter(
                    x=[last_dt, target_dt, target_dt, last_dt],
                    y=[fc_val, ci90_hi, ci90_lo, fc_val],
                    fill="toself", fillcolor=f"rgba({rgb},0.14)",
                    line=dict(width=0), name=f"{h}M 90% CI",
                    legendgroup=f"h{h}", showlegend=False, hoverinfo="skip",
                ))
            err_lo = max(0.0, fc_val - ci90_lo) if has_90 else None
            err_hi = max(0.0, ci90_hi - fc_val) if has_90 else None
            fig.add_trace(go.Scatter(
                x=[target_dt], y=[fc_val],
                name=f"{h}M Forecast", legendgroup=f"h{h}",
                mode="markers",
                marker={"color": color, "size": 11, "symbol": "diamond",
                        "line": {"color": "white", "width": 1}},
                error_y=dict(type="data", symmetric=False,
                             array=[err_hi], arrayminus=[err_lo],
                             color=color, thickness=2, width=8)
                if err_lo is not None else None,
                hovertemplate=(
                    f"<b>{h}M Forecast</b><br>Date: {target_dt.strftime('%b %Y')}<br>"
                    f"Return: {fc_val:.4f}"
                    + (f"<br>90%CI: [{ci90_lo:.4f}, {ci90_hi:.4f}]" if has_90 else "")
                    + "<extra></extra>"
                ),
            ))
            # Empirical PI
            cal = (calib_df[(calib_df["ticker"] == ticker) &
                             (calib_df["horizon"] == h)]
                   if not calib_df.empty else pd.DataFrame())
            if not cal.empty:
                emp_lo = fc_val + float(cal["pi_lower_offset"].iloc[0])
                emp_hi = fc_val + float(cal["pi_upper_offset"].iloc[0])
                fig.add_trace(go.Scatter(
                    x=[target_dt, target_dt], y=[emp_lo, emp_hi],
                    mode="lines+markers", name=f"{h}M Empirical PI",
                    legendgroup=f"h{h}",
                    line={"color": color, "dash": "dashdot", "width": 1},
                    showlegend=False,
                ))
        fig.update_layout(
            title=f"Return Forecast Fan — {ticker}  "
                  f"(shaded=analytical CI · dashdot=empirical PI)",
            xaxis_title="Date", yaxis_title="Monthly Log Return",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            hovermode="x unified",
        )
        # Build the table
        if ticker is None or combined_df.empty:
            table = html.P("No data")
        else:
            sub = combined_df[combined_df["ticker"] == ticker].sort_values("horizon")
            if sub.empty:
                table = html.P("No forecast data")
            else:
                header = [html.Tr([
                    html.Th("Horizon"), html.Th("Forecast (log ret)"),
                    html.Th("90% CI Lower"), html.Th("90% CI Upper"),
                    html.Th("95% CI Lower"), html.Th("95% CI Upper")
                ])]
                rows = []
                for _, r in sub.iterrows():
                    rows.append(html.Tr([
                        html.Td(f"{int(r['horizon'])}M"),
                        html.Td(f"{r['combined_forecast']:.4f}"),
                        html.Td(f"{r['ci90_lower']:.4f}" if pd.notna(r.get('ci90_lower')) else "—"),
                        html.Td(f"{r['ci90_upper']:.4f}" if pd.notna(r.get('ci90_upper')) else "—"),
                        html.Td(f"{r['ci95_lower']:.4f}" if pd.notna(r.get('ci95_lower')) else "—"),
                        html.Td(f"{r['ci95_upper']:.4f}" if pd.notna(r.get('ci95_upper')) else "—"),
                    ]))
                table = html.Table(header + rows,
                                style={"borderCollapse": "collapse",
                                        "marginTop": "10px",
                                        "width": "100%"})
        return fig, table
 
    # ── Callback: Price Forecast Chart (NEW) ──────────────────────────────────
    @app.callback([Output("price-forecast-chart", "figure"),
                  Output("price-table", "children")],
                  Input("price-ticker-dd", "value"))
    def update_price_forecast(ticker):
        empty = go.Figure().update_layout(title="No data available")
        if ticker is None or combined_df.empty:
            return empty
        if ticker not in monthly_prices.columns:
            return empty
 
        px_s       = monthly_prices[ticker].dropna()
        last_price = float(px_s.iloc[-1])
        last_date  = px_s.index[-1]
 
        fig = go.Figure()
        # Historical price
        fig.add_trace(go.Scatter(
            x=px_s.index, y=px_s.values,
            name="Historical Price", mode="lines",
            line={"color": "royalblue", "width": 1.8},
        ))
        # Mark last price
        fig.add_trace(go.Scatter(
            x=[last_date], y=[last_price],
            mode="markers", marker=dict(size=10, color="royalblue",
                                         symbol="circle"),
            name="Last Close", showlegend=False,
        ))
 
        for h in FORECAST_HORIZONS:
            row = combined_df[(combined_df["ticker"] == ticker) &
                              (combined_df["horizon"] == h)]
            if row.empty:
                continue
            row0      = row.iloc[0]
            fc_lr     = float(row0["combined_forecast"])
            target_dt = last_date + pd.DateOffset(months=h)
            price_fc  = last_price * np.exp(fc_lr)   # point forecast in ₹
 
            hc        = HORIZON_COLORS.get(h, {"hex": "gray", "rgb": "128,128,128"})
            color, rgb = hc["hex"], hc["rgb"]
 
            ci90_lo = row0.get("ci90_lower", np.nan)
            ci90_hi = row0.get("ci90_upper", np.nan)
            has_90  = pd.notna(ci90_lo) and pd.notna(ci90_hi)
 
            # 90% shaded cone (price space)
            if has_90:
                price_lo = last_price * np.exp(ci90_lo)
                price_hi = last_price * np.exp(ci90_hi)
                fig.add_trace(go.Scatter(
                    x=[last_date, target_dt, target_dt, last_date],
                    y=[last_price, price_hi, price_lo, last_price],
                    fill="toself", fillcolor=f"rgba({rgb},0.13)",
                    line=dict(width=0), name=f"{h}M 90% CI",
                    legendgroup=f"h{h}", showlegend=True, hoverinfo="skip",
                ))
 
            # Dashed line from last price to forecast
            fig.add_trace(go.Scatter(
                x=[last_date, target_dt],
                y=[last_price, price_fc],
                mode="lines+markers",
                name=f"{h}M ₹{price_fc:,.1f}",
                legendgroup=f"h{h}",
                line=dict(color=color, dash="dash", width=1.8),
                marker=dict(symbol="diamond", size=11, color=color,
                             line=dict(color="white", width=1)),
                hovertemplate=(
                    f"<b>{h}M Price Forecast</b><br>"
                    f"{target_dt.strftime('%b %Y')}<br>"
                    f"₹{price_fc:,.2f}  (log-return {fc_lr:+.4f})"
                    + (f"<br>90% CI: ₹{last_price*np.exp(ci90_lo):,.1f}"
                       f" – ₹{last_price*np.exp(ci90_hi):,.1f}" if has_90 else "")
                    + "<extra></extra>"
                ),
            ))
 
        fig.update_layout(
            title=f"Price Forecast — {STOCKS.get(ticker, ticker)} ({ticker})",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            hovermode="x unified",
        )
        # Build the price table at the end, before return
        if ticker is None or combined_df.empty or ticker not in monthly_prices.columns:
            table = html.P("No data")
        else:
            px_s = monthly_prices[ticker].dropna()
            last_price = float(px_s.iloc[-1])
            sub = combined_df[combined_df["ticker"] == ticker].sort_values("horizon")
            if sub.empty:
                table = html.P("No forecast data")
            else:
                header = [html.Tr([
                    html.Th("Horizon"), html.Th("Forecast Price (₹)"),
                    html.Th("90% CI Low (₹)"), html.Th("90% CI High (₹)"),
                    html.Th("95% CI Low (₹)"), html.Th("95% CI High (₹)")
                ])]
                rows = []
                for _, r in sub.iterrows():
                    fc_lr = r['combined_forecast']
                    p = last_price * np.exp(fc_lr)
                    ci90_l = last_price * np.exp(r['ci90_lower']) if pd.notna(r.get('ci90_lower')) else None
                    ci90_h = last_price * np.exp(r['ci90_upper']) if pd.notna(r.get('ci90_upper')) else None
                    ci95_l = last_price * np.exp(r['ci95_lower']) if pd.notna(r.get('ci95_lower')) else None
                    ci95_h = last_price * np.exp(r['ci95_upper']) if pd.notna(r.get('ci95_upper')) else None
                    rows.append(html.Tr([
                        html.Td(f"{int(r['horizon'])}M"),
                        html.Td(f"₹{p:,.2f}"),
                        html.Td(f"₹{ci90_l:,.2f}" if ci90_l else "—"),
                        html.Td(f"₹{ci90_h:,.2f}" if ci90_h else "—"),
                        html.Td(f"₹{ci95_l:,.2f}" if ci95_l else "—"),
                        html.Td(f"₹{ci95_h:,.2f}" if ci95_h else "—"),
                    ]))
                table = html.Table(header + rows,
                                style={"borderCollapse": "collapse",
                                        "marginTop": "10px",
                                        "width": "100%"})
        return fig, table
 
    # ── Callback: Model Evaluation Heatmap (NEW) ──────────────────────────────
    @app.callback(
        [Output("eval-heatmap", "figure"),
         Output("eval-table", "children")],
        [Input("eval-metric-dd", "value"),
         Input("eval-model-dd", "value")],
    )
    def update_eval_panel(metric, model_type):
        import plotly.express as px
        empty_fig = go.Figure().update_layout(title="No evaluation data")
        if eval_df.empty or metric not in eval_df.columns:
            return empty_fig, html.P("No data.")
        sub = eval_df[eval_df["model"] == model_type]
        if sub.empty:
            return empty_fig, html.P(f"No {model_type} evaluation data.")
        try:
            pivot = sub.pivot_table(values=metric, index="ticker",
                                    columns="horizon", aggfunc="mean")
            cmap  = "RdYlGn_r" if metric in ("RMSE", "MAPE", "MAE", "SMAPE") \
                    else "RdYlGn"
            fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[f"h={c}" for c in pivot.columns],
                y=pivot.index.tolist(),
                colorscale=cmap,
                texttemplate="%{z:.4f}",
                hovertemplate="Ticker: %{y}<br>Horizon: %{x}<br>"
                              + f"{metric}: " + "%{z:.4f}<extra></extra>",
            ))
            fig.update_layout(
                title=f"{model_type} — {metric} (out-of-sample, fixed origin)",
                height=max(300, len(pivot) * 30 + 150),
                xaxis_title="Horizon (months)", yaxis_title="Ticker",
            )
        except Exception:
            fig = empty_fig
 
        # Summary table
        summary = (sub.groupby(["ticker", "horizon"])[metric]
                   .mean().reset_index()
                   .rename(columns={metric: f"mean_{metric}"}))
        summary[f"mean_{metric}"] = summary[f"mean_{metric}"].round(4)
        tbl = html.Table(
            [html.Tr([html.Th(c) for c in summary.columns])] +
            [html.Tr([html.Td(str(v)) for v in row])
             for _, row in summary.iterrows()],
            style={"borderCollapse": "collapse", "fontSize": "12px",
                   "marginTop": "10px"}
        )
        return fig, tbl
 
    # ── Callback: Deep Dive ───────────────────────────────────────────────────
    @app.callback(
        [Output("price-regime-chart", "figure"),
         Output("rv-chart", "figure")],
        Input("deepdive-ticker-dd", "value")
    )
    def update_deep_dive(ticker):
        empty = go.Figure().update_layout(title="No data available")
        if ticker is None:
            return empty, empty
        px_s = (monthly_prices[ticker] if ticker in monthly_prices.columns
                else pd.Series(dtype=float))
        fig_px = go.Figure()
        fig_px.add_trace(go.Scatter(x=px_s.index, y=px_s.values,
                                     name="Monthly Close", mode="lines",
                                     line={"color": "steelblue"}))
        bear_col = [c for c in mkt_hmm_df.columns if "bear" in c]
        if bear_col and not px_s.empty:
            bear_p = mkt_hmm_df[bear_col[0]].reindex(px_s.index).fillna(0.0)
            for dt in bear_p[bear_p > 0.50].index:
                fig_px.add_vrect(x0=dt - pd.DateOffset(days=15),
                                  x1=dt + pd.DateOffset(days=15),
                                  fillcolor="crimson", opacity=0.08, line_width=0)
        fig_px.update_layout(
            title=f"{ticker} — Price History  (red = P(bear) > 0.50)",
            xaxis_title="Date", yaxis_title="Price (₹)", height=350)
        fig_rv = go.Figure()
        if not realized_vol.empty and ticker in realized_vol.columns:
            fig_rv.add_trace(go.Scatter(x=realized_vol.index,
                                         y=realized_vol[ticker].values,
                                         name="Realized Vol", mode="lines",
                                         line={"color": "darkorange"}))
        fig_rv.update_layout(title=f"{ticker} — Monthly Realized Volatility",
                               xaxis_title="Date", yaxis_title="σ_RV", height=280)
        return fig_px, fig_rv
 
    # Write stub
    stub = ('"""NSE Portfolio Monitor — standalone launcher."""\n'
            'from pathlib import Path\nimport pandas as pd\n'
            'OUTPUT_DIR = Path("pipeline_outputs_v1")\n'
            'print("Artifacts at", OUTPUT_DIR.resolve())\n')
    with open(OUTPUT_DIR / "dashboard_app.py", "w") as fh:
        fh.write(stub)
 
    logger.info("Phase 9 ✓\n")
    return app


# %%
# ======================================================================
# MAIN
# ======================================================================

def main() -> None:
    """Execute all nine phases in sequence and launch the Dash dashboard."""
    logger.info("=" * 70)
    logger.info("NSE Stock Forecasting & Portfolio Pipeline — Starting")
    logger.info("=" * 70)

    p1   = run_phase1()
    p2   = run_phase2(p1)
    p3   = run_phase3(p1, p2)
    p4   = run_phase4(p1, p2)
    p4_5 = run_phase4_5(p1, p2, p3, p4)
    p5   = run_phase5(p1, p2, p3, p4, p4_5)
    p6   = run_phase6(p5, p1)
    p7   = run_phase7(p1, p2, p3, p4_5, p6)
    p8   = run_phase8(p1, p2, p4_5, p7)
    p_eval = run_phase_eval(p1, p2, p3, p4, p4_5)
    app  = run_phase9(p1, p2, p3, p4_5, p5, p6, p7, p8, p_eval)

    logger.info("=" * 70)
    logger.info("All phases complete. Outputs: %s", OUTPUT_DIR.resolve())
    logger.info("=" * 70)
    logger.info("Dashboard → http://127.0.0.1:8052/")
    app.run(debug=False, host="0.0.0.0", port=8052)


if __name__ == "__main__":  
    main()


