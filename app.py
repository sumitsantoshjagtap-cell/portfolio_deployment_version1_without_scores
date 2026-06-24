"""Render‑ready Dash server – loads pre‑computed pipeline outputs."""
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Import the dashboard builder from your main script
# (rename your original file to pipeline.py if it's still called dashboard_app.py)
from pipeline import run_phase9, STOCKS, FORECAST_HORIZONS, HORIZON_COLORS, MIN_TRAIN_MONTHS

OUTPUT_DIR = Path("pipeline_outputs")
ANNUAL_RISK_FREE = 0.065

# ─── Load all saved outputs ───────────────────────────────────────────
monthly_prices = pd.read_parquet(OUTPUT_DIR / "monthly_prices.parquet")
mlr_clean = pd.read_parquet(OUTPUT_DIR / "monthly_log_returns_clean.parquet")
bench_returns = pd.read_parquet(OUTPUT_DIR / "benchmark_returns.parquet").squeeze()
active_tickers = list(monthly_prices.columns)
monthly_rf = (1 + ANNUAL_RISK_FREE) ** (1/12) - 1

realized_vol = pd.read_parquet(OUTPUT_DIR / "realized_volatility.parquet")
market_indicators = pd.read_parquet(OUTPUT_DIR / "market_indicators.parquet")

# market HMM probs (columns starting with 'market_hmm_')
all_probs = pd.read_parquet(OUTPUT_DIR / "filtered_probs_all.parquet")
market_filtered_df = all_probs.filter(regex="^market_hmm_")

perf_df = pd.read_csv(OUTPUT_DIR / "performance_metrics.csv")
calib_df = pd.read_csv(OUTPUT_DIR / "calibration_results.csv")

combined_df = pd.read_parquet(OUTPUT_DIR / "combined_forecasts.parquet")

weights_df = pd.read_csv(OUTPUT_DIR / "portfolio_weights.csv")
weights_clean = dict(zip(weights_df["ticker"], weights_df["weight"]))

portfolio_rets = pd.read_parquet(OUTPUT_DIR / "portfolio_returns.parquet").squeeze()
portfolio_val = pd.read_parquet(OUTPUT_DIR / "portfolio_value.parquet").squeeze()
drawdown_s = pd.read_parquet(OUTPUT_DIR / "drawdown_series.parquet").squeeze()
perf_summary_df = pd.read_csv(OUTPUT_DIR / "performance_summary.csv")
perf_summary = perf_summary_df.iloc[0].to_dict() if not perf_summary_df.empty else {}

eval_df = pd.read_csv(OUTPUT_DIR / "model_evaluation.csv")

# ─── Reconstruct the dictionaries exactly as run_phase9 expects ────────
p1 = {
    "monthly_prices": monthly_prices,
    "benchmark_returns": bench_returns,
    "active_tickers": active_tickers,
    "monthly_rf": monthly_rf,
}
p2 = {"monthly_log_returns_clean": mlr_clean}
p3 = {
    "realized_volatility": realized_vol,
    "market_indicators": market_indicators,
}
p4_5 = {"market_filtered_df": market_filtered_df}
p5 = {"perf_df": perf_df, "calib_df": calib_df}
p6 = {"combined_df": combined_df}
p7 = {"weights_clean": weights_clean}
p8 = {
    "portfolio_returns": portfolio_rets,
    "portfolio_value": portfolio_val,
    "drawdown_series": drawdown_s,
    "perf_summary": perf_summary,
}
p_eval = {"eval_df": eval_df}

# ─── Build the Dash app and expose the Flask server ───────────────────
app = run_phase9(p1, p2, p3, p4_5, p5, p6, p7, p8, p_eval)
server = app.server

if __name__ == "__main__":
    # Local test: python app.py
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)
