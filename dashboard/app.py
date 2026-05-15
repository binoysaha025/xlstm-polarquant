import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import matplotlib.pyplot as plt
from PIL import Image
from datetime import timedelta

API_URL = "http://localhost:8000"

st.set_page_config(page_title="xLSTM + PolarQuant", layout="wide")
st.title("xLSTM + PolarQuant — Financial Prediction Demo")
st.markdown("Predict next-day closing price using vanilla xLSTM, xLSTM+PolarQuant, and Transformer.")

# ── SIDEBAR ───────────────────────────────────────────────────────────
st.sidebar.header("Input")
ticker          = st.sidebar.text_input("Ticker", value="AAPL")
prediction_date = st.sidebar.date_input("Prediction date", value=pd.Timestamp('2023-06-01'))
run             = st.sidebar.button("Run Predictions")
st.sidebar.markdown("---")
st.sidebar.caption("Model pulls 60 trading days before the prediction date as input. Actual closing price for that date is fetched to compute accuracy.")

# ── MAIN ──────────────────────────────────────────────────────────────
if run:
    pred_ts   = pd.Timestamp(prediction_date)
    start_ts  = pred_ts - timedelta(days=120)  # extra buffer for 60 trading days

    with st.spinner("Fetching data..."):
        # fetch data including prediction date to get actual price
        df_full = yf.download(ticker,
                              start=start_ts.strftime('%Y-%m-%d'),
                              end=(pred_ts + timedelta(days=5)).strftime('%Y-%m-%d'),
                              auto_adjust=True)
        df_full = df_full[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        if len(df_full) < 61:
            st.error(f"Not enough trading days — got {len(df_full)}, need at least 61.")
            st.stop()

        # last 60 days before prediction date = input sequence
        # the day ON prediction date = actual price to compare against
        input_df   = df_full.iloc[-61:-1]   # 60 days input
        actual_row = df_full.iloc[-1]        # prediction date actual price
        actual_close = float(actual_row['Close'])
        sequence = input_df.values.tolist()

    with st.spinner("Running models..."):
        resp = requests.post(f"{API_URL}/predict",
                             json={"sequence": sequence})
        if resp.status_code != 200:
            st.error(f"API error: {resp.text}")
            st.stop()
        result = resp.json()

    # ── RESULTS ───────────────────────────────────────────────────────
    st.subheader(f"Predictions for {ticker} on {prediction_date}")
    st.markdown(f"**Actual closing price: ${actual_close:.2f}**")

    col1, col2, col3 = st.columns(3)

    def pct_error(pred, actual):
        return abs(pred - actual) / actual * 100

    with col1:
        pred  = result['vanilla_xlstm']
        err   = pct_error(pred, actual_close)
        delta = pred - actual_close
        st.metric("vanilla xLSTM", f"${pred:.2f}", f"{delta:+.2f}")
        st.caption(f"error: {err:.2f}% | val loss: 0.1707 | float32")

    with col2:
        pred  = result['polarquant_xlstm']
        err   = pct_error(pred, actual_close)
        delta = pred - actual_close
        st.metric("xLSTM + PolarQuant", f"${pred:.2f}", f"{delta:+.2f}")
        st.caption(f"error: {err:.2f}% | val loss: 0.0998 | 32x compressed ✓")

    with col3:
        pred  = result['transformer']
        err   = pct_error(pred, actual_close)
        delta = pred - actual_close
        st.metric("Transformer", f"${pred:.2f}", f"{delta:+.2f}")
        st.caption(f"error: {err:.2f}% | val loss: 0.1208 | float32")

    if result['drift_detected']:
        st.warning("⚠️ Data drift detected — market regime differs from training data (2010-2022). Predictions may be less reliable.")
    else:
        st.success("✓ No data drift detected — market regime consistent with training data.")

    # ── PRICE CHART ───────────────────────────────────────────────────
    st.subheader(f"{ticker} — 60 Day Input Window + Predictions vs Actual")
    fig, ax = plt.subplots(figsize=(12, 4))
    closes = input_df['Close'].values
    ax.plot(range(60), closes, color='steelblue', label='Input Window (Actual Close)')
    ax.axvline(x=59, color='gray', linestyle='--', alpha=0.5)

    # actual price
    ax.scatter(60, actual_close, color='green', s=150, zorder=6,
               marker='*', label=f'Actual ${actual_close:.2f}')

    # predictions
    ax.scatter(60, result['vanilla_xlstm'],    color='steelblue',  s=100, zorder=5, label=f'vanilla xLSTM ${result["vanilla_xlstm"]:.2f}')
    ax.scatter(60, result['polarquant_xlstm'], color='darkorange', s=100, zorder=5, label=f'PolarQuant ${result["polarquant_xlstm"]:.2f}')
    ax.scatter(60, result['transformer'],      color='crimson',    s=100, zorder=5, label=f'Transformer ${result["transformer"]:.2f}')

    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # ── ACCURACY TABLE ────────────────────────────────────────────────
    st.subheader("Accuracy Summary")
    acc_df = pd.DataFrame({
        'Model':          ['vanilla xLSTM', 'xLSTM + PolarQuant', 'Transformer'],
        'Predicted':      [f"${result['vanilla_xlstm']:.2f}",
                           f"${result['polarquant_xlstm']:.2f}",
                           f"${result['transformer']:.2f}"],
        'Actual':         [f"${actual_close:.2f}"] * 3,
        'Error ($)':      [f"{result['vanilla_xlstm'] - actual_close:+.2f}",
                           f"{result['polarquant_xlstm'] - actual_close:+.2f}",
                           f"{result['transformer'] - actual_close:+.2f}"],
        '% Error':        [f"{pct_error(result['vanilla_xlstm'], actual_close):.2f}%",
                           f"{pct_error(result['polarquant_xlstm'], actual_close):.2f}%",
                           f"{pct_error(result['transformer'], actual_close):.2f}%"],
        'Memory':         ['float32', 'int8 (4x compressed)', 'float32'],
        'Inference':      ['O(n)', 'O(n)', 'O(n²)'],
    })
    st.dataframe(acc_df, hide_index=True, use_container_width=True)

    # ── BENCHMARK CHARTS ──────────────────────────────────────────────
    st.subheader("Benchmark Results")
    bench_img = Image.open('/Users/binoy.saha/xlstm-polarquant/xlstm-polarquant/benchmark/results.png')
    st.image(bench_img, use_container_width=True)