"""pages/8_forecast.py — Forecast & Trend"""
import streamlit as st
st.set_page_config(page_title="AXON · Forecast", page_icon="◫", layout="wide")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from utils import inject_css, init_session, sidebar_nav, require_auth, require_analysis

inject_css(); init_session()
if not require_auth(): st.stop()
sidebar_nav("8_forecast")
if not require_analysis(): st.stop()

memory   = st.session_state.memory
enhanced = st.session_state.enhanced_result or {}
fc       = memory.get("forecast_intelligence")

st.markdown('<div class="ax-title">Forecast & Trends</div>', unsafe_allow_html=True)
st.markdown('<div class="ax-subtitle">Multi-method time-series forecasting with confidence bands and seasonality analysis</div>', unsafe_allow_html=True)

# ── Core forecast ─────────────────────────────────────────────────────────────
st.markdown('<div class="ax-section ax-section-green">Core Forecast Engine</div>', unsafe_allow_html=True)
if fc and fc.get("forecast_values"):
    trend = fc.get("trend_direction", "Stable")
    tc = "trend-up" if trend == "Upward" else "trend-down" if trend == "Downward" else "trend-stable"
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="fc-chip"><div class="fc-chip-lbl">Trend</div><div class="fc-chip-val {tc}">{trend}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="fc-chip"><div class="fc-chip-lbl">Confidence</div><div class="fc-chip-val">{fc.get("forecast_confidence",0):.0%}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="fc-chip"><div class="fc-chip-lbl">Volatility</div><div class="fc-chip-val" style="color:#ffd166;">{fc.get("volatility_score",0):.3f}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="fc-chip"><div class="fc-chip-lbl">Horizon</div><div class="fc-chip-val">{fc.get("forecast_horizon",0)} periods</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div style="font-size:12px;color:#8b97ae;margin:14px 0 6px;">Target: <strong style="color:#dde4f0;">{fc.get("target_column","")}</strong></div>', unsafe_allow_html=True)

    vals = fc["forecast_values"]
    cb   = fc.get("confidence_band", {})
    chart_df = pd.DataFrame({
        "Forecast": vals,
        "Lower (5%)":  cb.get("lower", []),
        "Upper (95%)": cb.get("upper", []),
    })
    st.line_chart(chart_df, height=280)

    if trend == "Upward":
        st.success("📈 Upward trend detected — positive momentum in the measured system.")
    elif trend == "Downward":
        st.error("📉 Downward trend detected — declining performance may require attention.")
    else:
        st.info("📊 Stable trend — the system is in equilibrium.")
else:
    st.markdown('<div class="ax-card" style="text-align:center;padding:40px;">'
                '<div style="font-size:13px;color:#8b97ae;">No time-series detected. The dataset needs a datetime column for forecasting.</div>'
                '</div>', unsafe_allow_html=True)

# ── Enhanced forecast (Plotly with CI) ───────────────────────────────────────
forecast_blocks = [b for b in enhanced.get("visual_blocks", []) if b.get("type") == "forecast"]
if forecast_blocks:
    st.markdown("---")
    st.markdown('<div class="ax-section">Advanced Forecast (Linear + Exponential Smoothing)</div>', unsafe_allow_html=True)
    for block in forecast_blocks:
        if block.get("chart"):
            st.plotly_chart(block["chart"], use_container_width=True)

        data = block.get("data", {})
        if data:
            m1, m2, m3 = st.columns(3)
            m1.metric("MAPE",  f"{data.get('mape',0):.1f}%",  help="Mean Absolute Percentage Error — lower is better")
            m2.metric("RMSE",  f"{data.get('rmse',0):.2f}",   help="Root Mean Squared Error")
            m3.metric("CI Level", f"{data.get('confidence_level',95)}%")

        seasonality = block.get("seasonality", {})
        if seasonality:
            st.markdown("---")
            st.markdown('<div class="ax-section">Seasonality Analysis</div>', unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            s1.metric("Seasonality Score", seasonality.get("score", 0))
            s2.metric("Confidence",         seasonality.get("confidence","").capitalize())
            s3.metric("Significant Lags",   str(seasonality.get("significant_lags", [])))

            score = seasonality.get("score", 0)
            if score > 60:
                st.success(f"Strong seasonal patterns detected (score: {score}). Consider seasonal models like SARIMA or Prophet.")
            elif score > 30:
                st.warning(f"Moderate seasonality (score: {score}). Incorporate periodic features in models.")
            else:
                st.info(f"Weak seasonality (score: {score}). Trend and irregular components dominate.")