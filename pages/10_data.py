"""pages/10_data.py — Data Explorer"""
import streamlit as st
st.set_page_config(page_title="AXON · Data", page_icon="◻", layout="wide")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from utils import inject_css, init_session, sidebar_nav, require_auth, require_analysis, badge

inject_css(); init_session()
if not require_auth(): st.stop()
sidebar_nav("10_data")
if not require_analysis(): st.stop()

memory   = st.session_state.memory
df       = st.session_state.df
awareness= st.session_state.awareness_result or {}
prof     = awareness.get("data_profile", {})

st.markdown('<div class="ax-title">Data Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="ax-subtitle">Dataset preview, column profiling, cardinality analysis, and pipeline audit</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Column Profile", "Cardinality", "Audit Log"])

# ── PREVIEW TAB ───────────────────────────────────────────────────────────────
with tab1:
    if df is not None:
        st.markdown(f'<div style="font-size:12px;color:#8b97ae;margin-bottom:10px;">{len(df):,} rows × {len(df.columns)} columns</div>', unsafe_allow_html=True)
        n_rows = st.slider("Rows to preview", 5, min(100, len(df)), 10)
        st.dataframe(df.head(n_rows), use_container_width=True)

        # Basic stats
        st.markdown("---")
        st.markdown('<div class="ax-section">Descriptive Statistics</div>', unsafe_allow_html=True)
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            st.dataframe(numeric.describe().round(3), use_container_width=True)
    else:
        st.info("No dataset in session — re-upload to explore.")

# ── COLUMN PROFILE TAB ────────────────────────────────────────────────────────
with tab2:
    if df is not None:
        selected_col = st.selectbox("Select column to profile", df.columns.tolist())
        if selected_col:
            col_data = df[selected_col]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Type",          str(col_data.dtype))
            c2.metric("Unique Values", col_data.nunique())
            c3.metric("Missing",       f"{col_data.isna().sum()} ({col_data.isna().mean():.1%})")
            c4.metric("Cardinality",   f"{col_data.nunique()/len(df):.3f}")

            if pd.api.types.is_numeric_dtype(col_data):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mean",   f"{col_data.mean():.3f}")
                m2.metric("Median", f"{col_data.median():.3f}")
                m3.metric("Std",    f"{col_data.std():.3f}")
                m4.metric("Skew",   f"{col_data.skew():.3f}")
                st.line_chart(col_data.dropna().reset_index(drop=True), height=200)
            else:
                st.markdown('<div class="ax-section" style="margin-top:14px;">Top Values</div>', unsafe_allow_html=True)
                vc = col_data.value_counts().head(15)
                st.bar_chart(vc)
    else:
        st.info("No dataset loaded.")

# ── CARDINALITY TAB ───────────────────────────────────────────────────────────
with tab3:
    cardinality = prof.get("cardinality", {})
    if cardinality:
        rows = []
        for col, d in cardinality.items():
            ratio = d.get("cardinality_ratio", 0)
            col_type = "identifier" if ratio > 0.9 else "categorical" if ratio < 0.05 else "numeric"
            badge_kind = "red" if ratio > 0.9 else "green" if ratio < 0.05 else "blue"
            rows.append({
                "Column": col,
                "Unique Values": d.get("unique_values", 0),
                "Cardinality Ratio": round(ratio, 4),
                "Type": col_type,
            })
        cdf = pd.DataFrame(rows).sort_values("Cardinality Ratio", ascending=False)
        st.dataframe(cdf, use_container_width=True)

        identifiers = [r["Column"] for r in rows if r["Type"] == "identifier"]
        categoricals = [r["Column"] for r in rows if r["Type"] == "categorical"]
        if identifiers:
            st.markdown("---")
            st.markdown('<div class="ax-section ax-section-warn">Identifier columns (excluded from model)</div>', unsafe_allow_html=True)
            st.markdown(" ".join(badge(c, "red") for c in identifiers), unsafe_allow_html=True)
        if categoricals:
            st.markdown('<div class="ax-section ax-section-green">Low-cardinality categorical columns</div>', unsafe_allow_html=True)
            st.markdown(" ".join(badge(c, "green") for c in categoricals), unsafe_allow_html=True)
    else:
        st.info("Cardinality data not available — re-run analysis.")

# ── AUDIT LOG TAB ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="ax-section">Pipeline Audit Log</div>', unsafe_allow_html=True)
    html = '<div class="ax-log">'
    for line in memory.get("audit_log", []):
        html += f'<div class="ax-log-line">{line}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    # Dropped columns
    dcp = memory.get("data_profile", {})
    dropped = dcp.get("dropped_columns", []) if isinstance(dcp, dict) else []
    if dropped:
        st.markdown("---")
        st.markdown('<div class="ax-section ax-section-warn">Columns dropped during cleaning</div>', unsafe_allow_html=True)
        st.markdown(" ".join(badge(c, "warn") for c in dropped), unsafe_allow_html=True)