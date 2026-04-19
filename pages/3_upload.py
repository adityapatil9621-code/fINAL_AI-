"""pages/3_upload.py — Upload & Pipeline Runner"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
st.set_page_config(page_title="AXON · Upload", page_icon="☁", layout="wide")

import time


import pandas as pd
from utils import inject_css, init_session, sidebar_nav, require_auth

from modules.core_engine  import SmartAIEngine
from modules.enhanced_data_analysis  import run_analysis, generate_insights
from modules.enhanced_data_awareness import generate_awareness_report
from modules.enhanced_semantic_layer import build_semantic_narrative, generate_executive_summary, rank_insights

inject_css()
init_session()
if not require_auth(): st.stop()
sidebar_nav("3_upload")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="ax-title">New Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="ax-subtitle">Upload a CSV dataset and run the full AI intelligence pipeline</div>', unsafe_allow_html=True)

# ── Feature tiles ─────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
tiles = [
    ("☁", "Auto-Cleaning", "Deduplication, type inference, outlier IQR capping, constant column removal"),
    ("◈", "Multi-Model Ensemble", "GBM + RF + XGBoost + LightGBM with weighted cross-validation"),
    ("◬", "Advanced Forecasting", "Lag + linear + exponential smoothing with 95% confidence bands"),
    ("◉", "Ensemble Anomaly Detection", "Isolation Forest + LOF + Elliptic Envelope majority vote"),
    ("◷", "Semantic Narratives", "Context-aware AI narratives with executive summary & ranked insights"),
    ("◧", "Driver Analysis", "RF importance + Pearson correlation with cross-validated R² metrics"),
]
for i, (icon, title, desc) in enumerate(tiles):
    col = [c1, c2, c3][i % 3]
    col.markdown(
        f'<div class="ax-card" style="padding:16px;">'
        f'<div style="font-size:20px;margin-bottom:8px;color:#4f9eff;">{icon}</div>'
        f'<div style="font-family:Syne,sans-serif;font-size:13px;font-weight:700;margin-bottom:5px;color:#dde4f0;">{title}</div>'
        f'<div style="font-size:12px;color:#8b97ae;line-height:1.55;">{desc}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ── File uploader ─────────────────────────────────────────────────────────────
st.markdown('<div class="ax-section">Dataset Upload</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop a CSV file here or click to browse", type=["csv"], label_visibility="collapsed")

if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        st.error("Could not parse the file. Please upload a valid CSV."); st.stop()

    if df.empty or len(df) < 10:
        st.error("Dataset too small — minimum 10 rows required."); st.stop()

    st.markdown(
        f'<div class="ax-card-sm" style="display:flex;gap:20px;align-items:center;">'
        f'<span style="font-size:20px;">📊</span>'
        f'<div>'
        f'<div style="font-weight:500;color:#dde4f0;">{uploaded.name}</div>'
        f'<div style="font-size:12px;color:#8b97ae;">{len(df):,} rows × {len(df.columns)} columns · {uploaded.size/1024:.1f} KB</div>'
        f'</div></div>',
        unsafe_allow_html=True
    )
    st.dataframe(df.head(5), use_container_width=True)

    if st.button("Run Full AI Analysis", type="primary", use_container_width=True):
        # ── Pipeline steps UI ─────────────────────────────────────────────────
        STEPS = [
            ("Data Cleaning & Quality Check",      "core"),
            ("Data Understanding & Domain",        "core"),
            ("Feature Engineering",                "core"),
            ("Multi-Model Ensemble Training",      "core"),
            ("Insight & SHAP Extraction",          "core"),
            ("Core Forecasting Engine",            "core"),
            ("Intelligence Scoring",               "core"),
            ("Advanced Statistical Analysis",      "enhanced"),
            ("Ensemble Anomaly Detection",         "awareness"),
            ("Semantic Narrative Generation",      "semantic"),
        ]

        progress_bar = st.progress(0)
        step_container = st.empty()

        def render_steps(current):
            html = '<div class="ax-card" style="padding:16px;">'
            for i, (lbl, _) in enumerate(STEPS):
                if i < current:
                    s, dot, lbl_cls = "done", "pipe-dot-done", "pipe-label-done"
                    status = "✓ done"
                elif i == current:
                    s, dot, lbl_cls = "active", "pipe-dot-active", "pipe-label-active"
                    status = "running…"
                else:
                    s, dot, lbl_cls = "wait", "pipe-dot-wait", "pipe-label-wait"
                    status = "—"
                html += (f'<div class="pipe-step pipe-step-{s}">'
                         f'<div class="pipe-dot {dot}"></div>'
                         f'<span class="{lbl_cls}">{lbl}</span>'
                         f'<span style="margin-left:auto;font-size:11px;font-family:\'DM Mono\',monospace;color:{"#00d4aa" if s=="done" else "#4f9eff" if s=="active" else "#4e5a70"};">{status}</span>'
                         f'</div>')
            html += '</div>'
            step_container.markdown(html, unsafe_allow_html=True)

        error_occurred = False
        try:
            # Stage 1: Core pipeline (steps 0–6)
            render_steps(0); progress_bar.progress(5)
            engine = SmartAIEngine()

            for step_idx in range(1, 7):
                render_steps(step_idx)
                progress_bar.progress(5 + step_idx * 10)
                time.sleep(0.2)

            memory = engine.run_pipeline(df)
            render_steps(7); progress_bar.progress(72)

            # Stage 2: Enhanced analysis (step 7)
            enhanced = run_analysis(df)
            enhanced["insights_text"] = generate_insights(enhanced)
            render_steps(8); progress_bar.progress(84)

            # Stage 3: Awareness (step 8)
            awareness = generate_awareness_report(df, enhanced)
            render_steps(9); progress_bar.progress(92)

            # Stage 4: Semantic (step 9)
            narratives, suggestions = build_semantic_narrative(enhanced, awareness)
            exec_summary = generate_executive_summary(enhanced, awareness)
            ranked = rank_insights(narratives, enhanced)
            progress_bar.progress(100)
            render_steps(10)

            # Store everything in session state
            st.session_state.memory           = memory
            st.session_state.df               = df
            st.session_state.enhanced_result  = enhanced
            st.session_state.awareness_result = awareness
            st.session_state.semantic_result  = {
                "narratives":      narratives,
                "suggestions":     suggestions,
                "exec_summary":    exec_summary,
                "ranked_insights": ranked,
            }
            st.session_state.chat_history         = []
            st.session_state.suggested_questions  = []

        except Exception as e:
            error_occurred = True
            st.error(f"Analysis failed: {e}")

        if not error_occurred:
            st.success("✅ Full analysis complete! Navigate using the sidebar.")
            c1, c2 = st.columns(2)
            c1.button("→ View Dashboard", type="primary",
                      on_click=lambda: st.switch_page("pages/4_dashboard.py"))
            c2.button("→ Chat with Data",
                      on_click=lambda: st.switch_page("pages/11_chat.py"))
else:
    st.markdown(
        '<div class="ax-card" style="text-align:center;padding:52px 32px;">'
        '<div style="font-size:44px;margin-bottom:14px;">☁️</div>'
        '<div style="font-family:Syne,sans-serif;font-size:18px;font-weight:700;margin-bottom:6px;color:#dde4f0;">Drop your CSV dataset</div>'
        '<div style="font-size:13.5px;color:#8b97ae;">or use the uploader above · CSV files only</div>'
        '</div>',
        unsafe_allow_html=True
    )