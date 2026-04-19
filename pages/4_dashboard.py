"""pages/4_dashboard.py — Executive Dashboard"""
import streamlit as st
st.set_page_config(page_title="AXON · Dashboard", page_icon="◈", layout="wide")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from utils import inject_css, init_session, sidebar_nav, require_auth, require_analysis, metric_card, badge, grade_color

inject_css()
init_session()
if not require_auth(): st.stop()
sidebar_nav("4_dashboard")
if not require_analysis(): st.stop()

memory   = st.session_state.memory
enhanced = st.session_state.enhanced_result or {}
awareness= st.session_state.awareness_result or {}
semantic = st.session_state.semantic_result or {}
meta     = memory["metadata"]
model    = memory["model_intelligence"]
insight  = memory["insight_intelligence"]
score    = memory["intelligence_score"]
fc       = memory.get("forecast_intelligence")
dom      = awareness.get("domain_detection", {})
dc       = awareness.get("decision_confidence", {})
anom     = awareness.get("anomaly_detection", {})
exec_sum = semantic.get("exec_summary", {})
ranked   = semantic.get("ranked_insights", [])

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="ax-title">Dashboard</div>', unsafe_allow_html=True)
if exec_sum.get("headline"):
    st.markdown(
        f'<div class="ax-card" style="border-left:3px solid #4f9eff;border-radius:0 14px 14px 0;">'
        f'<div style="font-family:Syne,sans-serif;font-size:17px;font-weight:700;margin-bottom:6px;color:#dde4f0;">{exec_sum["headline"]}</div>'
        f'<div style="font-size:12.5px;color:#8b97ae;">{exec_sum.get("confidence_assessment","")}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
gc = grade_color(score.get("grade", ""))
cards = [
    (c1, "Rows",           f"{meta['rows']:,}",                       f"{meta['columns']} columns",       "#dde4f0"),
    (c2, "Data Quality",   f"{meta['quality_score']:.0%}",            "Excellent" if meta['quality_score']>=.8 else "Good" if meta['quality_score']>=.6 else "Needs work", "#4f9eff"),
    (c3, "Intelligence",   score.get("grade","—"),                    f"{score.get('score',0):.0%} · {model.get('task_type','')}", gc),
    (c4, "Confidence",     f"{dc.get('score',0)}/100",                dc.get("level","—"),                "#00d4aa"),
    (c5, "Model CV",       f"{model.get('confidence',0):.0%}",        f"Stability {model.get('stability',0):.0%}", "#4f9eff"),
    (c6, "Anomaly Rate",   f"{anom.get('percentage',0)}%",            f"{anom.get('details',{}).get('ensemble',{}).get('methods_used',0)} methods", "#ffd166" if anom.get('percentage',0)>5 else "#00d4aa"),
]
for col, *args in cards:
    col.markdown(metric_card(*args), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Model + Score rings ───────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.markdown('<div class="ax-card">', unsafe_allow_html=True)
    st.markdown('<div class="ax-section">Ensemble Model</div>', unsafe_allow_html=True)
    ma, mb = st.columns([1, 2])
    ma.metric("Confidence", f"{model.get('confidence',0):.0%}")
    with mb:
        st.markdown(f'<div style="font-family:Syne,sans-serif;font-size:15px;font-weight:700;color:#dde4f0;margin-bottom:6px;">{model.get("selected_model","")}</div>', unsafe_allow_html=True)
        st.markdown(badge(model.get("task_type","regression"), "blue"), unsafe_allow_html=True)
        if "Ensemble" in str(model.get("selected_model", "")):
            st.markdown("<br>" + badge("Multi-Model", "purple"), unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;color:#8b97ae;margin-top:8px;">Stability: {model.get("stability",0):.0%}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="ax-card">', unsafe_allow_html=True)
    st.markdown('<div class="ax-section">Intelligence Score</div>', unsafe_allow_html=True)
    ra, rb = st.columns([1, 2])
    ra.metric("Grade", score.get("grade","—"))
    with rb:
        st.markdown(f'<div style="font-family:Syne,sans-serif;font-size:38px;font-weight:800;color:{gc};line-height:1;">{score.get("grade","—")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12.5px;color:#8b97ae;margin-top:6px;">{score.get("summary","Overall intelligence rating")}</div>', unsafe_allow_html=True)
        if insight.get("residual_bias_detected"):
            st.markdown(badge("Residual bias detected", "warn"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Forecast sparkline ────────────────────────────────────────────────────────
st.markdown('<div class="ax-section ax-section-green">Forecast</div>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="ax-card">', unsafe_allow_html=True)
    if fc and fc.get("forecast_values"):
        fa, fb, fc_col = st.columns(3)
        trend = fc.get("trend_direction","Stable")
        tc = "trend-up" if trend=="Upward" else "trend-down" if trend=="Downward" else "trend-stable"
        fa.markdown(f'<div class="fc-chip"><div class="fc-chip-lbl">Trend</div><div class="fc-chip-val {tc}">{trend}</div></div>', unsafe_allow_html=True)
        fb.markdown(f'<div class="fc-chip"><div class="fc-chip-lbl">Confidence</div><div class="fc-chip-val">{fc.get("forecast_confidence",0):.0%}</div></div>', unsafe_allow_html=True)
        fc_col.markdown(f'<div class="fc-chip"><div class="fc-chip-lbl">Volatility</div><div class="fc-chip-val" style="color:#ffd166;">{fc.get("volatility_score",0):.3f}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:12px;color:#8b97ae;margin-bottom:6px;">Target: <strong style="color:#dde4f0;">{fc.get("target_column","")}</strong> · {fc.get("forecast_horizon",0)} periods ahead</div>', unsafe_allow_html=True)
        chart_df = pd.DataFrame({"Forecast": fc["forecast_values"]})
        if fc.get("confidence_band"):
            chart_df["Lower"] = fc["confidence_band"]["lower"]
            chart_df["Upper"] = fc["confidence_band"]["upper"]
        st.line_chart(chart_df, height=220)
    else:
        st.info("No time-series detected in this dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Ranked Insights ───────────────────────────────────────────────────────────
if ranked:
    st.markdown('<div class="ax-section ax-section-warn">Key Insights</div>', unsafe_allow_html=True)
    for ins in ranked[:5]:
        imp = ins.get("importance","Medium")
        lvl = "high" if imp=="High" else "low" if imp=="Low" else "medium"
        st.markdown(
            f'<div class="ax-insight ax-insight-{lvl}">'
            f'{badge(imp, "red" if imp=="High" else "warn" if imp=="Medium" else "green")}&nbsp;'
            f'{badge(ins.get("actionability","")+" actionability","blue")}'
            f'<div style="margin-top:7px;">{ins.get("insight","")}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# ── Executive key findings & risks ───────────────────────────────────────────
if exec_sum.get("key_findings") or exec_sum.get("risk_factors"):
    left2, right2 = st.columns(2)
    with left2:
        if exec_sum.get("key_findings"):
            st.markdown('<div class="ax-section">Key Findings</div>', unsafe_allow_html=True)
            for f in exec_sum["key_findings"]:
                st.markdown(f'<div class="ax-card-sm" style="font-size:13px;">→ {f}</div>', unsafe_allow_html=True)
    with right2:
        if exec_sum.get("risk_factors"):
            st.markdown('<div class="ax-section ax-section-warn">Risk Factors</div>', unsafe_allow_html=True)
            for r in exec_sum["risk_factors"]:
                st.markdown(f'<div class="ax-card-sm" style="border-left:3px solid #ff6b6b;border-radius:0 9px 9px 0;font-size:13px;">⚠ {r}</div>', unsafe_allow_html=True)

# ── Audit log ────────────────────────────────────────────────────────────────
with st.expander("Pipeline Audit Log"):
    html = '<div class="ax-log">'
    for line in memory.get("audit_log", []):
        html += f'<div class="ax-log-line">{line}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ── Export ────────────────────────────────────────────────────────────────────
report_lines = [
    "═══════════════════════════════════",
    "     AXON AI INTELLIGENCE REPORT   ",
    "═══════════════════════════════════",
    "",
    exec_sum.get("headline",""),
    exec_sum.get("confidence_assessment",""),
    "",
    f"Dataset: {meta['rows']:,} rows × {meta['columns']} columns",
    f"Domain:  {meta.get('domain','general')}",
    f"Quality: {meta['quality_score']:.0%}",
    f"Grade:   {score.get('grade','—')}  ({score.get('score',0):.0%})",
    "",
    "── Key Findings ──",
    *[f"  • {f}" for f in exec_sum.get("key_findings",[])],
    "",
    "── Risk Factors ──",
    *[f"  ⚠ {r}" for r in exec_sum.get("risk_factors",[])],
    "",
    "── Pipeline Audit ──",
    *memory.get("audit_log",[]),
]
st.download_button("↓ Export Report", "\n".join(report_lines),
                   file_name="axon_report.txt", mime="text/plain")