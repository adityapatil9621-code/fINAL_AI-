import streamlit as st
st.set_page_config(page_title="AXON · Analysis", page_icon="◬", layout="wide")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import (
    inject_css, init_session, sidebar_nav,
    require_auth, require_analysis,
    driver_bar
)

# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
inject_css()
init_session()

if not require_auth(): st.stop()
sidebar_nav("5_analysis")
if not require_analysis(): st.stop()

# ─────────────────────────────────────────────
# LOAD SESSION DATA (ONLY ONCE)
# ─────────────────────────────────────────────
memory   = st.session_state.memory
enhanced = st.session_state.enhanced_result or {}
awareness= st.session_state.awareness_result or {}
semantic = st.session_state.semantic_result or {}

insight  = memory["insight_intelligence"]
model    = memory["model_intelligence"]
visuals  = memory.get("visual_intelligence", {}).get("figures", {})

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="ax-title">Analysis Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="ax-subtitle">Drivers · Charts · Awareness · Strategy</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Drivers", "Charts", "Awareness", "Suggestions"])

# =========================================================
# 🔹 TAB 1 — DRIVERS
# =========================================================
with tab1:

    st.markdown('<div class="ax-section">Driver Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Positive Drivers
    with col1:
        st.markdown('<div class="ax-section">Positive Drivers</div>', unsafe_allow_html=True)
        for i, d in enumerate(insight.get("top_positive_drivers", [])):
            st.markdown(driver_bar(i+1, d["feature"], d["impact"], True), unsafe_allow_html=True)

    # Negative Drivers
    with col2:
        st.markdown('<div class="ax-section ax-section-warn">Negative Drivers</div>', unsafe_allow_html=True)
        for i, d in enumerate(insight.get("top_negative_drivers", [])):
            st.markdown(driver_bar(i+1, d["feature"], d["impact"], False), unsafe_allow_html=True)

    # Model Performance
    st.markdown("---")
    st.markdown('<div class="ax-section">Model Performance</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", model.get("selected_model"))
    c2.metric("Confidence", f"{model.get('confidence',0):.2f}")
    c3.metric("Stability", f"{model.get('stability',0):.2f}")

    # Plots
    fi_plot = insight.get("feature_importance_plot")
    shap_plot = insight.get("shap_plot")

    if fi_plot or shap_plot:
        p1, p2 = st.columns(2)
        if fi_plot:
            with p1:
                st.markdown("Feature Importance")
                st.pyplot(fi_plot)
        if shap_plot:
            with p2:
                st.markdown("SHAP Values")
                st.pyplot(shap_plot)


# =========================================================
# 🔹 TAB 2 — CHARTS
# =========================================================
with tab2:

    st.markdown('<div class="ax-section">Visual Analysis</div>', unsafe_allow_html=True)

    # Plotly Charts
    for block in enhanced.get("visual_blocks", []):
        st.markdown(f"### {block.get('type','Analysis')}")
        if block.get("chart"):
            st.plotly_chart(block["chart"], use_container_width=True)

    # Pipeline Charts
    st.markdown("---")
    st.markdown('<div class="ax-section">Pipeline Charts</div>', unsafe_allow_html=True)

    for key, fig in visuals.items():
        if fig:
            st.markdown(f"**{key}**")
            st.pyplot(fig)


# =========================================================
# 🔹 TAB 3 — AWARENESS
# =========================================================
with tab3:

    st.markdown('<div class="ax-section">Data Awareness</div>', unsafe_allow_html=True)

    dom = awareness.get("domain_detection", {})
    anom = awareness.get("anomaly_detection", {})
    dc = awareness.get("decision_confidence", {})

    # Domain
    c1, c2 = st.columns(2)
    c1.metric("Domain", dom.get("detected_domain"))
    c2.metric("Confidence", dom.get("confidence"))

    # Anomaly
    st.markdown("---")
    st.markdown("### Anomaly Detection")

    st.metric("Anomaly %", f"{anom.get('percentage',0)}%")

    # Confidence
    st.markdown("---")
    st.markdown("### Decision Confidence")

    st.metric("Score", dc.get("score"))
    st.metric("Level", dc.get("level"))


# =========================================================
# 🔹 TAB 4 — SUGGESTIONS
# =========================================================
with tab4:

    st.markdown('<div class="ax-section">Strategic Suggestions</div>', unsafe_allow_html=True)

    exec_sum = semantic.get("exec_summary", {})
    ranked   = semantic.get("ranked_insights", [])
    sugs     = awareness.get("suggestions", [])

    # Executive Summary
    if exec_sum:
        st.markdown("### Executive Summary")
        st.write(exec_sum.get("headline", ""))
        st.write(exec_sum.get("confidence_assessment", ""))

    # Ranked Insights
    if ranked:
        st.markdown("### Key Insights")
        for ins in ranked[:5]:
            st.markdown(f"- {ins.get('insight')}")

    # Suggestions
    if sugs:
        st.markdown("### Recommendations")
        for s in sugs:
            st.markdown(f"• {s.get('suggestion','')}")