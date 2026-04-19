"""
utils.py — Shared helpers, CSS theme, and session state management
"""
import streamlit as st

# ── Global CSS ────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0a0d12;
    color: #dde4f0;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #10141c !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span { color: #8b97ae !important; }
section[data-testid="stSidebar"] .stSelectbox label { color: #8b97ae !important; }

/* ── Cards ── */
.ax-card {
    background: #1a2030;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.ax-card-sm {
    background: #1a2030;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.ax-card-accent {
    background: #1a2030;
    border-left: 3px solid #4f9eff;
    border-radius: 0 12px 12px 0;
    padding: 14px 16px;
    margin-bottom: 10px;
}

/* ── Metric cards ── */
.ax-metric {
    background: #1a2030;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
}
.ax-metric-label {
    font-size: 11px;
    color: #4e5a70;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 500;
    margin-bottom: 8px;
}
.ax-metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #dde4f0;
}
.ax-metric-sub { font-size: 11px; color: #4e5a70; margin-top: 4px; }

/* ── Headings ── */
.ax-title {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: #dde4f0;
    margin-bottom: 4px;
}
.ax-subtitle { font-size: 14px; color: #8b97ae; margin-bottom: 24px; }
.ax-section {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: #dde4f0;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.ax-section::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4f9eff;
    flex-shrink: 0;
}
.ax-section-green::before { background: #00d4aa; }
.ax-section-warn::before  { background: #ff6b6b; }
.ax-section-gold::before  { background: #ffd166; }

/* ── Badges ── */
.ax-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}
.ax-badge-blue   { background: rgba(79,158,255,.14); color: #4f9eff; }
.ax-badge-green  { background: rgba(0,212,170,.14);  color: #00d4aa; }
.ax-badge-warn   { background: rgba(255,209,102,.14); color: #ffd166; }
.ax-badge-red    { background: rgba(255,107,107,.14); color: #ff6b6b; }
.ax-badge-purple { background: rgba(124,92,252,.14); color: #7c5cfc; }

/* ── Priority badges ── */
.pri-High   { background: rgba(255,107,107,.14); color: #ff6b6b; }
.pri-Medium { background: rgba(255,209,102,.14); color: #ffd166; }
.pri-Low    { background: rgba(0,212,170,.14);  color: #00d4aa; }

/* ── Driver bars ── */
.ax-driver {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 13px;
    border-radius: 9px;
    background: #202838;
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 7px;
}
.ax-driver-rank { font-family: 'DM Mono', monospace; font-size: 11px; color: #4e5a70; width: 18px; text-align: right; }
.ax-driver-name { flex: 1; font-size: 13px; font-weight: 500; color: #dde4f0; }
.ax-driver-bar  { width: 80px; height: 4px; background: #28334a; border-radius: 2px; overflow: hidden; }
.ax-driver-fill-pos { height: 100%; background: linear-gradient(90deg,#4f9eff,#7c5cfc); border-radius: 2px; }
.ax-driver-fill-neg { height: 100%; background: linear-gradient(90deg,#ff6b6b,#ff8e53); border-radius: 2px; }
.ax-driver-impact { font-family: 'DM Mono', monospace; font-size: 11px; color: #8b97ae; width: 40px; text-align: right; }

/* ── Insight items ── */
.ax-insight {
    padding: 12px 16px;
    border-radius: 0 10px 10px 0;
    margin-bottom: 9px;
    background: #202838;
    font-size: 13.5px;
    line-height: 1.65;
    color: #dde4f0;
}
.ax-insight-high   { border-left: 3px solid #ff6b6b; }
.ax-insight-medium { border-left: 3px solid #ffd166; }
.ax-insight-low    { border-left: 3px solid #00d4aa; }
.ax-insight-info   { border-left: 3px solid #4f9eff; }

/* ── Suggestion cards ── */
.ax-sug {
    background: #202838;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 15px 17px;
    margin-bottom: 12px;
}
.ax-sug-header { display: flex; gap: 8px; margin-bottom: 8px; align-items: center; }
.ax-sug-text { font-size: 13.5px; color: #dde4f0; line-height: 1.6; margin-bottom: 6px; }
.ax-sug-rationale { font-size: 12px; color: #8b97ae; font-style: italic; margin-bottom: 8px; }
.ax-sug-action { font-size: 12px; color: #8b97ae; padding: 3px 0; }
.ax-sug-action::before { content: '→ '; color: #4f9eff; }

/* ── Auth ── */
.ax-auth-card {
    max-width: 420px;
    margin: 40px auto;
    background: #1a2030;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 22px;
    padding: 38px 34px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}
.ax-google-btn {
    display: flex; align-items: center; justify-content: center; gap: 10px;
    width: 100%; padding: 11px 16px;
    background: #202838; border: 1px solid rgba(255,255,255,0.11);
    border-radius: 9px; color: #dde4f0;
    font-family: 'DM Sans',sans-serif; font-size: 14px; font-weight: 500;
    cursor: pointer; text-decoration: none; transition: background .15s;
}
.ax-google-btn:hover { background: #28334a; }
.ax-divider { display:flex;align-items:center;gap:14px;color:#4e5a70;font-size:12px;margin:18px 0; }
.ax-divider::before,.ax-divider::after { content:'';flex:1;height:1px;background:rgba(255,255,255,.08); }

/* ── Pipeline steps ── */
.pipe-step { display:flex;align-items:center;gap:12px;padding:9px 13px;border-radius:9px;margin-bottom:5px;transition:all .3s; }
.pipe-step-done   { background:rgba(0,212,170,.06); }
.pipe-step-active { background:rgba(79,158,255,.07); }
.pipe-step-wait   { opacity:.35; }
.pipe-dot { width:8px;height:8px;border-radius:50%;flex-shrink:0; }
.pipe-dot-done   { background:#00d4aa; }
.pipe-dot-active { background:#4f9eff; animation:pulse 1.4s ease infinite; }
.pipe-dot-wait   { background:#28334a; }
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(79,158,255,.4);}50%{box-shadow:0 0 0 5px rgba(79,158,255,0);} }
.pipe-label-done   { font-size:13px;color:#00d4aa;font-weight:500; }
.pipe-label-active { font-size:13px;color:#4f9eff;font-weight:500; }
.pipe-label-wait   { font-size:13px;color:#4e5a70; }

/* ── Forecast chips ── */
.fc-chip { background:#202838;border-radius:9px;padding:12px 14px;text-align:center; }
.fc-chip-lbl { font-size:10.5px;color:#4e5a70;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px; }
.fc-chip-val { font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:#dde4f0; }
.trend-up   { color:#00d4aa!important; }
.trend-down { color:#ff6b6b!important; }
.trend-stable { color:#ffd166!important; }

/* ── Domain bar ── */
.domain-bar { display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid rgba(255,255,255,.05); }
.domain-name { font-size:12.5px;font-weight:500;color:#8b97ae;width:110px;text-transform:capitalize; }
.domain-bar-outer { flex:1;height:5px;background:#202838;border-radius:2px;overflow:hidden; }
.domain-bar-inner { height:100%;background:linear-gradient(90deg,#4f9eff,#7c5cfc);border-radius:2px; }
.domain-score { font-family:'DM Mono',monospace;font-size:11px;color:#4e5a70;width:20px;text-align:right; }

/* ── Audit log ── */
.ax-log { font-family:'DM Mono',monospace;font-size:12px;color:#8b97ae;line-height:2;background:#10141c;border-radius:10px;padding:14px 16px; }
.ax-log-line:nth-child(odd) { color:#4e5a70; }

/* ── Anomaly row ── */
.ax-anomaly { display:flex;align-items:center;gap:10px;padding:7px 12px;border-radius:8px;background:rgba(255,107,107,.05);border:1px solid rgba(255,107,107,.1);margin-bottom:5px; }

/* ── Chat ── */
.chat-bubble-user {
    background: linear-gradient(135deg,#4f9eff,#7c5cfc);
    color: #fff;
    padding: 11px 15px;
    border-radius: 13px 13px 4px 13px;
    font-size: 13.5px;
    line-height: 1.6;
    margin-left: auto;
    max-width: 80%;
    width: fit-content;
}
.chat-bubble-ai {
    background: #202838;
    border: 1px solid rgba(255,255,255,.08);
    color: #dde4f0;
    padding: 11px 15px;
    border-radius: 13px 13px 13px 4px;
    font-size: 13.5px;
    line-height: 1.6;
    max-width: 85%;
    width: fit-content;
}
.q-chip {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    background: #202838;
    border: 1px solid rgba(255,255,255,.1);
    font-size: 12px;
    color: #8b97ae;
    cursor: pointer;
    margin: 3px;
}

/* ── Streamlit overrides ── */
.stButton > button {
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border: 1px solid rgba(255,255,255,0.11) !important;
    background: #202838 !important;
    color: #dde4f0 !important;
    transition: all .15s !important;
}
.stButton > button:hover {
    background: #28334a !important;
    border-color: rgba(255,255,255,0.2) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#4f9eff,#3a8ef0) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 2px 12px rgba(79,158,255,.3) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 20px rgba(79,158,255,.45) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stMetric"] {
    background: #1a2030;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 14px 16px;
}
div[data-testid="stMetricLabel"] p { color: #8b97ae !important; font-size: 12px !important; }
div[data-testid="stMetricValue"]   { color: #dde4f0 !important; font-family: 'Syne',sans-serif !important; }
.stTextInput > div > div > input,
.stTextArea textarea {
    background: #202838 !important;
    border: 1px solid rgba(255,255,255,0.11) !important;
    border-radius: 9px !important;
    color: #dde4f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #4f9eff !important;
    box-shadow: 0 0 0 3px rgba(79,158,255,.12) !important;
}
.stSelectbox > div > div { background: #202838 !important; border-color: rgba(255,255,255,.11) !important; border-radius: 9px !important; color: #dde4f0 !important; }
.stFileUploader { background: #1a2030 !important; border: 2px dashed rgba(255,255,255,.12) !important; border-radius: 14px !important; }
.stFileUploader:hover { border-color: #4f9eff !important; }
.stTabs [data-baseweb="tab-list"] { background: #202838 !important; border-radius: 10px !important; padding: 4px !important; gap: 3px !important; }
.stTabs [data-baseweb="tab"] { border-radius: 7px !important; color: #8b97ae !important; font-family: 'DM Sans',sans-serif !important; font-size: 13px !important; }
.stTabs [aria-selected="true"] { background: #28334a !important; color: #dde4f0 !important; }
.stExpander { background: #1a2030 !important; border: 1px solid rgba(255,255,255,.07) !important; border-radius: 12px !important; }
.stAlert { border-radius: 10px !important; }
div[data-testid="stProgress"] > div { background: #202838 !important; border-radius: 4px !important; }
div[data-testid="stProgress"] > div > div { background: linear-gradient(90deg,#4f9eff,#7c5cfc) !important; border-radius: 4px !important; }
.stDataFrame { border-radius: 10px !important; border: 1px solid rgba(255,255,255,.07) !important; }
</style>
"""

GOOGLE_SVG = (
    '<svg width="16" height="16" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844a4.14 4.14 0 0 1-1.796 2.716v2.259h2.908c1.702-1.567 2.684-3.875 2.684-6.615z" fill="#4285F4"/>'
    '<path d="M9 18c2.43 0 4.467-.806 5.956-2.184l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z" fill="#34A853"/>'
    '<path d="M3.964 10.706A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.706V4.962H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.038l3.007-2.332z" fill="#FBBC05"/>'
    '<path d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.962L3.964 7.294C4.672 5.163 6.656 3.58 9 3.58z" fill="#EA4335"/>'
    '</svg>'
)


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def init_session():
    defaults = {
        "user_id":   None,
        "username":  None,
        "memory":    None,
        "df":        None,
        "analysis_id": None,
        "chat_history": [],
        "suggested_questions": [],
        "enhanced_result": None,
        "awareness_result": None,
        "semantic_result": None,
        "just_verified": False,
        "resend_mode": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def require_auth():
    """Redirect to login if not authenticated. Returns True if authenticated."""
    if not st.session_state.get("user_id"):
        st.switch_page("pages/1_login.py")
        return False
    return True


def require_analysis():
    """Show warning if no analysis has been run yet."""
    if not st.session_state.get("memory"):
        st.markdown('<div class="ax-card" style="text-align:center;padding:48px 24px;">'
                    '<div style="font-size:36px;margin-bottom:14px;">📊</div>'
                    '<div style="font-family:Syne,sans-serif;font-size:18px;font-weight:700;margin-bottom:8px;color:#dde4f0;">No analysis yet</div>'
                    '<div style="font-size:13.5px;color:#8b97ae;">Upload a dataset and run analysis first.</div>'
                    '</div>', unsafe_allow_html=True)
        if st.button("→ Go to Upload", type="primary"):
            st.switch_page("pages/3_upload.py")
        return False
    return True


def sidebar_nav(active: str):
    """Render the sidebar navigation."""
    username = st.session_state.get("username", "")
    has_analysis = st.session_state.get("memory") is not None

    with st.sidebar:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;padding:4px 0 20px;">'
            f'<div style="width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,#4f9eff,#7c5cfc);display:flex;align-items:center;justify-content:center;font-family:Syne,sans-serif;font-size:16px;font-weight:800;color:#fff;">A</div>'
            f'<div><div style="font-family:Syne,sans-serif;font-size:17px;font-weight:800;color:#dde4f0;">AXON</div>'
            f'<div style="font-size:10px;color:#4e5a70;letter-spacing:.1em;text-transform:uppercase;">AI Intelligence</div></div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div style="font-size:10px;color:#4e5a70;text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;">Workspace</div>', unsafe_allow_html=True)

        nav_items = [
            ("3_upload",   "☁",  "Upload",     "pages/3_upload.py"),
            ("4_dashboard","◈",  "Dashboard",  "pages/4_dashboard.py"),
            ("5_drivers",  "◧",  "Drivers",    "pages/5_drivers.py"),
            ("6_charts",   "◬",  "Charts",     "pages/6_charts.py"),
            ("7_awareness","◉",  "Awareness",  "pages/7_awareness.py"),
            ("8_forecast", "◫",  "Forecast",   "pages/8_forecast.py"),
            ("9_suggestions","◷","Suggestions","pages/9_suggestions.py"),
            ("10_data",    "◻",  "Data",       "pages/10_data.py"),
            ("11_chat",    "◎",  "AI Chat",    "pages/11_chat.py"),
        ]
        for page_id, icon, label, path in nav_items:
            is_active = active == page_id
            badge = ' <span style="float:right;font-size:10px;background:rgba(0,212,170,.12);color:#00d4aa;padding:1px 7px;border-radius:20px;">Ready</span>' if has_analysis and page_id != "3_upload" else ""
            style = "background:#28334a;color:#4f9eff;" if is_active else "color:#8b97ae;"
            if st.sidebar.button(f"{icon}  {label}", key=f"nav_{page_id}", use_container_width=True):
                st.switch_page(path)

        st.sidebar.markdown("---")
        st.sidebar.markdown(
            f'<div style="display:flex;align-items:center;gap:9px;padding:8px 10px;background:#202838;border-radius:9px;">'
            f'<div style="width:28px;height:28px;border-radius:7px;background:linear-gradient(135deg,#7c5cfc,#4f9eff);display:flex;align-items:center;justify-content:center;font-family:Syne,sans-serif;font-size:12px;font-weight:700;color:#fff;">{username[0].upper() if username else "U"}</div>'
            f'<div><div style="font-size:13px;font-weight:500;color:#dde4f0;">{username}</div>'
            f'<div style="font-size:11px;color:#4e5a70;">Analyst</div></div>'
            f'</div>',
            unsafe_allow_html=True
        )
        if st.sidebar.button("Logout", key="nav_logout", use_container_width=True):
            for k in ["user_id", "username", "memory", "df", "chat_history",
                      "suggested_questions", "enhanced_result", "awareness_result", "semantic_result"]:
                st.session_state[k] = None if k in ["user_id","username","memory","df"] else []
            st.switch_page("pages/1_login.py")


def metric_card(label, value, sub="", color="#dde4f0"):
    return (f'<div class="ax-metric">'
            f'<div class="ax-metric-label">{label}</div>'
            f'<div class="ax-metric-value" style="color:{color}">{value}</div>'
            f'{"<div class=ax-metric-sub>"+sub+"</div>" if sub else ""}'
            f'</div>')


def driver_bar(rank, name, impact, positive=True):
    pct = min(impact * 100, 100)
    fill_class = "ax-driver-fill-pos" if positive else "ax-driver-fill-neg"
    return (f'<div class="ax-driver">'
            f'<span class="ax-driver-rank">#{rank}</span>'
            f'<span class="ax-driver-name">{name}</span>'
            f'<div class="ax-driver-bar"><div class="{fill_class}" style="width:{pct}%"></div></div>'
            f'<span class="ax-driver-impact">{impact*100:.1f}%</span>'
            f'</div>')


def insight_card(text, level="info"):
    cls = f"ax-insight ax-insight-{level}"
    return f'<div class="{cls}">{text}</div>'


def badge(text, kind="blue"):
    return f'<span class="ax-badge ax-badge-{kind}">{text}</span>'


def grade_color(g):
    return {"A": "#00d4aa", "B": "#4f9eff", "C": "#ffd166", "D": "#ff6b6b", "F": "#ff6b6b"}.get(g, "#8b97ae")