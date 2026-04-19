"""
pages/11_chat.py — AXON AI Chat Analyst
Full analysis inside the chat: charts, forecasts, anomalies,
drivers, summaries, distribution — all rendered inline.
History stored in session state with save/restore.
"""

import streamlit as st
st.set_page_config(page_title="AXON · AI Chat", page_icon="◎", layout="wide")

import sys, os, random
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import inject_css, init_session, sidebar_nav, require_auth, require_analysis

try:
    from moduels.enhanced_data_analysis import (
        advanced_forecast, analyze_drivers, analyze_relationships, detect_seasonality
    )
    ENHANCED = True
except Exception:
    ENHANCED = False

try:
    from enhanced_data_awareness import detect_anomalies
    AWARENESS = True
except Exception:
    AWARENESS = False

try:
    from chat_engine import ChatEngine, generate_dynamic_questions
    OLLAMA = True
except Exception:
    OLLAMA = False

inject_css()
init_session()
if not require_auth(): st.stop()
sidebar_nav("11_chat")
if not require_analysis(): st.stop()

st.markdown("""
<style>
.msg-user-row{display:flex;justify-content:flex-end;margin-bottom:6px;}
.msg-ai-row{display:flex;gap:10px;margin-bottom:6px;align-items:flex-start;}
.msg-ai-avatar{width:30px;height:30px;border-radius:8px;flex-shrink:0;
  background:linear-gradient(135deg,#4f9eff,#7c5cfc);
  display:flex;align-items:center;justify-content:center;font-size:14px;margin-top:2px;}
.bubble-user{background:linear-gradient(135deg,#4f9eff,#7c5cfc);color:#fff;
  padding:11px 16px;border-radius:14px 14px 4px 14px;
  font-size:14px;line-height:1.6;max-width:76%;word-wrap:break-word;}
.bubble-ai{background:#202838;border:1px solid rgba(255,255,255,.08);color:#dde4f0;
  padding:13px 16px;border-radius:4px 14px 14px 14px;
  font-size:14px;line-height:1.7;max-width:100%;word-wrap:break-word;}
.bubble-ai strong{color:#4f9eff;}
.bubble-ai code{background:#10141c;border-radius:4px;padding:1px 5px;
  font-family:'DM Mono',monospace;font-size:12px;color:#00d4aa;}
.summary-box{background:rgba(79,158,255,.07);border:1px solid rgba(79,158,255,.2);
  border-radius:10px;padding:10px 14px;margin-top:10px;
  font-size:12.5px;color:#8b97ae;line-height:1.6;}
.summary-box strong{color:#4f9eff;}
.inline-insight{background:#1a2030;border-left:3px solid #4f9eff;
  border-radius:0 9px 9px 0;padding:10px 13px;margin:6px 0;
  font-size:13px;line-height:1.6;color:#dde4f0;}
.inline-insight.high{border-left-color:#ff6b6b;}
.inline-insight.medium{border-left-color:#ffd166;}
.inline-insight.low{border-left-color:#00d4aa;}
.intent-tag{display:inline-block;padding:2px 9px;border-radius:20px;
  font-size:10.5px;font-weight:600;margin-bottom:7px;margin-right:4px;
  background:rgba(79,158,255,.12);color:#4f9eff;}
.hist-item{padding:9px 12px;border-radius:9px;margin-bottom:5px;
  background:#202838;border:1px solid rgba(255,255,255,.06);}
.hist-item-q{font-size:12.5px;color:#dde4f0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.hist-item-meta{font-size:10.5px;color:#4e5a70;margin-top:3px;}
.mini-metric{display:inline-block;background:#28334a;border-radius:7px;
  padding:5px 10px;margin:3px;font-size:12px;}
.mini-metric span{color:#4f9eff;font-weight:600;}
.chat-empty{display:flex;flex-direction:column;align-items:center;
  justify-content:center;padding:60px 20px;text-align:center;}
.chat-empty-icon{font-size:44px;margin-bottom:14px;}
.chat-empty-title{font-family:'Syne',sans-serif;font-size:20px;font-weight:700;
  color:#dde4f0;margin-bottom:6px;}
.chat-empty-sub{font-size:13.5px;color:#8b97ae;}
</style>
""", unsafe_allow_html=True)

# ── Session defaults ──────────────────────────────────────────────────────────
for k, v in [("chat_history",[]), ("chat_sessions",[]), ("chat_qs",[]), ("chat_prefill","")]:
    if k not in st.session_state:
        st.session_state[k] = v

df       = st.session_state.df
memory   = st.session_state.memory
enhanced = st.session_state.get("enhanced_result") or {}
awareness= st.session_state.get("awareness_result") or {}
semantic = st.session_state.get("semantic_result") or {}
meta     = memory.get("metadata", {})
insight  = memory.get("insight_intelligence", {})
model_i  = memory.get("model_intelligence", {})
fc_mem   = memory.get("forecast_intelligence")
numeric_cols = df.select_dtypes(include=np.number).columns.tolist() if df is not None else []

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#8b97ae",
    template="plotly_dark",
)

# ── Intent detection ──────────────────────────────────────────────────────────
INTENTS = {
    "summary":     ["summary","overview","describe","tell me","explain","profile","what is this"],
    "correlation": ["correlation","correlate","heatmap","relationship","relate","connected","between"],
    "histogram":   ["histogram","distribution","spread","frequency","hist","how is.*distributed"],
    "boxplot":     ["boxplot","box plot","quartile","iqr","box"],
    "forecast":    ["forecast","predict","future","trend","projection","next","will","estimate"],
    "anomaly":     ["anomaly","anomalies","outlier","unusual","abnormal","strange","weird","detect"],
    "drivers":     ["driver","important","importance","feature","factor","influence","impact","shap","what drives"],
    "scatter":     ["scatter","plot.*vs","versus","against","compare two"],
    "stats":       ["mean","average","median","std","min","max","statistics","stats"],
    "missing":     ["missing","null","nan","empty","incomplete"],
    "domain":      ["domain","industry","sector","type of data"],
    "model":       ["model","accuracy","r2","score","performance","ensemble","confidence"],
    "suggestions": ["suggest","recommendation","advice","action","improve","should i"],
    "compare":     ["compare","top","bottom","highest","lowest","best","worst","rank"],
    "executive":   ["executive","report","brief","overview report"],
}

def detect_intent(q):
    ql = q.lower()
    found = [intent for intent, kws in INTENTS.items() if any(kw in ql for kw in kws)]
    return found if found else ["general"]

def extract_column(q):
    if df is None: return None
    ql = q.lower()
    # longest match first
    for col in sorted(df.columns, key=len, reverse=True):
        if col.lower() in ql:
            return col
    return None

# ── Response builder ──────────────────────────────────────────────────────────
def build_response(question):
    intents = detect_intent(question)
    col     = extract_column(question)
    ql      = question.lower()
    resp    = dict(text="", summary="", charts=[], tables=[], metrics=[], insights=[], intent=intents)

    # SUMMARY / EXECUTIVE
    if "summary" in intents or "executive" in intents:
        rows  = meta.get("rows", 0)
        ncols = meta.get("columns", 0)
        domain= meta.get("domain","general")
        qual  = meta.get("quality_score", 0)
        grade = memory.get("intelligence_score",{}).get("grade","—")
        exec_s= semantic.get("exec_summary",{})
        if exec_s.get("headline"):
            txt = f"**{exec_s['headline']}**\n\n{exec_s.get('confidence_assessment','')}"
            if exec_s.get("key_findings"):
                txt += "\n\n**Key Findings:**\n" + "\n".join(f"→ {f}" for f in exec_s["key_findings"])
            if exec_s.get("risk_factors"):
                txt += "\n\n**Risk Factors:**\n" + "\n".join(f"⚠ {r}" for r in exec_s["risk_factors"])
            resp["text"] = txt
        else:
            resp["text"] = (
                f"**Dataset Overview**\n\n"
                f"Your dataset has **{rows:,} rows × {ncols} columns** in the **{domain}** domain.\n\n"
                f"Data quality: **{qual:.0%}** · Intelligence grade: **{grade}**"
            )
        if df is not None:
            resp["tables"].append(df.describe().round(3))
        resp["metrics"] = [("Rows",f"{rows:,}"),("Columns",str(ncols)),("Quality",f"{qual:.0%}"),
                            ("Grade",grade),("Domain",domain.capitalize()),
                            ("Model",model_i.get("selected_model","—")[:20])]
        resp["summary"] = f"{domain.capitalize()} dataset — {rows:,} rows, quality {qual:.0%}, grade {grade}."

    # CORRELATION
    if "correlation" in intents:
        num = df.select_dtypes(include=np.number) if df is not None else pd.DataFrame()
        if not num.empty and len(num.columns) > 1:
            corr = num.corr().round(3)
            fig  = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                              title="Correlation Heatmap", zmin=-1, zmax=1, aspect="auto")
            fig.update_layout(**DARK_LAYOUT)
            resp["charts"].append(fig)
            corr_abs = corr.abs()
            np.fill_diagonal(corr_abs.values, 0)
            max_val = corr_abs.max().max()
            idx = np.where(corr_abs == max_val)
            c1, c2 = corr.columns[idx[0][0]], corr.columns[idx[1][0]]
            v = corr.loc[c1, c2]
            resp["text"] = (
                f"**Correlation Heatmap**\n\n"
                f"Strongest pair: **{c1}** vs **{c2}** (r = **{v:.3f}**) — "
                + ("positive relationship." if v > 0 else "negative relationship.")
            )
            resp["insights"].append((
                f"{'Strong' if abs(v)>.7 else 'Moderate' if abs(v)>.4 else 'Weak'} "
                f"{'positive' if v>0 else 'negative'} correlation: {c1} ↔ {c2} (r={v:.3f})",
                "high" if abs(v)>.7 else "medium"
            ))
            resp["summary"] = f"Correlation heatmap. Strongest pair: {c1} vs {c2} at r={v:.3f}."
        else:
            resp["text"] = "Not enough numeric columns to compute correlations."

    # HISTOGRAM / DISTRIBUTION
    if "histogram" in intents or ("distribution" in ql and "correlation" not in intents):
        target = col or (numeric_cols[-1] if numeric_cols else None)
        if target and df is not None and target in df.columns:
            skew = float(df[target].skew())
            kurt = float(df[target].kurtosis())
            fig  = make_subplots(rows=1, cols=2, subplot_titles=[f"Distribution — {target}", "Box Plot"])
            fig.add_trace(go.Histogram(x=df[target], marker_color="#4f9eff", nbinsx=30, name="Dist"), row=1, col=1)
            fig.add_trace(go.Box(y=df[target], marker_color="#7c5cfc", name="Box", boxpoints="outliers"), row=1, col=2)
            fig.update_layout(**DARK_LAYOUT, showlegend=False,
                               title=f"Distribution of {target} — Skew: {skew:.2f}, Kurt: {kurt:.2f}")
            resp["charts"].append(fig)
            resp["text"] = (
                f"**Distribution of `{target}`**\n\n"
                f"Mean: **{df[target].mean():.3f}** · Median: **{df[target].median():.3f}** · "
                f"Std: **{df[target].std():.3f}**\n\n"
                f"Skewness: **{skew:.2f}** "
                + ("→ right-skewed" if skew>1 else "→ left-skewed" if skew<-1 else "→ approximately symmetric")
            )
            resp["metrics"] = [("Mean",f"{df[target].mean():.3f}"),("Median",f"{df[target].median():.3f}"),
                                ("Std",f"{df[target].std():.3f}"),("Skewness",f"{skew:.3f}"),
                                ("Kurtosis",f"{kurt:.3f}"),("Missing",f"{df[target].isna().sum()}")]
            if abs(skew) > 1:
                resp["insights"].append((
                    f"`{target}` is {'right' if skew>0 else 'left'}-skewed (skew={skew:.2f}). "
                    "Consider log transformation for better model performance.", "medium"
                ))
            resp["summary"] = f"Distribution of {target}: mean={df[target].mean():.2f}, skew={skew:.2f}."

    # SCATTER
    if "scatter" in intents:
        mentioned = [c for c in (df.columns if df is not None else []) if c.lower() in ql]
        if len(mentioned) >= 2:
            x_c, y_c = mentioned[0], mentioned[1]
        elif len(numeric_cols) >= 2:
            x_c, y_c = numeric_cols[-2], numeric_cols[-1]
        else:
            x_c = y_c = None
        if x_c and y_c and df is not None:
            cv, pv = pearsonr(df[x_c].dropna(), df[y_c].dropna())
            fig = px.scatter(df, x=x_c, y=y_c, trendline="ols",
                              title=f"{x_c} vs {y_c} — r={cv:.3f}, p={pv:.4f}")
            fig.update_layout(**DARK_LAYOUT)
            resp["charts"].append(fig)
            resp["text"] = (
                f"**Scatter: {x_c} vs {y_c}**\n\n"
                f"r = **{cv:.3f}** — "
                + ("statistically significant (p<0.05)" if pv<0.05 else f"not significant (p={pv:.4f})")
            )
            resp["summary"] = f"Scatter {x_c} vs {y_c}, r={cv:.3f}."

    # BOX PLOT
    if "boxplot" in intents:
        cols_p = [col] if col else numeric_cols[:6]
        if cols_p and df is not None:
            fig = go.Figure()
            for c in cols_p:
                if c in df.columns:
                    fig.add_trace(go.Box(y=df[c], name=c, boxpoints="outliers"))
            fig.update_layout(**DARK_LAYOUT, title="Box Plot Comparison")
            resp["charts"].append(fig)
            resp["text"] = f"Box plots for **{len(cols_p)} column(s)**. Dots are outliers (>1.5×IQR)."
            resp["summary"] = f"Box plots: {', '.join(cols_p[:4])}."

    # STATS
    if "stats" in intents:
        target = col
        if target and df is not None and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
            s = df[target]
            resp["text"] = (
                f"**Statistics for `{target}`**\n\n"
                f"Count: {s.count():,} · Mean: {s.mean():.4f} · Median: {s.median():.4f}\n"
                f"Std: {s.std():.4f} · Min: {s.min():.4f} · Max: {s.max():.4f}\n"
                f"25th %ile: {s.quantile(0.25):.4f} · 75th %ile: {s.quantile(0.75):.4f}"
            )
            resp["metrics"] = [("Mean",f"{s.mean():.4f}"),("Median",f"{s.median():.4f}"),
                                ("Std",f"{s.std():.4f}"),("Min",f"{s.min():.4f}"),("Max",f"{s.max():.4f}")]
            resp["summary"] = f"Stats for {target}: mean={s.mean():.3f}, std={s.std():.3f}."
        elif df is not None:
            resp["tables"].append(df.describe().round(3))
            resp["text"] = "**Descriptive statistics for all numeric columns:**"
            resp["summary"] = "Full dataset statistics shown."

    # FORECAST
    if "forecast" in intents:
        target = col or (numeric_cols[-1] if numeric_cols else None)
        done = False
        if target and df is not None and target in df.columns and ENHANCED:
            series = df[target].dropna()
            fc_res = advanced_forecast(series, periods=20)
            if fc_res:
                hist_x = list(range(len(fc_res["historical"])))
                fc_x   = list(range(len(fc_res["historical"]), len(fc_res["historical"])+len(fc_res["forecast"])))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist_x, y=fc_res["historical"], name="Historical",
                                          line=dict(color="#4f9eff", width=2)))
                fig.add_trace(go.Scatter(x=fc_x, y=fc_res["forecast"], name="Forecast",
                                          line=dict(color="#00d4aa", width=2, dash="dash")))
                fig.add_trace(go.Scatter(x=fc_x, y=fc_res["forecast_upper"],
                                          line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=fc_x, y=fc_res["forecast_lower"],
                                          line=dict(width=0), fill="tonexty",
                                          fillcolor="rgba(0,212,170,0.12)", name="95% CI"))
                fig.update_layout(**DARK_LAYOUT, hovermode="x unified",
                                   title=f"Forecast: {target} — MAPE {fc_res['mape']}%")
                resp["charts"].append(fig)
                seas = detect_seasonality(series) if ENHANCED else {"score":0,"significant_lags":[]}
                resp["text"] = (
                    f"**Forecast: `{target}` — next 20 periods**\n\n"
                    f"MAPE: **{fc_res['mape']}%** · RMSE: **{fc_res['rmse']}** · CI: **95%**\n"
                    f"Seasonality score: **{seas['score']}** ({seas.get('confidence','—')} confidence)"
                )
                resp["metrics"] = [("MAPE",f"{fc_res['mape']}%"),("RMSE",str(fc_res['rmse'])),
                                    ("Horizon","20 periods"),("CI","95%"),("Seasonality",str(seas['score']))]
                resp["insights"].append((
                    f"Forecast accuracy is {'excellent' if fc_res['mape']<5 else 'good' if fc_res['mape']<15 else 'moderate'} "
                    f"(MAPE={fc_res['mape']}%). "
                    + ("Suitable for planning." if fc_res["mape"]<15 else "Consider advanced models (SARIMA, Prophet)."),
                    "low" if fc_res["mape"]<5 else "medium"
                ))
                resp["summary"] = f"Forecast {target}: MAPE={fc_res['mape']}%, 20 periods, CI=95%."
                done = True
        if not done and fc_mem and fc_mem.get("forecast_values"):
            vals = fc_mem["forecast_values"]
            cb   = fc_mem.get("confidence_band",{})
            chart_df = pd.DataFrame({"Forecast":vals,"Lower":cb.get("lower",[]),"Upper":cb.get("upper",[])})
            fig = px.line(chart_df, title=f"Pipeline Forecast — {fc_mem.get('target_column','')}")
            fig.update_layout(**DARK_LAYOUT)
            resp["charts"].append(fig)
            trend = fc_mem.get("trend_direction","Stable")
            resp["text"] = (
                f"**Pipeline Forecast — {fc_mem.get('target_column','')}**\n\n"
                f"Trend: **{trend}** · Confidence: **{fc_mem.get('forecast_confidence',0):.0%}** · "
                f"Volatility: **{fc_mem.get('volatility_score',0):.3f}**"
            )
            resp["summary"] = f"Forecast trend: {trend}."
        elif not done:
            resp["text"] = "Specify a column: 'forecast sales' or 'forecast [column name]'"

    # ANOMALY
    if "anomaly" in intents:
        if AWARENESS and df is not None:
            anom_pct, anom_scores, anom_details = detect_anomalies(df)
            top_idx = anom_scores.nlargest(10).index.tolist()
            num = df.select_dtypes(include=np.number)
            if len(num.columns) >= 2:
                colors = ["#ff6b6b" if i in top_idx else "#4f9eff" for i in range(len(df))]
                fig = go.Figure(go.Scatter(
                    x=num.iloc[:,0], y=num.iloc[:,1], mode="markers",
                    marker=dict(color=colors, size=7, opacity=0.75),
                    text=[f"Row {i}" for i in range(len(df))],
                ))
                fig.update_layout(**DARK_LAYOUT,
                                   title=f"Anomaly Detection — {anom_pct}% anomalies (red = flagged)",
                                   xaxis_title=num.columns[0], yaxis_title=num.columns[1])
                resp["charts"].append(fig)
            resp["text"] = (
                f"**Ensemble Anomaly Detection**\n\n"
                f"Overall anomaly rate: **{anom_pct}%** (3 methods: Isolation Forest + LOF + Elliptic Envelope)\n\n"
                + "\n".join(f"• **{m.replace('_',' ').title()}**: {d.get('anomalies_detected',0)} anomalies ({d.get('percentage',0)}%)"
                             for m, d in anom_details.items())
            )
            resp["metrics"] = [("Anomaly Rate",f"{anom_pct}%"),("Methods","3"),
                                ("Top Flagged Index",str(top_idx[0]) if top_idx else "—")]
            lvl = "high" if anom_pct>10 else "medium" if anom_pct>5 else "low"
            resp["insights"].append((
                f"{'High' if anom_pct>10 else 'Moderate' if anom_pct>5 else 'Low'} anomaly rate ({anom_pct}%). "
                + ("⚠️ Investigate flagged rows before decisions." if anom_pct>5 else "Data consistency is good."),
                lvl
            ))
            resp["summary"] = f"Anomaly detection: {anom_pct}% anomalous rows across 3 methods."
        else:
            if df is not None:
                num = df.select_dtypes(include=np.number)
                Q1, Q3 = num.quantile(0.25), num.quantile(0.75)
                IQR = Q3 - Q1
                out = ((num<(Q1-1.5*IQR))|(num>(Q3+1.5*IQR))).any(axis=1)
                pct = out.mean()*100
                resp["text"] = f"IQR-based outlier detection: **{pct:.1f}%** of rows are potential outliers."
                resp["summary"] = f"IQR outliers: {pct:.1f}%."

    # DRIVERS
    if "drivers" in intents:
        pos = insight.get("top_positive_drivers", [])
        neg = insight.get("top_negative_drivers", [])
        if pos or neg:
            all_d = [(d["feature"], d["impact"], True) for d in pos] + [(d["feature"], d["impact"], False) for d in neg]
            names   = [d[0] for d in all_d]
            impacts = [d[1] if d[2] else -d[1] for d in all_d]
            colors  = ["#4f9eff" if d[2] else "#ff6b6b" for d in all_d]
            fig = go.Figure(go.Bar(x=impacts, y=names, orientation="h",
                                    marker_color=colors,
                                    text=[f"{abs(v):.1%}" for v in impacts],
                                    textposition="outside"))
            fig.update_layout(**DARK_LAYOUT, title="Feature Drivers (Blue=Positive / Red=Negative)",
                               xaxis_title="Impact Score")
            resp["charts"].append(fig)
            top = pos[0] if pos else None
            resp["text"] = (
                f"**Driver Analysis**\n\n"
                + (f"Top driver: **{top['feature']}** ({top['impact']:.1%} impact)\n\n" if top else "")
                + f"Model: **{model_i.get('selected_model','')}** · "
                + f"Confidence: **{model_i.get('confidence',0):.0%}**"
            )
            if top:
                resp["insights"].append((
                    f"**{top['feature']}** is the dominant driver ({top['impact']:.1%} impact). "
                    "Focus optimization here for maximum effect.", "high"
                ))
            resp["summary"] = f"Top driver: {top['feature'] if top else 'N/A'}."
        elif ENHANCED and df is not None and len(numeric_cols) >= 2:
            target = numeric_cols[-1]
            X = df[numeric_cols[:-1]].fillna(0)
            y = df[target].fillna(df[target].mean())
            try:
                dr = analyze_drivers(X, y, numeric_cols[:-1])
                feats = dr["importance"].head(10)
                fig = px.bar(x=feats.values, y=feats.index, orientation="h",
                              title=f"Top Drivers of {target} — R²={dr['r2_test']}")
                fig.update_layout(**DARK_LAYOUT)
                resp["charts"].append(fig)
                resp["text"] = (
                    f"**Enhanced Driver Analysis — `{target}`**\n\n"
                    f"Top driver: **{dr['top_driver']}** · Test R²: **{dr['r2_test']}** · "
                    f"CV R²: **{dr['r2_cv_mean']} ± {dr['r2_cv_std']}**"
                )
                resp["summary"] = f"Top driver: {dr['top_driver']}, R²={dr['r2_test']}."
            except Exception:
                resp["text"] = "Driver analysis could not complete with this dataset structure."

    # MODEL
    if "model" in intents:
        sc = memory.get("intelligence_score",{})
        resp["text"] = (
            f"**Model Intelligence**\n\n"
            f"Model: **{model_i.get('selected_model','')}**\n"
            f"Task: **{model_i.get('task_type','').capitalize()}**\n"
            f"CV Confidence: **{model_i.get('confidence',0):.0%}** · "
            f"Stability: **{model_i.get('stability',0):.0%}**\n\n"
            f"Intelligence Grade: **{sc.get('grade','—')}** ({sc.get('score',0):.0%})"
        )
        resp["metrics"] = [("Model",model_i.get("selected_model","—")[:20]),
                            ("Task",model_i.get("task_type","").capitalize()),
                            ("Confidence",f"{model_i.get('confidence',0):.0%}"),
                            ("Stability",f"{model_i.get('stability',0):.0%}"),
                            ("Grade",sc.get("grade","—"))]
        resp["summary"] = f"Model: {model_i.get('selected_model','')}, grade {sc.get('grade','—')}."

    # MISSING
    if "missing" in intents:
        if df is not None:
            miss = df.isna().sum()
            miss = miss[miss>0].sort_values(ascending=False)
            if miss.empty:
                resp["text"] = "✅ No missing values in your dataset!"
                resp["summary"] = "No missing values."
            else:
                fig = px.bar(x=miss.values, y=miss.index, orientation="h",
                              title="Missing Values by Column")
                fig.update_traces(marker_color="#ff6b6b")
                fig.update_layout(**DARK_LAYOUT)
                resp["charts"].append(fig)
                pct = df.isna().mean().mean()*100
                resp["text"] = (
                    f"**Missing Data — {pct:.2f}% overall**\n\n"
                    + "\n".join(f"• `{c}`: {n} missing ({n/len(df):.1%})" for c,n in miss.head(10).items())
                )
                resp["insights"].append((
                    f"{pct:.1f}% missing overall. "
                    + ("High — implement data collection improvements." if pct>15
                       else "Moderate — median imputation applied." if pct>5
                       else "Low — minimal impact."),
                    "high" if pct>15 else "medium"
                ))
                resp["summary"] = f"Missing: {pct:.1f}%, {len(miss)} columns affected."

    # SUGGESTIONS
    if "suggestions" in intents:
        sugs    = awareness.get("suggestions",[])
        ranked  = semantic.get("ranked_insights",[])
        exec_s  = semantic.get("exec_summary",{})
        txt = "**Strategic Insights & Recommendations**\n\n"
        if ranked:
            txt += "**Top Insights:**\n"
            for ins in ranked[:3]:
                txt += f"[{ins.get('importance','')}] {ins.get('insight','')}\n\n"
        if sugs:
            txt += "**Action Items:**\n"
            for s in sugs[:4]:
                txt += f"→ [{s.get('priority','')}] {s.get('suggestion','')}\n"
        resp["text"] = txt if txt.strip() != "**Strategic Insights & Recommendations**" else "Run full analysis to get suggestions."
        resp["insights"] = [(ins.get("insight",""), "high" if ins.get("importance")=="High" else "medium") for ins in ranked[:4]]
        resp["summary"] = f"{len(sugs)} strategic suggestions."

    # COMPARE
    if "compare" in intents and col and df is not None and col in df.columns:
        n = 5
        top_n = df.nlargest(n, col)
        bot_n = df.nsmallest(n, col)
        fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Top {n}",f"Bottom {n}"])
        fig.add_trace(go.Bar(y=top_n[col], marker_color="#00d4aa", name=f"Top {n}"), row=1, col=1)
        fig.add_trace(go.Bar(y=bot_n[col], marker_color="#ff6b6b", name=f"Bottom {n}"), row=1, col=2)
        fig.update_layout(**DARK_LAYOUT, title=f"Top vs Bottom {n} by {col}", showlegend=False)
        resp["charts"].append(fig)
        resp["text"] = f"**Top vs Bottom {n} rows by `{col}`**\n\nMax: **{df[col].max():.3f}** · Min: **{df[col].min():.3f}**"
        resp["summary"] = f"Top/bottom {n} by {col}."

    # OLLAMA fallback
    if (not resp["text"] or "general" in intents) and OLLAMA:
        try:
            answer = ChatEngine().respond(question, memory)
            if answer and not answer.startswith("⚠️"):
                resp["text"] = (resp["text"] + f"\n\n---\n**AI:** {answer}") if resp["text"] else answer
                if not resp["summary"]:
                    resp["summary"] = (answer[:140]+"…") if len(answer)>140 else answer
        except Exception:
            pass

    # Final fallback
    if not resp["text"]:
        resp["text"] = (
            "I can analyse your data! Try:\n\n"
            "• **summary** — full dataset overview\n"
            "• **correlation** — relationship heatmap\n"
            "• **histogram of [col]** — distribution chart\n"
            "• **forecast** — predict next 20 periods\n"
            "• **anomalies** — detect unusual data\n"
            "• **drivers** — top influencing features\n"
            "• **missing data** — null value analysis\n"
            "• **model** — model performance\n"
            "• **suggestions** — strategic recommendations"
        )
        resp["summary"] = "Help guide shown."

    return resp

# ── Render message ────────────────────────────────────────────────────────────
def render_message(msg):
    role = msg["role"]
    if role == "user":
        st.markdown(f'<div class="msg-user-row"><div class="bubble-user">{msg["text"]}</div></div>', unsafe_allow_html=True)
    else:
        av_col, content_col = st.columns([0.05, 0.95])
        with av_col:
            st.markdown('<div class="msg-ai-avatar">🧠</div>', unsafe_allow_html=True)
        with content_col:
            intents = msg.get("intent",[])
            if intents and intents != ["general"]:
                st.markdown(" ".join(f'<span class="intent-tag">{i}</span>' for i in intents), unsafe_allow_html=True)
            if msg.get("text"):
                txt = msg["text"].replace("\n","<br>")
                st.markdown(f'<div class="bubble-ai">{txt}</div>', unsafe_allow_html=True)
            if msg.get("metrics"):
                parts = " ".join(f'<div class="mini-metric">{l}: <span>{v}</span></div>' for l,v in msg["metrics"])
                st.markdown(f'<div style="display:flex;flex-wrap:wrap;margin:8px 0;">{parts}</div>', unsafe_allow_html=True)
            for fig in msg.get("charts",[]):
                st.plotly_chart(fig, use_container_width=True, key=f"c_{id(fig)}_{random.randint(0,99999)}")
            for tbl in msg.get("tables",[]):
                st.dataframe(tbl, use_container_width=True)
            for ins_text, ins_level in msg.get("insights",[]):
                st.markdown(f'<div class="inline-insight {ins_level}">{ins_text}</div>', unsafe_allow_html=True)
            if msg.get("summary"):
                st.markdown(f'<div class="summary-box">📝 <strong>Summary:</strong> {msg["summary"]}</div>', unsafe_allow_html=True)

# ── Suggested questions ───────────────────────────────────────────────────────
def get_questions():
    if st.session_state.chat_qs:
        return st.session_state.chat_qs
    base = [
        "Give me a full summary of this dataset",
        "Show the correlation heatmap",
        f"Show histogram of {numeric_cols[-1] if numeric_cols else 'target'}",
        "Forecast the next 20 periods",
        "Detect anomalies in the data",
        "What are the key drivers?",
        "Show missing data analysis",
        "What domain is this data from?",
        "Give me strategic suggestions",
        "How is the model performing?",
        "Show boxplot of numeric columns",
        "Give me the executive summary report",
    ]
    if OLLAMA:
        try:
            ai_qs = generate_dynamic_questions(memory)
            base = list(dict.fromkeys(base + ai_qs))[:16]
        except Exception:
            pass
    st.session_state.chat_qs = base
    return base

def save_session():
    if not st.session_state.chat_history:
        return
    name = f"Session {len(st.session_state.chat_sessions)+1} — {datetime.now().strftime('%H:%M')}"
    saved = []
    for m in st.session_state.chat_history:
        saved.append({k: v for k, v in m.items() if k not in ("charts","tables")})
    st.session_state.chat_sessions.append({
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "messages": saved,
        "q_count": sum(1 for m in st.session_state.chat_history if m["role"]=="user"),
    })
    st.session_state.chat_history = []
    st.session_state.chat_qs = []

# ══════════════════════════════════════════════════════════════════
# PAGE
# ══════════════════════════════════════════════════════════════════
st.markdown('<div class="ax-title">AI Chat Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="ax-subtitle">Charts · Forecasts · Anomalies · Drivers · Summaries — all inside the chat</div>', unsafe_allow_html=True)

hist_col, chat_col = st.columns([1, 3])

# ── HISTORY PANEL ─────────────────────────────────────────────────────────────
with hist_col:
    st.markdown('<div class="ax-section">History</div>', unsafe_allow_html=True)
    cur_count = sum(1 for m in st.session_state.chat_history if m["role"]=="user")
    if cur_count:
        st.markdown(
            f'<div class="hist-item" style="border-color:rgba(79,158,255,.3);">'
            f'<div class="hist-item-q">● Current session</div>'
            f'<div class="hist-item-meta">{cur_count} question{"s" if cur_count!=1 else ""} · Active</div>'
            f'</div>', unsafe_allow_html=True
        )
    sessions = st.session_state.chat_sessions
    if sessions:
        st.markdown('<div style="font-size:11px;color:#4e5a70;text-transform:uppercase;letter-spacing:.09em;margin:10px 0 6px;">Saved Sessions</div>', unsafe_allow_html=True)
        for i, sess in enumerate(reversed(sessions)):
            ri = len(sessions) - 1 - i
            with st.expander(f"📁 {sess['name']}", expanded=False):
                st.markdown(f'<div style="font-size:11px;color:#4e5a70;margin-bottom:8px;">{sess["q_count"]} questions · {sess["timestamp"][:16]}</div>', unsafe_allow_html=True)
                for m in sess["messages"]:
                    if m["role"]=="user":
                        st.markdown(f'<div style="font-size:12px;color:#8b97ae;margin-bottom:3px;">❓ {m["text"]}</div>', unsafe_allow_html=True)
                    elif m.get("summary"):
                        st.markdown(f'<div style="font-size:12px;color:#4e5a70;padding-left:12px;border-left:2px solid #28334a;margin-bottom:7px;">→ {m["summary"]}</div>', unsafe_allow_html=True)
                if st.button("↩ Restore", key=f"restore_{ri}", use_container_width=True):
                    st.session_state.chat_history = [{**m, "charts":[], "tables":[]} for m in sess["messages"]]
                    st.rerun()
    else:
        st.markdown('<div style="font-size:12px;color:#4e5a70;text-align:center;padding:12px 0;">No saved sessions yet.</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("💾 Save to History", use_container_width=True, key="save_hist"):
        save_session()
        st.success("Saved!")
        st.rerun()
    if st.button("🗑 Clear Chat", use_container_width=True, key="clr"):
        st.session_state.chat_history = []
        st.rerun()
    if st.button("🔄 Refresh Questions", use_container_width=True, key="rq"):
        st.session_state.chat_qs = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        f'<div style="font-size:11px;color:#4e5a70;line-height:2;">'
        f'📊 {meta.get("rows",0):,} rows · {meta.get("columns",0)} cols<br>'
        f'🔬 {meta.get("domain","general").capitalize()}<br>'
        f'⭐ Grade {memory.get("intelligence_score",{}).get("grade","—")}<br>'
        f'🤖 {model_i.get("selected_model","—")[:22]}</div>',
        unsafe_allow_html=True
    )

# ── CHAT AREA ─────────────────────────────────────────────────────────────────
with chat_col:
    # Suggested questions
    qs = get_questions()
    sample = random.sample(qs, min(9, len(qs)))
    rows_q = [sample[i:i+3] for i in range(0, len(sample), 3)]
    st.markdown('<div style="font-size:11px;color:#4e5a70;text-transform:uppercase;letter-spacing:.09em;margin-bottom:8px;">Suggested Questions</div>', unsafe_allow_html=True)
    for row_q in rows_q:
        c1, c2, c3 = st.columns(3)
        for qcol, q in zip([c1,c2,c3], row_q):
            if qcol.button(q, key=f"qbtn_{q[:28]}", use_container_width=True):
                st.session_state.chat_prefill = q

    st.markdown("---")

    # Messages
    if not st.session_state.chat_history:
        st.markdown(
            '<div class="chat-empty">'
            '<div class="chat-empty-icon">🧠</div>'
            '<div class="chat-empty-title">Ask me anything about your data</div>'
            '<div class="chat-empty-sub">Charts · Forecasts · Anomalies · Drivers — all generated inline</div>'
            '</div>', unsafe_allow_html=True
        )
    else:
        for msg in st.session_state.chat_history:
            render_message(msg)
            st.markdown("<br>", unsafe_allow_html=True)

    # Input form
    prefill = st.session_state.pop("chat_prefill", "")
    with st.form("chat_f", clear_on_submit=True):
        ic, bc = st.columns([5,1])
        user_q = ic.text_input("Q", value=prefill,
                                placeholder="e.g. show correlation, forecast, anomalies, drivers, summary…",
                                label_visibility="collapsed")
        sub = bc.form_submit_button("Send →", type="primary", use_container_width=True)

    if sub and user_q.strip():
        st.session_state.chat_history.append({
            "role":"user","text":user_q.strip(),
            "charts":[],"tables":[],"metrics":[],"insights":[],"summary":"","intent":[],
        })
        with st.spinner("Analysing…"):
            try:
                resp = build_response(user_q.strip())
            except Exception as e:
                resp = {"text":f"Error: {e}","summary":"Error.","charts":[],"tables":[],"metrics":[],"insights":[],"intent":["error"]}
        st.session_state.chat_history.append({
            "role":"ai","text":resp["text"],"summary":resp["summary"],
            "charts":resp["charts"],"tables":resp["tables"],
            "metrics":resp["metrics"],"insights":resp["insights"],"intent":resp["intent"],
        })
        st.rerun()

    # Quick action bar
    st.markdown("<br>", unsafe_allow_html=True)
    qa1,qa2,qa3,qa4,qa5 = st.columns(5)
    for qcol, lbl, q in [
        (qa1,"📊 Summary","Give me a full summary of this dataset"),
        (qa2,"🔗 Heatmap","Show the correlation heatmap"),
        (qa3,"🔮 Forecast","Forecast the next 20 periods"),
        (qa4,"⚠️ Anomalies","Detect anomalies in the data"),
        (qa5,"🎯 Drivers","What are the key drivers?"),
    ]:
        if qcol.button(lbl, use_container_width=True, key=f"qa_{lbl}"):
            st.session_state.chat_prefill = q
            st.rerun()