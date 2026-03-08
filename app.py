import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from screener import extract_text_from_pdf, predict_category, match_resume_to_job
from preprocess import clean_text

st.set_page_config(
    page_title="RecruitIQ",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Figtree:wght@300;400;500;600;700&display=swap');

/* ══════════════════════════════════════
   TOKENS — warm editorial palette
══════════════════════════════════════ */
:root {
    --paper:      #f7f4ef;
    --paper2:     #f0ece4;
    --paper3:     #e8e2d8;
    --white:      #fffdf9;
    --ink:        #1a1714;
    --ink2:       #3d3830;
    --ink3:       #7a7060;
    --ink4:       #b0a898;
    --rule:       rgba(26,23,20,0.09);
    --rule2:      rgba(26,23,20,0.15);
    --accent:     #c8401a;
    --accent-dim: rgba(200,64,26,0.08);
    --accent-s:   #e04d20;
    --emerald:    #1a7a52;
    --emerald-dim:rgba(26,122,82,0.09);
    --amber:      #b07020;
    --amber-dim:  rgba(176,112,32,0.09);
    --r:          10px;
    --r-sm:       6px;
    --r-lg:       16px;
    --shadow:     0 1px 3px rgba(26,23,20,0.08), 0 4px 16px rgba(26,23,20,0.05);
    --shadow-lg:  0 2px 8px rgba(26,23,20,0.1), 0 12px 40px rgba(26,23,20,0.08);
}

/* ══════════════════════════════════════
   GLOBAL
══════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

html, body,
.stApp, .stApp > div,
[class*="css"],
.main, .main > div,
section.main,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"] {
    background-color: var(--paper) !important;
    color: var(--ink) !important;
    font-family: 'Figtree', sans-serif !important;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--paper2); }
::-webkit-scrollbar-thumb { background: var(--ink4); border-radius: 99px; }

.main .block-container {
    padding: 0 3.5rem 5rem !important;
    max-width: 1240px !important;
}

/* ══════════════════════════════════════
   SIDEBAR — deep charcoal contrast
══════════════════════════════════════ */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div > div {
    background: var(--ink) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * {
    color: var(--ink4) !important;
    font-family: 'Figtree', sans-serif !important;
}
[data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }
[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border: none !important;
    border-radius: var(--r-sm) !important;
    padding: 10px 14px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #8a8070 !important;
    transition: all .15s;
    cursor: pointer;
    letter-spacing: 0 !important;
    text-transform: none !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(247,244,239,0.06) !important;
    color: #f7f4ef !important;
}

/* ══════════════════════════════════════
   FILE UPLOADER — warm paper override
══════════════════════════════════════ */
.stFileUploader,
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] > div > div,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] > div,
section[data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploaderDropzoneInput"],
[class*="uploadedFile"], [class*="fileUpload"], [class*="FileUpload"] {
    background: var(--white) !important;
    background-color: var(--white) !important;
    color: var(--ink3) !important;
    border-color: var(--rule2) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: var(--white) !important;
    border: 1.5px dashed var(--rule2) !important;
    border-radius: var(--r) !important;
    padding: 20px !important;
    box-shadow: none !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: var(--ink3) !important;
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span { color: var(--ink4) !important; font-size: 0.82rem !important; }
[data-testid="stFileUploaderDropzone"] button,
[data-baseweb="button"],
[data-testid="baseButton-secondary"] {
    background: var(--paper2) !important;
    background-color: var(--paper2) !important;
    border: 1px solid var(--rule2) !important;
    color: var(--ink2) !important;
    border-radius: var(--r-sm) !important;
    font-size: 0.8rem !important;
    font-family: 'Figtree', sans-serif !important;
    box-shadow: none !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: var(--paper3) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFile"] > div {
    background: var(--paper2) !important;
    border: 1px solid var(--rule) !important;
    border-radius: var(--r-sm) !important;
}
[data-testid="stFileUploaderFile"] * { color: var(--ink2) !important; }
.stFileUploader label,
[data-testid="stFileUploader"] label {
    font-size: 0.62rem !important; font-weight: 700 !important;
    letter-spacing: 2.2px !important; text-transform: uppercase !important;
    color: var(--ink3) !important; margin-bottom: 8px !important; display: block !important;
}

/* ══════════════════════════════════════
   TEXT AREA
══════════════════════════════════════ */
.stTextArea textarea {
    background: var(--white) !important;
    border: 1.5px solid var(--rule2) !important;
    border-radius: var(--r) !important;
    color: var(--ink) !important;
    font-family: 'Figtree', sans-serif !important;
    font-size: 0.9rem !important; line-height: 1.7 !important;
    padding: 14px 16px !important; caret-color: var(--accent) !important;
    transition: border-color .2s, box-shadow .2s !important;
    box-shadow: none !important;
}
.stTextArea textarea::placeholder { color: var(--ink4) !important; }
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(200,64,26,0.08) !important;
    outline: none !important;
}
.stTextArea label {
    font-size: 0.62rem !important; font-weight: 700 !important;
    letter-spacing: 2.2px !important; text-transform: uppercase !important;
    color: var(--ink3) !important; margin-bottom: 8px !important;
}

/* ══════════════════════════════════════
   BUTTONS
══════════════════════════════════════ */
.stButton > button {
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: var(--r-sm) !important;
    font-family: 'Figtree', sans-serif !important;
    font-weight: 600 !important; font-size: 0.875rem !important;
    padding: 12px 28px !important; letter-spacing: 0.3px !important;
    box-shadow: var(--shadow) !important;
    transition: all .2s ease !important; width: 100% !important;
}
.stButton > button:hover {
    background: var(--ink2) !important;
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-lg) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.stDownloadButton > button {
    background: transparent !important;
    border: 1.5px solid var(--rule2) !important;
    color: var(--ink2) !important; font-size: 0.85rem !important;
    box-shadow: none !important; width: auto !important;
}
.stDownloadButton > button:hover {
    border-color: var(--accent) !important; color: var(--accent) !important;
    background: var(--accent-dim) !important; transform: none !important;
}

/* ══════════════════════════════════════
   MISC WIDGETS
══════════════════════════════════════ */
.stDataFrame { border-radius: var(--r) !important; overflow: hidden !important; border: 1px solid var(--rule) !important; box-shadow: var(--shadow) !important; }
.stProgress > div { background: var(--paper3) !important; border-radius: 99px !important; height: 5px !important; }
.stProgress > div > div { background: var(--accent) !important; border-radius: 99px !important; }
.stSuccess { background: var(--emerald-dim) !important; border: 1px solid rgba(26,122,82,0.2) !important; border-radius: var(--r-sm) !important; color: var(--emerald) !important; }
.stWarning { background: var(--amber-dim) !important; border: 1px solid rgba(176,112,32,0.2) !important; border-radius: var(--r-sm) !important; color: var(--amber) !important; }
.stError   { background: var(--accent-dim) !important; border: 1px solid rgba(200,64,26,0.2) !important; border-radius: var(--r-sm) !important; color: var(--accent) !important; }
.stInfo    { background: rgba(26,23,20,0.04) !important; border: 1px solid var(--rule) !important; border-radius: var(--r-sm) !important; color: var(--ink2) !important; }
[data-testid="stSpinner"] * { color: var(--accent) !important; }
hr { border: none !important; border-top: 1px solid var(--rule) !important; margin: 28px 0 !important; }
h1, h2, h3, h4 { font-family: 'Playfair Display', serif !important; color: var(--ink) !important; }

/* ══════════════════════════════════════
   CUSTOM COMPONENTS
══════════════════════════════════════ */

/* ── Top band ── */
.topband {
    background: var(--ink);
    padding: 12px 3.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 -3.5rem 0;
}
.topband-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--paper);
    letter-spacing: 0.5px;
}
.topband-logo em { font-style: italic; color: var(--accent-s); }
.topband-tag {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #5a5048;
}

/* ── Hero ── */
.hero {
    padding: 56px 0 44px;
    border-bottom: 1px solid var(--rule);
    margin-bottom: 40px;
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 32px;
    align-items: end;
}
.hero-kicker {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.hero-kicker::before {
    content: '';
    width: 22px; height: 1.5px;
    background: var(--accent);
    display: inline-block;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.6rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.05;
    letter-spacing: -1px;
    margin-bottom: 16px;
}
.hero-title em { font-style: italic; color: var(--accent); }
.hero-sub {
    font-size: 1rem;
    font-weight: 400;
    color: var(--ink3);
    line-height: 1.72;
    max-width: 480px;
}
.hero-meta {
    text-align: right;
    padding-bottom: 4px;
}
.hero-meta-num {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1;
}
.hero-meta-lbl {
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--ink4);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

/* ── Section heading ── */
.sh {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--ink);
    margin: 32px 0 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--rule);
    display: flex;
    align-items: center;
    gap: 10px;
}
.sh-sm {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--ink3);
    margin: 20px 0 10px;
}

/* ── Page intro ── */
.page-intro {
    font-size: 0.9rem;
    color: var(--ink3);
    line-height: 1.7;
    margin-bottom: 28px;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--rule);
}

/* ── Cards ── */
.card {
    background: var(--white);
    border: 1px solid var(--rule);
    border-radius: var(--r);
    padding: 20px 22px;
    margin-bottom: 10px;
    box-shadow: var(--shadow);
    transition: box-shadow .2s, border-color .2s;
}
.card:hover { box-shadow: var(--shadow-lg); border-color: var(--rule2); }

.card-accent {
    background: var(--white);
    border: 1px solid var(--rule);
    border-left: 3px solid var(--accent);
    border-radius: var(--r);
    padding: 20px 22px;
    margin-bottom: 10px;
    box-shadow: var(--shadow);
}

/* ── Metric grid ── */
.mrow {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 10px;
    margin: 20px 0;
}
.mc {
    background: var(--white);
    border: 1px solid var(--rule);
    border-radius: var(--r);
    padding: 20px 16px;
    text-align: center;
    box-shadow: var(--shadow);
    transition: box-shadow .2s, transform .2s;
}
.mc:hover { box-shadow: var(--shadow-lg); transform: translateY(-2px); }
.mc-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.1;
    margin-bottom: 5px;
}
.mc-lbl {
    font-size: 0.62rem;
    font-weight: 700;
    color: var(--ink4);
    text-transform: uppercase;
    letter-spacing: 1.8px;
}

/* ── Progress ── */
.pw { background: var(--paper3); border-radius: 99px; height: 5px; overflow: hidden; margin: 7px 0; }
.pf { height: 100%; border-radius: 99px; transition: width .5s ease; }
.pf-g { background: var(--emerald); }
.pf-y { background: var(--amber); }
.pf-r { background: var(--accent); }
.pf-b { background: var(--ink2); }

/* ── Badges ── */
.badge { display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: var(--r-sm); font-size: 0.71rem; font-weight: 600; letter-spacing: 0.3px; }
.badge-g { background: var(--emerald-dim); color: var(--emerald); border: 1px solid rgba(26,122,82,0.2); }
.badge-y { background: var(--amber-dim);   color: var(--amber);   border: 1px solid rgba(176,112,32,0.2); }
.badge-r { background: var(--accent-dim);  color: var(--accent);  border: 1px solid rgba(200,64,26,0.2); }

/* ── Pills ── */
.pill-g { display: inline-block; background: var(--emerald-dim); color: var(--emerald); border: 1px solid rgba(26,122,82,0.15); padding: 2px 9px; border-radius: var(--r-sm); margin: 3px; font-size: 0.73rem; font-weight: 500; }
.pill-r { display: inline-block; background: var(--accent-dim);  color: var(--accent);  border: 1px solid rgba(200,64,26,0.15);  padding: 2px 9px; border-radius: var(--r-sm); margin: 3px; font-size: 0.73rem; font-weight: 500; }

/* ── Tip ── */
.tip { background: rgba(176,112,32,0.05); border-left: 2.5px solid var(--amber); border-radius: 0 var(--r-sm) var(--r-sm) 0; padding: 10px 14px; margin: 6px 0; font-size: 0.84rem; color: var(--ink2); line-height: 1.62; }

/* ── Winner ── */
.winner { background: linear-gradient(135deg, rgba(26,122,82,0.05), rgba(200,64,26,0.04)); border: 1px solid rgba(26,122,82,0.15); border-radius: var(--r-lg); padding: 32px; text-align: center; margin: 12px 0 20px; box-shadow: var(--shadow); }
.winner h3 { font-family: 'Playfair Display', serif; font-size: 1.7rem; font-weight: 700; color: var(--emerald); margin-bottom: 6px; font-style: italic; }
.winner p { color: var(--ink3); font-size: 0.875rem; margin-top: 4px; }

/* ── Empty state ── */
.empty { background: var(--white); border: 1.5px dashed var(--paper3); border-radius: var(--r-lg); padding: 64px 24px; text-align: center; }
.empty-ico { font-size: 2rem; margin-bottom: 14px; opacity: 0.35; }
.empty-title { font-family: 'Playfair Display', serif; font-size: 1.05rem; font-weight: 600; color: var(--ink2); margin-bottom: 6px; }
.empty-sub { font-size: 0.83rem; color: var(--ink4); }

/* ── Candidate label ── */
.cand-tag { display: inline-block; font-size: 0.62rem; font-weight: 700; letter-spacing: 2.5px; text-transform: uppercase; padding: 4px 12px; border-radius: var(--r-sm); margin-bottom: 12px; }
.cand-a { background: rgba(200,64,26,0.08); color: var(--accent); border: 1px solid rgba(200,64,26,0.15); }
.cand-b { background: rgba(26,122,82,0.08); color: var(--emerald); border: 1px solid rgba(26,122,82,0.15); }

/* ── Number rule callout ── */
.rule-callout {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    background: var(--white);
    border: 1px solid var(--rule);
    border-radius: var(--r);
    margin-bottom: 10px;
    box-shadow: var(--shadow);
}
.rule-num {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    min-width: 42px;
    line-height: 1;
}
.rule-body { flex: 1; }
.rule-title { font-weight: 600; font-size: 0.875rem; color: var(--ink); line-height: 1.3; }
.rule-sub { font-size: 0.78rem; color: var(--ink3); margin-top: 2px; }

</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────
def kw_analysis(resume_text, jd):
    cr = set(w for w in clean_text(resume_text).split() if len(w) > 3)
    cj = set(w for w in clean_text(jd).split() if len(w) > 3)
    return sorted(cr & cj), sorted(cj - cr)

def suggestions(missing, score):
    t = []
    if score < 40:   t.append(("Significant gap", "This resume needs substantial additions to be competitive for this role."))
    elif score < 50: t.append(("Partial match", "A few targeted additions could push this resume into shortlist territory."))
    else:            t.append(("Strong match", "Minor refinements will make this resume even more compelling."))
    if missing:      t.append(("Add missing keywords", f"Include these terms from the JD: {', '.join(missing[:8])}"))
    t.append(("Quantify everything", "Replace soft claims with numbers — 'Improved load time by 40%' beats 'improved performance'"))
    t.append(("Mirror the JD language", "Rewrite your summary to echo the exact phrasing used in the job description"))
    t.append(("Dedicated skills section", "Add a structured Skills block using keywords exactly as they appear in the JD"))
    if score < 50:   t.append(("Add proof of expertise", "Include certifications, courses, or projects directly relevant to this role"))
    return t

def sc(s):     return "#1a7a52" if s >= 50 else "#b07020" if s >= 25 else "#c8401a"
def slabel(s): return "✅ Shortlisted" if s >= 50 else "⚠️ Review" if s >= 25 else "❌ Rejected"
def sbadge(s):
    if s >= 50: return '<span class="badge badge-g">Shortlisted</span>'
    if s >= 25: return '<span class="badge badge-y">Review</span>'
    return '<span class="badge badge-r">Rejected</span>'
def pcls(s): return "pf-g" if s >= 50 else "pf-y" if s >= 25 else "pf-r"

def warm_ax(fig, ax):
    fig.patch.set_facecolor('#fffdf9')
    ax.set_facecolor('#fffdf9')
    ax.tick_params(colors='#7a7060', labelsize=8.5)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_color('#7a7060')
    ax.xaxis.label.set_color('#7a7060')
    ax.yaxis.label.set_color('#7a7060')
    ax.title.set_color('#1a1714')
    ax.title.set_fontsize(10.5)
    ax.title.set_fontfamily('serif')
    for sp in ax.spines.values():
        sp.set_edgecolor('#e8e2d8')
    return fig, ax

def prog(pct, cls):
    return f'<div class="pw"><div class="{cls} pf" style="width:{min(int(pct),100)}%"></div></div>'

def mc_html(val, lbl, color="#1a1714"):
    return f'<div class="mc"><div class="mc-val" style="color:{color}">{val}</div><div class="mc-lbl">{lbl}</div></div>'

MEDALS = ["01", "02", "03"]

# ── Top band ──────────────────────────────────────────────────
st.markdown("""
<div class="topband">
  <div class="topband-logo">Recruit<em>IQ</em></div>
  <div class="topband-tag">AI Resume Intelligence</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:24px 0 32px">
      <div style="font-family:'Playfair Display',serif;font-size:1.35rem;font-weight:700;color:#f7f4ef;font-style:italic">Recruit<span style="color:#e04d20">IQ</span></div>
      <div style="font-size:0.58rem;color:#3d3830;font-weight:700;letter-spacing:3px;text-transform:uppercase;margin-top:5px">Resume Intelligence</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("nav", [
        "→  Analyze Single",
        "→  Rank Multiple",
        "→  Compare Two",
        "→  Full Dashboard"
    ], label_visibility="collapsed")

    st.markdown('<div style="height:1px;background:rgba(247,244,239,0.06);margin:16px 0"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.58rem;color:#3d3830;font-weight:700;text-transform:uppercase;letter-spacing:2.5px;margin-bottom:12px">25 Supported Roles</div>', unsafe_allow_html=True)
    for r in ["Advocate","Arts","Automation Testing","Blockchain","Business Analyst",
              "Civil Engineer","Data Science","Database","DevOps Engineer","DotNet Developer",
              "ETL Developer","Electrical Engineering","HR","Hadoop","Health & Fitness",
              "Java Developer","Mechanical Engineer","Network Security","Operations Manager",
              "PMO","Python Developer","SAP Developer","Sales","Testing","Web Designing"]:
        st.markdown(f'<div style="font-size:0.72rem;color:#3d3830;padding:2.5px 0;line-height:1.7">{r}</div>', unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div>
    <div class="hero-kicker">AI-Powered Recruitment</div>
    <div class="hero-title">Smarter hiring,<br><em>faster decisions.</em></div>
    <div class="hero-sub">Screen resumes intelligently — match candidates to roles, uncover skill gaps, and build shortlists with confidence.</div>
  </div>
  <div class="hero-meta">
    <div class="hero-meta-num">25</div>
    <div class="hero-meta-lbl">Job Categories</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 1 — SINGLE ANALYZER
# ════════════════════════════════════════════════════════════
if page == "→  Analyze Single":
    st.markdown('<div class="sh">Single Resume Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-intro">Deep-dive analysis: match score, keyword gaps, top role predictions, and personalized improvement tips for one candidate.</p>', unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")
    with col_l:
        jd = st.text_area("Job Description", height=210, placeholder="Paste the full job description here…", key="jd1")
    with col_r:
        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
        f = st.file_uploader("Candidate Resume (PDF)", type=['pdf'], key="f1")

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    if f and jd:
        if st.button("Run Analysis", use_container_width=True):
            with st.spinner("Analyzing…"):
                rt = extract_text_from_pdf(f)
                cat, conf, top3 = predict_category(rt)
                score = match_resume_to_job(rt, jd)
                matched, missing = kw_analysis(rt, jd)
                tips = suggestions(missing, score)

            st.success("Analysis complete.")
            st.divider()

            st.markdown(f"""<div class="mrow">
                {mc_html(cat, "Predicted Role", "#c8401a")}
                {mc_html(f"{conf}%", "Model Confidence")}
                {mc_html(f"{score}%", "Match Score", sc(score))}
                {mc_html(f"{len(matched)}/{len(matched)+len(missing)}", "Keywords Found")}
            </div>""", unsafe_allow_html=True)

            verdict = ("Strong match — recommend for interview" if score >= 50
                      else "Moderate match — worth a second look" if score >= 25
                      else "Weak match — significant gaps identified")

            st.markdown(f"""
            <div class="card-accent" style="margin-top:8px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
                <span style="font-family:'Playfair Display',serif;font-weight:600;font-size:1rem;color:#1a1714">Overall Match</span>
                {sbadge(score)}
              </div>
              {prog(score, pcls(score))}
              <div style="display:flex;justify-content:space-between;margin-top:9px">
                <span style="font-size:0.82rem;color:#7a7060;font-style:italic">{verdict}</span>
                <span style="font-size:1rem;font-family:'Playfair Display',serif;font-weight:700;color:{sc(score)}">{score}%</span>
              </div>
            </div>""", unsafe_allow_html=True)

            st.divider()
            left, right = st.columns(2, gap="large")

            with left:
                st.markdown('<div class="sh">Role Predictions</div>', unsafe_allow_html=True)
                for i, (role, prob) in enumerate(top3):
                    st.markdown(f"""
                    <div class="card" style="padding:14px 18px;margin-bottom:8px">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:9px">
                        <div>
                          <span style="font-size:0.62rem;font-weight:700;color:#b0a898;letter-spacing:1.5px;margin-right:8px">{MEDALS[i]}</span>
                          <span style="font-weight:600;font-size:0.875rem;color:#1a1714">{role}</span>
                        </div>
                        <span style="color:{sc(prob)};font-family:'Playfair Display',serif;font-weight:700;font-size:1rem">{prob}%</span>
                      </div>
                      {prog(prob, pcls(prob))}
                    </div>""", unsafe_allow_html=True)

                st.markdown('<div class="sh" style="margin-top:24px">Keyword Analysis</div>', unsafe_allow_html=True)
                if matched:
                    st.markdown('<div class="sh-sm" style="color:#1a7a52">Present in Resume</div>', unsafe_allow_html=True)
                    st.markdown('<div style="line-height:2.3">' + " ".join(f'<span class="pill-g">{k}</span>' for k in matched[:18]) + '</div>', unsafe_allow_html=True)
                if missing:
                    st.markdown('<div class="sh-sm" style="color:#c8401a;margin-top:14px">Missing from Resume</div>', unsafe_allow_html=True)
                    st.markdown('<div style="line-height:2.3">' + " ".join(f'<span class="pill-r">{k}</span>' for k in missing[:18]) + '</div>', unsafe_allow_html=True)

            with right:
                st.markdown('<div class="sh">Improvement Guide</div>', unsafe_allow_html=True)
                for n, (title, body) in enumerate(tips, 1):
                    st.markdown(f"""
                    <div class="rule-callout">
                      <div class="rule-num">{n:02d}</div>
                      <div class="rule-body">
                        <div class="rule-title">{title}</div>
                        <div class="rule-sub">{body}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(5, 2.8))
                warm_ax(fig, ax)
                zones = [25, 50, 100]
                clrs  = ['#c8401a', '#b07020', '#1a7a52']
                lbls  = ['Weak', 'Moderate', 'Strong']
                bars  = ax.barh(lbls, zones, color=clrs, alpha=0.14, height=0.44, edgecolor='none')
                ax.axvline(score, color='#1a1714', lw=1.8, zorder=5)
                ax.text(min(score+1.5, 88), 2, f'{score}%', color='#1a1714', fontweight='bold', fontsize=10, fontfamily='serif')
                ax.set_xlim(0, 100); ax.set_xlabel('Match Score (%)')
                ax.set_title('Score Position', pad=10)
                plt.tight_layout(); st.pyplot(fig)
    else:
        st.markdown("""<div class="empty">
          <div class="empty-ico">📄</div>
          <div class="empty-title">Ready to analyze</div>
          <div class="empty-sub">Paste a job description and upload a PDF resume to begin</div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 — MULTI RANKER
# ════════════════════════════════════════════════════════════
elif page == "→  Rank Multiple":
    st.markdown('<div class="sh">Batch Resume Ranker</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-intro">Upload an entire candidate pool and get every resume automatically scored, ranked, and categorized — in one pass.</p>', unsafe_allow_html=True)

    jd = st.text_area("Job Description", height=160, placeholder="Paste job description…", key="jd2")
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    files = st.file_uploader("Upload Resumes (PDF, multiple)", type=['pdf'], accept_multiple_files=True, key="f2")
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    if files and jd:
        if st.button("Screen All Candidates", use_container_width=True):
            rows = []; pb = st.progress(0); stxt = st.empty()
            for i, file in enumerate(files):
                stxt.text(f"Analyzing {file.name}  ({i+1}/{len(files)})…")
                rt  = extract_text_from_pdf(file)
                cat, conf, _ = predict_category(rt)
                s   = match_resume_to_job(rt, jd)
                m, mis = kw_analysis(rt, jd)
                rows.append({'Candidate': file.name.replace('.pdf', ''), 'Predicted Role': cat,
                             'Confidence': f"{conf}%", 'Match Score': s,
                             'Keywords Matched': len(m), 'Keywords Missing': len(mis),
                             'Status': slabel(s)})
                pb.progress((i + 1) / len(files))
            stxt.empty(); pb.empty()

            df = pd.DataFrame(rows).sort_values('Match Score', ascending=False).reset_index(drop=True)
            df.index += 1
            st.success(f"Screened {len(files)} candidates.")
            st.divider()

            short = len(df[df['Match Score'] >= 50])
            rev   = len(df[(df['Match Score'] >= 25) & (df['Match Score'] < 50)])
            rej   = len(df[df['Match Score'] < 25])

            st.markdown(f"""<div class="mrow">
                {mc_html(len(df), "Total")}
                {mc_html(short, "Shortlisted", "#1a7a52")}
                {mc_html(rev,   "For Review",  "#b07020")}
                {mc_html(rej,   "Rejected",    "#c8401a")}
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sh">Ranked Results</div>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)

            st.markdown('<div class="sh">Score Distribution</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, max(4, len(df) * .62)))
            warm_ax(fig, ax)
            bars = ax.barh(df['Candidate'], df['Match Score'],
                          color=[sc(s) for s in df['Match Score']], edgecolor='none', height=0.48, alpha=0.85)
            ax.axvline(50, color='#1a7a52', ls='--', lw=1.2, alpha=0.6, label='Shortlist threshold')
            ax.axvline(25, color='#b07020', ls='--', lw=1.2, alpha=0.6, label='Review threshold')
            for bar, s in zip(bars, df['Match Score']):
                ax.text(bar.get_width() + .5, bar.get_y() + bar.get_height()/2,
                       f'{s}%', va='center', color='#7a7060', fontsize=8.5)
            ax.set_xlabel('Match Score (%)')
            ax.set_title('Candidate Rankings')
            ax.legend(facecolor='#fffdf9', edgecolor='#e8e2d8', labelcolor='#7a7060', fontsize=8.5)
            plt.tight_layout(); st.pyplot(fig)

            st.download_button("Download Rankings CSV", df.to_csv(index=True).encode(), "rankings.csv", "text/csv")
    else:
        st.markdown("""<div class="empty">
          <div class="empty-ico">👥</div>
          <div class="empty-title">Upload your candidate pool</div>
          <div class="empty-sub">Add a job description and 2 or more PDF resumes</div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 3 — COMPARISON
# ════════════════════════════════════════════════════════════
elif page == "→  Compare Two":
    st.markdown('<div class="sh">Head-to-Head Comparison</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-intro">Put two candidates side by side. See who matches better, where each falls short, and what differentiates them.</p>', unsafe_allow_html=True)

    jd = st.text_area("Job Description", height=130, placeholder="Paste job description…", key="jd3")
    st.divider()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<span class="cand-tag cand-a">Candidate A</span>', unsafe_allow_html=True)
        fa = st.file_uploader("Upload Resume A", type=['pdf'], key="ca")
    with c2:
        st.markdown('<span class="cand-tag cand-b">Candidate B</span>', unsafe_allow_html=True)
        fb = st.file_uploader("Upload Resume B", type=['pdf'], key="cb")

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    if fa and fb and jd:
        if st.button("Compare Candidates", use_container_width=True):
            with st.spinner("Comparing…"):
                ta = extract_text_from_pdf(fa); ca2, cfa, t3a = predict_category(ta)
                sa = match_resume_to_job(ta, jd); ma, mia = kw_analysis(ta, jd)
                tb = extract_text_from_pdf(fb); cb2, cfb, t3b = predict_category(tb)
                sb = match_resume_to_job(tb, jd); mb, mib = kw_analysis(tb, jd)

            na = fa.name.replace('.pdf', ''); nb = fb.name.replace('.pdf', '')

            if sa > sb:
                st.markdown(f'<div class="winner"><h3>{na} leads</h3><p>{sa}% vs {sb}% match score</p></div>', unsafe_allow_html=True)
            elif sb > sa:
                st.markdown(f'<div class="winner"><h3>{nb} leads</h3><p>{sb}% vs {sa}% match score</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="winner"><h3>It\'s a draw</h3><p>Both candidates score equally</p></div>', unsafe_allow_html=True)

            st.divider()
            col1, col2 = st.columns(2, gap="large")

            for col, name, cat, conf, score, top3, matched, missing, tag_cls in [
                (col1, na, ca2, cfa, sa, t3a, ma, mia, "cand-a"),
                (col2, nb, cb2, cfb, sb, t3b, mb, mib, "cand-b"),
            ]:
                with col:
                    st.markdown(f'<span class="cand-tag {tag_cls}">{name}</span>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="card">
                      <div class="mrow" style="gap:8px;margin:0 0 14px 0">
                        <div class="mc" style="min-width:0">
                          <div class="mc-val" style="font-size:1.7rem;color:{sc(score)}">{score}%</div>
                          <div class="mc-lbl">Match</div>
                        </div>
                        <div class="mc" style="min-width:0">
                          <div class="mc-val" style="font-size:1.7rem">{conf}%</div>
                          <div class="mc-lbl">Confidence</div>
                        </div>
                        <div class="mc" style="min-width:0">
                          <div class="mc-val" style="font-size:1.7rem">{len(matched)}</div>
                          <div class="mc-lbl">Keywords</div>
                        </div>
                      </div>
                      {prog(score, pcls(score))}
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:10px">
                        {sbadge(score)}
                        <span style="font-size:0.8rem;color:#7a7060;font-style:italic">{cat}</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown('<div class="sh-sm" style="margin-top:14px">Top Predictions</div>', unsafe_allow_html=True)
                    for i, (role, prob) in enumerate(top3):
                        st.markdown(f"""
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                          <span style="color:#7a7060;font-size:0.82rem"><span style="color:#b0a898;font-weight:700;margin-right:6px">{MEDALS[i]}</span>{role}</span>
                          <span style="color:{sc(prob)};font-family:'Playfair Display',serif;font-weight:700;font-size:0.92rem">{prob}%</span>
                        </div>{prog(prob, pcls(prob))}""", unsafe_allow_html=True)

                    if matched:
                        st.markdown('<div class="sh-sm" style="color:#1a7a52;margin-top:14px">Present</div>', unsafe_allow_html=True)
                        st.markdown('<div style="line-height:2.2">' + " ".join(f'<span class="pill-g">{k}</span>' for k in matched[:10]) + '</div>', unsafe_allow_html=True)
                    if missing:
                        st.markdown('<div class="sh-sm" style="color:#c8401a;margin-top:10px">Missing</div>', unsafe_allow_html=True)
                        st.markdown('<div style="line-height:2.2">' + " ".join(f'<span class="pill-r">{k}</span>' for k in missing[:10]) + '</div>', unsafe_allow_html=True)

            st.divider()
            st.markdown('<div class="sh">Comparison Chart</div>', unsafe_allow_html=True)
            kwt = max(len(ma) + len(mia), len(mb) + len(mib), 1)
            lbls = ['Match Score', 'Confidence', 'Keyword Coverage']
            va = [sa, cfa, round(len(ma) / kwt * 100, 1)]
            vb = [sb, cfb, round(len(mb) / kwt * 100, 1)]
            x = np.arange(len(lbls)); w = 0.3
            fig, ax = plt.subplots(figsize=(8, 3.8))
            warm_ax(fig, ax)
            ba = ax.bar(x - w/2, va, w, label=na, color='#c8401a', alpha=0.8, edgecolor='none')
            bb = ax.bar(x + w/2, vb, w, label=nb, color='#1a7a52', alpha=0.8, edgecolor='none')
            for bar in list(ba) + list(bb):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                       f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8.5, color='#7a7060')
            ax.set_xticks(x); ax.set_xticklabels(lbls)
            ax.set_title(f'{na}  ·  {nb}')
            ax.legend(facecolor='#fffdf9', edgecolor='#e8e2d8', labelcolor='#7a7060')
            ax.set_ylim(0, 118)
            plt.tight_layout(); st.pyplot(fig)
    else:
        st.markdown("""<div class="empty">
          <div class="empty-ico">⚖️</div>
          <div class="empty-title">Ready to compare</div>
          <div class="empty-sub">Add a job description and upload two PDF resumes</div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 4 — DASHBOARD
# ════════════════════════════════════════════════════════════
elif page == "→  Full Dashboard":
    st.markdown('<div class="sh">Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-intro">Complete candidate pool overview — role breakdown, score rankings, skill gap analysis, and exportable reports.</p>', unsafe_allow_html=True)

    jd = st.text_area("Job Description", height=130, placeholder="Paste job description…", key="jd4")
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    files = st.file_uploader("Upload All Resumes (PDF)", type=['pdf'], accept_multiple_files=True, key="f4")
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    if files and jd:
        if st.button("Generate Dashboard", use_container_width=True):
            rows = []; all_miss = []; pb = st.progress(0); stxt = st.empty()
            for i, file in enumerate(files):
                stxt.text(f"Processing {file.name}…")
                rt = extract_text_from_pdf(file)
                cat, conf, _ = predict_category(rt)
                s = match_resume_to_job(rt, jd)
                m, mis = kw_analysis(rt, jd)
                all_miss.extend(mis)
                rows.append({'Candidate': file.name.replace('.pdf', ''), 'Role': cat,
                             'Confidence': conf, 'Match Score': s,
                             'Keywords Matched': len(m), 'Keywords Missing': len(mis),
                             'Status': slabel(s)})
                pb.progress((i + 1) / len(files))
            stxt.empty(); pb.empty()

            df = pd.DataFrame(rows).sort_values('Match Score', ascending=False).reset_index(drop=True)
            st.success(f"Dashboard ready — {len(files)} candidates processed.")
            st.divider()

            short = len(df[df['Match Score'] >= 50])
            rev   = len(df[(df['Match Score'] >= 25) & (df['Match Score'] < 50)])
            rej   = len(df[df['Match Score'] < 25])

            st.markdown(f"""<div class="mrow">
                {mc_html(len(df), "Total Candidates")}
                {mc_html(short,  "Shortlisted",   "#1a7a52")}
                {mc_html(rev,    "For Review",    "#b07020")}
                {mc_html(rej,    "Rejected",      "#c8401a")}
                {mc_html(f"{df['Match Score'].mean():.1f}%", "Avg Score", "#1a1714")}
                {mc_html(f"{df['Match Score'].max():.1f}%",  "Top Score", "#1a7a52")}
            </div>""", unsafe_allow_html=True)

            st.divider()
            ch1, ch2 = st.columns(2, gap="large")

            with ch1:
                st.markdown('<div class="sh">Role Distribution</div>', unsafe_allow_html=True)
                rc = df['Role'].value_counts()
                fig, ax = plt.subplots(figsize=(6, 4)); warm_ax(fig, ax)
                pal = ['#c8401a','#1a7a52','#b07020','#3d3830','#7a7060','#a08060','#60806a','#806040']
                wedges, texts, autos = ax.pie(rc.values, labels=rc.index, autopct='%1.0f%%',
                    colors=pal[:len(rc)], startangle=90, wedgeprops=dict(edgecolor='#fffdf9', linewidth=2))
                for t in texts: t.set_color('#7a7060'); t.set_fontsize(8)
                for a in autos: a.set_color('#fffdf9'); a.set_fontsize(7.5); a.set_fontweight('bold')
                ax.set_title('Candidate Pool by Role')
                plt.tight_layout(); st.pyplot(fig)

            with ch2:
                st.markdown('<div class="sh">Screening Breakdown</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, 4)); warm_ax(fig, ax)
                bars = ax.bar(['Rejected\n(0–25%)', 'Review\n(25–50%)', 'Shortlisted\n(50–100%)'],
                           [rej, rev, short], color=['#c8401a', '#b07020', '#1a7a52'],
                           alpha=0.8, edgecolor='none', width=0.5)
                for bar, c in zip(bars, [rej, rev, short]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                           str(c), ha='center', va='bottom', fontweight='bold', fontsize=15, color='#1a1714', fontfamily='serif')
                ax.set_ylabel('Candidates'); ax.set_title('Screening Results')
                plt.tight_layout(); st.pyplot(fig)

            ch3, ch4 = st.columns(2, gap="large")

            with ch3:
                st.markdown('<div class="sh">Ranked Candidates</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(6, max(4, len(df) * .58))); warm_ax(fig, ax)
                bars = ax.barh(df['Candidate'], df['Match Score'],
                            color=[sc(s) for s in df['Match Score']], edgecolor='none', height=0.48, alpha=0.85)
                ax.axvline(50, color='#1a7a52', ls='--', lw=1.2, alpha=0.6, label='Shortlist')
                ax.axvline(25, color='#b07020', ls='--', lw=1.2, alpha=0.6, label='Review')
                for bar, s in zip(bars, df['Match Score']):
                    ax.text(bar.get_width() + .3, bar.get_y() + bar.get_height()/2,
                           f'{s}%', va='center', color='#7a7060', fontsize=8)
                ax.legend(facecolor='#fffdf9', edgecolor='#e8e2d8', labelcolor='#7a7060')
                ax.set_title('Match Score Rankings')
                plt.tight_layout(); st.pyplot(fig)

            with ch4:
                st.markdown('<div class="sh">Skill Gap Analysis</div>', unsafe_allow_html=True)
                if all_miss:
                    top_m = Counter(all_miss).most_common(10)
                    words, cnts = zip(*top_m)
                    fig, ax = plt.subplots(figsize=(6, 4)); warm_ax(fig, ax)
                    ax.barh(list(words)[::-1], list(cnts)[::-1], color='#c8401a', alpha=0.75, edgecolor='none', height=0.48)
                    ax.set_xlabel('Frequency missing'); ax.set_title('Most Missing Keywords')
                    plt.tight_layout(); st.pyplot(fig)

            st.divider()
            st.markdown('<div class="sh">Full Candidate Report</div>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
            st.download_button("Download Full Report (CSV)", df.to_csv(index=False).encode(), "report.csv", "text/csv")
    else:
        st.markdown("""<div class="empty">
          <div class="empty-ico">📊</div>
          <div class="empty-title">Ready to generate insights</div>
          <div class="empty-sub">Upload 3 or more resumes to see the full analytics dashboard</div>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:44px 0 16px;border-top:1px solid rgba(26,23,20,0.08);margin-top:52px">
  <div style="font-family:'Playfair Display',serif;font-size:0.95rem;font-style:italic;color:#b0a898">RecruitIQ · AI Resume Intelligence</div>
  <div style="font-size:0.68rem;color:#d4cfc8;margin-top:5px">Random Forest · TF-IDF · Scikit-learn · Streamlit</div>
</div>
""", unsafe_allow_html=True)