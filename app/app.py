import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Sentiment Analytics",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL STYLES (BRIGHT, CLEAN & INTERACTIVE)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

/* ── Root & background ── */
html, body, [class*="css"] {
    font-family: 'Roboto', sans-serif;
}

.stApp {
    background: #f9f9f9;
    color: #0f0f0f;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e5e5e5;
    box-shadow: 2px 0 15px rgba(0,0,0,0.03);
}

[data-testid="stSidebar"] .stRadio label {
    color: #606060 !important;
    font-size: 0.95rem;
    padding: 12px 16px;
    border-radius: 10px;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    cursor: pointer;
    margin-bottom: 4px;
}

[data-testid="stSidebar"] .stRadio label:hover {
    color: #ff0000 !important;
    background: #fff0f0;
    transform: translateX(6px);
}

/* ── Metric cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    cursor: default;
}

.metric-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 24px rgba(255, 0, 0, 0.08);
    border-color: #ff000033;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 5px;
    background: var(--accent, #ff0000);
    border-radius: 16px 16px 0 0;
    transition: height 0.3s ease;
}

.metric-card:hover::before {
    height: 8px;
}

.metric-icon {
    font-size: 2.5rem;
    margin-bottom: 12px;
    transition: transform 0.3s ease;
}

.metric-card:hover .metric-icon {
    transform: scale(1.15) rotate(5deg);
}

.metric-value {
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--accent, #ff0000);
    line-height: 1;
    margin-bottom: 6px;
}

.metric-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #606060;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Section headers ── */
.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0f0f0f;
    margin: 40px 0 20px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.section-title::after {
    content: '';
    flex: 1;
    height: 2px;
    background: linear-gradient(90deg, #e5e5e5, transparent);
    margin-left: 16px;
}

/* ── Page title ── */
.page-header {
    padding: 28px 36px;
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.02);
    border: 1px solid #e5e5e5;
    margin-bottom: 32px;
    border-left: 6px solid #ff0000;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.page-header:hover {
    transform: translateX(4px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.05);
}

.page-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #0f0f0f;
    margin: 0;
    letter-spacing: -0.5px;
}

.page-header p {
    color: #606060;
    font-size: 1.05rem;
    margin: 8px 0 0 0;
}

/* ── Insight cards ── */
.insight-box {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    border-left: 5px solid #ff0000;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    font-size: 0.95rem;
    color: #333333;
    line-height: 1.6;
    box-shadow: 0 4px 10px rgba(0,0,0,0.02);
    transition: all 0.3s ease;
}

.insight-box:hover {
    transform: translateX(6px);
    box-shadow: 0 6px 15px rgba(0,0,0,0.06);
}

.insight-box strong {
    color: #0f0f0f;
    font-size: 1.05rem;
}

/* ── Prediction box ── */
.pred-positive {
    background: linear-gradient(135deg, #ffffff 0%, #f1f8e9 100%);
    border: 1px solid #a5d6a7;
    border-radius: 16px;
    padding: 40px 24px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(43, 166, 64, 0.08);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.pred-positive:hover {
    transform: scale(1.03);
    box-shadow: 0 15px 35px rgba(43, 166, 64, 0.15);
}

.pred-negative {
    background: linear-gradient(135deg, #ffffff 0%, #ffebee 100%);
    border: 1px solid #ef9a9a;
    border-radius: 16px;
    padding: 40px 24px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(255, 0, 0, 0.08);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.pred-negative:hover {
    transform: scale(1.03);
    box-shadow: 0 15px 35px rgba(255, 0, 0, 0.15);
}

.pred-emoji {
    font-size: 4.5rem;
    margin-bottom: 16px;
    transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}

.pred-positive:hover .pred-emoji, .pred-negative:hover .pred-emoji {
    transform: scale(1.25) rotate(10deg);
}

.pred-label-pos {
    font-size: 1.8rem;
    font-weight: 700;
    color: #2ba640;
}

.pred-label-neg {
    font-size: 1.8rem;
    font-weight: 700;
    color: #ff0000;
}

/* ── Text area ── */
.stTextArea textarea {
    background: #ffffff !important;
    border: 2px solid #e5e5e5 !important;
    color: #0f0f0f !important;
    border-radius: 12px !important;
    font-family: 'Roboto', sans-serif !important;
    padding: 16px !important;
    font-size: 1.05rem !important;
    transition: all 0.3s ease !important;
    box-shadow: inset 0 2px 5px rgba(0,0,0,0.02) !important;
}

.stTextArea textarea:focus {
    border-color: #ff0000 !important;
    box-shadow: 0 0 0 4px rgba(255,0,0,0.1) !important;
    transform: translateY(-2px);
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 14px 32px !important;
    font-size: 1.05rem !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    box-shadow: 0 4px 15px rgba(255,0,0,0.2) !important;
}

.stButton > button:hover {
    transform: translateY(-4px) scale(1.02) !important;
    box-shadow: 0 10px 25px rgba(255,0,0,0.3) !important;
}

.stButton > button:active {
    transform: translateY(1px) !important;
    box-shadow: 0 2px 10px rgba(255,0,0,0.2) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 2px solid #e5e5e5 !important;
    color: #0f0f0f !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}

.stSelectbox > div > div:hover {
    border-color: #ff000055 !important;
}

/* ── About cards ── */
.about-card {
    background: #ffffff;
    border: 1px solid #e5e5e5;
    border-radius: 16px;
    padding: 32px;
    height: 100%;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    transition: all 0.3s ease;
}

.about-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
}

.about-card h3 {
    color: #ff0000;
    font-size: 1.1rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 20px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

.about-card p, .about-card li {
    color: #444444;
    font-size: 1.05rem;
    line-height: 1.8;
}

.about-card ul {
    padding-left: 20px;
    margin: 0;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #ff4d4d, #ff0000) !important;
    border-radius: 8px !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-size: 1.05rem !important;
    color: #606060 !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ff0000 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] > div {
    background-color: #ff0000 !important;
}
[data-testid="stTabs"] button:hover {
    color: #ff0000 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB THEME (BRIGHT & CLEAN)
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#ffffff",
    "axes.facecolor":    "#ffffff",
    "axes.edgecolor":    "#e5e5e5",
    "axes.labelcolor":   "#606060",
    "xtick.color":       "#606060",
    "ytick.color":       "#606060",
    "text.color":        "#0f0f0f",
    "grid.color":        "#f0f0f0",
    "grid.alpha":        0.8,
})

# ─────────────────────────────────────────────
#  DATA & MODEL
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("../data/label_youtube.csv")

@st.cache_resource
def load_model():
    model = joblib.load("../naive_bayes_model.pkl")
    tfidf = joblib.load("../tfidf.pkl")
    return model, tfidf

df = load_data()
model, tfidf = load_model()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 32px 0;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg" style="width: 150px; margin-bottom: 8px; transition: transform 0.3s ease; cursor: pointer;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'" alt="YouTube Logo">
        <div style="font-weight:700; font-size:1.15rem; color:#0f0f0f; margin-top:12px;">Sentiment Analytics</div>
        <div style="font-size:0.8rem; color:#888888; margin-top:4px;">Intelligence Dashboard v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "Navigasi",
        ["🏠  Dashboard", "☁️  WordCloud", "🎯  Evaluasi Model", "🤖  Prediksi", "📄  Dataset", "📚  Tentang"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div style="margin-top:auto; padding: 20px 0 8px 0; border-top:1px solid #e5e5e5;">
        <div style="font-size:0.85rem; color:#888888; text-align:center;">
            Naive Bayes · TF-IDF · YouTube Data
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SHARED DATA
# ─────────────────────────────────────────────
total      = len(df)
positif    = int((df['sentiment'] == 1).sum())
negatif    = int((df['sentiment'] == 0).sum())
pct_pos    = positif / total * 100
pct_neg    = negatif / total * 100

CM = np.array([[1377, 208], [324, 4091]])
dash_acc = (CM[0,0] + CM[1,1]) / CM.sum() * 100

# ══════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════
if "Dashboard" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>🏠 Dashboard Analisis Sentimen</h1>
        <p>Gambaran umum distribusi sentimen komentar YouTube</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric cards ──
    c1, c2, c3, c4 = st.columns(4)

    cards = [
        (c1, "📊", f"{total:,}", "Total Komentar", "#0f0f0f"),
        (c2, "👍", f"{positif:,}", "Sentimen Positif", "#2ba640"),
        (c3, "👎", f"{negatif:,}", "Sentimen Negatif", "#ff0000"),
        (c4, "🎯", f"{dash_acc:.2f}%", "Akurasi Model", "#3ea6ff"),
    ]

    for col, icon, value, label, color in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{color}">
                <div class="metric-icon">{icon}</div>
                <div class="metric-value" style="color:{color}">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Distribusi Sentimen</div>', unsafe_allow_html=True)

    col_chart, col_info = st.columns([3, 2], gap="large")

    with col_chart:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Bar chart
        ax = axes[0]
        bars = ax.bar(
            ["Positif", "Negatif"],
            [positif, negatif],
            color=["#2ba640", "#ff0000"],
            width=0.45,
            edgecolor="none",
            zorder=3
        )
        ax.set_ylabel("Jumlah Komentar", fontsize=10)
        ax.set_title("Distribusi Kelas", fontsize=12, fontweight='700', pad=12)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[['top','right','left']].set_visible(False)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (total*0.01),
                f"{int(bar.get_height()):,}",
                ha='center', va='bottom', fontsize=11, color='#0f0f0f', fontweight='700'
            )

        # Pie chart
        ax2 = axes[1]
        wedges, texts, autotexts = ax2.pie(
            [positif, negatif],
            labels=["Positif", "Negatif"],
            colors=["#2ba640", "#ff0000"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(edgecolor="#ffffff", linewidth=2.5),
            pctdistance=0.75
        )
        for at in autotexts:
            at.set_color("#ffffff")
            at.set_fontsize(11)
            at.set_fontweight('700')
        for t in texts:
            t.set_color("#606060")
            t.set_fontsize(11)
            t.set_fontweight('600')
        ax2.set_title("Proporsi Kelas", fontsize=12, fontweight='700', pad=12)

        fig.tight_layout(pad=2)
        st.pyplot(fig)

    with col_info:
        st.markdown("""
        <div class="insight-box" style="border-left-color:#3ea6ff">
            <strong>📌 Komposisi Dataset</strong><br>
            Dataset berisi komentar penonton dari video YouTube yang sudah dilabeli secara manual ke dalam dua kelas sentimen.
        </div>
        """, unsafe_allow_html=True)

        pos_pct = f"{pct_pos:.1f}%"
        neg_pct = f"{pct_neg:.1f}%"

        st.markdown(f"""
        <div class="insight-box" style="border-left-color:#2ba640">
            <strong>👍 Positif — {pos_pct}</strong><br>
            Sebagian besar audiens memberikan komentar positif, menunjukkan apresiasi terhadap kualitas konten.
        </div>
        <div class="insight-box" style="border-left-color:#ff0000">
            <strong>👎 Negatif — {neg_pct}</strong><br>
            Komentar negatif umumnya berupa kritik terhadap isi video, kualitas audio/visual, atau ketidaksetujuan opini.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">💬 Cuplikan Komentar YouTube</div>', unsafe_allow_html=True)
    
    col_pos, col_neg = st.columns(2)
    
    with col_pos:
        st.markdown("<h4 style='color:#2ba640; font-size:1.1rem; margin-bottom:12px;'>👍 Positif Teratas</h4>", unsafe_allow_html=True)
        # Ambil 3 sampel komentar positif acak
        pos_samples = df[df['sentiment'] == 1]['content'].dropna().sample(min(3, len(df[df['sentiment'] == 1])), random_state=42).tolist()
        for c in pos_samples:
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e5e5e5; border-radius:10px; padding:16px; margin-bottom:12px; box-shadow:0 2px 8px rgba(0,0,0,0.02); transition: transform 0.2s ease;" onmouseover="this.style.transform='translateX(4px)'" onmouseout="this.style.transform='translateX(0)'">
                <div style="color:#333333; font-size:0.95rem; line-height:1.5;">"{c}"</div>
            </div>
            """, unsafe_allow_html=True)
            
    with col_neg:
        st.markdown("<h4 style='color:#ff0000; font-size:1.1rem; margin-bottom:12px;'>👎 Negatif Teratas</h4>", unsafe_allow_html=True)
        # Ambil 3 sampel komentar negatif acak
        neg_samples = df[df['sentiment'] == 0]['content'].dropna().sample(min(3, len(df[df['sentiment'] == 0])), random_state=42).tolist()
        for c in neg_samples:
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #e5e5e5; border-radius:10px; padding:16px; margin-bottom:12px; box-shadow:0 2px 8px rgba(0,0,0,0.02); transition: transform 0.2s ease;" onmouseover="this.style.transform='translateX(4px)'" onmouseout="this.style.transform='translateX(0)'">
                <div style="color:#333333; font-size:0.95rem; line-height:1.5;">"{c}"</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE: WORDCLOUD
# ══════════════════════════════════════════════
elif "WordCloud" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>☁️ Word Cloud Visualisasi</h1>
        <p>Kata-kata yang paling sering muncul dalam komentar positif dan negatif</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["👍  Komentar Positif", "👎  Komentar Negatif"])

    def make_wordcloud(freqs, colormap, bg="#ffffff"):
        wc = WordCloud(
            width=1000,
            height=420,
            background_color=bg,
            colormap=colormap,
            max_words=120,
            prefer_horizontal=0.85,
            margin=8
        ).generate_from_frequencies(freqs)
        return wc

    feature_names = tfidf.get_feature_names_out()
    pos_freqs_array = model.feature_count_[1]
    neg_freqs_array = model.feature_count_[0]
    
    pos_freqs = {feature_names[i]: pos_freqs_array[i] for i in range(len(feature_names)) if pos_freqs_array[i] > 0}
    neg_freqs = {feature_names[i]: neg_freqs_array[i] for i in range(len(feature_names)) if neg_freqs_array[i] > 0}

    with tab1:
        wc = make_wordcloud(pos_freqs, "Greens")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig.patch.set_facecolor("#ffffff")
        fig.tight_layout(pad=0)
        st.pyplot(fig)

        # Top words
        top_words = sorted(pos_freqs.items(), key=lambda x: x[1], reverse=True)[:10]
        st.markdown('<div class="section-title">Top 10 Kata — Positif</div>', unsafe_allow_html=True)
        tw_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.barh(tw_df["Kata"][::-1], tw_df["Frekuensi"][::-1], color="#2ba640", edgecolor="none")
        ax2.set_xlabel("Frekuensi (TF-IDF)", fontsize=10)
        ax2.xaxis.grid(True)
        ax2.set_axisbelow(True)
        ax2.spines[['top','right','bottom']].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)

    with tab2:
        wc = make_wordcloud(neg_freqs, "Reds")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig.patch.set_facecolor("#ffffff")
        fig.tight_layout(pad=0)
        st.pyplot(fig)

        # Top words
        top_words = sorted(neg_freqs.items(), key=lambda x: x[1], reverse=True)[:10]
        st.markdown('<div class="section-title">Top 10 Kata — Negatif</div>', unsafe_allow_html=True)
        tw_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.barh(tw_df["Kata"][::-1], tw_df["Frekuensi"][::-1], color="#ff0000", edgecolor="none")
        ax2.set_xlabel("Frekuensi (TF-IDF)", fontsize=10)
        ax2.xaxis.grid(True)
        ax2.set_axisbelow(True)
        ax2.spines[['top','right','bottom']].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)

# ══════════════════════════════════════════════
#  PAGE: EVALUASI MODEL
# ══════════════════════════════════════════════
elif "Evaluasi" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>🎯 Evaluasi Model</h1>
        <p>Performa model Naive Bayes pada data uji</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric row ──
    TP, FN = CM[1][1], CM[1][0]
    FP, TN = CM[0][1], CM[0][0]

    accuracy  = (TP + TN) / CM.sum()
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f1        = 2 * precision * recall / (precision + recall)

    m1, m2, m3, m4 = st.columns(4)
    for col, label, val, color in [
        (m1, "Accuracy",  f"{accuracy*100:.2f}%",  "#3ea6ff"),
        (m2, "Precision", f"{precision*100:.2f}%", "#2ba640"),
        (m3, "Recall",    f"{recall*100:.2f}%",    "#f5a623"),
        (m4, "F1-Score",  f"{f1*100:.2f}%",        "#c47aff"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card" style="--accent:{color}">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)

    col_cm, col_report = st.columns([1, 1], gap="large")

    with col_cm:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            CM,
            annot=True,
            fmt='d',
            cmap='Reds',
            linewidths=2,
            linecolor="#ffffff",
            cbar=False,
            ax=ax,
            annot_kws={"size": 15, "weight": "bold"}
        )
        ax.set_xlabel("Predicted Label", fontsize=12, labelpad=12, fontweight='bold')
        ax.set_ylabel("Actual Label", fontsize=12, labelpad=12, fontweight='bold')
        ax.set_xticklabels(["Negatif", "Positif"], fontsize=11)
        ax.set_yticklabels(["Negatif", "Positif"], fontsize=11, rotation=0)
        fig.tight_layout()
        st.pyplot(fig)

    with col_report:
        st.markdown(f"""
        <div class="insight-box" style="border-left-color:#2ba640">
            <strong>True Positive (TP)</strong> — {TP:,}<br>
            Komentar positif yang diprediksi benar sebagai positif.
        </div>
        <div class="insight-box" style="border-left-color:#0f0f0f">
            <strong>True Negative (TN)</strong> — {TN:,}<br>
            Komentar negatif yang diprediksi benar sebagai negatif.
        </div>
        <div class="insight-box" style="border-left-color:#ff0000">
            <strong>False Positive (FP)</strong> — {FP:,}<br>
            Komentar negatif yang keliru diprediksi sebagai positif.
        </div>
        <div class="insight-box" style="border-left-color:#ff0000">
            <strong>False Negative (FN)</strong> — {FN:,}<br>
            Komentar positif yang keliru diprediksi sebagai negatif.
        </div>
        """, unsafe_allow_html=True)

    # Classification report per class
    st.markdown('<div class="section-title">Laporan Per Kelas</div>', unsafe_allow_html=True)

    precision_0 = TN / (TN + FN)
    recall_0 = TN / (TN + FP)
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0)

    report_data = {
        "Kelas":     ["Negatif (0)", "Positif (1)"],
        "Precision": [f"{precision_0*100:.1f}%", f"{precision*100:.1f}%"],
        "Recall":    [f"{recall_0*100:.1f}%",  f"{recall*100:.1f}%"],
        "F1-Score":  [f"{f1_0*100:.1f}%", f"{f1*100:.1f}%"],
        "Support":   [TN+FP, TP+FN],
    }
    st.dataframe(pd.DataFrame(report_data), hide_index=True)

# ══════════════════════════════════════════════
#  PAGE: PREDIKSI
# ══════════════════════════════════════════════
elif "Prediksi" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>🤖 Prediksi Sentimen</h1>
        <p>Masukkan komentar YouTube untuk diklasifikasikan oleh model</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        review = st.text_area(
            "Tulis komentar YouTube di sini…",
            placeholder="Contoh: Video ini sangat informatif, terima kasih pembahasannya!",
            height=180
        )

        examples = [
            "Penjelasannya sangat detail dan mudah dipahami, konten yang bagus!",
            "Kualitas audio sangat buruk, terlalu banyak noise sehingga tidak terdengar jelas.",
            "Visualnya keren, semangat terus berkarya!",
        ]

        st.markdown("<div style='font-size:0.95rem; color:#606060; margin:20px 0 10px 0; font-weight:600;'>✨ Coba contoh komentar:</div>", unsafe_allow_html=True)
        for ex in examples:
            if st.button(ex[:45] + "…" if len(ex) > 45 else ex, key=ex):
                review = ex

        predict_btn = st.button("🔍  Analisis Sentimen", width='stretch')

    with col_result:
        if review.strip():

            import re
            slangwords = {"@": "di", "abis": "habis", "wtb": "beli", "masi": "masih", "wts": "jual", "wtt": "tukar", "bgt": "banget", "maks": "maksimal", "plisss": "tolong", "bgttt": "banget", "indo": "indonesia", "bgtt": "banget", "ad": "ada", "rv": "redvelvet", "plis": "tolong", "pls": "tolong", "cr": "sumber", "cod": "bayar ditempat", "adlh": "adalah", "afaik": "as far as i know", "ahaha": "haha", "aj": "saja", "ajep-ajep": "dunia gemerlap", "ak": "saya", "akika": "aku", "akkoh": "aku", "akuwh": "aku", "alay": "norak", "alow": "halo", "ambilin": "ambilkan", "ancur": "hancur", "anjrit": "anjing", "anter": "antar", "ap2": "apa-apa", "apasih": "apa sih", "apes": "sial", "aps": "apa", "aq": "saya", "aquwh": "aku", "asbun": "asal bunyi", "aseekk": "asyik", "asekk": "asyik", "asem": "asam", "aspal": "asli tetapi palsu", "astul": "asal tulis", "ato": "atau", "au ah": "tidak mau tahu", "awak": "saya", "ay": "sayang", "ayank": "sayang", "b4": "sebelum", "bakalan": "akan", "bandes": "bantuan desa", "bangedh": "banget", "banpol": "bantuan polisi", "banpur": "bantuan tempur", "basbang": "basi", "bcanda": "bercanda", "bdg": "bandung", "begajulan": "nakal", "beliin": "belikan", "bencong": "banci", "bentar": "sebentar", "ber3": "bertiga", "beresin": "membereskan", "bete": "bosan", "beud": "banget", "bg": "abang", "bgmn": "bagaimana", "bgt": "banget", "bijimane": "bagaimana", "bintal": "bimbingan mental", "bkl": "akan", "bknnya": "bukannya", "blegug": "bodoh", "blh": "boleh", "bln": "bulan", "blum": "belum", "bnci": "benci", "bnran": "yang benar", "bodor": "lucu", "bokap": "ayah", "boker": "buang air besar", "bokis": "bohong", "boljug": "boleh juga", "bonek": "bocah nekat", "boyeh": "boleh", "br": "baru", "brg": "bareng", "bro": "saudara laki-laki", "bru": "baru", "bs": "bisa", "bsen": "bosan", "bt": "buat", "btw": "ngomong-ngomong", "buaya": "tidak setia", "bubbu": "tidur", "bubu": "tidur", "bumil": "ibu hamil", "bw": "bawa", "bwt": "buat", "byk": "banyak", "byrin": "bayarkan", "cabal": "sabar", "cadas": "keren", "calo": "makelar", "can": "belum", "capcus": "pergi", "caper": "cari perhatian", "ce": "cewek", "cekal": "cegah tangkal", "cemen": "penakut", "cengengesan": "tertawa", "cepet": "cepat", "cew": "cewek", "chuyunk": "sayang", "cimeng": "ganja", "cipika cipiki": "cium pipi kanan cium pipi kiri", "ciyh": "sih", "ckepp": "cakep", "ckp": "cakep", "cmiiw": "correct me if i'm wrong", "cmpur": "campur", "cong": "banci", "conlok": "cinta lokasi", "cowwyy": "maaf", "cp": "siapa", "cpe": "capek", "cppe": "capek", "cucok": "cocok", "cuex": "cuek", "cumi": "Cuma miscall", "cups": "culun", "curanmor": "pencurian kendaraan bermotor", "curcol": "curahan hati colongan", "cwek": "cewek", "cyin": "cinta", "d": "di", "dah": "deh", "dapet": "dapat", "de": "adik", "dek": "adik", "demen": "suka", "deyh": "deh", "dgn": "dengan", "diancurin": "dihancurkan", "dimaafin": "dimaafkan", "dimintak": "diminta", "disono": "di sana", "dket": "dekat", "dkk": "dan kawan-kawan", "dll": "dan lain-lain", "dlu": "dulu", "dngn": "dengan", "dodol": "bodoh", "doku": "uang", "dongs": "dong", "dpt": "dapat", "dri": "dari", "drmn": "darimana", "drtd": "dari tadi", "dst": "dan seterusnya", "dtg": "datang", "duh": "aduh", "duren": "durian", "ed": "edisi", "egp": "emang gue pikirin", "eke": "aku", "elu": "kamu", "emangnya": "memangnya", "emng": "memang", "endak": "tidak", "enggak": "tidak", "envy": "iri", "ex": "mantan", "fax": "facsimile", "fifo": "first in first out", "folbek": "follow back", "fyi": "sebagai informasi", "gaada": "tidak ada uang", "gag": "tidak", "gaje": "tidak jelas", "gak papa": "tidak apa-apa", "gan": "juragan", "gaptek": "gagap teknologi", "gatek": "gagap teknologi", "gawe": "kerja", "gbs": "tidak bisa", "gebetan": "orang yang disuka", "geje": "tidak jelas", "gepeng": "gelandangan dan pengemis", "ghiy": "lagi", "gile": "gila", "gimana": "bagaimana", "gino": "gigi nongol", "githu": "gitu", "gj": "tidak jelas", "gmana": "bagaimana", "gn": "begini", "goblok": "bodoh", "golput": "golongan putih", "gowes": "mengayuh sepeda", "gpny": "tidak punya", "gr": "gede rasa", "gretongan": "gratisan", "gtau": "tidak tahu", "gua": "saya", "guoblok": "goblok", "gw": "saya", "ha": "tertawa", "haha": "tertawa", "hallow": "halo", "hankam": "pertahanan dan keamanan", "hehe": "he", "helo": "halo", "hey": "hai", "hlm": "halaman", "hny": "hanya", "hoax": "isu bohong", "hr": "hari", "hrus": "harus", "hubdar": "perhubungan darat", "huff": "mengeluh", "hum": "rumah", "humz": "rumah", "ilang": "hilang", "ilfil": "tidak suka", "imho": "in my humble opinion", "imoetz": "imut", "item": "hitam", "itungan": "hitungan", "iye": "iya", "ja": "saja", "jadiin": "jadi", "jaim": "jaga image", "jayus": "tidak lucu", "jdi": "jadi", "jem": "jam", "jga": "juga", "jgnkan": "jangankan", "jir": "anjing", "jln": "jalan", "jomblo": "tidak punya pacar", "jubir": "juru bicara", "jutek": "galak", "k": "ke", "kab": "kabupaten", "kabor": "kabur", "kacrut": "kacau", "kadiv": "kepala divisi", "kagak": "tidak", "kalo": "kalau", "kampret": "sialan", "kamtibmas": "keamanan dan ketertiban masyarakat", "kamuwh": "kamu", "kanwil": "kantor wilayah", "karna": "karena", "kasubbag": "kepala subbagian", "katrok": "kampungan", "kayanya": "kayaknya", "kbr": "kabar", "kdu": "harus", "kec": "kecamatan", "kejurnas": "kejuaraan nasional", "kekeuh": "keras kepala", "kel": "kelurahan", "kemaren": "kemarin", "kepengen": "mau", "kepingin": "mau", "kepsek": "kepala sekolah", "kesbang": "kesatuan bangsa", "kesra": "kesejahteraan rakyat", "ketrima": "diterima", "kgiatan": "kegiatan", "kibul": "bohong", "kimpoi": "kawin", "kl": "kalau", "klianz": "kalian", "kloter": "kelompok terbang", "klw": "kalau", "km": "kamu", "kmps": "kampus", "kmrn": "kemarin", "knal": "kenal", "knp": "kenapa", "kodya": "kota madya", "komdis": "komisi disiplin", "komsov": "komunis sovyet", "kongkow": "kumpul bareng teman-teman", "kopdar": "kopi darat", "korup": "korupsi", "kpn": "kapan", "krenz": "keren", "krm": "kirim", "kt": "kita", "ktmu": "ketemu", "ktr": "kantor", "kuper": "kurang pergaulan", "kw": "imitasi", "kyk": "seperti", "la": "lah", "lam": "salam", "lamp": "lampiran", "lanud": "landasan udara", "latgab": "latihan gabungan", "lebay": "berlebihan", "leh": "boleh", "lelet": "lambat", "lemot": "lambat", "lgi": "lagi", "lgsg": "langsung", "liat": "lihat", "litbang": "penelitian dan pengembangan", "lmyn": "lumayan", "lo": "kamu", "loe": "kamu", "lola": "lambat berfikir", "louph": "cinta", "low": "kalau", "lp": "lupa", "luber": "langsung, umum, bebas, dan rahasia", "luchuw": "lucu", "lum": "belum", "luthu": "lucu", "lwn": "lawan", "maacih": "terima kasih", "mabal": "bolos", "macem": "macam", "macih": "masih", "maem": "makan", "magabut": "makan gaji buta", "maho": "homo", "mak jang": "kaget", "maksain": "memaksa", "malem": "malam", "mam": "makan", "maneh": "kamu", "maniez": "manis", "mao": "mau", "masukin": "masukkan", "melu": "ikut", "mepet": "dekat sekali", "mgu": "minggu", "migas": "minyak dan gas bumi", "mikol": "minuman beralkohol", "miras": "minuman keras", "mlah": "malah", "mngkn": "mungkin", "mo": "mau", "mokad": "mati", "moso": "masa", "mpe": "sampai", "msk": "masuk", "mslh": "masalah", "mt": "makan teman", "mubes": "musyawarah besar", "mulu": "melulu", "mumpung": "selagi", "munas": "musyawarah nasional", "muntaber": "muntah dan berak", "musti": "mesti", "muupz": "maaf", "mw": "now watching", "n": "dan", "nanam": "menanam", "nanya": "bertanya", "napa": "kenapa", "napi": "narapidana", "napza": "narkotika, alkohol, psikotropika, dan zat adiktif ", "narkoba": "narkotika, psikotropika, dan obat terlarang", "nasgor": "nasi goreng", "nda": "tidak", "ndiri": "sendiri", "ne": "ini", "nekolin": "neokolonialisme", "nembak": "menyatakan cinta", "ngabuburit": "menunggu berbuka puasa", "ngaku": "mengaku", "ngambil": "mengambil", "nganggur": "tidak punya pekerjaan", "ngapah": "kenapa", "ngaret": "terlambat", "ngasih": "memberikan", "ngebandel": "berbuat bandel", "ngegosip": "bergosip", "ngeklaim": "mengklaim", "ngeksis": "menjadi eksis", "ngeles": "berkilah", "ngelidur": "menggigau", "ngerampok": "merampok", "ngga": "tidak", "ngibul": "berbohong", "ngiler": "mau", "ngiri": "iri", "ngisiin": "mengisikan", "ngmng": "bicara", "ngomong": "bicara", "ngubek2": "mencari-cari", "ngurus": "mengurus", "nie": "ini", "nih": "ini", "niyh": "nih", "nmr": "nomor", "nntn": "nonton", "nobar": "nonton bareng", "np": "now playing", "ntar": "nanti", "ntn": "nonton", "numpuk": "bertumpuk", "nutupin": "menutupi", "nyari": "mencari", "nyekar": "menyekar", "nyicil": "mencicil", "nyoblos": "mencoblos", "nyokap": "ibu", "ogah": "tidak mau", "ol": "online", "ongkir": "ongkos kirim", "oot": "out of topic", "org2": "orang-orang", "ortu": "orang tua", "otda": "otonomi daerah", "otw": "on the way, sedang di jalan", "pacal": "pacar", "pake": "pakai", "pala": "kepala", "pansus": "panitia khusus", "parpol": "partai politik", "pasutri": "pasangan suami istri", "pd": "pada", "pede": "percaya diri", "pelatnas": "pemusatan latihan nasional", "pemda": "pemerintah daerah", "pemkot": "pemerintah kota", "pemred": "pemimpin redaksi", "penjas": "pendidikan jasmani", "perda": "peraturan daerah", "perhatiin": "perhatikan", "pesenan": "pesanan", "pgang": "pegang", "pi": "tapi", "pilkada": "pemilihan kepala daerah", "pisan": "sangat", "pk": "penjahat kelamin", "plg": "paling", "pmrnth": "pemerintah", "polantas": "polisi lalu lintas", "ponpes": "pondok pesantren", "pp": "pulang pergi", "prg": "pergi", "prnh": "pernah", "psen": "pesan", "pst": "pasti", "pswt": "pesawat", "pw": "posisi nyaman", "qmu": "kamu", "rakor": "rapat koordinasi", "ranmor": "kendaraan bermotor", "re": "reply", "ref": "referensi", "rehab": "rehabilitasi", "rempong": "sulit", "repp": "balas", "restik": "reserse narkotika", "rhs": "rahasia", "rmh": "rumah", "ru": "baru", "ruko": "rumah toko", "rusunawa": "rumah susun sewa", "ruz": "terus", "saia": "saya", "salting": "salah tingkah", "sampe": "sampai", "samsek": "sama sekali", "sapose": "siapa", "satpam": "satuan pengamanan", "sbb": "sebagai berikut", "sbh": "sebuah", "sbnrny": "sebenarnya", "scr": "secara", "sdgkn": "sedangkan", "sdkt": "sedikit", "se7": "setuju", "sebelas dua belas": "mirip", "sembako": "sembilan bahan pokok", "sempet": "sempat", "sendratari": "seni drama tari", "sgt": "sangat", "shg": "sehingga", "siech": "sih", "sikon": "situasi dan kondisi", "sinetron": "sinema elektronik", "siramin": "siramkan", "sj": "saja", "skalian": "sekalian", "sklh": "sekolah", "skt": "sakit", "slesai": "selesai", "sll": "selalu", "slma": "selama", "slsai": "selesai", "smpt": "sempat", "smw": "semua", "sndiri": "sendiri", "soljum": "sholat jumat", "songong": "sombong", "sory": "maaf", "sosek": "sosial-ekonomi", "sotoy": "sok tahu", "spa": "siapa", "sppa": "siapa", "spt": "seperti", "srtfkt": "sertifikat", "stiap": "setiap", "stlh": "setelah", "suk": "masuk", "sumpek": "sempit", "syg": "sayang", "t4": "tempat", "tajir": "kaya", "tau": "tahu", "taw": "tahu", "td": "tadi", "tdk": "tidak", "teh": "kakak perempuan", "telat": "terlambat", "telmi": "telat berpikir", "temen": "teman", "tengil": "menyebalkan", "tepar": "terkapar", "tggu": "tunggu", "tgu": "tunggu", "thankz": "terima kasih", "thn": "tahun", "tilang": "bukti pelanggaran", "tipiwan": "TvOne", "tks": "terima kasih", "tlp": "telepon", "tls": "tulis", "tmbah": "tambah", "tmen2": "teman-teman", "tmpah": "tumpah", "tmpt": "tempat", "tngu": "tunggu", "tnyta": "ternyata", "tokai": "tai", "toserba": "toko serba ada", "tpi": "tapi", "trdhulu": "terdahulu", "trima": "terima kasih", "trm": "terima", "trs": "terus", "trutama": "terutama", "ts": "penulis", "tst": "tahu sama tahu", "ttg": "tentang", "tuch": "tuh", "tuir": "tua", "tw": "tahu", "u": "kamu", "ud": "sudah", "udah": "sudah", "ujg": "ujung", "ul": "ulangan", "unyu": "lucu", "uplot": "unggah", "urang": "saya", "usah": "perlu", "utk": "untuk", "valas": "valuta asing", "w/": "dengan", "wadir": "wakil direktur", "wamil": "wajib militer", "warkop": "warung kopi", "warteg": "warung tegal", "wat": "buat", "wkt": "waktu", "wtf": "what the fuck", "xixixi": "tertawa", "ya": "iya", "yap": "iya", "yaudah": "ya sudah", "yawdah": "ya sudah", "yg": "yang", "yl": "yang lain", "yo": "iya", "yowes": "ya sudah", "yup": "iya", "7an": "tujuan", "ababil": "abg labil", "acc": "accord", "adlah": "adalah", "adoh": "aduh", "aha": "tertawa", "aing": "saya", "aja": "saja", "ajj": "saja", "aka": "dikenal juga sebagai", "akko": "aku", "akku": "aku", "akyu": "aku", "aljasa": "asal jadi saja", "ama": "sama", "ambl": "ambil", "anjir": "anjing", "ank": "anak", "ap": "apa", "apaan": "apa", "ape": "apa", "aplot": "unggah", "apva": "apa", "aqu": "aku", "asap": "sesegera mungkin", "aseek": "asyik", "asek": "asyik", "aseknya": "asyiknya", "asoy": "asyik", "astrojim": "astagfirullahaladzim", "ath": "kalau begitu", "atuh": "kalau begitu", "ava": "avatar", "aws": "awas", "ayang": "sayang", "ayok": "ayo", "bacot": "banyak bicara", "bales": "balas", "bangdes": "pembangunan desa", "bangkotan": "tua", "banpres": "bantuan presiden", "bansarkas": "bantuan sarana kesehatan", "bazis": "badan amal, zakat, infak, dan sedekah", "bcoz": "karena", "beb": "sayang", "bejibun": "banyak", "belom": "belum", "bener": "benar", "ber2": "berdua", "berdikari": "berdiri di atas kaki sendiri", "bet": "banget", "beti": "beda tipis", "beut": "banget", "bgd": "banget", "bgs": "bagus", "bhubu": "tidur", "bimbuluh": "bimbingan dan penyuluhan", "bisi": "kalau-kalau", "bkn": "bukan", "bl": "beli", "blg": "bilang", "blm": "belum", "bls": "balas", "bnchi": "benci", "bngung": "bingung", "bnyk": "banyak", "bohay": "badan aduhai", "bokep": "porno", "bokin": "pacar", "bole": "boleh", "bolot": "bodoh", "bonyok": "ayah ibu", "bpk": "bapak", "brb": "segera kembali", "brngkt": "berangkat", "brp": "berapa", "brur": "saudara laki-laki", "bsa": "bisa", "bsk": "besok", "bu_bu": "tidur", "bubarin": "bubarkan", "buber": "buka bersama", "bujubune": "luar biasa", "buser": "buru sergap", "bwhn": "bawahan", "byar": "bayar", "byr": "bayar", "c8": "chat", "cabut": "pergi", "caem": "cakep", "cama-cama": "sama-sama", "cangcut": "celana dalam", "cape": "capek", "caur": "jelek", "cekak": "tidak ada uang", "cekidot": "coba lihat", "cemplungin": "cemplungkan", "ceper": "pendek", "ceu": "kakak perempuan", "cewe": "cewek", "cibuk": "sibuk", "cin": "cinta", "ciye": "cie", "ckck": "ck", "clbk": "cinta lama bersemi kembali", "cmpr": "campur", "cnenk": "senang", "congor": "mulut", "cow": "cowok", "coz": "karena", "cpa": "siapa", "gokil": "gila", "gombal": "suka merayu", "gpl": "tidak pakai lama", "gpp": "tidak apa-apa", "gretong": "gratis", "gt": "begitu", "gtw": "tidak tahu", "gue": "saya", "guys": "teman-teman", "gws": "cepat sembuh", "haghaghag": "tertawa", "hakhak": "tertawa", "handak": "bahan peledak", "hansip": "pertahanan sipil", "hellow": "halo", "helow": "halo", "hi": "hai", "hlng": "hilang", "hnya": "hanya", "houm": "rumah", "hrs": "harus", "hubad": "hubungan angkatan darat", "hubla": "perhubungan laut", "huft": "mengeluh", "humas": "hubungan masyarakat", "idk": "saya tidak tahu", "ilfeel": "tidak suka", "imba": "jago sekali", "imoet": "imut", "info": "informasi", "itung": "hitung", "isengin": "bercanda", "iyala": "iya lah", "iyo": "iya", "jablay": "jarang dibelai", "jadul": "jaman dulu", "jancuk": "anjing", "jd": "jadi", "jdikan": "jadikan", "jg": "juga", "jgn": "jangan", "jijay": "jijik", "jkt": "jakarta", "jnj": "janji", "jth": "jatuh", "jurdil": "jujur adil", "jwb": "jawab", "ka": "kakak", "kabag": "kepala bagian", "kacian": "kasihan", "kadit": "kepala direktorat", "kaga": "tidak", "kaka": "kakak", "kamtib": "keamanan dan ketertiban", "kamuh": "kamu", "kamyu": "kamu", "kapt": "kapten", "kasat": "kepala satuan", "kasubbid": "kepala subbidang", "kau": "kamu", "kbar": "kabar", "kcian": "kasihan", "keburu": "terlanjur", "kedubes": "kedutaan besar", "kek": "seperti", "keknya": "kayaknya", "keliatan": "kelihatan", "keneh": "masih", "kepikiran": "terpikirkan", "kepo": "mau tahu urusan orang", "kere": "tidak punya uang", "kesian": "kasihan", "ketauan": "ketahuan", "keukeuh": "keras kepala", "khan": "kan", "kibus": "kaki busuk", "kk": "kakak", "klian": "kalian", "klo": "kalau", "kluarga": "keluarga", "klwrga": "keluarga", "kmari": "kemari", "kmpus": "kampus", "kn": "kan", "knl": "kenal", "knpa": "kenapa", "kog": "kok", "kompi": "komputer", "komtiong": "komunis Tiongkok", "konjen": "konsulat jenderal", "koq": "kok", "kpd": "kepada", "kptsan": "keputusan", "krik": "garing", "krn": "karena", "ktauan": "ketahuan", "ktny": "katanya", "kudu": "harus", "kuq": "kok", "ky": "seperti", "kykny": "kayanya", "laka": "kecelakaan", "lambreta": "lambat", "lansia": "lanjut usia", "lapas": "lembaga pemasyarakatan", "lbur": "libur", "lekong": "laki-laki", "lg": "lagi", "lgkp": "lengkap", "lht": "lihat", "linmas": "perlindungan masyarakat", "lmyan": "lumayan", "lngkp": "lengkap", "loch": "loh", "lol": "tertawa", "lom": "belum", "loupz": "cinta", "lowh": "kamu", "lu": "kamu", "luchu": "lucu", "luff": "cinta", "luph": "cinta", "lw": "kamu", "lwt": "lewat", "maaciw": "terima kasih", "mabes": "markas besar", "macem-macem": "macam-macam", "madesu": "masa depan suram", "maen": "main", "mahatma": "maju sehat bersama", "mak": "ibu", "makasih": "terima kasih", "malah": "bahkan", "malu2in": "memalukan", "mamz": "makan", "manies": "manis", "mantep": "mantap", "markus": "makelar kasus", "mba": "mbak", "mending": "lebih baik", "mgkn": "mungkin", "mhn": "mohon", "miker": "minuman keras", "milis": "mailing list", "mksd": "maksud", "mls": "malas", "mnt": "minta", "moge": "motor gede", "mokat": "mati", "mosok": "masa", "msh": "masih", "mskpn": "meskipun", "msng2": "masing-masing", "muahal": "mahal", "muker": "musyawarah kerja", "mumet": "pusing", "muna": "munafik", "munaslub": "musyawarah nasional luar biasa", "musda": "musyawarah daerah", "muup": "maaf", "muuv": "maaf", "nal": "kenal", "nangis": "menangis", "naon": "apa", "napol": "narapidana politik", "naq": "anak", "narsis": "bangga pada diri sendiri", "nax": "anak", "ndak": "tidak", "ndut": "gendut", "nekolim": "neokolonialisme", "nelfon": "menelepon", "ngabis2in": "menghabiskan", "ngakak": "tertawa", "ngambek": "marah", "ngampus": "pergi ke kampus", "ngantri": "mengantri", "ngapain": "sedang apa", "ngaruh": "berpengaruh", "ngawur": "berbicara sembarangan", "ngeceng": "kumpul bareng-bareng", "ngeh": "sadar", "ngekos": "tinggal di kos", "ngelamar": "melamar", "ngeliat": "melihat", "ngemeng": "bicara terus-terusan", "ngerti": "mengerti", "nggak": "tidak", "ngikut": "ikut", "nginep": "menginap", "ngisi": "mengisi", "ngmg": "bicara", "ngocol": "lucu", "ngomongin": "membicarakan", "ngumpul": "berkumpul", "ni": "ini", "nyasar": "tersesat", "nyariin": "mencari", "nyiapin": "mempersiapkan", "nyiram": "menyiram", "nyok": "ayo", "o/": "oleh", "ok": "ok", "priksa": "periksa", "pro": "profesional", "psn": "pesan", "psti": "pasti", "puanas": "panas", "qmo": "kamu", "qt": "kita", "rame": "ramai", "raskin": "rakyat miskin", "red": "redaksi", "reg": "register", "rejeki": "rezeki", "renstra": "rencana strategis", "reskrim": "reserse kriminal", "sni": "sini", "somse": "sombong sekali", "sorry": "maaf", "sosbud": "sosial-budaya", "sospol": "sosial-politik", "sowry": "maaf", "spd": "sepeda", "sprti": "seperti", "spy": "supaya", "stelah": "setelah", "subbag": "subbagian", "sumbangin": "sumbangkan", "sy": "saya", "syp": "siapa", "tabanas": "tabungan pembangunan nasional", "tar": "nanti", "taun": "tahun", "tawh": "tahu", "tdi": "tadi", "te2p": "tetap", "tekor": "rugi", "telkom": "telekomunikasi", "telp": "telepon", "temen2": "teman-teman", "tengok": "menjenguk", "terbitin": "terbitkan", "tgl": "tanggal", "thanks": "terima kasih", "thd": "terhadap", "thx": "terima kasih", "tipi": "TV", "tkg": "tukang", "tll": "terlalu", "tlpn": "telepon", "tman": "teman", "tmbh": "tambah", "tmn2": "teman-teman", "tmph": "tumpah", "tnda": "tanda", "tnh": "tanah", "togel": "toto gelap", "tp": "tapi", "tq": "terima kasih", "trgntg": "tergantung", "trims": "terima kasih", "cb": "coba", "y": "ya", "munfik": "munafik", "reklamuk": "reklamasi", "sma": "sama", "tren": "trend", "ngehe": "kesal", "mz": "mas", "analisise": "analisis", "sadaar": "sadar", "sept": "september", "nmenarik": "menarik", "zonk": "bodoh", "rights": "benar", "simiskin": "miskin", "ngumpet": "sembunyi", "hardcore": "keras", "akhirx": "akhirnya", "solve": "solusi", "watuk": "batuk", "ngebully": "intimidasi", "masy": "masyarakat", "still": "masih", "tauk": "tahu", "mbual": "bual", "tioghoa": "tionghoa", "ngentotin": "senggama", "kentot": "senggama", "faktakta": "fakta", "sohib": "teman", "rubahnn": "rubah", "trlalu": "terlalu", "nyela": "cela", "heters": "pembenci", "nyembah": "sembah", "most": "paling", "ikon": "lambang", "light": "terang", "pndukung": "pendukung", "setting": "atur", "seting": "akting", "next": "lanjut", "waspadalah": "waspada", "gantengsaya": "ganteng", "parte": "partai", "nyerang": "serang", "nipu": "tipu", "ktipu": "tipu", "jentelmen": "berani", "buangbuang": "buang", "tsangka": "tersangka", "kurng": "kurang", "ista": "nista", "less": "kurang", "koar": "teriak", "paranoid": "takut", "problem": "masalah", "tahi": "kotoran", "tirani": "tiran", "tilep": "tilap", "happy": "bahagia", "tak": "tidak", "penertiban": "tertib", "uasai": "kuasa", "mnolak": "tolak", "trending": "trend", "taik": "tahi", "wkwkkw": "tertawa", "ahokncc": "ahok", "istaa": "nista", "benarjujur": "jujur", "mgkin": "mungkin"}
            
            clean_rev = str(review)
            clean_rev = re.sub(r'@[A-Za-z0-9_]+', ' ', clean_rev)
            clean_rev = re.sub(r'#[A-Za-z0-9_]+', ' ', clean_rev)
            clean_rev = re.sub(r'http\S+|www\S+', ' ', clean_rev)
            clean_rev = re.sub(r'RT[\s]+', ' ', clean_rev)
            clean_rev = re.sub(r'\d+', ' ', clean_rev)
            clean_rev = re.sub(r'[^\w\s]', ' ', clean_rev)
            clean_rev = clean_rev.replace('\n', ' ')
            clean_rev = re.sub(r'\s+', ' ', clean_rev).strip()
            
            clean_rev = clean_rev.lower()
            
            words = clean_rev.split()
            fixed_words = [slangwords.get(word, word) for word in words]
            clean_rev = " ".join(fixed_words)
            
            vector     = tfidf.transform([clean_rev])
            pred       = model.predict(vector)[0]
            prob       = model.predict_proba(vector)[0]
            confidence = max(prob) * 100

            if pred == 1:
                st.markdown(f"""
                <div class="pred-positive">
                    <div class="pred-emoji">👍</div>
                    <div class="pred-label-pos">Sentimen Positif</div>
                    <div style="color:#444444; font-size:1rem; margin-top:8px;">
                        Model memprediksi komentar ini bernada positif
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-negative">
                    <div class="pred-emoji">👎</div>
                    <div class="pred-label-neg">Sentimen Negatif</div>
                    <div style="color:#444444; font-size:1rem; margin-top:8px;">
                        Model memprediksi komentar ini bernada negatif
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:28px;">
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="font-size:0.95rem; color:#606060; font-weight:600;">Confidence Score</span>
                    <span style="font-size:1.1rem; color:#0f0f0f; font-weight:700;">{confidence:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(confidence / 100)

            # Prob bars
            st.markdown("<div style='margin-top:20px;'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 1.5))
            classes = ["Negatif", "Positif"]
            colors  = ["#ff0000", "#2ba640"]
            bars = ax.barh(classes, [prob[0]*100, prob[1]*100], color=colors, edgecolor="none", height=0.45)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probabilitas (%)", fontsize=10)
            ax.spines[['top','right','bottom']].set_visible(False)
            for bar, p in zip(bars, [prob[0]*100, prob[1]*100]):
                ax.text(p + 1, bar.get_y() + bar.get_height()/2, f"{p:.1f}%", va='center', fontsize=10, color='#0f0f0f', fontweight='600')
            fig.tight_layout()
            st.pyplot(fig)

        elif predict_btn and not review.strip():
            st.warning("⚠️ Masukkan komentar terlebih dahulu.")
        else:
            st.markdown("""
            <div style="text-align:center; padding:80px 20px; color:#aaaaaa;">
                <div style="font-size:4rem; margin-bottom:16px; opacity:0.5;">🤖</div>
                <div style="font-size:1.1rem;">Hasil prediksi akan muncul di sini</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  PAGE: DATASET
# ══════════════════════════════════════════════
elif "Dataset" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>📄 Dataset Explorer</h1>
        <p>Jelajahi dan filter dataset komentar YouTube</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        search_term = st.text_area("🔍 Cari kata dalam komentar", placeholder="Ketik kata kunci (tekan Ctrl+Enter untuk mencari)…", height=68)
    with c2:
        pilihan = st.selectbox("Filter Sentimen", ["Semua", "Positif 👍", "Negatif 👎"])
    with c3:
        n_rows = st.selectbox("Tampilkan", [50, 100, 200, 500, "Semua"], index=1)

    data = df.copy()
    if "Positif" in pilihan:
        data = data[data['sentiment'] == 1]
    elif "Negatif" in pilihan:
        data = data[data['sentiment'] == 0]

    if search_term:
        data = data[data['content'].str.contains(search_term, case=False, na=False)]

    if n_rows != "Semua":
        data = data.head(int(n_rows))

    st.markdown(f"""
    <div style="font-size:0.95rem; color:#606060; margin-bottom:16px;">
        Menampilkan <strong style="color:#0f0f0f">{len(data):,}</strong> baris
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        data.reset_index(drop=True),
        use_container_width=True,
        height=500
    )

    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️  Download CSV",
        data=csv,
        file_name="youtube_sentiment_filtered.csv",
        mime="text/csv"
    )

# ══════════════════════════════════════════════
#  PAGE: TENTANG
# ══════════════════════════════════════════════
elif "Tentang" in menu:

    st.markdown("""
    <div class="page-header">
        <h1>📚 Tentang Penelitian</h1>
        <p>Informasi metodologi dan detail teknis proyek ini</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div class="about-card">
            <h3>🎯 Tujuan Penelitian</h3>
            <p>
                Mengklasifikasikan sentimen komentar penonton YouTube
                ke dalam dua kelas: <strong style="color:#2ba640">positif</strong> dan
                <strong style="color:#ff0000">negatif</strong>, untuk membantu memahami
                persepsi audiens secara otomatis menggunakan pendekatan machine learning.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="about-card">
            <h3>📊 Dataset</h3>
            <ul>
                <li>Sumber: API YouTube / Web Scraping</li>
                <li>Total: <strong style="color:#0f0f0f">{total:,} komentar</strong></li>
                <li>Label: Positif & Negatif</li>
                <li>Metode labeling: Semi-otomatis + verifikasi manual</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.markdown("""
        <div class="about-card">
            <h3>⚙️ Pipeline NLP</h3>
            <ul>
                <li>Preprocessing: case folding, stopword removal, stemming</li>
                <li>Ekstraksi fitur: <strong style="color:#0f0f0f">TF-IDF Vectorizer</strong></li>
                <li>Classifier: <strong style="color:#0f0f0f">Multinomial Naive Bayes</strong></li>
                <li>Validasi: train-test split 80:20</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        TP_t, FN_t = CM[1][1], CM[1][0]
        FP_t, TN_t = CM[0][1], CM[0][0]
        accuracy_t  = (TP_t + TN_t) / CM.sum()
        precision_t = TP_t / (TP_t + FP_t)
        recall_t    = TP_t / (TP_t + FN_t)
        f1_t        = 2 * precision_t * recall_t / (precision_t + recall_t)
        
        st.markdown(f"""
        <div class="about-card">
            <h3>📈 Hasil Evaluasi</h3>
            <ul>
                <li>Accuracy: <strong style="color:#3ea6ff">{accuracy_t*100:.2f}%</strong></li>
                <li>Precision: <strong style="color:#2ba640">{precision_t*100:.2f}%</strong></li>
                <li>Recall: <strong style="color:#f5a623">{recall_t*100:.2f}%</strong></li>
                <li>F1-Score: <strong style="color:#c47aff">{f1_t*100:.2f}%</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#ffffff; border:1px solid #e5e5e5; border-radius:12px; padding:24px; text-align:center; box-shadow:0 4px 15px rgba(0,0,0,0.02);">
        <span style="font-size:0.95rem; color:#606060;">
            Dibuat dengan ❤️ menggunakan Streamlit · Naive Bayes · TF-IDF
        </span>
    </div>
    """, unsafe_allow_html=True)