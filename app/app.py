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
        (c4, "🎯", "91.13%", "Akurasi Model", "#3ea6ff"),
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

    def make_wordcloud(text, colormap, bg="#ffffff"):
        wc = WordCloud(
            width=1000,
            height=420,
            background_color=bg,
            colormap=colormap,
            max_words=120,
            prefer_horizontal=0.85,
            collocations=False,
            margin=8
        ).generate(text)
        return wc

    with tab1:
        positive_text = " ".join(df[df['sentiment'] == 1]['content'].dropna())
        wc = make_wordcloud(positive_text, "Greens")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig.patch.set_facecolor("#ffffff")
        fig.tight_layout(pad=0)
        st.pyplot(fig)

        # Top words
        words = positive_text.lower().split()
        top_words = Counter(words).most_common(10)
        st.markdown('<div class="section-title">Top 10 Kata — Positif</div>', unsafe_allow_html=True)
        tw_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.barh(tw_df["Kata"][::-1], tw_df["Frekuensi"][::-1], color="#2ba640", edgecolor="none")
        ax2.set_xlabel("Frekuensi", fontsize=10)
        ax2.xaxis.grid(True)
        ax2.set_axisbelow(True)
        ax2.spines[['top','right','bottom']].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)

    with tab2:
        negative_text = " ".join(df[df['sentiment'] == 0]['content'].dropna())
        wc = make_wordcloud(negative_text, "Reds")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig.patch.set_facecolor("#ffffff")
        fig.tight_layout(pad=0)
        st.pyplot(fig)

        words = negative_text.lower().split()
        top_words = Counter(words).most_common(10)
        st.markdown('<div class="section-title">Top 10 Kata — Negatif</div>', unsafe_allow_html=True)
        tw_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.barh(tw_df["Kata"][::-1], tw_df["Frekuensi"][::-1], color="#ff0000", edgecolor="none")
        ax2.set_xlabel("Frekuensi", fontsize=10)
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
        st.markdown("""
        <div class="insight-box" style="border-left-color:#2ba640">
            <strong>True Positive (TP)</strong> — 4,091<br>
            Komentar positif yang diprediksi benar sebagai positif.
        </div>
        <div class="insight-box" style="border-left-color:#0f0f0f">
            <strong>True Negative (TN)</strong> — 1,377<br>
            Komentar negatif yang diprediksi benar sebagai negatif.
        </div>
        <div class="insight-box" style="border-left-color:#ff0000">
            <strong>False Positive (FP)</strong> — 208<br>
            Komentar negatif yang keliru diprediksi sebagai positif.
        </div>
        <div class="insight-box" style="border-left-color:#ff0000">
            <strong>False Negative (FN)</strong> — 324<br>
            Komentar positif yang keliru diprediksi sebagai negatif.
        </div>
        """, unsafe_allow_html=True)

    # Classification report per class
    st.markdown('<div class="section-title">Laporan Per Kelas</div>', unsafe_allow_html=True)

    report_data = {
        "Kelas":     ["Negatif (0)", "Positif (1)"],
        "Precision": [f"{TN/(TN+FN)*100:.1f}%", f"{precision*100:.1f}%"],
        "Recall":    [f"{TN/(TN+FP)*100:.1f}%",  f"{recall*100:.1f}%"],
        "F1-Score":  ["—", f"{f1*100:.1f}%"],
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

        predict_btn = st.button("🔍  Analisis Sentimen", use_container_width=True)

    with col_result:
        if review.strip():
            vector     = tfidf.transform([review])
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
        search_term = st.text_input("🔍 Cari kata dalam komentar", placeholder="Ketik kata kunci…")
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
                <li>Total: <strong style="color:#0f0f0f">30.000 komentar</strong></li>
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
        st.markdown("""
        <div class="about-card">
            <h3>📈 Hasil Evaluasi</h3>
            <ul>
                <li>Accuracy: <strong style="color:#3ea6ff">91.13%</strong></li>
                <li>Precision: <strong style="color:#2ba640">95.2%</strong></li>
                <li>Recall: <strong style="color:#f5a623">92.7%</strong></li>
                <li>F1-Score: <strong style="color:#c47aff">93.9%</strong></li>
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