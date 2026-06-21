"""
app.py — Streamlit UI untuk Cyberbullying Detection
Menampilkan hasil prediksi dari Tier 1 (individual) hingga Tier 3 (Enhanced BERT).
"""

import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go

st.set_page_config(
    page_title="Cyberbullying Detector",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
    .prediction-card {
        margin-bottom: 0.2rem;
    }

    div[class*="st-key-mb-"][class*="-bully"] {
        background-color: #F5EFEB;
        border-left: 5px solid #2F4156;
        border-radius: 12px;
        padding: 1rem 1.2rem 0.6rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    div[class*="st-key-mb-"][class*="-safe"] {
        background-color: #FFFFFF;
        border-left: 5px solid #567C8D;
        border-radius: 12px;
        padding: 1rem 1.2rem 0.6rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    div[class*="st-key-mb-"][class*="-neutral"] {
        background-color: #C8D9E6;
        border-left: 5px solid #567C8D;
        border-radius: 12px;
        padding: 1rem 1.2rem 0.6rem 1.2rem;
        margin-bottom: 0.6rem;
    }

    .card-model-name {
        font-size: 0.75rem;
        font-weight: 700;
        color: #567C8D;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }
    .card-prediction {
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        line-height: 1.2;
    }
    .card-bully  .card-prediction { color: #2F4156; }
    .card-safe   .card-prediction { color: #567C8D; }
    .card-neutral .card-prediction { color: #2F4156; }

    .card-confidence {
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        background-color: rgba(47,65,86,0.1);
        color: #2F4156;
    }

    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2F4156;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #C8D9E6;
    }

    div[data-testid="stExpander"] summary p {
        font-size: 1.15rem;
        font-weight: 700;
        color: #2F4156;
    }

    div[data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #C8D9E6;
        border-radius: 10px;
    }

    div[data-testid="stExpander"] summary {
        display: flex;
        flex-direction: row-reverse;
        justify-content: space-between;
    }

    h1, h2, h3 { color: #2F4156; }

    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaption"] p,
    [data-testid="stCaptionContainer"] span,
    [data-testid="stCaption"] span {
        font-size: 1.2rem !important;
    }

    [data-testid="stTextArea"] textarea {
        font-size: 1.1rem !important;
    }

    [data-testid="stButton"] button p {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] { background-color: #C8D9E6; }
    [data-testid="stSidebar"] * { color: #2F4156; }
</style>
""", unsafe_allow_html=True)


def is_bullying(label: str):
    """Return True jika label menandakan bullying, False jika aman, None jika tidak diketahui."""
    label_lower = label.lower()
    if any(k in label_lower for k in ["bully", "hate", "toxic", "offensive", "harassment", "abuse", "cyberbullying"]):
        return True
    if any(k in label_lower for k in ["not", "safe", "normal", "benign", "neutral", "clean"]):
        return False
    return None


def card_css(label: str) -> str:
    b = is_bullying(label)
    return "card-bully" if b is True else ("card-safe" if b is False else "card-neutral")


def render_card(model_name: str, label: str, confidence):
    css = card_css(label)
    conf_text = f"{confidence:.1%}" if confidence else "—"
    st.markdown(f"""
    <div class="prediction-card {css}">
        <div class="card-model-name">{model_name}</div>
        <div class="card-prediction">{label}</div>
        <span class="card-confidence">Confidence: {conf_text}</span>
    </div>
    """, unsafe_allow_html=True)


def render_chart(probabilities: dict, height: int = 180, key=None):
    if not probabilities:
        return
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors = []
    for lbl in labels:
        b = is_bullying(lbl)
        colors.append("#2F4156" if b is True else ("#567C8D" if b is False else "#C8D9E6"))

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in values],
        textposition="auto",
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=0, r=6, t=6, b=0),
        height=height,
        xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(size=11),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)


def render_model_block(model_name: str, res: dict, chart_height: int = 170, block_id=None):
    if "error" in res:
        st.warning(res["error"])
        return
    css = card_css(res["label"])
    kind = css.split("-")[-1]  # bully | safe | neutral
    slug = "".join(c if c.isalnum() else "-" for c in str(block_id))
    with st.container(key=f"mb-{slug}-{kind}"):
        render_card(model_name, res["label"], res.get("confidence"))
        if res.get("probabilities"):
            render_chart(res["probabilities"], height=chart_height, key=f"chart-{slug}-{kind}")


def row_chunks(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


enable_bert = True

st.sidebar.title("ℹ️ Tier Overview")
st.sidebar.markdown("""
- **Tier 1** — Individual: LR, NB, SVM, RF, KNN
- **Tier 2A** — Weighted Soft Voting
- **Tier 2B** — Hard Voting (DT+RF+XGB)
- **Tier 2C** — Stacking (LR+NB+SVM → LR)
- **Tier 2D** — Stacking (DT+RF+XGB → RF)
- **Tier 3A** — BERT + Weighted Loss
- **Tier 3B** — Enhanced BERT (BERT + ML → Meta-LR)
""")


@st.cache_resource
def load_predictor(use_bert: bool):
    from predictor import CyberbullyingPredictor
    return CyberbullyingPredictor(enable_bert=use_bert)


with st.spinner("Loading models... (pertama kali bisa 30-60 detik)"):
    predictor = load_predictor(enable_bert)


st.title("Cyberbullying Detection")
st.caption("Multi-tier classification: masukkan teks dan lihat prediksi dari semua model.")

text_input = st.text_area(
    "Masukkan teks yang ingin dianalisis:",
    height=120,
    placeholder="Contoh: You're such a loser, nobody likes you...",
)

_, col_btn = st.columns([3, 1])
with col_btn:
    run = st.button("🔍 Analyze", type="primary", use_container_width=True)


if run and text_input.strip():
    start = time.time()
    with st.spinner("Running all tiers..."):
        results = predictor.predict_all(text_input.strip())
    elapsed = time.time() - start

    st.success(f"Selesai dalam {elapsed:.2f} detik")

    st.markdown("---")
    st.markdown('<div class="section-header">Ringkasan Semua Tier</div>', unsafe_allow_html=True)

    summary_rows = []

    for name, res in results["tier1"].items():
        summary_rows.append({
            "Tier": "Tier 1",
            "Model": name,
            "Prediction": res["label"],
            "Confidence": f"{res['confidence']:.1%}" if res.get("confidence") else "—",
        })

    for tier, model, res in [
        ("Tier 2A", "WSV (LR+NB+SVM+RF+KNN)",    results["tier2a_wsv"]),
        ("Tier 2B", "Hard Voting (DT+RF+XGB)",    results["tier2b_hard_voting"]),
        ("Tier 2C", "Stacking (LR+NB+SVM → LR)", results["tier2c_stacking"]),
        ("Tier 2D", "Stacking (DT+RF+XGB → RF)", results["tier2d_stacking"]),
    ]:
        summary_rows.append({
            "Tier": tier,
            "Model": model,
            "Prediction": res["label"],
            "Confidence": f"{res['confidence']:.1%}" if res.get("confidence") else "N/A",
        })

    if enable_bert and "tier3a_bert" in results:
        for tier, model, key in [
            ("Tier 3A", "BERT + Weighted Loss", "tier3a_bert"),
            ("Tier 3B", "Enhanced BERT",        "tier3b_enhanced"),
        ]:
            res = results[key]
            if "error" not in res:
                summary_rows.append({
                    "Tier": tier,
                    "Model": model,
                    "Prediction": res["label"],
                    "Confidence": f"{res['confidence']:.1%}" if res.get("confidence") else "—",
                })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


    st.markdown("---")
    with st.expander("Tier 1 — Individual Models (lihat detail)", expanded=False):
        tier1_items = list(results["tier1"].items())
        for chunk in row_chunks(tier1_items, 3):
            cols = st.columns(len(chunk))
            for col, (name, res) in zip(cols, chunk):
                with col:
                    render_model_block(name, res, chart_height=160, block_id=f"t1-{name}")

    with st.expander("Tier 2 — Ensemble Models (lihat detail)", expanded=False):
        t2_data = {
            "Tier 2A — WSV":                results["tier2a_wsv"],
            "Tier 2B — Hard Voting":        results["tier2b_hard_voting"],
            "Tier 2C — Stacking (→ LR)":    results["tier2c_stacking"],
            "Tier 2D — Stacking (→ RF)":    results["tier2d_stacking"],
        }
        for chunk in row_chunks(list(t2_data.items()), 2):
            cols2 = st.columns(len(chunk))
            for col, (name, res) in zip(cols2, chunk):
                with col:
                    render_model_block(name, res, chart_height=160, block_id=f"t2-{name}")

    if enable_bert and "tier3a_bert" in results:
        with st.expander("Tier 3 — BERT Models (lihat detail)", expanded=False):
            t3_data = {
                "Tier 3A — BERT + Weighted Loss": results["tier3a_bert"],
                "Tier 3B — Enhanced BERT":        results["tier3b_enhanced"],
            }
            cols3 = st.columns(2)
            for col, (name, res) in zip(cols3, t3_data.items()):
                with col:
                    render_model_block(name, res, chart_height=200, block_id=f"t3-{name}")

elif run and not text_input.strip():
    st.warning("Masukkan teks terlebih dahulu.")
