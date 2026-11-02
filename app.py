import html
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Next Word Prediction", page_icon="🧠", layout="centered")

# ------------------------------
# Styling
# ------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #0B0F2A;
        --card: rgba(255,255,255,0.06);
        --card-border: rgba(255,255,255,0.12);
        --text: #EAF0FF;
        --muted: #9AA7D0;
        --violet: #8B5CF6;
        --cyan: #22D3EE;
        --pink: #F472B6;
    }

    html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; color: var(--text); }

    /* Aurora background */
    .stApp {
        background:
            radial-gradient(900px 500px at 12% -5%, rgba(139,92,246,.35), transparent 60%),
            radial-gradient(800px 500px at 95% 5%, rgba(34,211,238,.22), transparent 60%),
            radial-gradient(700px 500px at 50% 110%, rgba(244,114,182,.18), transparent 60%),
            var(--bg);
    }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 780px; padding-top: 2.5rem; padding-bottom: 4rem; }
    section[data-testid="stSidebar"] { background: rgba(11,15,42,.92); border-right: 1px solid var(--card-border); }

    h1, h2, h3, label, p, span, div { color: var(--text); }
    [data-testid="stCaptionContainer"], .stCaption, small { color: var(--muted) !important; }

    /* Hero */
    .hero { text-align: center; margin-bottom: 1.8rem; }
    .badge { display: inline-block; padding: .3rem .9rem; border-radius: 999px;
             border: 1px solid var(--card-border); background: var(--card);
             color: var(--cyan); font-size: .82rem; font-weight: 500; margin-bottom: 1rem; }
    .hero-title { font-family: 'Sora', sans-serif; font-weight: 800; font-size: 3rem;
                  line-height: 1.1; margin: 0 0 .6rem 0;
                  background: linear-gradient(90deg, #C4B5FD 0%, #67E8F9 55%, #F9A8D4 100%);
                  -webkit-background-clip: text; background-clip: text;
                  -webkit-text-fill-color: transparent; color: transparent; }
    .hero-sub { color: var(--muted); font-size: 1.05rem; max-width: 520px; margin: 0 auto; }

    /* Text input */
    div[data-baseweb="input"] { background: var(--card) !important; border: 1px solid var(--card-border) !important;
                                border-radius: 14px !important; }
    div[data-baseweb="input"]:focus-within { border-color: var(--cyan) !important;
                                             box-shadow: 0 0 0 3px rgba(34,211,238,.25) !important; }
    div[data-baseweb="input"] input { color: var(--text) !important; font-size: 1.05rem; padding: .85rem 1rem; }
    div[data-baseweb="input"] input::placeholder { color: #6C79A8; }

    /* Glass cards */
    .glass { background: var(--card); border: 1px solid var(--card-border); border-radius: 20px;
             padding: 1.4rem 1.6rem; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); }
    .preview-label { color: var(--muted); font-size: .85rem; margin-bottom: .5rem; }
    .preview-text { font-family: 'Sora', sans-serif; font-size: 1.6rem; font-weight: 600; line-height: 1.6; }
    .gen { background: linear-gradient(90deg, var(--violet), var(--cyan)); color: #fff;
           padding: .1rem .7rem; border-radius: 10px; margin-left: .2rem;
           box-shadow: 0 6px 24px rgba(139,92,246,.45); }
    .cursor { display: inline-block; width: 3px; height: 1.3em; background: var(--cyan);
              vertical-align: text-bottom; margin-left: 4px; animation: blink 1s steps(1) infinite; }
    @keyframes blink { 50% { opacity: 0; } }
    @media (prefers-reduced-motion: reduce) { .cursor { animation: none; } }
    .empty { color: #6C79A8; font-weight: 400; font-size: 1.1rem; }

    /* Stat row */
    .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0 .5rem 0; }
    .stat-title { color: var(--muted); font-size: .85rem; margin-bottom: .3rem; }
    .stat-word { font-family: 'Sora', sans-serif; font-size: 2rem; font-weight: 700; }
    .conf { display: flex; align-items: center; gap: 1rem; }
    .ring { --p: 0; width: 76px; height: 76px; border-radius: 50%; flex: none;
            background: conic-gradient(var(--cyan) calc(var(--p) * 1%), rgba(255,255,255,.1) 0);
            display: grid; place-items: center; }
    .ring::before { content: attr(data-label); display: grid; place-items: center;
                    width: 58px; height: 58px; border-radius: 50%; background: #121842;
                    font-family: 'Sora', sans-serif; font-weight: 700; font-size: .95rem; }
    .conf-note { color: var(--muted); font-size: .9rem; }

    /* Probability bars */
    .section-title { font-family: 'Sora', sans-serif; font-size: 1.15rem; font-weight: 700; margin: 1.6rem 0 .3rem 0; }
    .bar-row { display: grid; grid-template-columns: 90px 1fr 52px; align-items: center; gap: .8rem; margin: .55rem 0; }
    .bar-word { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .bar-track { height: 12px; border-radius: 999px; background: rgba(255,255,255,.08); overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--violet), var(--cyan)); }
    .bar-row.top .bar-fill { background: linear-gradient(90deg, var(--pink), var(--violet), var(--cyan));
                             box-shadow: 0 0 14px rgba(34,211,238,.55); }
    .bar-pct { color: var(--muted); font-size: .9rem; text-align: right; }

    /* Buttons */
    .stButton > button { border-radius: 12px; font-weight: 600; color: var(--text);
                         background: var(--card); border: 1px solid var(--card-border); transition: all .15s ease; }
    .stButton > button:hover { border-color: var(--cyan); color: #fff; background: rgba(34,211,238,.12); }
    .stButton > button:focus-visible { outline: 3px solid var(--cyan); outline-offset: 2px; }
    .stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(90deg, var(--violet), var(--cyan)); border: none; color: #fff;
        padding: .65rem 1.2rem; box-shadow: 0 8px 28px rgba(139,92,246,.4); }

    /* Result + expander */
    div[data-testid="stAlert"] { background: var(--card); border: 1px solid var(--card-border); border-radius: 14px; }
    details { background: var(--card); border: 1px solid var(--card-border) !important; border-radius: 14px; }
    hr { border-color: var(--card-border); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_resources():
    model = load_model("lstm_model (1).h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    index_word = {i: w for w, i in tokenizer.word_index.items()}
    return model, tokenizer, max_len, index_word


try:
    model, tokenizer, max_len, index_word = load_resources()
except Exception as e:
    st.error(
        "Couldn't load the model files. Check that `lstm_model (1).h5`, "
        "`tokenizer.pkl` and `max_len.pkl` are in the same folder as this app."
    )
    st.exception(e)
    st.stop()

# ------------------------------
# Prediction helpers
# ------------------------------
def predict_top_k(text, k=5):
    seq = tokenizer.texts_to_sequences([text])[0]
    if not seq:
        return []
    seq = pad_sequences([seq], maxlen=max_len - 1, padding="pre")
    probs = model.predict(seq, verbose=0)[0]
    top = np.argsort(probs)[::-1][: k + 1]
    results = [(index_word[i], float(probs[i])) for i in top if i in index_word]
    return results[:k]


def generate(text, n_words):
    out, current = [], text
    for _ in range(n_words):
        preds = predict_top_k(current, k=1)
        if not preds:
            break
        out.append(preds[0][0])
        current = f"{current} {preds[0][0]}"
    return out


# ------------------------------
# Session state
# ------------------------------
st.session_state.setdefault("text", "")
st.session_state.setdefault("history", [])

EXAMPLES = ["to be or not to", "the quick brown", "once upon a", "i think that"]


def use_example(example):
    st.session_state.text = example


def add_word(word):
    st.session_state.text = f"{st.session_state.text.strip()} {word}"


# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Suggestions to show", 1, 10, 5)
    n_words = st.slider("Words to generate", 1, 10, 3)
    st.caption(f"The model reads up to the last {max_len - 1} words.")
    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []

# ------------------------------
# Hero
# ------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="badge">🧠 LSTM neural network</div>
        <div class="hero-title">Next word prediction</div>
        <div class="hero-sub">Type the start of a sentence and watch the model guess what comes next.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.text_input(
    "Your text",
    key="text",
    placeholder="Start typing, for example: to be or not to",
    label_visibility="collapsed",
)

cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    col.button(ex, key=f"ex_{ex}", on_click=use_example, args=(ex,), use_container_width=True)

text = st.session_state.text.strip()
st.write("")

# ------------------------------
# Results
# ------------------------------
if not text:
    st.markdown(
        '<div class="glass"><div class="preview-label">Live preview</div>'
        '<div class="preview-text"><span class="empty">Your sentence will appear here</span>'
        '<span class="cursor"></span></div></div>',
        unsafe_allow_html=True,
    )
else:
    preds = predict_top_k(text, top_k)

    if not preds:
        st.markdown(
            f'<div class="glass"><div class="preview-label">Live preview</div>'
            f'<div class="preview-text">{html.escape(text)}</div></div>',
            unsafe_allow_html=True,
        )
        st.warning("None of these words are in the model's vocabulary. Try different or more common words.")
    else:
        best_word, best_prob = preds[0]
        pct = round(best_prob * 100)

        st.markdown(
            f'<div class="glass"><div class="preview-label">Live preview</div>'
            f'<div class="preview-text">{html.escape(text)} '
            f'<span class="gen">{html.escape(best_word)}</span></div></div>',
            unsafe_allow_html=True,
        )

        note = "Very sure" if pct >= 60 else "Fairly sure" if pct >= 30 else "Just a guess"
        st.markdown(
            f"""
            <div class="stats">
                <div class="glass">
                    <div class="stat-title">Most likely next word</div>
                    <div class="stat-word">{html.escape(best_word)}</div>
                </div>
                <div class="glass conf">
                    <div class="ring" style="--p:{pct}" data-label="{pct}%"></div>
                    <div>
                        <div class="stat-title">Confidence</div>
                        <div class="conf-note">{note}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Clickable suggestions
        st.markdown('<div class="section-title">Pick a word</div>', unsafe_allow_html=True)
        st.caption("Select a suggestion to add it to your sentence.")
        word_cols = st.columns(min(len(preds), 5))
        for i, (w, p) in enumerate(preds):
            word_cols[i % len(word_cols)].button(
                w, key=f"pick_{i}_{w}", on_click=add_word, args=(w,), use_container_width=True
            )

        # Probability bars
        st.markdown('<div class="section-title">How likely is each word?</div>', unsafe_allow_html=True)
        rows = ""
        for i, (w, p) in enumerate(preds):
            width = max(p / best_prob * 100, 2)
            cls = "bar-row top" if i == 0 else "bar-row"
            rows += (
                f'<div class="{cls}"><div class="bar-word">{html.escape(w)}</div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>'
                f'<div class="bar-pct">{p:.0%}</div></div>'
            )
        st.markdown(f'<div class="glass">{rows}</div>', unsafe_allow_html=True)

        # Generate several words
        st.markdown('<div class="section-title">Continue the sentence</div>', unsafe_allow_html=True)
        if st.button(f"Generate next {n_words} words", type="primary"):
            with st.spinner("Generating…"):
                words = generate(text, n_words)
            if words:
                full = f"{text} {' '.join(words)}"
                st.success(full)
                st.session_state.history.insert(0, {"input": text, "output": full})
            else:
                st.warning("Couldn't generate a continuation for this text.")

# ------------------------------
# History
# ------------------------------
if st.session_state.history:
    st.write("")
    with st.expander(f"History ({len(st.session_state.history)})"):
        for item in st.session_state.history[:10]:
            st.markdown(f"**{item['input']}** → {item['output']}")

st.markdown("---")
st.caption("LSTM-based next word prediction, built with Streamlit and TensorFlow.")