import os
import time
import streamlit as st
import joblib
import yaml
import numpy as np
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = 'models'
FAQS_PATH = 'app/faqs/faqs_en.yaml'

st.set_page_config(page_title="Sentiment Chatbot", layout="centered")
st.title("Customer Support Chatbot (EN)")

# Sidebar: select model, show probabilities checkbox
def load_models():
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_pipeline.joblib')]
    models = {f.replace('_pipeline.joblib',''): joblib.load(os.path.join(MODELS_DIR, f)) for f in files}
    return models

@st.cache_resource
def load_faqs():
    if os.path.exists(FAQS_PATH):
        with open(FAQS_PATH, 'r') as f:
            return yaml.safe_load(f)
    return {}

@st.cache_resource
def build_faq_index(faqs: dict):
    questions = list(faqs.keys()) if faqs else []
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform([q.lower() for q in questions]) if questions else None
    return vec, X, questions

MODELS = load_models()
FAQS = load_faqs()
VEC, FAQ_X, FAQ_QUESTIONS = build_faq_index(FAQS)

if 'latencies_ms' not in st.session_state:
    st.session_state.latencies_ms = []

model_name = st.sidebar.selectbox('Pick model:', list(MODELS.keys()))
show_prob = st.sidebar.checkbox('Show prediction details', value=True)
escalation_threshold = st.sidebar.slider('Escalation threshold (confidence/margin)', 0.1, 2.5, 0.7, 0.1)
faq_threshold = st.sidebar.slider('FAQ similarity threshold', 0.1, 0.9, 0.45, 0.05)
if st.session_state.latencies_ms:
    avg_ms = sum(st.session_state.latencies_ms) / len(st.session_state.latencies_ms)
    st.sidebar.write(f"Avg latency: {avg_ms:.1f} ms over {len(st.session_state.latencies_ms)} msgs")
model = MODELS[model_name]

example = st.chat_input('Write here (e.g. "Why is my flight delayed?")')
if example:
    user_msg = example.strip()
    t0 = time.perf_counter()
    pred = model.predict([user_msg])[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    st.session_state.latencies_ms.append(elapsed_ms)

    confidence = None
    margin = None
    if hasattr(model.named_steps['clf'], 'predict_proba'):
        proba = model.named_steps['clf'].predict_proba(model.named_steps['tfidf'].transform([user_msg]))[0]
        confidence = float(np.max(proba))
        if show_prob:
            st.write(f"Probabilities: {dict(zip(model.classes_, np.round(proba,2)))}")
    elif hasattr(model.named_steps['clf'], 'decision_function'):
        df = model.named_steps['clf'].decision_function(model.named_steps['tfidf'].transform([user_msg]))
        if df.ndim == 1:
            margin = float(np.max(df))
        else:
            margin = float(np.max(df))

    st.markdown(f"**Model prediction:** <span style='color:orange;font-weight:bold'>{pred.upper()}</span>", unsafe_allow_html=True)

    # Tone-mapping
    if pred == 'negative':
        preamble = "We're sorry about your experience. We understand your frustration."
    elif pred == 'neutral':
        preamble = "Thank you for your question."
    else:
        preamble = "We're glad to help!"

    # FAQ retrieval via TF-IDF cosine similarity
    ans = None
    if FAQS and FAQ_X is not None:
        q_vec = VEC.transform([user_msg.lower()])
        sims = cosine_similarity(q_vec, FAQ_X)[0]
        idx = int(np.argmax(sims)) if sims.size > 0 else -1
        if sims.size > 0 and sims[idx] >= faq_threshold:
            q_match = FAQ_QUESTIONS[idx]
            ans = FAQS[q_match]

    if ans:
        st.markdown(f"{preamble} {ans}")
    else:
        should_escalate = False
        if pred == 'negative':
            if confidence is not None and confidence >= escalation_threshold:
                should_escalate = True
            if margin is not None and margin >= escalation_threshold:
                should_escalate = True
        if should_escalate:
            st.markdown(f"{preamble} Would you like to be connected with a human agent?")
        else:
            st.markdown(f"{preamble} Please let us know how we can assist further.")
