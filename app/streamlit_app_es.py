import os
import time
import streamlit as st
import joblib
import yaml
import numpy as np
from deep_translator import GoogleTranslator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = 'models'
FAQS_PATH = 'app/faqs/faqs_es.yaml'

st.set_page_config(page_title="Chatbot de Soporte (ES)", layout="centered")
st.title("Chatbot de Soporte al Cliente (ES)")

# Carga de modelos entrenados (EN)
def load_models():
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_pipeline.joblib')]
    models = {f.replace('_pipeline.joblib',''): joblib.load(os.path.join(MODELS_DIR, f)) for f in files}
    return models

@st.cache_resource
def load_faqs():
    if os.path.exists(FAQS_PATH):
        with open(FAQS_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

@st.cache_resource
def build_faq_index(faqs: dict):
    preguntas = list(faqs.keys()) if faqs else []
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform([q.lower() for q in preguntas]) if preguntas else None
    return vec, X, preguntas

MODELS = load_models()
FAQS = load_faqs()
VEC, FAQ_X, FAQ_PREGUNTAS = build_faq_index(FAQS)

if 'latencies_ms' not in st.session_state:
    st.session_state.latencies_ms = []

model_name = st.sidebar.selectbox('Modelo:', list(MODELS.keys()))
mostrar_detalle = st.sidebar.checkbox('Mostrar detalles de predicción', value=False)
umbral_escalamiento = st.sidebar.slider('Umbral de escalamiento (confianza/margen)', 0.1, 2.5, 0.7, 0.1)
umbral_faq = st.sidebar.slider('Umbral de similitud FAQ', 0.1, 0.9, 0.45, 0.05)
if st.session_state.latencies_ms:
    avg_ms = sum(st.session_state.latencies_ms) / len(st.session_state.latencies_ms)
    st.sidebar.write(f"Latencia promedio: {avg_ms:.1f} ms en {len(st.session_state.latencies_ms)} mensajes")
model = MODELS[model_name]

translator_es_en = GoogleTranslator(source='auto', target='en')
translator_en_es = GoogleTranslator(source='auto', target='es')

mensaje = st.chat_input('Escribe aquí (ej. "Mi pedido no ha llegado")')
if mensaje:
    msg_es = mensaje.strip()
    # Traducir al inglés para clasificar
    t0 = time.perf_counter()
    msg_en = translator_es_en.translate(msg_es)
    pred = model.predict([msg_en])[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    st.session_state.latencies_ms.append(elapsed_ms)

    confianza = None
    margen = None
    if hasattr(model.named_steps['clf'], 'predict_proba'):
        proba = model.named_steps['clf'].predict_proba(model.named_steps['tfidf'].transform([msg_en]))[0]
        confianza = float(np.max(proba))
        if mostrar_detalle:
            st.write(f"Probabilidades EN: {dict(zip(model.classes_, np.round(proba,2)))}")
    elif hasattr(model.named_steps['clf'], 'decision_function'):
        df = model.named_steps['clf'].decision_function(model.named_steps['tfidf'].transform([msg_en]))
        if df.ndim == 1:
            margen = float(np.max(df))
        else:
            margen = float(np.max(df))

    st.markdown(f"**Predicción (EN):** {pred.upper()}")

    # Plantillas de tono ES
    if pred == 'negative':
        preambulo = "Lamentamos lo sucedido. Entendemos tu frustración."
    elif pred == 'neutral':
        preambulo = "Gracias por tu mensaje."
    else:
        preambulo = "Con gusto te ayudamos."

    # FAQs con similitud TF-IDF
    respuesta = None
    if FAQS and FAQ_X is not None:
        q_vec = VEC.transform([msg_es.lower()])
        sims = cosine_similarity(q_vec, FAQ_X)[0]
        idx = int(np.argmax(sims)) if sims.size > 0 else -1
        if sims.size > 0 and sims[idx] >= umbral_faq:
            q_match = FAQ_PREGUNTAS[idx]
            respuesta = FAQS[q_match]

    if respuesta:
        st.markdown(f"{preambulo} {respuesta}")
    else:
        escalar = False
        if pred == 'negative':
            if confianza is not None and confianza >= umbral_escalamiento:
                escalar = True
            if margen is not None and margen >= umbral_escalamiento:
                escalar = True
        if escalar:
            st.markdown(f"{preambulo} ¿Deseas que escalemos tu caso con un agente humano?")
        else:
            st.markdown(f"{preambulo} Cuéntanos cómo podemos ayudarte con más detalle.")
