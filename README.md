# Sentiment-Aware Customer Support Chatbot

A project implementing a customer-support chatbot with classic machine learning and emotion detection, using the Kaggle airline sentiment dataset.

## Why this project?
Traditional chatbots answer correctly but ignore user emotions. This project builds and evaluates a chatbot that detects sentiment in user messages (positive, neutral, negative) and adapts its reply: empathetic tone for negatives  and optional escalation to a human agent.

## Objectives 
- Detect emotions in customer messages: {positive, neutral, negative}.
- Adapt responses by emotion, with empathetic tone for negatives.
- Recommend escalation when frustration is high (threshold-based).
- Provide a real-time web UI (Streamlit), simple and evaluable.
- Compare three classic ML models on the same dataset: Logistic Regression, Linear SVM, Multinomial Naive Bayes.

## System overview
- Dataset: Twitter US Airline Sentiment (~14.6k tweets).
- Preprocessing: lowercase, remove URLs, mentions, hashtags; keep negation; split 80/20 stratified.
- Features: TF–IDF (1–2 grams), min_df=3, max_df=0.95, sublinear_tf=True.
- Models: LR (class_weight='balanced'), LinearSVC (class_weight='balanced'), MultinomialNB.
- Training & Eval: 5-fold stratified CV (macro F1 primary), hold-out test; artifacts saved with joblib.
- UI: Streamlit EN/ES; FAQs retrieval via TF–IDF cosine similarity; escalation by threshold on probability or margin; sidebar shows running average latency.

## Getting Started

1. Create and activate a virtual environment
   - Windows (PowerShell):
     ```
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - macOS/Linux (bash/zsh):
     ```
     python3 -m venv .venv
     source .venv/bin/activate
     ```
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Train models (English dataset):
   ```
   python scripts/train_all.py
   ```
4. Launch UI (English):
   ```
   streamlit run app/streamlit_app.py
   ```
5. Launch UI (Spanish):
   ```
   streamlit run app/streamlit_app_es.py
   ```

## What the apps do (EN/ES)
- Sentiment classification of each user message using the selected model.
- Response policy:
  - negative: empathetic preamble; if confidence/margin ≥ threshold ⇒ suggest escalation to human.
  - neutral/positive: informative preamble.
- FAQs: 10–15 questions answered by TF–IDF similarity to user message (configurable similarity threshold).
- Latency: shows session average latency in the sidebar.
- Spanish app translates ES→EN for classification and replies in Spanish (with Spanish FAQs).

## Defaults used in demos/reports
- Escalation threshold (confidence/margin): 0.7
- FAQ similarity threshold (cosine on TF‑IDF): 0.45
You can adjust both in the sidebar.

## Dataset and preprocessing
- Source: Kaggle Twitter US Airline Sentiment (labels: negative, neutral, positive).
- Class distribution is imbalanced (negative ≈ 62–63%). We emphasize macro F1.
- Leakage prevention: split train/test before vectorization; TF–IDF fit only on train; drop label-related columns.

## Models, training and artifacts
- Pipelines: `tfidf` + classifier; saved to `models/*_pipeline.joblib` (loadable with `joblib.load`).
- Scripts:
  - `scripts/train_all.py`: trains LR/SVM/NB with 5-fold CV, evaluates on test, saves reports.
  - `scripts/eval.py`: loads pipelines and prints reports/plots on test split.
- Reports: `reports/metrics.json`, `reports/classification_*.csv`, `reports/confusion_*.csv`.

## Results summary
From `reports/metrics.json` of a representative run:
- Logistic Regression: Accuracy 0.7845, Macro F1 0.7368
- Linear SVM: Accuracy 0.7941, Macro F1 0.7360
- Multinomial NB: Accuracy 0.7186, Macro F1 0.5426
Interpretation: Linear SVM attains the best accuracy; Logistic Regression offers a strong balance (fast, competitive). NB is fastest but less balanced across classes.

## EDA
Open `notebooks/eda.ipynb`:
- Class distribution and imbalance.
- Message length and token counts by class.
- Cleaning preview (URLs, mentions, hashtags removal) and justification for keeping negations.
- Rationale for TF–IDF n‑grams and macro F1 as the primary metric.

## Reproducibility and quality
- Seeds fixed (`random_state=42`).
- Train/test split before any text fitting.
- Class weights used for LR/SVM due to imbalance.
- All artifacts regenerable from scripts; only essential data (`data/Tweets.csv`) is versioned.

## Limitations and future work 
- English-only training data; Spanish UI relies on translation.
- Three coarse sentiment classes only (no fine-grained emotions, no sarcasm handling).
- FAQ coverage limited to 10–15 items; simple TF–IDF retrieval.
- Future: expand languages and data, add more emotion categories, sarcasm heuristics, larger FAQ base, optional context memory.

## References (selected)
- Liu (2012), Pang & Lee (2008), Go et al. (2009), Ravi & Ravi (2015).
- scikit-learn (Pedregosa et al., 2011) for all classic ML components.

## Project information
- Authors: Paula Llanos, Samuel Rivero, Sara López — Universidad EAFIT
- Dataset: Kaggle Twitter US Airline Sentiment
