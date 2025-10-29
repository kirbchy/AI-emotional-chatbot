import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import time

DATA_PATH = 'data/Tweets.csv'
MODELS_DIR = 'models'
REPORTS_DIR = 'reports'
SEED = 42
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ------------- Data Loading and Basic Cleaning
print("Loading data...")
df = pd.read_csv(DATA_PATH)
# Only keep relevant columns, drop NA
cols_needed = ['airline_sentiment', 'text']
df = df[cols_needed].dropna()

# Simple clean: lowercase, remove URLs/mentions/hashtags
import re
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text'] = df['text'].apply(clean_text)

# ------------- Stratified Train/Test Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(sss.split(df['text'], df['airline_sentiment']))
df_train = df.iloc[train_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)
df_train.to_csv(os.path.join(REPORTS_DIR, 'train_split.csv'), index=False)
df_test.to_csv(os.path.join(REPORTS_DIR, 'test_split.csv'), index=False)

# ------------- Feature extraction: TF-IDF
print("Fitting TF-IDF vectorizer (Vocab from train only)...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True)
X_train = tfidf.fit_transform(df_train['text'])
X_test = tfidf.transform(df_test['text'])
y_train = df_train['airline_sentiment'].values
y_test = df_test['airline_sentiment'].values
joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))

# ------------- Define models to train
pipelines = {
    'logreg': Pipeline([
        ('tfidf', tfidf),
        ('clf', LogisticRegression(C=1.0, max_iter=300, class_weight='balanced', random_state=SEED))
    ]),
    'svm': Pipeline([
        ('tfidf', tfidf),
        ('clf', LinearSVC(C=1.0, class_weight='balanced', random_state=SEED))
    ]),
    'nb': Pipeline([
        ('tfidf', tfidf),
        ('clf', MultinomialNB(alpha=1.0))
    ])
}

results = {}
for name, pipe in pipelines.items():
    print(f"Training model: {name}")
    start = time.time()
    # Cross-validate
    scores = cross_validate(pipe, df_train['text'], y_train, cv=5, scoring=['f1_macro', 'accuracy'], n_jobs=-1, return_train_score=True)
    pipe.fit(df_train['text'], y_train)
    fit_time = time.time() - start
    y_pred = pipe.predict(df_test['text'])
    report = classification_report(y_test, y_pred, output_dict=True)
    conf = confusion_matrix(y_test, y_pred)
    # Save model and reports
    joblib.dump(pipe, os.path.join(MODELS_DIR, f'{name}_pipeline.joblib'))
    results[name] = {
        'cv_f1_macro_mean': np.mean(scores['test_f1_macro']),
        'cv_acc_mean': np.mean(scores['test_accuracy']),
        'fit_time_sec': fit_time,
        'test_f1_macro': f1_score(y_test, y_pred, average='macro'),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'classification_report': report,
        'confusion_matrix': conf.tolist()
    }
    print(f"{name} | Train F1_macro: {np.mean(scores['test_f1_macro']):.3f} | Test F1_macro: {results[name]['test_f1_macro']:.3f}")
    # Write confusion to CSV
    pd.DataFrame(conf, columns=pipe.classes_, index=pipe.classes_).to_csv(os.path.join(REPORTS_DIR, f'confusion_{name}.csv'))
    # Detailed per-model report
    pd.DataFrame(report).to_csv(os.path.join(REPORTS_DIR, f'classification_{name}.csv'))

# Save all metrics
import json
with open(os.path.join(REPORTS_DIR, 'metrics.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("All models trained and results saved (see models/ and reports/).")
