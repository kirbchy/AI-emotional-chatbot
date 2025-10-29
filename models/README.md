# Model Artifacts

This directory contains saved models and utilities.

- *_pipeline.joblib: includes both the text vectorizer and classifier for each algorithm (e.g., logreg_pipeline.joblib, svm_pipeline.joblib, nb_pipeline.joblib)
- tfidf_vectorizer.joblib: fitted TfidfVectorizer (duplicate of what's in pipeline, separated for analytic use)

All files can be loaded with `joblib.load` in Python. Each pipeline supports `.predict()` directly on text.

Metrics and configs are documented in `../reports/metrics.json`

## Model cards (training configuration and summary metrics)

Common preprocessing:
- Vectorizer: TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.95, sublinear_tf=True)
- Split: stratified 80/20, random_state=42
- CV: Stratified 5-fold, primary metric F1 macro

### Logistic Regression (logreg_pipeline.joblib)
- Params: C=1.0, max_iter=300, class_weight='balanced', solver=liblinear or lbfgs (auto-selected by scikit-learn)
- CV mean F1 macro: 0.7381
- CV mean accuracy: 0.7856
- Test F1 macro: 0.7368
- Test accuracy: 0.7845
- Fit time (s): 13.33

### Linear SVM (svm_pipeline.joblib)
- Params: LinearSVC(C=1.0), class_weight='balanced', random_state=42
- CV mean F1 macro: 0.7319
- CV mean accuracy: 0.7914
- Test F1 macro: 0.7360
- Test accuracy: 0.7941
- Fit time (s): 8.00

### Multinomial Naive Bayes (nb_pipeline.joblib)
- Params: alpha=1.0
- CV mean F1 macro: 0.5261
- CV mean accuracy: 0.7110
- Test F1 macro: 0.5426
- Test accuracy: 0.7186
- Fit time (s): 2.99

Notes:
- Due to class imbalance, macro F1 is emphasized. Accuracy is reported as secondary.
- Confusion matrices per model are available in `../reports/confusion_*.csv`.
