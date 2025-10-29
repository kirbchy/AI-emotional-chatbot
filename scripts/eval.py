import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

REPORTS_DIR = 'reports'
MODELS_DIR = 'models'

# Load test split
test_path = os.path.join(REPORTS_DIR, 'test_split.csv')
df_test = pd.read_csv(test_path)

model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('_pipeline.joblib')]
print("Available models:", model_files)

for fname in model_files:
    print(f"\n-----\nEvaluating: {fname}")
    pipe = joblib.load(os.path.join(MODELS_DIR, fname))
    y_true = df_test['airline_sentiment']
    y_pred = pipe.predict(df_test['text'])
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=pipe.classes_)
    df_cm = pd.DataFrame(cm, index=pipe.classes_, columns=pipe.classes_)
    print("Confusion matrix:")
    print(df_cm)
    try:
        plt.figure(figsize=(6,4))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.title(fname)
        plt.show()
    except Exception:
        print("Matplotlib or seaborn may not be available.")
