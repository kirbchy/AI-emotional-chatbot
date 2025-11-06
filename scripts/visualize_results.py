import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPORTS_DIR = 'reports'
PLOTS_DIR = 'reports/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- 1. Gráfico de Barras Comparativo de Métricas Principales ---
print("Generating model comparison plot...")
with open(os.path.join(REPORTS_DIR, 'metrics.json'), 'r') as f:
    metrics_data = json.load(f)

# Convertir a DataFrame para facilitar el ploteo
metrics_df = pd.DataFrame.from_dict(metrics_data, orient='index')
metrics_to_plot = metrics_df[['test_accuracy', 'test_f1_macro']].reset_index().rename(columns={'index': 'model'})

# Reorganizar el DataFrame para el ploteo con seaborn
df_plot = pd.melt(metrics_to_plot, id_vars='model', var_name='metric', value_name='score')

# Crear el gráfico
plt.figure(figsize=(10, 6))
barplot = sns.barplot(data=df_plot, x='model', y='score', hue='metric')
plt.title('Model Performance Comparison (Test Set)')
plt.ylabel('Score')
plt.xlabel('Model')
plt.ylim(0, 1)

# Añadir etiquetas de valor en las barras
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '.4f'),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center',
                     xytext=(0, 9),
                     textcoords='offset points')

comparison_plot_path = os.path.join(PLOTS_DIR, 'model_comparison.png')
plt.savefig(comparison_plot_path)
plt.close()
print(f"Saved comparison plot to {comparison_plot_path}")


# --- 2. Heatmaps de las Matrices de Confusión ---
print("\nGenerating confusion matrix heatmaps...")
confusion_files = [f for f in os.listdir(REPORTS_DIR) if f.startswith('confusion_') and f.endswith('.csv')]

for fname in confusion_files:
    model_name = fname.replace('confusion_', '').replace('.csv', '')
    file_path = os.path.join(REPORTS_DIR, fname)
    df_cm = pd.read_csv(file_path, index_col=0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix: {model_name.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    heatmap_path = os.path.join(PLOTS_DIR, f'confusion_matrix_{model_name}.png')
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved heatmap for {model_name} to {heatmap_path}")

print("\nAll plots saved in reports/plots/ directory.")