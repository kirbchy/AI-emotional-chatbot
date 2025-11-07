# Chatbot de Atención al Cliente con Sensibilidad al Sentimiento

Un proyecto que implementa un chatbot de atención al cliente con aprendizaje automático clásico y detección de emociones, utilizando el conjunto de datos de sentimiento de aerolíneas de Kaggle.

## ¿Por qué este proyecto?

Los chatbots tradicionales responden correctamente, pero ignoran las emociones del usuario. Este proyecto crea y evalúa un chatbot que detecta el sentimiento en los mensajes de los usuarios (positivo, neutro, negativo) y adapta su respuesta: tono empático para los negativos y, opcionalmente, derivación a un agente humano.

## Objetivos
- Detectar las emociones en los mensajes de los clientes: {positivas, neutras, negativas}.

- Adaptar las respuestas según la emoción, con tono empático para los negativos.

- Recomendar la derivación cuando la frustración sea alta (basado en umbrales).

- Proporcionar una interfaz web en tiempo real (Streamlit), sencilla y evaluable.

- Comparar tres modelos clásicos de aprendizaje automático en el mismo conjunto de datos: Regresión Logística, SVM Lineal y Naive Bayes Multinomial.

## Descripción general del sistema
- Conjunto de datos: Análisis de sentimiento de aerolíneas estadounidenses en Twitter (~14.600 tuits).

- Preprocesamiento: conversión a minúsculas, eliminación de URL, menciones y hashtags; conservar negaciones; segmentación estratificada 80/20.

- Características: TF-IDF (1-2 gramos), min_df=3, max_df=0.95, sublinear_tf=True.

- Modelos: LR (class_weight='balanced'), LinearSVC (class_weight='balanced'), MultinomialNB.

- Entrenamiento y evaluación: validación cruzada estratificada de 5 pliegues (F1 macro como criterio principal), prueba de reserva; artefactos guardados con joblib.

- Interfaz de usuario: Streamlit (inglés/español); recuperación de preguntas frecuentes mediante similitud coseno TF-IDF; escalado por umbral de probabilidad o margen; la barra lateral muestra la latencia promedio.

## Primeros pasos

1. Crear y activar un entorno virtual

- Windows (PowerShell):

``

python -m venv .venv

.venv\Scripts\Activate.ps1

``

- macOS/Linux (bash/zsh):

``

python3 -m venv .venv

source .venv/bin/activate

``
2. Instalar los requisitos:

``

pip install -r requirements.txt

``
3. Entrenar los modelos (conjunto de datos en inglés):

``

python scripts/train_all.py

``
4. Iniciar la interfaz de usuario (inglés):

``
streamlit run app/streamlit_app.py

``
5. Iniciar la interfaz de usuario (español):

``

streamlit run app/streamlit_app_es.py

``

## Funciones de las aplicaciones (EN/ES)

- Clasificación de sentimiento de cada mensaje de usuario mediante el modelo seleccionado. - Política de respuesta:

- Negativa: introducción empática; si la confianza/margen ≥ umbral ⇒ sugerir escalamiento a un agente humano.

- Neutral/Positiva: introducción informativa.

- Preguntas frecuentes: 10-15 preguntas respondidas mediante la similitud TF-IDF con el mensaje del usuario (umbral de similitud configurable).

- Latencia: muestra la latencia promedio de la sesión en la barra lateral.

- La aplicación en español traduce ES→EN para la clasificación y responde en español (con preguntas frecuentes en español).

## Valores predeterminados utilizados en demostraciones/informes

- Umbral de escalamiento (confianza/margen): 0.7

- Umbral de similitud de preguntas frecuentes (coseno en TF-IDF): 0.45
Puede ajustarlos en la barra lateral.

## Conjunto de datos y preprocesamiento

- Fuente: Kaggle Twitter US Airline Sentiment (etiquetas: negativa, neutral, positiva).

- La distribución de clases está desequilibrada (negativa ≈ 62-63%). Damos prioridad a la macro F1.

- Prevención de fugas de datos: dividir los conjuntos de entrenamiento y prueba antes de la vectorización; ajustar TF-IDF solo en el conjunto de entrenamiento; eliminar las columnas relacionadas con las etiquetas.

## Modelos, entrenamiento y artefactos
- Pipelines: `tfidf` + clasificador; guardados en `models/*_pipeline.joblib` (cargables con `joblib.load`).

- Scripts:

- `scripts/train_all.py`: entrena LR/SVM/NB con validación cruzada de 5 pliegues, evalúa en el conjunto de prueba y guarda los informes.

- `scripts/eval.py`: carga los pipelines e imprime informes y gráficos en el conjunto de prueba.

- Informes: `reports/metrics.json`, `reports/classification_*.csv`, `reports/confusion_*.csv`.

## Resumen de resultados
Del archivo `reports/metrics.json` de una ejecución representativa:

- Regresión logística: Precisión 0.7845, Macro F1 0.7368

- SVM lineal: Precisión 0.7941, Macro F1 0.7360

- NB multinomial: Precisión 0.7186, Macro F1 0.5426

Interpretación: SVM lineal alcanza la mejor precisión; la regresión logística ofrece un buen equilibrio (rápida y competitiva). NB es la más rápida, pero menos equilibrada entre clases.
