# WikiFlow — Wikipedia Editor Dropout Prediction at Scale
> MET CS 777 Big Data Analytics · Boston University · Spring 2026

End-to-end distributed ML pipeline on **GCP** predicting whether a new Wikipedia
editor will stop editing within 6 months. Processed **1.16M editors** from **25 GB**
of revision history (2021–2025) using PySpark on Dataproc.

| Model | AUC-ROC | Notes |
|-------|---------|-------|
| Custom Logistic Regression (RDD) | 0.845 | Hand-rolled, no MLlib |
| Custom K-Means (RDD) | Silhouette 0.335 | 4 editor archetypes |
| **Elephas Keras DNN** | **0.909** | Distributed across Spark workers |
