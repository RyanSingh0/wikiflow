# WikiFlow — Wikipedia Editor Dropout Prediction at Scale

> **MET CS 777 Big Data Analytics · Boston University · Spring 2026**  
> Team: Aryan Meena · Kunj Manish Kumar Patel

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PySpark](https://img.shields.io/badge/PySpark-3.4-orange)
![GCP](https://img.shields.io/badge/GCP-Dataproc%20%7C%20BigQuery-blue)
![AUC](https://img.shields.io/badge/Best%20AUC--ROC-0.909-brightgreen)
![Scale](https://img.shields.io/badge/Editors%20Processed-1.16M-green)

---

## Overview

WikiFlow is a distributed analytics pipeline on **Google Cloud Platform** that ingests
the public English Wikipedia revision history (2026-03 snapshot, Jan 2021 – Dec 2025),
builds per-editor behavioral feature vectors, and predicts whether a newly registered
editor will **stop contributing within six months of their first ten edits**.

**Why this matters:** English Wikipedia has been losing active editors since 2007. A
high-confidence dropout score on a brand-new editor can trigger an early mentorship
intervention — operationally useful, and a clean fit for distributed computing since
1.16 million editors and 60 million revision events do not fit in memory on a single machine.

**Scale:** ~25 GB compressed · 60M+ revision events · 1,158,248 unique editors · runs fully on GCP with zero local data movement.

---

## System Architecture

```
Wikipedia Dumps (HTTPS, ~25 GB compressed, 60 monthly TSV.bz2 files)
        │
        ▼  00_ingest_dumps.py  (concurrent download, idempotent)
   Google Cloud Storage (GCS)
        │
        ▼  02_features.py  (PySpark on Dataproc)
   Feature Engineering  ──────────────────────────→  BigQuery results table
   13 features × 1,158,248 editors                   + Plotly dashboard on GCS
        │
        ├──▶ 03_logreg_rdd.py   →  Custom Logistic Regression (RDD mini-batch SGD)
        ├──▶ 04_kmeans_rdd.py   →  Custom K-Means (RDD Lloyd iteration)
        └──▶ 05_elephas_dnn.py  →  Elephas Keras DNN (synchronous distributed training)
```

Full pipeline runs end-to-end through GCP web console + Cloud Shell.
Bottleneck: ingestion (~1–2 hrs). All three models train in **under 7 minutes** total.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Wikimedia MediaWiki history dumps |
| Snapshot | 2026-03 (English Wikipedia) |
| Event months | January 2021 – December 2025 |
| Raw size | ~25 GB compressed (60 monthly files, 200–600 MB each) |
| Revision events | ~60 million |
| Schema | 76 pre-computed columns per revision |
| Final cohort | **1,158,248** unique registered, non-bot editors |

**Label:** Binary. `1` = editor made **zero edits** in the 180 days after their first 10-edit observation window. `0` = still active.

---

## Feature Engineering (13 features)

Features are computed from **each editor's first 10 edits only** — nothing outside the observation window, preventing label leakage.

| Feature | What it captures | Direction |
|---------|-----------------|-----------|
| `session_count` | Distinct editing sessions in first 10 edits | ↑ sessions → retention (**strongest signal**) |
| `velocity` | Edits per day in observation window | ↑ velocity → retention |
| `revert_rate` | Fraction of edits reverted by community | ↑ reverts → dropout |
| `talk_page_frac` | Fraction of edits on talk pages | ↑ talk → retention |
| `namespace_entropy` | Shannon entropy across namespaces | ↑ diversity → retention |
| `summary_present_frac` | Fraction of edits with an edit summary | Signals investment |
| `first_edit_size` | Byte size of first edit | ↑ size → retention (weak) |
| `peak_hour` | Most active hour of day | Time-of-day control |
| `weekend_frac` | Fraction of edits on weekends | Behavioral control |
| `avg_bytes_changed` | Mean absolute bytes changed | Edit scope |
| `minor_edit_frac` | Fraction flagged as minor edits | Editing style |
| `reverts_received` | Count of reverts received | Community friction |
| `articles_touched` | Distinct articles edited | Scope breadth |

**Selection:** 40+ candidates evaluated against three filters: (1) computable from first 10 edits only, (2) no author identity, (3) backed by retention literature. Final 13 directly cover the signals Halfaker et al. (2013) identified as strongest.

---

## Models

All three models are **built from scratch** — no MLlib — to demonstrate the underlying distributed computation pattern.

### 1 — Custom Logistic Regression (RDD mini-batch SGD)

Mini-batch SGD on PySpark RDDs. Each epoch samples a training fraction; every partition computes a partial gradient; `treeAggregate` sums gradients in O(log N) hops; driver applies one SGD step on broadcast weights.

| Metric | Value |
|--------|-------|
| **AUC-ROC** | **0.845** |
| Recall (dropouts) | **90.8%** |
| Precision | 77.2% |

**Weight signs confirm retention theory:**
- `session_count` = **−0.45** (strongest): multi-session editors stay
- `velocity` = **−0.23**: faster early editing → retention
- `revert_rate` = **+0.16**: community pushback → dropout (Halfaker's frustrated-newcomer hypothesis, replicated at scale on 1.16M editors)

**Best use case:** When missing a dropout is more costly than a false alarm (low intervention cost).

---

### 2 — Custom K-Means (RDD Lloyd iteration)

Standard Lloyd iteration on RDDs. Centroids broadcast each iteration; points compute closest centroid in a map; `reduceByKey` accumulates per-cluster sums; driver divides for new centroids. Stops when centroid shift < 1e-4 or WCSS improvement < 0.1%.

**Quality:** Silhouette = **0.335** (50k-row sample). WCSS = 11.58M on standardized features.

**4 Editor Archetypes Discovered:**

| Cluster | Size | Behavior | Label |
|---------|------|----------|-------|
| C0 | 308K (26.6%) | 5.4 edits/day · 50% revert rate · article-only edits | **"Frustrated newcomer"** — Halfaker's pattern at scale |
| C1 | 402K (34.7%) | 0.49 edits/day · 12% revert rate · 5+ sessions | **"The keeper"** — slow, careful, multi-session |
| C2 | 140K (12.1%) | Namespace entropy 0.73 · 46% talk-page edits | **"Community builder"** — high-value; flag for mentors |
| C3 | 307K (26.5%) | 3.1 edits/day · narrow scope · fixed peak hour | **"Focused/bot-like"** — single-purpose or unflagged bots |

---

### 3 — Elephas Distributed Keras DNN

Elephas serializes a Keras `Sequential` model, ships it to every Spark worker, each worker trains on its local partition with full TensorFlow autodiff, and weight updates are merged synchronously per epoch. Same pattern as Horovod and TF's MultiWorkerMirroredStrategy, expressed inside Spark.

**Architecture:** Input(13) → Dense(64, ReLU) → Dense(32, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)  
**Optimizer:** Adam · **Epochs:** 15 · **Batch size:** 256

| Metric | Value |
|--------|-------|
| **AUC-ROC** | **0.909** |
| Precision | **0.873** |
| Recall | **0.871** |

DNN beats LR by **+6.4 AUC points** and ~10 precision points. DNN AUC of 0.91 **exceeds** the 0.72–0.78 range Halfaker et al. (2013) reported on similar features, single-machine — bigger model + more data helped.

**Best use case:** Balanced cost — when false intervention cost ≈ missed-dropout cost.

---

## Full Results Summary

| Model | AUC-ROC | Precision | Recall | Training time |
|-------|---------|-----------|--------|--------------|
| Custom LR (RDD) | 0.845 | 0.772 | **0.908** | ~1 min |
| Custom K-Means | Silhouette: 0.335 | — | — | ~1 min |
| **Elephas Keras DNN** | **0.909** | **0.873** | 0.871 | ~5 min |
| — | — | — | — | **Total < 7 min** |

---

## Key Findings

1. **Session spread beats edit volume.** Editors who spread 10 edits across multiple sessions stay; single-sitting editors leave. Strongest feature by weight (−0.45).

2. **Community friction drives dropout at scale.** `revert_rate` as a dropout predictor, confirming Halfaker et al.'s frustrated-newcomer hypothesis, is replicated here on 1.16M editors — not a sample.

3. **C2 editors (12.1% of cohort)** are disproportionately valuable. Immediate talk-page engagement + high namespace diversity signals genuine community investment. Worth flagging to Wikimedia mentorship programs.

4. **DNN consistently outperforms LR** but LR converges to a useful complementary operating point: very high recall (90.8%) when you need to catch almost everyone who might leave.

---

## Runtime & Cost (GCP)

| Stage | Wall time |
|-------|-----------|
| Ingestion (60 files, 4 concurrent streams) | ~1–2 hours |
| Feature engineering | ~25–40 minutes |
| All 3 models | < 7 minutes |
| **Total** | **~2 hours** |

Cluster: 1 master + 4 workers (n1-standard-4). Estimated cost: ~$2.50 per full run.

---

## How to Run

```bash
pip install -r requirements.txt

# Step 1: Download Wikipedia dumps to GCS (run on Dataproc)
python src/00_ingest_dumps.py --start 2021-01 --end 2025-12

# Step 2: Feature engineering
python src/02_features.py

# Step 3: Train models
python src/03_logreg_rdd.py      # Custom LR (AUC 0.845)
python src/04_kmeans_rdd.py      # Custom K-Means (silhouette 0.335)
python src/05_elephas_dnn.py     # Elephas DNN (AUC 0.909)
```

> Full pipeline requires a GCP project with Dataproc enabled.
> A local sample (50k rows) can be generated from the feature output CSV for testing.

---

## References

1. Halfaker, A., Geiger, R.S., Morgan, J.T., & Riedl, J. (2013). *The Rise and Decline of an Open Collaboration System.* American Behavioral Scientist, 57(5), 664–688.
2. Zaharia, M. et al. (2016). *Apache Spark: A Unified Engine for Big Data Processing.* CACM, 59(11).
3. Cahall, D. *Elephas: Distributed Deep Learning with Keras and Spark.* github.com/danielenricocahall/elephas
4. Wikimedia Analytics. *MediaWiki History documentation.* wikitech.wikimedia.org

---

**Aryan Meena** · [LinkedIn](https://linkedin.com/in/aryan-meena-32685415a) · araj7042@gmail.com · Boston, MA
