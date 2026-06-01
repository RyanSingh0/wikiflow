import argparse
import json
import time

import numpy as np

from pyspark.sql import SparkSession


FEATURE_COLS = ["velocity", "revert_rate", "namespace_diversity",
                "avg_bytes_added", "avg_bytes_removed", "talk_page_ratio",
                "peak_hour_sin", "peak_hour_cos", "weekend_ratio",
                "edit_summary_rate", "minor_edit_ratio",
                "session_count", "first_edit_size"]


def build_model(input_dim):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    m = Sequential()
    m.add(Dense(64, activation="relu", input_dim=input_dim))
    m.add(Dropout(0.2))
    m.add(Dense(32, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def to_numpy(rows):
    X = np.array([[float(r[c]) for c in FEATURE_COLS] for r in rows],
                 dtype=np.float32)
    y = np.array([float(r["dropout"]) for r in rows], dtype=np.float32)
    return X, y


def auc_via_sample(y_true, y_score, n_pos=10000, n_neg=10000, seed=7):
    rng = np.random.default_rng(seed)
    pos = y_score[y_true == 1]; neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    if len(pos) > n_pos: pos = rng.choice(pos, n_pos, replace=False)
    if len(neg) > n_neg: neg = rng.choice(neg, n_neg, replace=False)
    wins = ties = 0
    for p in pos:
        wins += int(np.sum(p > neg))
        ties += int(np.sum(p == neg))
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    spark = (SparkSession.builder
             .appName("WikiFlow-05-elephas-dnn")
             .config("spark.driver.memory", "4g")
             .getOrCreate())
    sc = spark.sparkContext

    df = spark.read.parquet(args.features)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=args.seed)
    train_df = train_df.cache(); test_df = test_df.cache()
    n_train = train_df.count(); n_test = test_df.count()
    print(f"n_train={n_train:,}  n_test={n_test:,}", flush=True)

    X_train, y_train = to_numpy(train_df.collect())
    X_test, y_test = to_numpy(test_df.collect())

    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds == 0] = 1.0
    X_train = (X_train - means) / stds
    X_test = (X_test - means) / stds

    model = build_model(len(FEATURE_COLS))
    model.summary(print_fn=lambda s: print(s, flush=True))

    from elephas.utils.rdd_utils import to_simple_rdd
    train_rdd = to_simple_rdd(sc, X_train, y_train)

    from elephas.spark_model import SparkModel
    spark_model = SparkModel(model, frequency='epoch', mode='synchronous')

    t0 = time.time()
    spark_model.fit(train_rdd, epochs=args.epochs, batch_size=args.batch,
                    verbose=1, validation_split=0.1)
    train_seconds = time.time() - t0

    final_model = spark_model.master_network
    p_test = final_model.predict(X_test, batch_size=4096, verbose=0).reshape(-1)
    yhat = (p_test >= 0.5).astype(np.int32)
    y_int = y_test.astype(np.int32)

    tp = int(np.sum((yhat == 1) & (y_int == 1)))
    fp = int(np.sum((yhat == 1) & (y_int == 0)))
    tn = int(np.sum((yhat == 0) & (y_int == 0)))
    fn = int(np.sum((yhat == 0) & (y_int == 1)))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    acc = (tp + tn) / max(1, tp + fp + tn + fn)
    auc = auc_via_sample(y_int, p_test)

    metrics = {
        "model": "elephas_dnn", "n_train": int(n_train), "n_test": int(n_test),
        "n_features": len(FEATURE_COLS), "epochs": args.epochs, "batch": args.batch,
        "train_seconds": train_seconds,
        "auc_roc": auc, "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }

    local_path = "/tmp/wikiflow_dnn.h5"
    final_model.save(local_path)
    hconf = sc._jsc.hadoopConfiguration()
    fs = (sc._jvm.org.apache.hadoop.fs.FileSystem
          .get(sc._jvm.java.net.URI.create(args.out), hconf))
    src = sc._jvm.org.apache.hadoop.fs.Path("file://" + local_path)
    dst = sc._jvm.org.apache.hadoop.fs.Path(args.out.rstrip("/") + "/wikiflow_dnn.h5")
    fs.copyFromLocalFile(False, True, src, dst)

    sc.parallelize([json.dumps(metrics, indent=2)], 1).saveAsTextFile(args.metrics + ".tmp")
    print("DNN metrics:", json.dumps(metrics, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
