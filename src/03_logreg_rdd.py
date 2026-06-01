import argparse
import json
import math
import time
import random

from pyspark.sql import SparkSession


FEATURE_COLS = ["velocity", "revert_rate", "namespace_diversity",
                "avg_bytes_added", "avg_bytes_removed", "talk_page_ratio",
                "peak_hour_sin", "peak_hour_cos", "weekend_ratio",
                "edit_summary_rate", "minor_edit_ratio",
                "session_count", "first_edit_size"]


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def standardize_stats(rdd_features):
    n = rdd_features.count()
    sums = rdd_features.reduce(lambda a, b: [x + y for x, y in zip(a, b)])
    means = [s / n for s in sums]
    sq = rdd_features.map(lambda v: [(x - m) ** 2 for x, m in zip(v, means)])
    sqs = sq.reduce(lambda a, b: [x + y for x, y in zip(a, b)])
    stds = [math.sqrt(s / n) if s > 0 else 1.0 for s in sqs]
    return means, stds


def gradient_for_batch(batch, w, b):
    g_w = [0.0] * len(w)
    g_b = 0.0
    for label, x in batch:
        z = b + sum(wi * xi for wi, xi in zip(w, x))
        err = sigmoid(z) - label
        for i in range(len(w)):
            g_w[i] += err * x[i]
        g_b += err
    return g_w, g_b, len(batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("WikiFlow-03-logreg-rdd").getOrCreate()
    sc = spark.sparkContext

    df = spark.read.parquet(args.features)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=args.seed)
    train_df.cache(); test_df.cache()

    def row_to_pair(r):
        return (float(r["dropout"]), [float(r[c]) for c in FEATURE_COLS])

    train = train_df.rdd.map(row_to_pair)
    test = test_df.rdd.map(row_to_pair)

    feat_only = train.map(lambda x: x[1]); feat_only.cache()
    means, stds = standardize_stats(feat_only)
    feat_only.unpersist()

    bc_means = sc.broadcast(means); bc_stds = sc.broadcast(stds)

    def scale(pair):
        label, x = pair
        m, s = bc_means.value, bc_stds.value
        return (label, [(xi - mi) / si for xi, mi, si in zip(x, m, s)])

    train = train.map(scale).cache()
    test = test.map(scale).cache()

    n_features = len(FEATURE_COLS)
    rng = random.Random(args.seed)
    w = [rng.gauss(0.0, 0.01) for _ in range(n_features)]
    b = 0.0

    n_train = train.count()
    print(f"Training rows: {n_train}")

    start = time.time()
    for epoch in range(args.epochs):
        sample_frac = min(1.0, args.batch * 100.0 / max(1, n_train))
        sample = train.sample(False, sample_frac, seed=args.seed + epoch)

        bc_w = sc.broadcast(w); bc_b = sc.broadcast(b)

        def per_partition(it):
            batch = list(it)
            if not batch:
                return iter([([0.0] * n_features, 0.0, 0)])
            return iter([gradient_for_batch(batch, bc_w.value, bc_b.value)])

        partials = sample.mapPartitions(per_partition)

        def combine(a, b_):
            return ([x + y for x, y in zip(a[0], b_[0])],
                    a[1] + b_[1],
                    a[2] + b_[2])

        g_w, g_b, n_seen = partials.treeAggregate(
            ([0.0] * n_features, 0.0, 0), combine, combine, depth=2)

        if n_seen == 0:
            continue
        for i in range(n_features):
            w[i] -= args.lr * (g_w[i] / n_seen)
        b -= args.lr * (g_b / n_seen)

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}: |w|={sum(abs(x) for x in w):.3f}  b={b:.3f}")

    train_seconds = time.time() - start

    bc_w = sc.broadcast(w); bc_b = sc.broadcast(b)

    def predict(pair):
        label, x = pair
        z = bc_b.value + sum(wi * xi for wi, xi in zip(bc_w.value, x))
        return (label, sigmoid(z))

    scored = test.map(predict).cache()

    def cm_partition(it):
        tp = fp = tn = fn = 0
        for y, p in it:
            yhat = 1 if p >= 0.5 else 0
            if y == 1 and yhat == 1: tp += 1
            elif y == 0 and yhat == 1: fp += 1
            elif y == 0 and yhat == 0: tn += 1
            else: fn += 1
        yield (tp, fp, tn, fn)

    cm = scored.mapPartitions(cm_partition).reduce(
        lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]))
    tp, fp, tn, fn = cm
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    acc = (tp + tn) / max(1, tp + fp + tn + fn)

    pos = scored.filter(lambda x: x[0] == 1).map(lambda x: x[1]).take(20000)
    neg = scored.filter(lambda x: x[0] == 0).map(lambda x: x[1]).take(20000)
    if pos and neg:
        wins = ties = 0
        for p in pos:
            for q in neg:
                if p > q: wins += 1
                elif p == q: ties += 1
        auc = (wins + 0.5 * ties) / (len(pos) * len(neg))
    else:
        auc = float("nan")

    metrics = {
        "model": "logreg_rdd", "n_train": n_train, "n_features": n_features,
        "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
        "train_seconds": train_seconds,
        "auc_roc": auc, "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
    weights = {"intercept": b, "weights": dict(zip(FEATURE_COLS, w))}
    scaler = {"means": dict(zip(FEATURE_COLS, means)),
              "stds": dict(zip(FEATURE_COLS, stds))}

    sc.parallelize([json.dumps(metrics, indent=2)], 1).saveAsTextFile(args.metrics + ".tmp")
    sc.parallelize([json.dumps(weights, indent=2)], 1).saveAsTextFile(args.out.rstrip("/") + "/weights.tmp")
    sc.parallelize([json.dumps(scaler, indent=2)], 1).saveAsTextFile(args.out.rstrip("/") + "/scaler.tmp")

    print("Final metrics:", json.dumps(metrics, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
