import argparse
import json
import math
import time

from pyspark.sql import SparkSession


FEATURE_COLS = ["velocity", "revert_rate", "namespace_diversity",
                "avg_bytes_added", "avg_bytes_removed", "talk_page_ratio",
                "peak_hour_sin", "peak_hour_cos", "weekend_ratio",
                "edit_summary_rate", "minor_edit_ratio",
                "session_count", "first_edit_size"]


def closest(point, centers):
    best_i, best_d = 0, float("inf")
    for i, c in enumerate(centers):
        d = sum((a - b) ** 2 for a, b in zip(point, c))
        if d < best_d:
            best_d = d
            best_i = i
    return best_i, best_d


def standardize_stats(rdd):
    n = rdd.count()
    sums = rdd.reduce(lambda a, b: [x + y for x, y in zip(a, b)])
    means = [s / n for s in sums]
    sq = rdd.map(lambda v: [(x - m) ** 2 for x, m in zip(v, means)])
    sqs = sq.reduce(lambda a, b: [x + y for x, y in zip(a, b)])
    stds = [math.sqrt(s / n) if s > 0 else 1.0 for s in sqs]
    return means, stds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("WikiFlow-04-kmeans-rdd").getOrCreate()
    sc = spark.sparkContext

    df = spark.read.parquet(args.features)
    pts = df.rdd.map(lambda r: [float(r[c]) for c in FEATURE_COLS]).cache()

    means, stds = standardize_stats(pts)
    bc_means = sc.broadcast(means); bc_stds = sc.broadcast(stds)
    pts = pts.map(
        lambda v: [(x - m) / s
                   for x, m, s in zip(v, bc_means.value, bc_stds.value)]).cache()

    centers = [list(p) for p in pts.takeSample(False, args.k, seed=args.seed)]

    start = time.time()
    last_cost = float("inf")
    for it in range(args.iters):
        bc_c = sc.broadcast(centers)

        def assign(p):
            i, d = closest(p, bc_c.value)
            return (i, (p, 1, d))

        def merge(a, b):
            return ([x + y for x, y in zip(a[0], b[0])],
                    a[1] + b[1], a[2] + b[2])

        agg = pts.map(assign).reduceByKey(merge).collect()
        new_centers = list(centers)
        cost = 0.0
        for cluster_id, (s, n, c) in agg:
            cost += c
            new_centers[cluster_id] = [x / n for x in s]

        shift = sum(math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))
                    for p, q in zip(centers, new_centers))
        centers = new_centers
        print(f"iter {it}: cost={cost:.1f} shift={shift:.4f}")
        if shift < 1e-4 or abs(last_cost - cost) < 1e-3 * max(1.0, last_cost):
            break
        last_cost = cost

    train_seconds = time.time() - start

    bc_c = sc.broadcast(centers)
    sample = pts.sample(False, min(1.0, 50000.0 / max(1, pts.count())),
                        seed=args.seed).collect()

    sil_scores = []
    for p in sample:
        dists = [sum((p[i] - c[i]) ** 2 for i in range(len(p))) for c in centers]
        a_idx = dists.index(min(dists))
        a = dists[a_idx]
        others = [d for i, d in enumerate(dists) if i != a_idx]
        b = min(others) if others else a
        denom = max(a, b)
        sil_scores.append(0.0 if denom == 0 else (b - a) / denom)
    silhouette = sum(sil_scores) / max(1, len(sil_scores))

    def assign_cluster(p):
        i, _ = closest(p, bc_c.value)
        return (i, (p, 1))

    def merge2(a, b):
        return ([x + y for x, y in zip(a[0], b[0])], a[1] + b[1])

    summaries = pts.map(assign_cluster).reduceByKey(merge2).collect()

    profile = {}
    for cluster_id, (s, n) in summaries:
        scaled_mean = [x / n for x in s]
        original_mean = [m_ + s_ * v
                         for m_, s_, v in zip(means, stds, scaled_mean)]
        profile[str(cluster_id)] = {
            "size": n,
            "feature_mean": dict(zip(FEATURE_COLS, original_mean)),
        }

    metrics = {
        "model": "kmeans_rdd", "k": args.k, "train_seconds": train_seconds,
        "silhouette": silhouette, "wcss": last_cost,
        "cluster_profile": profile,
    }
    centroids_payload = {
        "centers_standardized": centers,
        "feature_order": FEATURE_COLS,
        "scaler": {"means": means, "stds": stds},
    }

    sc.parallelize([json.dumps(metrics, indent=2)], 1).saveAsTextFile(args.metrics + ".tmp")
    sc.parallelize([json.dumps(centroids_payload, indent=2)], 1).saveAsTextFile(args.out.rstrip("/") + "/centers.tmp")

    print("Final K-Means metrics:", json.dumps(metrics, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
