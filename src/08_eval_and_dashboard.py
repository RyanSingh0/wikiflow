import argparse
import json

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def read_json_folder(spark, path):
    rdd = spark.sparkContext.textFile(path)
    return json.loads("\n".join(rdd.collect()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--dashboard", required=True)
    p.add_argument("--bq-dataset", required=True)
    p.add_argument("--bq-table", required=True)
    args = p.parse_args()

    spark = SparkSession.builder.appName("WikiFlow-08-eval-dashboard").getOrCreate()
    sc = spark.sparkContext

    logreg = read_json_folder(spark, args.metrics + "logreg.json.tmp")
    kmeans = read_json_folder(spark, args.metrics + "kmeans.json.tmp")
    try: dnn = read_json_folder(spark, args.metrics + "dnn.json.tmp")
    except Exception: dnn = None

    feats = spark.read.parquet(args.features)
    n_editors = feats.count()
    dropout_rate = feats.agg(F.avg("dropout")).collect()[0][0]

    summary = {"n_editors": n_editors, "dropout_rate": dropout_rate,
               "logreg": logreg, "kmeans": kmeans}
    if dnn is not None:
        summary["dnn"] = dnn

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Model AUC-ROC comparison",
                        "Confusion matrix (logistic regression)",
                        "Cluster sizes",
                        "Cluster mean: revert_rate vs velocity"),
        specs=[[{"type": "bar"}, {"type": "heatmap"}],
               [{"type": "bar"}, {"type": "scatter"}]])

    names = ["LogReg"]; aucs = [logreg["auc_roc"]]
    if dnn is not None: names.append("Elephas DNN"); aucs.append(dnn["auc_roc"])
    fig.add_trace(go.Bar(x=names, y=aucs, name="AUC-ROC"), row=1, col=1)

    cm = logreg["confusion"]
    fig.add_trace(go.Heatmap(
        z=[[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        x=["pred 0", "pred 1"], y=["actual 0", "actual 1"],
        colorscale="Blues", showscale=False,
        text=[[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
        texttemplate="%{text}"), row=1, col=2)

    profile = kmeans["cluster_profile"]
    cluster_ids = sorted(profile.keys())
    sizes = [profile[c]["size"] for c in cluster_ids]
    fig.add_trace(go.Bar(x=[f"C{c}" for c in cluster_ids], y=sizes,
                         name="cluster size"), row=2, col=1)

    rv = [profile[c]["feature_mean"]["revert_rate"] for c in cluster_ids]
    vv = [profile[c]["feature_mean"]["velocity"] for c in cluster_ids]
    fig.add_trace(go.Scatter(x=rv, y=vv, mode="markers+text",
                             text=[f"C{c}" for c in cluster_ids],
                             marker=dict(size=18, color=sizes, colorscale="Viridis"),
                             name="clusters"), row=2, col=2)
    fig.update_xaxes(title_text="revert_rate (cluster mean)", row=2, col=2)
    fig.update_yaxes(title_text="velocity (edits/day)", row=2, col=2)

    fig.update_layout(
        title_text=(f"WikiFlow Results | editors: {n_editors:,} | "
                    f"dropout rate: {dropout_rate:.2%}"),
        height=800, showlegend=False)

    html = fig.to_html(include_plotlyjs="cdn", full_html=True)

    out_path = args.dashboard.rstrip("/") + "/index.html"
    hconf = sc._jsc.hadoopConfiguration()
    fs = (sc._jvm.org.apache.hadoop.fs.FileSystem
          .get(sc._jvm.java.net.URI.create(args.dashboard), hconf))
    out_stream = fs.create(sc._jvm.org.apache.hadoop.fs.Path(out_path), True)
    out_stream.write(html.encode("utf-8")); out_stream.close()

    j_stream = fs.create(sc._jvm.org.apache.hadoop.fs.Path(
        args.dashboard.rstrip("/") + "/summary.json"), True)
    j_stream.write(json.dumps(summary, indent=2).encode("utf-8")); j_stream.close()

    rows = []
    for c in cluster_ids:
        rec = {
            "cluster_id": int(c),
            "cluster_size": int(profile[c]["size"]),
            "model_auc_logreg": float(logreg["auc_roc"]),
            "model_auc_dnn": float(dnn["auc_roc"]) if dnn else None,
            "kmeans_silhouette": float(kmeans["silhouette"]),
            "logreg_train_seconds": float(logreg["train_seconds"]),
            "kmeans_train_seconds": float(kmeans["train_seconds"]),
        }
        for k, v in profile[c]["feature_mean"].items():
            rec["mean_" + k] = float(v)
        rows.append(rec)

    results_df = spark.createDataFrame(rows)
    (results_df.write
        .format("bigquery")
        .option("table", f"{args.bq_dataset}.{args.bq_table}")
        .option("writeMethod", "direct")
        .mode("overwrite").save())

    spark.stop()


if __name__ == "__main__":
    main()
