import argparse
import math

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--horizon-days", type=int, default=180)
    args = parser.parse_args()

    spark = (SparkSession.builder
             .appName("WikiFlow-02-features")
             .config("spark.sql.shuffle.partitions", "400")
             .getOrCreate())

    df = spark.read.parquet(args.clean).filter(F.col("event_user_id").isNotNull())

    w = Window.partitionBy("event_user_id").orderBy("ts")
    df = df.withColumn("edit_rank", F.row_number().over(w))

    first_window = df.filter(F.col("edit_rank") <= args.window)
    post_window = df.filter(F.col("edit_rank") > args.window)

    win_bounds = (first_window
                  .groupBy("event_user_id")
                  .agg(F.min("ts").alias("first_ts"),
                       F.max("ts").alias("last_window_ts"),
                       F.count("*").alias("window_edits"))
                  .filter(F.col("window_edits") >= 3))

    fw = first_window.join(win_bounds, "event_user_id")
    fw = fw.withColumn("window_days",
                       F.greatest(F.lit(1.0),
                                  (F.unix_timestamp("last_window_ts") -
                                   F.unix_timestamp("first_ts")) / 86400.0))

    velocity = (fw.groupBy("event_user_id")
                .agg((F.count("*") / F.first("window_days")).alias("velocity")))

    revert_rate = (fw.groupBy("event_user_id")
                   .agg(F.avg(F.col("revision_is_identity_reverted").cast("double"))
                        .alias("revert_rate")))

    ns = (fw.groupBy("event_user_id", "page_namespace").count())
    ns_total = ns.groupBy("event_user_id").agg(F.sum("count").alias("total"))
    ns = ns.join(ns_total, "event_user_id")
    ns = ns.withColumn("p", F.col("count") / F.col("total"))
    ns = ns.withColumn("plog", -F.col("p") * F.log2("p"))
    ns_div = ns.groupBy("event_user_id").agg(F.sum("plog").alias("namespace_diversity"))

    bytes_stats = (fw.groupBy("event_user_id")
                   .agg(F.avg(F.when(F.col("revision_text_bytes_diff") > 0,
                                     F.col("revision_text_bytes_diff"))
                              .otherwise(0)).alias("avg_bytes_added"),
                        F.avg(F.when(F.col("revision_text_bytes_diff") < 0,
                                     -F.col("revision_text_bytes_diff"))
                              .otherwise(0)).alias("avg_bytes_removed")))

    talk_ratio = (fw.groupBy("event_user_id")
                  .agg(F.avg((F.col("page_namespace") == 1).cast("double"))
                       .alias("talk_page_ratio")))

    fw2 = fw.withColumn("hour", F.hour("ts"))
    hour_modes = (fw2.groupBy("event_user_id", "hour")
                  .count()
                  .withColumn("rk",
                              F.row_number().over(
                                  Window.partitionBy("event_user_id")
                                  .orderBy(F.desc("count"))))
                  .filter(F.col("rk") == 1)
                  .select("event_user_id", "hour"))
    hour_modes = hour_modes.withColumn(
        "peak_hour_sin", F.sin(2 * math.pi * F.col("hour") / 24.0))
    hour_modes = hour_modes.withColumn(
        "peak_hour_cos", F.cos(2 * math.pi * F.col("hour") / 24.0))
    hour_modes = hour_modes.select("event_user_id", "peak_hour_sin", "peak_hour_cos")

    weekend = (fw.withColumn("dow", F.dayofweek("ts"))
               .groupBy("event_user_id")
               .agg(F.avg(F.when(F.col("dow").isin(1, 7), 1.0)
                          .otherwise(0.0)).alias("weekend_ratio")))

    summary = (fw.groupBy("event_user_id")
               .agg(F.avg(F.when((F.col("event_comment").isNotNull()) &
                                 (F.col("event_comment") != ""), 1.0)
                          .otherwise(0.0)).alias("edit_summary_rate")))

    minor = (fw.groupBy("event_user_id")
             .agg(F.avg(F.col("revision_minor_edit").cast("double"))
                  .alias("minor_edit_ratio")))

    fw3 = fw.withColumn(
        "prev_ts",
        F.lag("ts").over(Window.partitionBy("event_user_id").orderBy("ts")))
    fw3 = fw3.withColumn(
        "gap_min",
        (F.unix_timestamp("ts") - F.unix_timestamp("prev_ts")) / 60.0)
    fw3 = fw3.withColumn("new_session",
                         F.when((F.col("gap_min").isNull()) |
                                (F.col("gap_min") > 30), 1).otherwise(0))
    sessions = (fw3.groupBy("event_user_id")
                .agg(F.sum("new_session").alias("session_count")))

    first_size = (fw.filter(F.col("edit_rank") == 1)
                  .select("event_user_id",
                          F.coalesce(F.col("revision_text_bytes"),
                                     F.lit(0)).alias("first_edit_size")))

    horizon = args.horizon_days * 86400
    pw = post_window.join(win_bounds, "event_user_id")
    pw = pw.withColumn(
        "within_horizon",
        F.when((F.unix_timestamp("ts") -
                F.unix_timestamp("last_window_ts")) <= horizon, 1).otherwise(0))
    label = pw.groupBy("event_user_id").agg(F.max("within_horizon").alias("kept"))

    feats = (win_bounds
             .join(velocity, "event_user_id", "left")
             .join(revert_rate, "event_user_id", "left")
             .join(ns_div, "event_user_id", "left")
             .join(bytes_stats, "event_user_id", "left")
             .join(talk_ratio, "event_user_id", "left")
             .join(hour_modes, "event_user_id", "left")
             .join(weekend, "event_user_id", "left")
             .join(summary, "event_user_id", "left")
             .join(minor, "event_user_id", "left")
             .join(sessions, "event_user_id", "left")
             .join(first_size, "event_user_id", "left")
             .join(label, "event_user_id", "left"))

    feats = feats.withColumn("dropout",
                             F.when(F.col("kept") == 1, 0).otherwise(1))

    feature_cols = ["velocity", "revert_rate", "namespace_diversity",
                    "avg_bytes_added", "avg_bytes_removed", "talk_page_ratio",
                    "peak_hour_sin", "peak_hour_cos", "weekend_ratio",
                    "edit_summary_rate", "minor_edit_ratio",
                    "session_count", "first_edit_size"]
    for c in feature_cols:
        feats = feats.withColumn(c,
                                 F.coalesce(F.col(c), F.lit(0.0)).cast("double"))

    out = feats.select("event_user_id", "dropout", *feature_cols)
    out.write.mode("overwrite").parquet(args.out)
    spark.stop()


if __name__ == "__main__":
    main()
