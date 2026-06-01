import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    spark = (SparkSession.builder
             .appName("WikiFlow-01-clean")
             .config("spark.sql.shuffle.partitions", "400")
             .config("spark.sql.files.maxPartitionBytes", "256m")
             .getOrCreate())

    df = (spark.read
          .option("sep", "\t")
          .option("header", "false")
          .option("quote", "")
          .option("escape", "")
          .csv(args.raw.rstrip("/") + "/*.tsv.bz2"))

    df = df.filter(F.col("_c1") == "revision")
    df = df.filter((F.col("_c14").isNull()) | (F.trim(F.col("_c14")) == ""))
    df = df.filter(F.col("_c18") == "false")
    df = df.filter(F.col("_c31").isin("0", "1"))

    df = df.select(
        F.lit("enwiki").alias("wiki_db"),
        F.to_timestamp("_c3").alias("ts"),
        F.col("_c4").alias("event_comment"),
        F.col("_c6").alias("event_user_id"),
        F.col("_c8").alias("event_user_text"),
        F.to_timestamp("_c21").alias("event_user_registration_timestamp"),
        F.to_timestamp("_c23").alias("event_user_first_edit_timestamp"),
        F.col("_c24").cast("long").alias("event_user_revision_count"),
        F.col("_c26").cast("long").alias("page_id"),
        F.col("_c31").cast("int").alias("page_namespace"),
        F.col("_c58").cast("long").alias("revision_id"),
        (F.col("_c60") == "true").alias("revision_minor_edit"),
        F.col("_c63").cast("long").alias("revision_text_bytes"),
        F.col("_c64").cast("long").alias("revision_text_bytes_diff"),
        (F.col("_c70") == "true").alias("revision_is_identity_reverted"),
        (F.col("_c73") == "true").alias("revision_is_identity_revert"),
    )

    df = df.filter(F.col("ts").isNotNull())
    df = df.withColumn("year", F.year("ts"))
    df = df.withColumn("month", F.month("ts"))

    (df.write
       .mode("overwrite")
       .partitionBy("year", "month")
       .parquet(args.out))

    spark.stop()


if __name__ == "__main__":
    main()
