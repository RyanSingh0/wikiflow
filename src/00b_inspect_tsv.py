import argparse

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    args = ap.parse_args()

    spark = SparkSession.builder.appName("WikiFlow-00b-inspect").getOrCreate()
    df = (spark.read
          .option("sep", "\t").option("header", "false")
          .option("quote", "").option("escape", "")
          .csv(args.file))
    n_cols = len(df.columns); n_rows = df.count()
    print(f"FILE   : {args.file}", flush=True)
    print(f"ROWS   : {n_rows:,}", flush=True)
    print(f"COLUMNS: {n_cols}", flush=True)
    sample = df.limit(3).collect()
    for i, row in enumerate(sample):
        print(f"\n--- ROW {i + 1} ---", flush=True)
        for j, col in enumerate(df.columns):
            val = row[col]
            disp = "<null>" if val is None else (str(val)[:80])
            print(f"  c{j:02d}  {disp}", flush=True)
    spark.stop()


if __name__ == "__main__":
    main()
