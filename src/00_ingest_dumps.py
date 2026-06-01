import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from pyspark.sql import SparkSession


BASE = "https://dumps.wikimedia.org/other/mediawiki_history"


def months_between(start, end):
    sy, sm = [int(x) for x in start.split("-")]
    ey, em = [int(x) for x in end.split("-")]
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


def fs_for(sc, gcs_path):
    return (sc._jvm.org.apache.hadoop.fs.FileSystem
            .get(sc._jvm.java.net.URI.create(gcs_path),
                 sc._jsc.hadoopConfiguration()))


def gcs_size(sc, gcs_path):
    fs = fs_for(sc, gcs_path)
    p = sc._jvm.org.apache.hadoop.fs.Path(gcs_path)
    if not fs.exists(p):
        return -1
    return int(fs.getFileStatus(p).getLen())


def gcs_delete(sc, gcs_path):
    fs = fs_for(sc, gcs_path)
    p = sc._jvm.org.apache.hadoop.fs.Path(gcs_path)
    if fs.exists(p):
        fs.delete(p, False)


def gcs_rename(sc, src, dst):
    fs = fs_for(sc, src)
    fs.rename(sc._jvm.org.apache.hadoop.fs.Path(src),
              sc._jvm.org.apache.hadoop.fs.Path(dst))


def stream_one(sc, url, final_path, retries):
    if gcs_size(sc, final_path) > 0:
        return ("SKIP", gcs_size(sc, final_path), "already present")

    part_path = final_path + ".part"
    for attempt in range(1, retries + 1):
        gcs_delete(sc, part_path)
        fs = fs_for(sc, part_path)
        out = fs.create(sc._jvm.org.apache.hadoop.fs.Path(part_path), True)
        wrote = 0
        try:
            req = Request(url, headers={
                "User-Agent": "WikiFlow/1.0 CS777",
                "Accept-Encoding": "identity",
            })
            with urlopen(req, timeout=900) as r:
                expected = r.headers.get("Content-Length")
                expected = int(expected) if expected else None
                while True:
                    chunk = r.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    wrote += len(chunk)
            out.close()
            if expected is not None and wrote != expected:
                raise IOError(f"short read: {wrote}/{expected}")
            gcs_delete(sc, final_path)
            gcs_rename(sc, part_path, final_path)
            return ("OK", wrote, f"attempt {attempt}")
        except (HTTPError, URLError, IOError, Exception) as e:
            try: out.close()
            except Exception: pass
            gcs_delete(sc, part_path)
            if attempt == retries:
                return ("FAIL", wrote, f"after {attempt} tries: {e!r}")
            time.sleep(min(60, 2 ** attempt))
    return ("FAIL", 0, "unreachable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="2026-03")
    ap.add_argument("--start", default="2021-01")
    ap.add_argument("--end", default="2025-12")
    ap.add_argument("--wiki", default="enwiki")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()

    spark = SparkSession.builder.appName("WikiFlow-00-ingest").getOrCreate()
    sc = spark.sparkContext

    pairs = []
    for y, m in months_between(args.start, args.end):
        ym = f"{y:04d}-{m:02d}"
        url = (f"{BASE}/{args.snapshot}/{args.wiki}/"
               f"{args.snapshot}.{args.wiki}.{ym}.tsv.bz2")
        out_path = (args.out.rstrip("/") + "/"
                    f"{args.snapshot}.{args.wiki}.{ym}.tsv.bz2")
        pairs.append((url, out_path))

    print(f"Targets: {len(pairs)} files. Threads: {args.threads}.", flush=True)

    t0 = time.time()
    counts = {"OK": 0, "SKIP": 0, "FAIL": 0}
    failed = []

    def task(pair):
        return (pair[0], pair[1]) + stream_one(sc, pair[0], pair[1], args.retries)

    done = 0
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futs = [pool.submit(task, pr) for pr in pairs]
        for fut in as_completed(futs):
            url, out_path, status, nbytes, msg = fut.result()
            done += 1
            counts[status] = counts.get(status, 0) + 1
            mb = nbytes / 1024 / 1024
            print(f"[{done}/{len(pairs)}] {status} {mb:8.1f} MB  "
                  f"{url.rsplit('/', 1)[-1]}  -- {msg}", flush=True)
            if status == "FAIL":
                failed.append((url, msg))

    print(f"\nFinished in {time.time() - t0:.0f}s. "
          f"OK={counts['OK']}  SKIP={counts['SKIP']}  FAIL={counts['FAIL']}",
          flush=True)
    if failed:
        for url, msg in failed:
            print(f"  {url.rsplit('/', 1)[-1]}  ::  {msg}", flush=True)
        spark.stop()
        sys.exit(1)
    spark.stop()


if __name__ == "__main__":
    main()
