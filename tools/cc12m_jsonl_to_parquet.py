from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(
        description="Stream a (gzipped) JSON-lines image dataset into a compact url+caption "
        "parquet that img2dataset can shard efficiently."
    )
    parser.add_argument("--jsonl", required=True, help="Path to train.jsonl or train.jsonl.gz")
    parser.add_argument("--out", required=True, help="Output parquet file path")
    parser.add_argument("--url-col", default="url")
    parser.add_argument("--caption-col", default="caption_llava")
    parser.add_argument(
        "--status-col",
        default="status",
        help="Keep only rows where this column equals --status-ok (empty string to disable).",
    )
    parser.add_argument("--status-ok", default="success")
    parser.add_argument("--limit", type=int, default=0, help="Stop after writing this many rows (0 = all).")
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()

    src = Path(args.jsonl)
    opener = gzip.open if src.suffix == ".gz" else open

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([(args.url_col, pa.string()), (args.caption_col, pa.string())])
    writer = pq.ParquetWriter(out, schema)

    urls: list[str] = []
    caps: list[str] = []
    written = 0
    seen = 0
    skipped_status = 0
    skipped_missing = 0
    checked_cols = False

    def flush():
        nonlocal urls, caps
        if not urls:
            return
        writer.write_table(pa.table({args.url_col: urls, args.caption_col: caps}, schema=schema))
        urls, caps = [], []

    with opener(src, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not checked_cols:
                missing = [c for c in (args.url_col, args.caption_col) if c not in row]
                if missing:
                    writer.close()
                    raise SystemExit(
                        f"ERROR: column(s) {missing} not in JSONL. Available keys: {list(row.keys())}"
                    )
                checked_cols = True
            seen += 1
            if args.status_col and row.get(args.status_col) != args.status_ok:
                skipped_status += 1
                continue
            url = row.get(args.url_col)
            caption = row.get(args.caption_col)
            if not url or caption is None:
                skipped_missing += 1
                continue
            urls.append(str(url))
            caps.append(str(caption))
            written += 1
            if len(urls) >= args.batch_size:
                flush()
            if args.limit and written >= args.limit:
                break

    flush()
    writer.close()
    print(f"read {seen} rows; wrote {written} -> {out}", flush=True)
    print(
        f"skipped: status!={args.status_ok}={skipped_status}, missing url/caption={skipped_missing}",
        flush=True,
    )


if __name__ == "__main__":
    main()
