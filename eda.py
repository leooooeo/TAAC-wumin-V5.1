#!/usr/bin/env python3
"""EDA for two questions:

(A) Which user features are "stable" — i.e. present (non-null / non-empty /
    non-zero) for basically every row. Output a sortable table so we can pick
    which features to feed into the item-history-user aggregator.

(B) For each row, count how many *prior* interactions the same item_id has
    seen, broken down by label_type (==2 positive conversion, ==1 click/expo,
    ==other). Print percentile + bucketed distributions so we can decide:
      - whether the item-history-user signal exists at all on this data,
      - what max-K to truncate at (sampling cap for online build),
      - cold-item rate (items with 0 prior history at sample time).

Reads parquet column-by-column to keep memory low. NO files are saved —
only prints. Logs go to stdout.

Usage
-----
  python3 eda.py --data_dir $TRAIN_DATA_PATH
  python3 eda.py --data_dir $TRAIN_DATA_PATH --max_files 2   # quick sanity
"""

import argparse
import glob
import logging
import os
import re
from typing import List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(message)s",
    datefmt="%H:%M:%S",
)


# ─────────────────────── helpers ──────────────────────────────────────────


def list_parquet_files(data_dir: str, max_files: int) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet in {data_dir}")
    if max_files > 0:
        files = files[:max_files]
    return files


def detect_feature_cols(files: List[str]) -> Tuple[List[str], List[str]]:
    """Return (user_int_cols, user_dense_cols) — sorted by fid."""
    pf = pq.ParquetFile(files[0])
    names = pf.schema_arrow.names
    int_cols = [n for n in names if n.startswith("user_int_feats_")]
    dense_cols = [n for n in names if n.startswith("user_dense_feats_")]

    def fid_of(n: str) -> int:
        m = re.search(r"_(\d+)$", n)
        return int(m.group(1)) if m else -1

    int_cols.sort(key=fid_of)
    dense_cols.sort(key=fid_of)
    return int_cols, dense_cols


# ─────────────────────── (A) feature stability ────────────────────────────


def feature_stability_one_column(
    files: List[str], col_name: str
) -> Tuple[int, int, int, int, int, float, float]:
    """Return per-column counters:
      n_rows, n_null, n_empty, n_nonpos, n_present, mean_len, p99_len

    Definitions:
      - n_null     : Arrow null (top-level)
      - n_empty    : list<...> column with length-0 element (non-null but empty)
      - n_nonpos   : for scalar int columns, value <= 0 (counted as "missing")
      - n_present  : rows that are usable for downstream aggregation
      - mean_len / p99_len: for list columns only, length stats over non-null rows
    """
    n_rows = 0
    n_null = 0
    n_empty = 0
    n_nonpos = 0
    len_sum = 0
    len_count = 0
    # reservoir for p99 length
    len_samples: List[int] = []
    max_samples = 200_000

    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=[col_name])
            col = tbl.column(0)
            B = len(col)
            n_rows += B
            # null mask
            null_mask = col.is_null().to_numpy(zero_copy_only=False)
            n_null += int(null_mask.sum())

            t = col.type
            if pa.types.is_list(t) or pa.types.is_large_list(t):
                # length per row using offsets; handle chunks
                # chunked array → iterate chunks
                if isinstance(col, pa.ChunkedArray):
                    chunks = col.chunks
                else:
                    chunks = [col]
                offset_lens: List[np.ndarray] = []
                for ch in chunks:
                    offs = np.asarray(ch.offsets, dtype=np.int64)
                    lens = np.diff(offs)
                    offset_lens.append(lens)
                lens_all = (
                    np.concatenate(offset_lens) if offset_lens else np.array([], np.int64)
                )
                # for null rows, length is 0 in arrow offsets but the row is "null"
                # we want non-null AND length>0 = "present"
                nonnull = ~null_mask
                empty_mask = nonnull & (lens_all == 0)
                n_empty += int(empty_mask.sum())
                present_lens = lens_all[nonnull & (lens_all > 0)]
                len_sum += int(present_lens.sum())
                len_count += int(present_lens.size)
                # reservoir
                if len_samples.__len__() < max_samples and present_lens.size > 0:
                    take = min(max_samples - len(len_samples), present_lens.size)
                    idx = np.random.randint(0, present_lens.size, size=take)
                    len_samples.extend(present_lens[idx].tolist())
            else:
                # scalar int / float column → count non-positive as "missing" for int
                arr = col.to_numpy(zero_copy_only=False)
                if np.issubdtype(arr.dtype, np.integer):
                    nonpos = (arr <= 0) & (~null_mask)
                    n_nonpos += int(nonpos.sum())
                else:
                    # float scalar: treat NaN as null-equivalent
                    nan_mask = np.isnan(arr.astype(np.float64, copy=False)) & (~null_mask)
                    n_nonpos += int(nan_mask.sum())

    n_present = n_rows - n_null - n_empty - n_nonpos
    mean_len = (len_sum / len_count) if len_count > 0 else 0.0
    p99_len = float(np.percentile(len_samples, 99)) if len_samples else 0.0
    return n_rows, n_null, n_empty, n_nonpos, n_present, mean_len, p99_len


def report_feature_stability(files: List[str]) -> None:
    int_cols, dense_cols = detect_feature_cols(files)
    logging.info(
        f"[A] Scanning {len(int_cols)} user_int_feats_* and "
        f"{len(dense_cols)} user_dense_feats_* columns over {len(files)} file(s)"
    )

    rows = []
    for col in int_cols + dense_cols:
        n, nn, ne, np_, npres, mlen, p99 = feature_stability_one_column(files, col)
        present_pct = (npres / n * 100.0) if n > 0 else 0.0
        rows.append(
            (col, n, nn, ne, np_, npres, present_pct, mlen, p99)
        )
        logging.info(
            f"  {col:30s}  rows={n:>10,}  null={nn:>9,}  empty={ne:>9,}  "
            f"nonpos={np_:>9,}  present={present_pct:6.2f}%  "
            f"mean_len={mlen:5.2f}  p99_len={p99:5.1f}"
        )

    print("\n" + "=" * 92)
    print("[A] USER FEATURE STABILITY — sorted by present% desc")
    print("=" * 92)
    print(
        f"{'column':32s}  {'rows':>10s}  {'null':>9s}  {'empty':>9s}  "
        f"{'nonpos':>9s}  {'present%':>8s}  {'mlen':>6s}  {'p99':>6s}"
    )
    rows.sort(key=lambda r: r[6], reverse=True)
    for col, n, nn, ne, np_, npres, present_pct, mlen, p99 in rows:
        print(
            f"{col:32s}  {n:>10,}  {nn:>9,}  {ne:>9,}  "
            f"{np_:>9,}  {present_pct:7.2f}%  {mlen:6.2f}  {p99:6.1f}"
        )

    # Quick filter view: which features are present for >= X% of rows
    print("\n" + "-" * 92)
    print("[A] STABLE FEATURE BUCKETS (candidates for item-history-user aggregator)")
    print("-" * 92)
    for thr in (99.5, 99.0, 95.0, 90.0, 80.0):
        keep = [r[0] for r in rows if r[6] >= thr]
        print(f"  present >= {thr:5.1f}% : {len(keep):3d} features")
        for c in keep:
            print(f"      {c}")
        print()


# ─────────────────────── (B) per-item prior interaction counts ─────────────


def collect_meta(files: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read item_id, timestamp, label_type into three np arrays."""
    iid_parts, ts_parts, lt_parts = [], [], []
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(
                rg, columns=["item_id", "timestamp", "label_type"]
            )
            iid_parts.append(
                tbl.column("item_id").to_numpy(zero_copy_only=False).astype(np.int64)
            )
            ts_parts.append(
                tbl.column("timestamp").to_numpy(zero_copy_only=False).astype(np.int64)
            )
            # label_type can be null → fill 0
            lt = tbl.column("label_type").fill_null(0).to_numpy(zero_copy_only=False)
            lt_parts.append(lt.astype(np.int64))
    iid = np.concatenate(iid_parts)
    ts = np.concatenate(ts_parts)
    lt = np.concatenate(lt_parts)
    logging.info(
        f"[B] meta loaded: {len(iid):,} rows; "
        f"item unique={np.unique(iid).size:,}"
    )
    return iid, ts, lt


def report_item_history(files: List[str]) -> None:
    iid, ts, lt = collect_meta(files)
    N = len(iid)
    # tie-break by original order so equal timestamps preserve a deterministic order
    order = np.lexsort((np.arange(N), ts, iid))
    iid_s = iid[order]
    ts_s = ts[order]
    lt_s = lt[order]

    # Per-group cumulative counts of label==2 and label==1, BEFORE current row.
    # group_starts: where iid changes
    grp_change = np.concatenate(([True], iid_s[1:] != iid_s[:-1]))
    # within-group index (0 = first occurrence of that item)
    grp_id = np.cumsum(grp_change) - 1
    # cumulative count of label==2 within group, exclusive of current row
    is_lab2 = (lt_s == 2).astype(np.int64)
    is_lab1 = (lt_s == 1).astype(np.int64)
    is_lab0 = ((lt_s != 1) & (lt_s != 2)).astype(np.int64)

    # within-group cumulative sums (inclusive), then shift right by 1 within group
    def cum_excl(values: np.ndarray) -> np.ndarray:
        cum = np.cumsum(values)
        # subtract the group's starting offset (cum value just before group start)
        # equivalent: for each row, prior_in_group = cum[i] - values[i] - cum[group_start-1]
        # Compute group offset = cum[group_start_idx-1]
        gs_idx = np.where(grp_change)[0]
        # cum_at_start_minus1[k] = cum[gs_idx[k]-1] if k>0 else 0
        offsets = np.zeros(len(gs_idx), dtype=np.int64)
        offsets[1:] = cum[gs_idx[1:] - 1]
        # per-row offset
        per_row_offset = offsets[grp_id]
        inclusive_in_grp = cum - per_row_offset
        exclusive_in_grp = inclusive_in_grp - values
        return exclusive_in_grp

    prior_pos = cum_excl(is_lab2)
    prior_lab1 = cum_excl(is_lab1)
    prior_neg = cum_excl(is_lab0)
    prior_all = prior_pos + prior_lab1 + prior_neg

    # Row-view distributions
    def percentiles(x: np.ndarray, name: str) -> None:
        ps = [0, 25, 50, 75, 90, 95, 99, 99.9, 100]
        vals = np.percentile(x, ps)
        s = "  ".join(f"p{p}={int(v)}" for p, v in zip(ps, vals))
        print(f"  [{name}]  mean={x.mean():.2f}  {s}")

    def bucket_hist(x: np.ndarray, name: str) -> None:
        edges = [0, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 10**9]
        labels = [
            "0", "1", "2-4", "5-9", "10-19", "20-49",
            "50-99", "100-499", "500-999", ">=1000",
        ]
        counts = np.zeros(len(labels), dtype=np.int64)
        # bucket index = first i s.t. x < edges[i+1]
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            counts[i] = int(((x >= lo) & (x < hi)).sum())
        total = counts.sum()
        print(f"  [{name}] row-view bucket histogram (n={total:,})")
        for lab, c in zip(labels, counts):
            pct = c / total * 100 if total else 0
            print(f"    {lab:>10s} : {c:>12,}  ({pct:5.2f}%)")

    print("\n" + "=" * 92)
    print("[B] PER-ITEM PRIOR INTERACTION COUNTS (at each row's timestamp)")
    print("=" * 92)
    print(f"  total rows = {N:,}")
    print(
        f"  rows with zero prior history (cold)  = "
        f"{int((prior_all == 0).sum()):,}  "
        f"({(prior_all == 0).mean() * 100:.2f}%)"
    )
    print(
        f"  rows with >=1 prior label==2 (pos)   = "
        f"{int((prior_pos >= 1).sum()):,}  "
        f"({(prior_pos >= 1).mean() * 100:.2f}%)"
    )
    print(
        f"  rows with >=1 prior label==1         = "
        f"{int((prior_lab1 >= 1).sum()):,}  "
        f"({(prior_lab1 >= 1).mean() * 100:.2f}%)"
    )
    print(
        f"  rows with >=1 prior label==0/other   = "
        f"{int((prior_neg >= 1).sum()):,}  "
        f"({(prior_neg >= 1).mean() * 100:.2f}%)"
    )

    print("\n-- percentiles over rows --")
    percentiles(prior_pos, "prior_label==2 (positive)")
    percentiles(prior_lab1, "prior_label==1")
    percentiles(prior_neg, "prior_label==other/0")
    percentiles(prior_all, "prior_total")

    print("\n-- distributions --")
    bucket_hist(prior_pos, "prior_label==2")
    bucket_hist(prior_lab1, "prior_label==1")
    bucket_hist(prior_neg, "prior_label==other/0")

    # Suggestion of K caps
    print("\n-- coverage of fixed-K truncation (positive column) --")
    for K in (4, 8, 16, 32, 64, 128):
        cap = np.minimum(prior_pos, K)
        retain = cap.sum() / max(prior_pos.sum(), 1) * 100
        print(
            f"    K={K:>4d}: rows-with-full-pool={(prior_pos <= K).mean() * 100:5.2f}%  "
            f"interactions-retained={retain:5.2f}%"
        )
    print("\n-- coverage of fixed-K truncation (label==1 column) --")
    for K in (4, 8, 16, 32, 64, 128):
        cap = np.minimum(prior_lab1, K)
        retain = cap.sum() / max(prior_lab1.sum(), 1) * 100
        print(
            f"    K={K:>4d}: rows-with-full-pool={(prior_lab1 <= K).mean() * 100:5.2f}%  "
            f"interactions-retained={retain:5.2f}%"
        )


# ─────────────────────── main ─────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data_dir",
        default=os.environ.get("TRAIN_DATA_PATH", ""),
        help="Directory of *.parquet (env TRAIN_DATA_PATH).",
    )
    ap.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Limit number of parquet files (0 = all).",
    )
    ap.add_argument("--skip_a", action="store_true", help="Skip feature-stability EDA")
    ap.add_argument("--skip_b", action="store_true", help="Skip item-history EDA")
    args = ap.parse_args()

    if not args.data_dir:
        raise SystemExit("--data_dir or env TRAIN_DATA_PATH required")

    files = list_parquet_files(args.data_dir, args.max_files)
    logging.info(f"data_dir = {args.data_dir}")
    logging.info(f"using {len(files)} parquet file(s)")

    if not args.skip_a:
        report_feature_stability(files)
    if not args.skip_b:
        report_item_history(files)

    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
