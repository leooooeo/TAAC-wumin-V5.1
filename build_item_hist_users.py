#!/usr/bin/env python3
"""Build an item-keyed history table over the training interactions.

For every training row r with (item_id, timestamp, label_type, user_*) we
record a single interaction entry. Entries are grouped by item_id and sorted
by timestamp ascending. Dataset / inference code then looks up an item slice
at runtime and samples pos/neg historical users with a temporal cut, so the
sampling policy (k_pos, k_neg, time_gap, etc.) lives at runtime — not here.

Output directory layout
-----------------------
  item_ids.npy           (I,)       int64   — unique item ids, sorted ascending
  offsets.npy            (I+1,)     int64   — CSR offsets into the (M,)-arrays
  int_ts.npy             (M,)       int32   — Unix-second timestamp of each event
  int_label.npy          (M,)       int8    — label_type (1 = shown-no-conv, 2 = conv)
  int_user_scalars.npy   (M, 12)    int32   — 12 stable user scalars
  int_user_dense61.npy   (M, 256)   float16 — user_dense_feats_61 (ID embedding)
  int_user_dense87.npy   (M, 320)   float16 — user_dense_feats_87 (behavior emb)
  meta.json                          — schema info (scalar fids, dense fids/dims, …)

The two dense files are kept separate (not concatenated) so the model can
apply two independent projection heads — fid=61 is a collaborative ID
embedding, fid=87 is the user's behavior-history embedding; they live in
different semantic spaces and benefit from their own LayerNorm/Linear.

Within each item slice [offsets[i] : offsets[i+1]) the rows are guaranteed
sorted by timestamp ascending; ``int_label`` distinguishes pos (==2) from
neg (==1) entries. The same file is used by train AND infer — inference rows'
timestamps are later than every training timestamp, so the temporal cut at
runtime trivially exposes the full training history.

Inference rows are NOT written here. The intentional property is: the file
contains only training-set interactions; test rows query it by item_id.
"""
import argparse
import glob
import json
import logging
import os
import time
from typing import List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# Selected user features (chosen from EDA):
#   - 12 scalars with present% >= 97% — pure user identity / demographics
#   - dense_61 (256d): collaborative ID embedding, 99.87% present
#   - dense_87 (320d): user behavior-history embedding, 99.35% present
# PAIR features 62-66/89-91 are excluded because they are user-item crosses,
# not user identity. fid 89 multi-hot is held back as a future M2 add-on.
USER_SCALAR_FIDS: List[int] = [1, 3, 4, 48, 49, 50, 51, 52, 53, 55, 56, 57]
USER_DENSE_FID_61: int = 61
USER_DENSE_DIM_61: int = 256
USER_DENSE_FID_87: int = 87
USER_DENSE_DIM_87: int = 320


def list_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no .parquet in {data_dir}")
    return files


def _as_single_array(col: "pa.ChunkedArray | pa.Array") -> "pa.Array":
    """Flatten a Table column (always ChunkedArray) to a single Array so
    ``.offsets`` / ``.values`` are addressable."""
    if isinstance(col, pa.ChunkedArray):
        if col.num_chunks == 1:
            return col.chunk(0)
        return pa.concat_arrays(col.chunks)
    return col


def _fill_dense_list_col(
    tbl: "pa.Table",
    col_name: str,
    out: np.ndarray,
    out_offset: int,
    dim: int,
) -> None:
    """Materialize a list<float> column into a slice of ``out[out_offset:, :dim]``,
    truncating to ``dim`` and downcasting to float16."""
    dcol = _as_single_array(tbl.column(col_name))
    offs = dcol.offsets.to_numpy()
    vals = dcol.values.to_numpy().astype(np.float32, copy=False)
    B = tbl.num_rows
    for i in range(B):
        s, e = int(offs[i]), int(offs[i + 1])
        n = min(e - s, dim)
        if n > 0:
            out[out_offset + i, :n] = vals[s:s + n].astype(np.float16)


def scan_interactions(
    files: List[str],
    total_rows: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single pass over the training parquets. Returns six flat arrays of
    length N (= total_rows), in parquet read order."""
    scalar_cols = [f"user_int_feats_{fid}" for fid in USER_SCALAR_FIDS]
    dense_col_61 = f"user_dense_feats_{USER_DENSE_FID_61}"
    dense_col_87 = f"user_dense_feats_{USER_DENSE_FID_87}"
    needed = scalar_cols + [
        dense_col_61, dense_col_87, "item_id", "timestamp", "label_type",
    ]

    item_ids = np.zeros(total_rows, dtype=np.int64)
    timestamps = np.zeros(total_rows, dtype=np.int64)
    labels = np.zeros(total_rows, dtype=np.int8)
    user_scalars = np.zeros((total_rows, len(USER_SCALAR_FIDS)), dtype=np.int32)
    user_dense_61 = np.zeros((total_rows, USER_DENSE_DIM_61), dtype=np.float16)
    user_dense_87 = np.zeros((total_rows, USER_DENSE_DIM_87), dtype=np.float16)

    offset = 0
    for f in files:
        pf = pq.ParquetFile(f)
        names = pf.schema_arrow.names
        for col in needed:
            if col not in names:
                raise ValueError(f"{f}: missing column {col}")

        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=needed)
            B = tbl.num_rows

            for j, col_name in enumerate(scalar_cols):
                arr = (
                    tbl.column(col_name).fill_null(0)
                    .to_numpy(zero_copy_only=False).astype(np.int64)
                )
                arr[arr < 0] = 0
                user_scalars[offset:offset + B, j] = arr.astype(np.int32)

            _fill_dense_list_col(tbl, dense_col_61, user_dense_61, offset, USER_DENSE_DIM_61)
            _fill_dense_list_col(tbl, dense_col_87, user_dense_87, offset, USER_DENSE_DIM_87)

            item_ids[offset:offset + B] = (
                tbl.column("item_id").fill_null(0)
                .to_numpy(zero_copy_only=False).astype(np.int64)
            )
            timestamps[offset:offset + B] = (
                tbl.column("timestamp").fill_null(0)
                .to_numpy(zero_copy_only=False).astype(np.int64)
            )
            labels[offset:offset + B] = (
                tbl.column("label_type").fill_null(0)
                .to_numpy(zero_copy_only=False).astype(np.int8)
            )

            offset += B

    assert offset == total_rows, (offset, total_rows)
    return item_ids, timestamps, labels, user_scalars, user_dense_61, user_dense_87


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.environ.get("TRAIN_DATA_PATH"))
    parser.add_argument(
        "--out_dir", required=True,
        help="output directory, typically $USER_CACHE_PATH/item_hist",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.data_dir:
        raise ValueError("--data_dir (or $TRAIN_DATA_PATH) required")
    os.makedirs(args.out_dir, exist_ok=True)

    files = list_files(args.data_dir)
    total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    logging.info(
        f"data_dir={args.data_dir}, files={len(files)}, total_rows={total_rows:,}"
    )

    t0 = time.time()
    (item_ids, timestamps, labels, user_scalars,
     user_dense_61, user_dense_87) = scan_interactions(files, total_rows)
    logging.info(f"scan done in {time.time()-t0:.1f}s")

    # Drop entries with item_id == 0 (treated as missing/padding). Keeping them
    # would create a fake "item 0" with millions of unrelated users.
    keep = item_ids > 0
    if not keep.all():
        dropped = int((~keep).sum())
        logging.info(f"dropping {dropped:,} rows with item_id<=0")
        item_ids = item_ids[keep]
        timestamps = timestamps[keep]
        labels = labels[keep]
        user_scalars = user_scalars[keep]
        user_dense_61 = user_dense_61[keep]
        user_dense_87 = user_dense_87[keep]

    # Sort globally by (item_id, ts ascending). lexsort uses the LAST key as
    # the primary key, so put item_id last.
    t0 = time.time()
    order = np.lexsort((timestamps, item_ids))
    item_ids = item_ids[order]
    timestamps = timestamps[order].astype(np.int32, copy=False)
    labels = labels[order]
    user_scalars = user_scalars[order]
    user_dense_61 = user_dense_61[order]
    user_dense_87 = user_dense_87[order]
    logging.info(f"sort done in {time.time()-t0:.1f}s")

    # CSR offsets per unique item.
    unique_items, first_idx = np.unique(item_ids, return_index=True)
    offsets = np.append(first_idx, len(item_ids)).astype(np.int64)
    num_items = len(unique_items)
    num_interactions = len(item_ids)
    logging.info(
        f"items={num_items:,}, interactions={num_interactions:,}, "
        f"avg_per_item={num_interactions / max(num_items, 1):.1f}"
    )

    np.save(os.path.join(args.out_dir, "item_ids.npy"), unique_items.astype(np.int64))
    np.save(os.path.join(args.out_dir, "offsets.npy"), offsets)
    np.save(os.path.join(args.out_dir, "int_ts.npy"), timestamps)
    np.save(os.path.join(args.out_dir, "int_label.npy"), labels)
    np.save(os.path.join(args.out_dir, "int_user_scalars.npy"), user_scalars)
    np.save(os.path.join(args.out_dir, "int_user_dense61.npy"), user_dense_61)
    np.save(os.path.join(args.out_dir, "int_user_dense87.npy"), user_dense_87)

    meta = {
        "num_items": int(num_items),
        "num_interactions": int(num_interactions),
        "scalar_fids": USER_SCALAR_FIDS,
        "dense_fid_61": USER_DENSE_FID_61,
        "dense_dim_61": USER_DENSE_DIM_61,
        "dense_fid_87": USER_DENSE_FID_87,
        "dense_dim_87": USER_DENSE_DIM_87,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    # Per-item pos/neg coverage stats — helps tune k_pos / k_neg at runtime.
    pos_per_item = np.zeros(num_items, dtype=np.int64)
    neg_per_item = np.zeros(num_items, dtype=np.int64)
    is_pos = (labels == 2)
    is_neg = (labels == 1)
    # bincount by item-position (NOT item_id, which may be sparse).
    grp = np.repeat(np.arange(num_items), np.diff(offsets))
    np.add.at(pos_per_item, grp, is_pos.astype(np.int64))
    np.add.at(neg_per_item, grp, is_neg.astype(np.int64))
    logging.info(
        f"pos pool: mean={pos_per_item.mean():.1f}, "
        f"p50={int(np.median(pos_per_item))}, p95={int(np.percentile(pos_per_item, 95))}, "
        f"max={int(pos_per_item.max())}"
    )
    logging.info(
        f"neg pool: mean={neg_per_item.mean():.1f}, "
        f"p50={int(np.median(neg_per_item))}, p95={int(np.percentile(neg_per_item, 95))}, "
        f"max={int(neg_per_item.max())}"
    )
    logging.info(f"done. output: {args.out_dir}")


if __name__ == "__main__":
    main()
