#!/usr/bin/env python3
"""Build an item-keyed history table over the training interactions.

For every training row r with (item_id, timestamp, label_type, user_*) we
record one interaction entry. Entries are grouped by item_id and sorted by
timestamp ascending. Dataset / inference code looks up an item slice at
runtime and samples pos/neg historical users with a temporal cut, so the
sampling policy (k_pos / k_neg / time_gap) lives at runtime.

Each historical interaction stores the FULL user-side feature set so the
hist module can encode historical users through the same path that backbone
uses for the current user — strict Q/KV symmetry. Specifically:

  int_user_int      (M, user_int_total_dim)   int32  — non-pair user_int fids,
                                                       laid out exactly as
                                                       dataset.user_int_schema
  int_user_dense    (M, user_dense_total_dim) fp16   — non-pair user_dense fids
  int_pair_int      (M, pair_int_total_dim)   int32  — pair user-item int crosses
  int_pair_dense    (M, pair_dense_total_dim) fp16   — pair user-item dense crosses

  item_ids   (I,)   int64  — sorted unique item ids
  offsets    (I+1,) int64  — CSR offsets into the (M,) arrays
  int_ts     (M,)   int32  — Unix-second timestamp per event (ascending within item)
  int_label  (M,)   int8   — label_type (1 = shown-no-conv, 2 = converted)

Within each item slice [offsets[i] : offsets[i+1]) the rows are guaranteed
sorted by timestamp ascending. The temporal split between train and valid
is irrelevant here — the file is built over the full TRAIN_DATA_PATH; at
infer time the test row's ts > all train ts, so the temporal cut at runtime
trivially exposes the full training history.

The output is wiped at the start of every run (see main()) so switching
schemas doesn't leave orphan arrays behind.
"""
import argparse
import glob
import json
import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Reuse dataset constants + helpers so build/runtime are guaranteed in sync.
from dataset import PAIR_FEATURES, pad_varlen_int_jit, pad_varlen_float_jit


def list_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no .parquet in {data_dir}")
    return files


def _as_single_array(col: "pa.ChunkedArray | pa.Array") -> "pa.Array":
    if isinstance(col, pa.ChunkedArray):
        if col.num_chunks == 1:
            return col.chunk(0)
        return pa.concat_arrays(col.chunks)
    return col


# ---------------------------------------------------------------------------
# Schema parsing: derive the same per-fid (offset, length) layout that
# dataset._load_schema produces, so build artifacts plug 1:1 into the live
# dataset's user_int / user_dense / pair_int / pair_dense buffers.
# ---------------------------------------------------------------------------

class SchemaPlan:
    """Plan for filling one of the four buffers (user_int / pair_int / user_dense / pair_dense)."""

    def __init__(self) -> None:
        # Each entry: (fid, vocab_size, length, offset). vocab_size == 0 means
        # no clip; for dense buffers vocab_size is meaningless and stored as 0.
        self.entries: List[Tuple[int, int, int, int]] = []
        self.total_dim: int = 0

    def add(self, fid: int, vs: int, length: int) -> None:
        self.entries.append((fid, vs, int(length), self.total_dim))
        self.total_dim += int(length)


def parse_schema(schema_path: str) -> Tuple[SchemaPlan, SchemaPlan, SchemaPlan, SchemaPlan]:
    """Parse schema.json and return four plans laid out exactly as
    dataset.PCVRParquetDataset._load_schema does."""
    with open(schema_path, "r") as f:
        raw = json.load(f)

    user_int_plan = SchemaPlan()
    pair_int_plan = SchemaPlan()
    for fid, vs, dim in raw["user_int"]:
        if fid in PAIR_FEATURES:
            pair_int_plan.add(int(fid), int(vs), int(dim))
        else:
            user_int_plan.add(int(fid), int(vs), int(dim))

    user_dense_plan = SchemaPlan()
    pair_dense_plan = SchemaPlan()
    for fid, dim in raw["user_dense"]:
        if fid in PAIR_FEATURES:
            pair_dense_plan.add(int(fid), 0, int(dim))
        else:
            user_dense_plan.add(int(fid), 0, int(dim))

    return user_int_plan, pair_int_plan, user_dense_plan, pair_dense_plan


# ---------------------------------------------------------------------------
# Column-level extractors that mirror dataset._convert_batch behavior:
#   - null / non-positive int → 0
#   - vocab clip when vs > 0 (else force to 0)
#   - list<int>  → pad to length
#   - list<float>→ pad to length, write directly
# ---------------------------------------------------------------------------

def _fill_int_column(
    tbl: "pa.Table",
    col_name: str,
    vs: int,
    length: int,
    out: np.ndarray,
    out_offset: int,
    col_offset: int,
) -> None:
    """Write one int column (scalar or multi-hot) into out[row_off : row_off+B,
    col_off : col_off+length]. Null/<=0 → 0; out-of-vocab → 0; ``vs<=0`` zeros
    the whole slice (no-vocab feature)."""
    col = tbl.column(col_name)
    B = tbl.num_rows
    if vs <= 0:
        return  # out already zero; nothing to write
    if length == 1:
        arr = (
            col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
        )
        arr[arr <= 0] = 0
        arr[arr >= vs] = 0
        out[out_offset:out_offset + B, col_offset] = arr.astype(np.int32)
    else:
        list_arr = _as_single_array(col)
        offsets = list_arr.offsets.to_numpy()
        values = list_arr.values.to_numpy()
        padded, _ = pad_varlen_int_jit(offsets, values, length)  # (B, length) int64
        padded[padded >= vs] = 0
        out[out_offset:out_offset + B, col_offset:col_offset + length] = padded.astype(np.int32)


def _fill_dense_column(
    tbl: "pa.Table",
    col_name: str,
    length: int,
    out: np.ndarray,
    out_offset: int,
    col_offset: int,
) -> None:
    """Write one dense (list<float>) column into out[row_off : row_off+B,
    col_off : col_off+length]. Missing entries are left at 0."""
    list_arr = _as_single_array(tbl.column(col_name))
    offsets = list_arr.offsets.to_numpy()
    values = list_arr.values.to_numpy()
    padded = pad_varlen_float_jit(offsets, values, length)        # (B, length) fp32
    B = tbl.num_rows
    out[out_offset:out_offset + B, col_offset:col_offset + length] = padded.astype(np.float16)


# ---------------------------------------------------------------------------
# Main scan: produce per-interaction arrays in parquet read order.
# ---------------------------------------------------------------------------

def scan_interactions(
    files: List[str],
    total_rows: int,
    user_int_plan: SchemaPlan,
    pair_int_plan: SchemaPlan,
    user_dense_plan: SchemaPlan,
    pair_dense_plan: SchemaPlan,
) -> Dict[str, np.ndarray]:
    needed: List[str] = []
    for fid, _, _, _ in user_int_plan.entries:
        needed.append(f"user_int_feats_{fid}")
    for fid, _, _, _ in pair_int_plan.entries:
        needed.append(f"user_int_feats_{fid}")
    for fid, _, _, _ in user_dense_plan.entries:
        needed.append(f"user_dense_feats_{fid}")
    for fid, _, _, _ in pair_dense_plan.entries:
        needed.append(f"user_dense_feats_{fid}")
    needed += ["item_id", "timestamp", "label_type"]

    out = {
        "item_ids":   np.zeros(total_rows, dtype=np.int64),
        "timestamps": np.zeros(total_rows, dtype=np.int64),
        "labels":     np.zeros(total_rows, dtype=np.int8),
        "user_int":   np.zeros((total_rows, user_int_plan.total_dim),   dtype=np.int32),
        "pair_int":   np.zeros((total_rows, pair_int_plan.total_dim),   dtype=np.int32),
        "user_dense": np.zeros((total_rows, user_dense_plan.total_dim), dtype=np.float16),
        "pair_dense": np.zeros((total_rows, pair_dense_plan.total_dim), dtype=np.float16),
    }

    offset = 0
    t_per_log = time.time()
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        names = pf.schema_arrow.names
        for col in needed:
            if col not in names:
                raise ValueError(f"{f}: missing column {col}")

        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=needed)
            B = tbl.num_rows

            for fid, vs, length, col_off in user_int_plan.entries:
                _fill_int_column(tbl, f"user_int_feats_{fid}", vs, length,
                                 out["user_int"], offset, col_off)
            for fid, vs, length, col_off in pair_int_plan.entries:
                _fill_int_column(tbl, f"user_int_feats_{fid}", vs, length,
                                 out["pair_int"], offset, col_off)
            for fid, _, length, col_off in user_dense_plan.entries:
                _fill_dense_column(tbl, f"user_dense_feats_{fid}", length,
                                   out["user_dense"], offset, col_off)
            for fid, _, length, col_off in pair_dense_plan.entries:
                _fill_dense_column(tbl, f"user_dense_feats_{fid}", length,
                                   out["pair_dense"], offset, col_off)

            out["item_ids"][offset:offset + B] = (
                tbl.column("item_id").fill_null(0)
                .to_numpy(zero_copy_only=False).astype(np.int64)
            )
            out["timestamps"][offset:offset + B] = (
                tbl.column("timestamp").fill_null(0)
                .to_numpy(zero_copy_only=False).astype(np.int64)
            )
            out["labels"][offset:offset + B] = (
                tbl.column("label_type").fill_null(0)
                .to_numpy(zero_copy_only=False).astype(np.int8)
            )

            offset += B

        if (fi + 1) % 100 == 0:
            logging.info(
                f"scan: {fi+1}/{len(files)} files, rows={offset:,}, "
                f"dt={time.time()-t_per_log:.1f}s"
            )
            t_per_log = time.time()

    assert offset == total_rows, (offset, total_rows)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.environ.get("TRAIN_DATA_PATH"))
    parser.add_argument(
        "--out_dir", required=True,
        help="output directory, typically $USER_CACHE_PATH/item_hist_${HIST_TAG}",
    )
    parser.add_argument(
        "--schema_path",
        default=None,
        help="schema.json path (defaults to <data_dir>/schema.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.data_dir:
        raise ValueError("--data_dir (or $TRAIN_DATA_PATH) required")
    schema_path = args.schema_path or os.path.join(args.data_dir, "schema.json")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"missing schema.json at {schema_path}")

    os.makedirs(args.out_dir, exist_ok=True)
    # Wipe stale artifacts (different schemas leave orphan .npy that would
    # confuse the loader).
    _OWNED = {"meta.json"}
    for name in os.listdir(args.out_dir):
        if name.endswith(".npy") or name in _OWNED:
            os.remove(os.path.join(args.out_dir, name))
    logging.info(f"cleaned stale artifacts in {args.out_dir}")

    user_int_plan, pair_int_plan, user_dense_plan, pair_dense_plan = parse_schema(schema_path)
    logging.info(
        f"schema: user_int={user_int_plan.total_dim} (n_fids={len(user_int_plan.entries)}), "
        f"pair_int={pair_int_plan.total_dim} (n_fids={len(pair_int_plan.entries)}), "
        f"user_dense={user_dense_plan.total_dim} (n_fids={len(user_dense_plan.entries)}), "
        f"pair_dense={pair_dense_plan.total_dim} (n_fids={len(pair_dense_plan.entries)})"
    )

    files = list_files(args.data_dir)
    total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    logging.info(
        f"data_dir={args.data_dir}, files={len(files)}, total_rows={total_rows:,}"
    )

    t0 = time.time()
    arr = scan_interactions(
        files, total_rows,
        user_int_plan, pair_int_plan, user_dense_plan, pair_dense_plan,
    )
    logging.info(f"scan done in {time.time()-t0:.1f}s")

    # Drop entries with item_id == 0 (treated as missing/padding).
    keep = arr["item_ids"] > 0
    if not keep.all():
        dropped = int((~keep).sum())
        logging.info(f"dropping {dropped:,} rows with item_id<=0")
        for k in arr:
            arr[k] = arr[k][keep]

    # Sort globally by (item_id, ts asc). lexsort uses LAST key as primary.
    t0 = time.time()
    order = np.lexsort((arr["timestamps"], arr["item_ids"]))
    item_ids   = arr["item_ids"][order]
    timestamps = arr["timestamps"][order].astype(np.int32, copy=False)
    labels     = arr["labels"][order]
    user_int   = arr["user_int"][order]
    pair_int   = arr["pair_int"][order]
    user_dense = arr["user_dense"][order]
    pair_dense = arr["pair_dense"][order]
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

    np.save(os.path.join(args.out_dir, "item_ids.npy"),       unique_items.astype(np.int64))
    np.save(os.path.join(args.out_dir, "offsets.npy"),        offsets)
    np.save(os.path.join(args.out_dir, "int_ts.npy"),         timestamps)
    np.save(os.path.join(args.out_dir, "int_label.npy"),      labels)
    np.save(os.path.join(args.out_dir, "int_user_int.npy"),   user_int)
    np.save(os.path.join(args.out_dir, "int_pair_int.npy"),   pair_int)
    np.save(os.path.join(args.out_dir, "int_user_dense.npy"), user_dense)
    np.save(os.path.join(args.out_dir, "int_pair_dense.npy"), pair_dense)

    def plan_to_meta(plan: SchemaPlan) -> Dict:
        return {
            "total_dim": plan.total_dim,
            "entries": [
                {"fid": fid, "vocab_size": vs, "length": length, "offset": off}
                for (fid, vs, length, off) in plan.entries
            ],
        }

    meta = {
        "num_items": int(num_items),
        "num_interactions": int(num_interactions),
        "user_int":   plan_to_meta(user_int_plan),
        "pair_int":   plan_to_meta(pair_int_plan),
        "user_dense": plan_to_meta(user_dense_plan),
        "pair_dense": plan_to_meta(pair_dense_plan),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    # Per-item pos/neg coverage stats — helps tune k_pos / k_neg at runtime.
    pos_per_item = np.zeros(num_items, dtype=np.int64)
    neg_per_item = np.zeros(num_items, dtype=np.int64)
    grp = np.repeat(np.arange(num_items), np.diff(offsets))
    np.add.at(pos_per_item, grp, (labels == 2).astype(np.int64))
    np.add.at(neg_per_item, grp, (labels == 1).astype(np.int64))
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
