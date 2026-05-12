#!/usr/bin/env python3
"""Build per-row item-history-user pools for the PCVRHyFormer hist module.

For every row r with (item_id=X, timestamp=T) this script records two pools of
past interactions with item X:

  pos pool (label_type==2 / conversion):
      take all if |pool|<=k_pos, else (latest k_pos/2) + (random k_pos/2 from rest)
  neg pool (label_type==1 / shown-but-no-conversion):
      take all if |pool|<=k_neg, else time-decay weighted reservoir of k_neg

Both pools enforce a hard temporal cut: only entries with ts < T - time_gap are
considered, to block target-encoding leakage. The same per-row temporal filter
is applied uniformly to every row regardless of train/valid membership, so a
valid sample never sees a future train interaction and vice-versa.

Output (typically under ``$USER_CACHE_PATH/item_hist_${HIST_TAG}/``):
  user_lookup_scalars.npy   (N_total, 7)    int32   ← 7 stable user scalars
  user_lookup_dense61.npy   (N_total, 256)  float16 ← user_dense_feats_61
  hist_pos_indices.npy      (N_total, k_pos) int32  (-1 padding)
  hist_neg_indices.npy      (N_total, k_neg) int32
  hist_pos_lens.npy         (N_total,)      int8
  hist_neg_lens.npy         (N_total,)      int8
  meta.json                 row-group layout, hyperparams

dataset.py mmap-loads these files and recomputes per-batch global_row_idx by
matching its own (sorted basename, rg_idx) iteration against meta.json.
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


# Selected user features (chosen from EDA on present% >= 99.5%; PAIR features
# 62-66/89-91 excluded because they are user-item crosses, not user identity).
USER_SCALAR_FIDS: List[int] = [1, 48, 49, 50, 51, 52, 53]
USER_DENSE_FID: int = 61
USER_DENSE_DIM: int = 256


def list_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no .parquet in {data_dir}")
    return files


def _as_single_array(col: "pa.ChunkedArray | pa.Array") -> "pa.Array":
    """Return a single Arrow Array regardless of whether ``col`` came in as a
    ChunkedArray (from ``Table.column``) or as a plain Array (from
    ``RecordBatch.column``). ``read_row_group`` returns a Table whose columns
    are always ChunkedArrays, which is why this helper is needed.
    """
    if isinstance(col, pa.ChunkedArray):
        if col.num_chunks == 1:
            return col.chunk(0)
        return pa.concat_arrays(col.chunks)
    return col


def read_pass1(
    files: List[str],
    out_scalars: np.ndarray,
    out_dense: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, int, int, int]]]:
    """First pass: dump per-row (scalars, dense61) and collect (item, ts, lt)."""
    scalar_cols = [f"user_int_feats_{fid}" for fid in USER_SCALAR_FIDS]
    dense_col = f"user_dense_feats_{USER_DENSE_FID}"
    needed_cols = scalar_cols + [dense_col, "item_id", "timestamp", "label_type"]

    iid_all, ts_all, lt_all = [], [], []
    rg_layout: List[Tuple[str, int, int, int]] = []
    offset = 0

    for f in files:
        pf = pq.ParquetFile(f)
        names = pf.schema_arrow.names
        for col in needed_cols:
            if col not in names:
                raise ValueError(f"{f}: missing column {col}")

        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=needed_cols)
            B = tbl.num_rows

            # 7 user scalars: null/<0 → 0, cast to int32
            for j, col_name in enumerate(scalar_cols):
                arr = (
                    tbl.column(col_name)
                    .fill_null(0)
                    .to_numpy(zero_copy_only=False)
                    .astype(np.int64)
                )
                arr[arr < 0] = 0
                out_scalars[offset:offset + B, j] = arr.astype(np.int32)

            # dense_61: list<float> of length 256 per row (or null).
            # Table.column returns a ChunkedArray; flatten to a single
            # ListArray so .offsets / .values are addressable.
            dcol = _as_single_array(tbl.column(dense_col))
            offs = dcol.offsets.to_numpy()
            vals = dcol.values.to_numpy().astype(np.float32, copy=False)
            for i in range(B):
                s, e = int(offs[i]), int(offs[i + 1])
                n = min(e - s, USER_DENSE_DIM)
                if n > 0:
                    out_dense[offset + i, :n] = vals[s:s + n].astype(np.float16)

            iid_all.append(
                tbl.column("item_id").fill_null(0)
                .to_numpy(zero_copy_only=False).astype(np.int64)
            )
            ts_all.append(
                tbl.column("timestamp").fill_null(0)
                .to_numpy(zero_copy_only=False).astype(np.int64)
            )
            lt_all.append(
                tbl.column("label_type").fill_null(0)
                .to_numpy(zero_copy_only=False).astype(np.int8)
            )

            rg_layout.append((os.path.basename(f), rg, offset, B))
            offset += B

    return (
        np.concatenate(iid_all),
        np.concatenate(ts_all),
        np.concatenate(lt_all),
        rg_layout,
    )


def _weighted_reservoir(values: np.ndarray, weights: np.ndarray, k: int,
                       rng: np.random.Generator) -> np.ndarray:
    """A-Res weighted reservoir sampling: keys = u^(1/w), keep top-k by key."""
    n = len(values)
    if n <= k:
        return values
    eps = 1e-12
    u = np.clip(rng.random(n), eps, 1.0)
    w = np.maximum(weights.astype(np.float64), eps)
    log_keys = np.log(u) / w
    top_idx = np.argpartition(log_keys, -k)[-k:]
    return values[top_idx]


def build_pass2(
    item_ids: np.ndarray,
    timestamps: np.ndarray,
    label_types: np.ndarray,
    k_pos: int,
    k_neg: int,
    time_gap: int,
    tau: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Second pass: per-row (pos_indices, neg_indices) via the agreed sampling."""
    N = len(item_ids)
    rng = np.random.default_rng(seed)

    # Sort all rows by (item_id, timestamp) so each item's rows are contiguous
    # and chronological. Use stable np.lexsort: last key is primary.
    order = np.lexsort((timestamps, item_ids))
    iid_s = item_ids[order]
    ts_s = timestamps[order]
    lt_s = label_types[order]
    ridx_s = order.astype(np.int32)  # global row index in sorted position

    grp_change = np.concatenate(([True], iid_s[1:] != iid_s[:-1]))
    grp_starts = np.where(grp_change)[0]
    grp_ends = np.concatenate((grp_starts[1:], [N]))

    hist_pos_idx = np.full((N, k_pos), -1, dtype=np.int32)
    hist_neg_idx = np.full((N, k_neg), -1, dtype=np.int32)
    hist_pos_len = np.zeros(N, dtype=np.int8)
    hist_neg_len = np.zeros(N, dtype=np.int8)

    half_pos = k_pos // 2  # latest half + random half from the rest

    t0 = time.time()
    n_groups = len(grp_starts)
    for gi in range(n_groups):
        gs, ge = grp_starts[gi], grp_ends[gi]
        ts_grp = ts_s[gs:ge]
        lt_grp = lt_s[gs:ge]
        ridx_grp = ridx_s[gs:ge]
        n = ge - gs

        pos_local = np.where(lt_grp == 2)[0]
        neg_local = np.where(lt_grp == 1)[0]
        # ts and global row idx for label==2 / label==1 entries (already sorted
        # ascending by ts because ts_grp is)
        pos_ts = ts_grp[pos_local]
        neg_ts = ts_grp[neg_local]
        pos_g = ridx_grp[pos_local]
        neg_g = ridx_grp[neg_local]

        for j in range(n):
            cur_ts = int(ts_grp[j])
            cutoff = cur_ts - time_gap
            row = int(ridx_grp[j])

            # pos pool
            p_end = int(np.searchsorted(pos_ts, cutoff, side="left"))
            if p_end > 0:
                p_idx = pos_g[:p_end]
                if p_end <= k_pos:
                    chosen = p_idx
                else:
                    latest = p_idx[-half_pos:]
                    rest = p_idx[:-half_pos]
                    pick_k = k_pos - half_pos
                    if rest.size <= pick_k:
                        chosen = np.concatenate([rest, latest])
                    else:
                        picked = rng.choice(rest.size, size=pick_k, replace=False)
                        chosen = np.concatenate([rest[picked], latest])
                kk = chosen.size
                hist_pos_idx[row, :kk] = chosen
                hist_pos_len[row] = kk

            # neg pool
            q_end = int(np.searchsorted(neg_ts, cutoff, side="left"))
            if q_end > 0:
                q_idx = neg_g[:q_end]
                if q_end <= k_neg:
                    chosen = q_idx
                else:
                    q_t = neg_ts[:q_end]
                    weights = np.exp(-(cur_ts - q_t).astype(np.float64) / tau)
                    chosen = _weighted_reservoir(q_idx, weights, k_neg, rng)
                kk = chosen.size
                hist_neg_idx[row, :kk] = chosen
                hist_neg_len[row] = kk

        if (gi + 1) % 20000 == 0:
            logging.info(
                f"pass2: {gi+1}/{n_groups} items, elapsed={time.time()-t0:.1f}s"
            )

    return hist_pos_idx, hist_neg_idx, hist_pos_len, hist_neg_len


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.environ.get("TRAIN_DATA_PATH"))
    parser.add_argument(
        "--out_dir", required=True,
        help="output directory, typically $USER_CACHE_PATH/item_hist_${HIST_TAG}",
    )
    parser.add_argument("--k_pos", type=int, default=16)
    parser.add_argument("--k_neg", type=int, default=32)
    parser.add_argument(
        "--time_gap", type=int, default=3600,
        help="seconds; only hist with ts < t_sample - gap is used",
    )
    parser.add_argument(
        "--tau", type=float, default=7 * 86400.0,
        help="decay constant (seconds) for neg-pool weighted reservoir",
    )
    parser.add_argument("--seed", type=int, default=42)
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
    total_rows = 0
    for f in files:
        pf = pq.ParquetFile(f)
        total_rows += pf.metadata.num_rows
    logging.info(
        f"data_dir={args.data_dir}, files={len(files)}, total_rows={total_rows:,}"
    )
    logging.info(
        f"k_pos={args.k_pos}, k_neg={args.k_neg}, time_gap={args.time_gap}s, "
        f"tau={args.tau:.0f}s"
    )

    out_scalars = np.zeros((total_rows, len(USER_SCALAR_FIDS)), dtype=np.int32)
    out_dense = np.zeros((total_rows, USER_DENSE_DIM), dtype=np.float16)

    t0 = time.time()
    item_ids, timestamps, label_types, rg_layout = read_pass1(
        files, out_scalars, out_dense
    )
    logging.info(f"pass1 done in {time.time()-t0:.1f}s")

    np.save(os.path.join(args.out_dir, "user_lookup_scalars.npy"), out_scalars)
    np.save(os.path.join(args.out_dir, "user_lookup_dense61.npy"), out_dense)
    del out_scalars, out_dense
    logging.info("saved user_lookup_*.npy")

    t0 = time.time()
    hist_pos_idx, hist_neg_idx, hist_pos_len, hist_neg_len = build_pass2(
        item_ids, timestamps, label_types,
        k_pos=args.k_pos, k_neg=args.k_neg,
        time_gap=args.time_gap, tau=args.tau, seed=args.seed,
    )
    logging.info(f"pass2 done in {time.time()-t0:.1f}s")

    np.save(os.path.join(args.out_dir, "hist_pos_indices.npy"), hist_pos_idx)
    np.save(os.path.join(args.out_dir, "hist_neg_indices.npy"), hist_neg_idx)
    np.save(os.path.join(args.out_dir, "hist_pos_lens.npy"), hist_pos_len)
    np.save(os.path.join(args.out_dir, "hist_neg_lens.npy"), hist_neg_len)

    meta = {
        "total_rows": int(total_rows),
        "k_pos": int(args.k_pos),
        "k_neg": int(args.k_neg),
        "time_gap": int(args.time_gap),
        "tau": float(args.tau),
        "scalar_fids": USER_SCALAR_FIDS,
        "dense_fid": USER_DENSE_FID,
        "dense_dim": USER_DENSE_DIM,
        "rg_layout": [
            {"file": b, "rg": r, "row_start": s, "num_rows": n}
            for (b, r, s, n) in rg_layout
        ],
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    n_cold_pos = int((hist_pos_len == 0).sum())
    n_cold_neg = int((hist_neg_len == 0).sum())
    logging.info(
        f"summary: rows={total_rows:,}, "
        f"pos: mean={float(hist_pos_len.mean()):.2f} "
        f"cold={n_cold_pos:,} ({n_cold_pos/total_rows*100:.2f}%), "
        f"neg: mean={float(hist_neg_len.mean()):.2f} "
        f"cold={n_cold_neg:,} ({n_cold_neg/total_rows*100:.2f}%)"
    )
    logging.info(f"done. output: {args.out_dir}")


if __name__ == "__main__":
    main()
