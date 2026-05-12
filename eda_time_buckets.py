"""Time-diff bucket EDA for seq_a / seq_b / seq_c / seq_d.

Streams a sample of rows from the training parquet, computes
``time_diff = sample_ts - event_ts`` per sequence domain, and reports:

  1. Per-domain percentile summary (p1..p99.9) and basic stats.
  2. Current per-domain bucket occupancy + empty / saturated buckets.
  3. Proposed quantile-based boundaries (64 edges) per domain.

Env vars:
  TRAIN_DATA_PATH  parquet directory or single file
  EDA_SCHEMA_PATH  schema.json path (default: $TRAIN_DATA_PATH/schema.json)
  EDA_MAX_ROWS     row cap across all row groups (default: 200000)
  EDA_OUT_JSON     optional path to dump the full report as JSON
"""

import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq

# Reuse the production bucket boundaries so the report matches what the model sees.
from dataset import BUCKET_BOUNDARIES_BY_DOMAIN


PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]


def _human_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s/60:.1f}min"
    if s < 86400:
        return f"{s/3600:.2f}h"
    if s < 2592000:
        return f"{s/86400:.2f}d"
    if s < 31536000:
        return f"{s/2592000:.2f}mo"
    return f"{s/31536000:.2f}y"


def _resolve_parquet_files(path: str) -> List[str]:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet files in {path}")
        return files
    return [path]


def _load_schema(schema_path: str) -> Dict[str, Dict]:
    with open(schema_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    seq_cfg = raw["seq"]
    domains: Dict[str, Dict] = {}
    for domain, cfg in seq_cfg.items():
        ts_fid = cfg["ts_fid"]
        prefix = cfg["prefix"]
        domains[domain] = {
            "ts_col": f"{prefix}_{ts_fid}",
        }
    return domains


def _collect_time_diffs(
    parquet_files: List[str],
    domains: Dict[str, Dict],
    max_rows: int,
) -> Dict[str, np.ndarray]:
    """Stream parquet row-groups and accumulate flat time_diff arrays per domain."""
    needed_cols = ["timestamp"] + [d["ts_col"] for d in domains.values()]
    buf: Dict[str, list] = {d: [] for d in domains}
    rows_seen = 0

    for path in parquet_files:
        pf = pq.ParquetFile(path)
        # Filter columns that actually exist in this file's schema.
        schema_names = set(pf.schema_arrow.names)
        cols = [c for c in needed_cols if c in schema_names]
        if "timestamp" not in cols:
            raise KeyError(f"`timestamp` column missing from {path}")

        for rg in range(pf.metadata.num_row_groups):
            if rows_seen >= max_rows:
                break
            tbl = pf.read_row_group(rg, columns=cols)
            sample_ts = tbl.column("timestamp").to_numpy().astype(np.int64)
            n = sample_ts.shape[0]

            for domain, meta in domains.items():
                ts_col_name = meta["ts_col"]
                if ts_col_name not in schema_names:
                    continue
                # ``tbl.column(...)`` returns a ChunkedArray; collapse to a single
                # ListArray so we can read ``.offsets`` / ``.values`` directly.
                col = tbl.column(ts_col_name).combine_chunks()
                offsets = col.offsets.to_numpy()
                values = col.values.to_numpy().astype(np.int64)
                # Expand each row's event timestamps and broadcast sample_ts.
                row_idx = np.repeat(np.arange(n, dtype=np.int64),
                                    offsets[1:] - offsets[:-1])
                if values.size == 0:
                    continue
                diffs = sample_ts[row_idx] - values
                # Keep only valid event slots (event_ts > 0) and non-negative diffs.
                mask = (values > 0) & (diffs >= 0)
                if mask.any():
                    buf[domain].append(diffs[mask])

            rows_seen += n
        if rows_seen >= max_rows:
            break

    out: Dict[str, np.ndarray] = {}
    for domain, chunks in buf.items():
        out[domain] = (
            np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
        )
    return out, rows_seen


def _current_bucket_occupancy(
    diffs: np.ndarray, boundaries: np.ndarray
) -> np.ndarray:
    """Replicates dataset.py bucketing: searchsorted(left), clip, +1, padding=0."""
    if diffs.size == 0:
        return np.zeros(len(boundaries) + 1, dtype=np.int64)
    raw = np.searchsorted(boundaries, diffs, side="left")
    np.clip(raw, 0, len(boundaries) - 1, out=raw)
    buckets = raw + 1  # slot 0 reserved for padding
    counts = np.bincount(buckets, minlength=len(boundaries) + 1)
    return counts


def _quantile_boundaries(diffs: np.ndarray, n_edges: int = 64) -> List[int]:
    """Equal-frequency boundaries; dedupes collisions in low-resolution regions."""
    if diffs.size == 0:
        return []
    qs = np.linspace(0.0, 1.0, n_edges + 2)[1:-1]  # interior quantiles
    edges = np.quantile(diffs, qs).astype(np.int64)
    edges = np.unique(edges)
    return edges.tolist()


def _print_percentile_table(diffs: np.ndarray, name: str) -> Dict:
    print(f"\n=== {name}  (n_events={diffs.size:,}) ===")
    if diffs.size == 0:
        print("  (no valid events)")
        return {"n": 0}
    stats = {
        "n": int(diffs.size),
        "min": int(diffs.min()),
        "max": int(diffs.max()),
        "mean": float(diffs.mean()),
        "percentiles": {},
    }
    print(f"  min={stats['min']}s ({_human_seconds(stats['min'])})  "
          f"max={stats['max']}s ({_human_seconds(stats['max'])})  "
          f"mean={stats['mean']:.0f}s ({_human_seconds(stats['mean'])})")
    print("  percentiles:")
    for p in PERCENTILES:
        v = float(np.percentile(diffs, p))
        stats["percentiles"][str(p)] = v
        print(f"    p{p:>5}: {int(v):>12,d}s   ({_human_seconds(v)})")
    return stats


def _print_bucket_occupancy(
    diffs: np.ndarray,
    name: str,
    boundaries: np.ndarray,
    top_k: int = 10,
) -> Dict:
    counts = _current_bucket_occupancy(diffs, boundaries)
    total = counts.sum()
    n_buckets = len(counts)
    n_real = n_buckets - 1
    if total == 0:
        print(f"\n[{name}] bucket occupancy: empty.")
        return {"counts": counts.tolist()}

    pct = counts / total * 100
    empty = int((counts[1:] == 0).sum())  # ignore padding slot 0
    print(f"\n[{name}] current {n_buckets}-bucket occupancy "
          f"(non-padding empty buckets: {empty}/{n_real}):")
    order = np.argsort(counts)[::-1]
    print(f"  top {top_k} buckets by mass:")
    for rank, bidx in enumerate(order[:top_k]):
        if bidx == 0:
            continue
        # bucket b (1..n_real) covers [boundaries[b-2], boundaries[b-1])
        lo = 0 if bidx == 1 else int(boundaries[bidx - 2])
        if bidx - 1 < len(boundaries):
            hi = int(boundaries[bidx - 1])
            rng = f"[{_human_seconds(lo)}, {_human_seconds(hi)})"
        else:
            rng = f"[{_human_seconds(lo)}, +inf)"
        print(f"    #{rank+1:<2}  bucket {bidx:>2}  {rng:<28}  "
              f"count={int(counts[bidx]):>10,}  ({pct[bidx]:5.2f}%)")
    return {
        "counts": counts.tolist(),
        "empty_non_padding": empty,
        "n_buckets": n_buckets,
    }


def _print_quantile_proposal(diffs: np.ndarray, name: str) -> Dict:
    edges = _quantile_boundaries(diffs, n_edges=64)
    print(f"\n[{name}] proposed quantile-based boundaries "
          f"({len(edges)} unique edges, target 64):")
    # Print compactly: 8 per line, with human-readable trail every 16 edges.
    line: List[str] = []
    for i, e in enumerate(edges):
        line.append(f"{e:>10d}")
        if (i + 1) % 8 == 0:
            print("    " + ", ".join(line))
            line = []
    if line:
        print("    " + ", ".join(line))
    if edges:
        print(f"  span: {_human_seconds(edges[0])}  ->  {_human_seconds(edges[-1])}")
    return {"edges": edges}


def main() -> int:
    data_path = os.environ.get("TRAIN_DATA_PATH")
    if not data_path:
        print("ERROR: TRAIN_DATA_PATH must be set.", file=sys.stderr)
        return 1
    schema_path = os.environ.get("EDA_SCHEMA_PATH") or os.path.join(
        data_path if os.path.isdir(data_path) else os.path.dirname(data_path),
        "schema.json",
    )
    if not os.path.exists(schema_path):
        print(f"ERROR: schema not found at {schema_path}", file=sys.stderr)
        return 1
    max_rows = int(os.environ.get("EDA_MAX_ROWS", "200000"))
    out_json = os.environ.get("EDA_OUT_JSON", "")

    parquet_files = _resolve_parquet_files(data_path)
    domains = _load_schema(schema_path)

    print(f"[eda] data_path     = {data_path}")
    print(f"[eda] schema_path   = {schema_path}")
    print(f"[eda] parquet files = {len(parquet_files)}")
    print(f"[eda] domains       = {sorted(domains.keys())}")
    print(f"[eda] row cap       = {max_rows:,}")
    print(f"[eda] per-domain BUCKET_BOUNDARIES_BY_DOMAIN: "
          + ", ".join(
              f"{d}={len(b)} edges"
              for d, b in BUCKET_BOUNDARIES_BY_DOMAIN.items()
          ))

    diffs, rows_seen = _collect_time_diffs(parquet_files, domains, max_rows)
    print(f"\n[eda] sampled rows  = {rows_seen:,}")

    report: Dict[str, Dict] = {
        "rows_sampled": rows_seen,
        "current_boundaries_by_domain": {
            d: [int(x) for x in b.tolist()]
            for d, b in BUCKET_BOUNDARIES_BY_DOMAIN.items()
        },
        "domains": {},
    }

    for domain in sorted(diffs.keys()):
        d = diffs[domain]
        boundaries = BUCKET_BOUNDARIES_BY_DOMAIN.get(
            domain, np.zeros(0, dtype=np.int64)
        )
        section = {}
        section["stats"] = _print_percentile_table(d, domain)
        section["current_occupancy"] = _print_bucket_occupancy(d, domain, boundaries)
        section["proposed_quantile"] = _print_quantile_proposal(d, domain)
        report["domains"][domain] = section

    if out_json:
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[eda] wrote JSON report -> {out_json}")

    print("\n[eda] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
