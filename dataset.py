"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.
"""

import os
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timezone
from numba import njit

# numpy.typing is available since numpy >= 1.20; on older numpy fall back to a
# no-op shim so that forward-referenced annotations like ``npt.NDArray[np.int64]``
# keep working as plain strings without raising at import time.
try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover

    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


@njit(cache=True, fastmath=True, parallel=False)
def hash_ids_inplace(x, hash_size):
    mask = hash_size - 1

    for i in range(x.shape[0]):
        v = x[i]
        if v <= 0:
            continue

        v = np.uint64(v)
        v ^= v >> 33
        v *= np.uint64(0xFF51AFD7ED558CCD)
        v ^= v >> 33
        v *= np.uint64(0xC4CEB9FE1A85EC53)
        v ^= v >> 33

        x[i] = (v & mask) + 1


@njit(cache=True, fastmath=True)
def pad_varlen_int_jit(offsets, values, max_len):
    B = offsets.shape[0] - 1
    padded = np.zeros((B, max_len), dtype=np.int64)
    lengths = np.zeros(B, dtype=np.int64)

    for i in range(B):
        start = offsets[i]
        end = offsets[i + 1]
        raw_len = end - start

        if raw_len <= 0:
            continue

        use_len = raw_len if raw_len < max_len else max_len

        for j in range(use_len):
            v = values[start + j]
            if v > 0:
                padded[i, j] = v
            else:
                padded[i, j] = 0

        lengths[i] = use_len

    return padded, lengths


@njit(cache=True, fastmath=True)
def pad_varlen_float_jit(offsets, values, max_dim):
    B = offsets.shape[0] - 1
    padded = np.zeros((B, max_dim), dtype=np.float32)

    for i in range(B):
        start = offsets[i]
        end = offsets[i + 1]
        raw_len = end - start

        if raw_len <= 0:
            continue

        use_len = raw_len if raw_len < max_dim else max_dim

        for j in range(use_len):
            padded[i, j] = values[start + j]

    return padded


@njit(cache=True, fastmath=True)
def fill_seq_buffer_jit(out, lengths, offsets_tuple, values_tuple, max_len, vocab_sizes):
    B = out.shape[0]
    C = len(offsets_tuple)

    for c in range(C):
        offsets = offsets_tuple[c]
        values = values_tuple[c]
        vs = vocab_sizes[c]

        for i in range(B):
            s = offsets[i]
            e = offsets[i + 1]
            rl = e - s

            if rl <= 0:
                continue

            ul = rl if rl < max_len else max_len

            for j in range(ul):
                v = values[s + j]
                if vs > 0:
                    if v > 0 and v < vs:
                        out[i, c, j] = v
                    else:
                        out[i, c, j] = 0
                else:
                    out[i, c, j] = 0

            if ul > lengths[i]:
                lengths[i] = ul


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        # Ordered list of (feature_id, offset, length).
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        # Quick lookup from fid to its (offset, length).
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        """Append a feature to the schema."""
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        """Get ``(offset, length)`` for a feature_id."""
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        """Return all feature_ids in their insertion order."""
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (for JSON dumping)."""
        return {
            "entries": self.entries,
            "total_dim": self.total_dim,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureSchema":
        """Reconstruct a :class:`FeatureSchema` from its dict form."""
        schema = cls()
        for fid, offset, length in d["entries"]:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d["total_dim"]
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)


# Use filesystem-based tensor sharing (instead of /dev/shm) to avoid running
# out of shared memory when many DataLoader workers are active.
torch.multiprocessing.set_sharing_strategy("file_system")

# Time-delta bucket boundaries (64 edges -> 65 buckets: 0=padding, 1..64).
# fmt: off
BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

PAIR_FEATURES = [62, 63, 64, 65, 66, 89, 90, 91]
# fmt: on

# Total number of time-bucket embedding slots (= number of boundaries + 1, with
# padding=0 included).
#
# This constant is uniquely determined by the length of BUCKET_BOUNDARIES; on
# the model side, ``nn.Embedding(num_embeddings=NUM_TIME_BUCKETS)`` must match
# this value exactly, otherwise an IndexError may be raised at runtime.
#
# That is why ``train.py`` / ``infer.py`` only expose the boolean flag
# ``--use_time_buckets`` and derive the concrete bucket count from here.
NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1

# Number of scalars in ts_stat_feats per sequence domain.
# Consumers (model.py / trainer.py / infer.py) import this constant so that
# changing the stat feature set only requires editing dataset.py.
TS_STAT_DIM = 6

# Number of channels in ts_float_feats per sequence position.
TS_FLOAT_DIM = 8

# Trailing user_dense columns (not from Parquet): sample ``timestamp`` -> hod cos/sin.
USER_DENSE_TIMESTAMP_HOD_FID = -10001
USER_DENSE_TIMESTAMP_HOD_DIM = 2

# Virtual fid IDs for item_int null indicators (negative to avoid collision with real fids).
# vocab_size=2 → embedding table has 3 rows: 0=padding, 1=null/missing, 2=present.
_ITEM_NULL_FID_11 = -21011
_ITEM_NULL_FID_83 = -21083
_ITEM_NULL_FID_84 = -21084
_ITEM_NULL_FID_85 = -21085
_ITEM_NULL_VOCAB_SIZE = 2
_ITEM_NULL_SOURCE_FIDS = [
    (11, _ITEM_NULL_FID_11),
    (83, _ITEM_NULL_FID_83),
    (84, _ITEM_NULL_FID_84),
    (85, _ITEM_NULL_FID_85),
]


def hour_decimal_cos_sin_from_unix_sec(
    ts_sec: Any,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hour-of-day with one decimal place in ``[0.0, 23.9]``, then cos/sin (24h period).

    ``seconds_within_day / 3600``, rounded to 1 decimal. Same encoding for sample
    ``timestamp`` and per-slot sequence event times (synced).
    """
    t = np.asarray(ts_sec, dtype=np.float64)
    sec_in_day = np.mod(t, 86400.0)
    hour_dec = np.round((sec_in_day / 3600.0) * 10.0) / 10.0
    hour_dec = np.clip(hour_dec, 0.0, 23.9)
    ang = (2.0 * np.pi / 24.0) * hour_dec
    c = np.cos(ang).astype(np.float32)
    s = np.sin(ang).astype(np.float32)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        c = np.where(vm, c, np.float32(0.0))
        s = np.where(vm, s, np.float32(0.0))
    return c, s


class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    - int features: scalar or list (multi-hot); values <= 0 are mapped to 0 (padding).
    - dense features: ``list<float>``, variable-length padded up to ``max_dim``.
    - sequence features: ``list<int64>``, grouped by domain; includes side-info
      columns and an optional timestamp column (used for time-bucketing).
    - label: mapped from ``label_type == 2``.
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        split_ts_threshold: Optional[int] = None,
        split_side: Optional[str] = None,
        hist_users_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            parquet_path: either a directory containing ``*.parquet`` files or
                a single parquet file path.
            schema_path: path of the schema JSON describing feature layouts.
            batch_size: fixed batch size used for the pre-allocated buffers.
            seq_max_lens: optional per-domain override of sequence truncation,
                e.g. ``{'seq_d': 256}``. Domains not listed fall back to the
                schema default of 256.
            shuffle: whether to shuffle within a ``buffer_batches``-sized window.
            buffer_batches: shuffle buffer size in units of batches.
            row_group_range: ``(start, end)`` slice of Row Groups; ``None`` to
                use all Row Groups.
            clip_vocab: if True, clip out-of-bound ids to 0; if False, raise.
            is_training: if True, derive ``label`` from ``label_type == 2``;
                if False, return an all-zeros label column.
            split_ts_threshold: optional Unix-second cutoff (int) for row-level
                time filtering together with ``split_side``.
            split_side: ``'train'`` keeps rows with ``timestamp < split_ts_threshold``;
                ``'valid'`` keeps ``timestamp >= split_ts_threshold``. Both must be
                set together, or both omitted (no time-based row filter).
            hist_users_dir: optional directory produced by
                ``build_item_hist_users.py`` (see meta.json + *.npy inside). When
                provided, every yielded batch carries six extra tensors used by
                the model's ``ItemHistUserModule``:
                ``hist_pos_scalars`` (B, k_pos, 7) int64,
                ``hist_pos_dense``   (B, k_pos, 256) float32,
                ``hist_neg_scalars`` (B, k_neg, 7) int64,
                ``hist_neg_dense``   (B, k_neg, 256) float32,
                ``hist_pos_lens`` / ``hist_neg_lens`` (B,) int32.
                The directory must match ``parquet_path`` (same sorted basenames
                and row-group counts); otherwise __init__ raises.
        """
        super().__init__()

        # Accept either a directory or a single file path.
        if os.path.isdir(parquet_path):
            import glob

            files = sorted(glob.glob(os.path.join(parquet_path, "*.parquet")))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        if (split_ts_threshold is None) ^ (split_side is None):
            raise ValueError(
                "split_ts_threshold and split_side must both be None or both set "
                f"(got threshold={split_ts_threshold!r}, side={split_side!r})."
            )
        if split_side is not None and split_side not in ("train", "valid"):
            raise ValueError(
                f"split_side must be 'train' or 'valid', got {split_side!r}"
            )
        self._split_ts_threshold: Optional[int] = (
            int(split_ts_threshold) if split_ts_threshold is not None else None
        )
        self._split_side: Optional[str] = split_side
        # Out-of-bound statistics:
        #   {(group, col_idx): {'count': N, 'max': M, 'min_oob': M, 'vocab': V}}
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        # Build the list of Row Groups.
        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        # Pre-compute global row-start offset for every (file, rg) BEFORE the
        # row_group_range slice, so that train/valid splits (or any subset)
        # still agree with the build_item_hist_users.py global indexing.
        self._rg_global_offsets: Dict[Tuple[str, int], int] = {}
        _cum = 0
        for f, i, n in self._rg_list:
            self._rg_global_offsets[(os.path.basename(f), i)] = _cum
            _cum += n
        self._total_rows_unfiltered = _cum

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        # Load item-history-user lookup tables (mmap'd) if a directory is given.
        # If not, every emitted batch simply omits the six hist_* keys and the
        # downstream model falls back to the baseline path.
        self.hist_users_dir = hist_users_dir
        self._hist_loaded = False
        if hist_users_dir is not None:
            self._load_hist_users(hist_users_dir)

        # Load schema.json.
        self._load_schema(schema_path, seq_max_lens or {})

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- Pre-allocate numpy buffers ----
        B = batch_size
        self._buf_user_int = np.zeros(
            (B, self.user_int_schema.total_dim), dtype=np.int64
        )
        self._buf_item_int = np.zeros(
            (B, self.item_int_schema.total_dim), dtype=np.int64
        )
        self._buf_user_dense = np.zeros(
            (B, self.user_dense_schema.total_dim), dtype=np.float32
        )
        self._buf_item_dense = np.zeros(
            (B, self.item_dense_schema.total_dim), dtype=np.float32
        )
        self._buf_seq = {}
        self._buf_seq_tb = {}
        self._buf_seq_lens = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        self._buf_pair_int = np.zeros(
            (B, self.pair_int_schema.total_dim), dtype=np.int64
        )
        self._buf_pair_dense = np.zeros(
            (B, self.pair_dense_schema.total_dim), dtype=np.float32
        )
        self._buf_event_ts: Dict[str, np.ndarray] = {}
        self._buf_ts_float: Dict[str, np.ndarray] = {}
        self._buf_ts_stat: Dict[str, np.ndarray] = {}
        self._buf_shifted: Dict[str, np.ndarray] = {}
        self._buf_idx: Dict[str, np.ndarray] = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            self._buf_event_ts[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_ts_float[domain] = np.zeros((B, TS_FLOAT_DIM, max_len), dtype=np.float32)
            self._buf_ts_stat[domain] = np.zeros((B, TS_STAT_DIM), dtype=np.float32)
            self._buf_shifted[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_idx[domain] = np.arange(max_len, dtype=np.int64)[None, :]

        # ---- Pre-compute (col_idx, offset, vocab_size) plans for int columns ----
        self.idx_to_fid = {}
        self._user_int_plan = []  # [(col_idx, dim, offset, vocab_size), ...]
        self._pair_int_plan = []
        offset = 0
        pair_offset = 0
        for fid, vs, dim in self._user_int_cols:
            ci = self._col_idx.get(f"user_int_feats_{fid}")
            if fid in PAIR_FEATURES:
                self._pair_int_plan.append((ci, dim, pair_offset, vs))
                pair_offset += dim
            else:
                self._user_int_plan.append((ci, dim, offset, vs))
                offset += dim
            self.idx_to_fid[ci] = f"user_int_feats_{fid}"

        self._item_int_plan = []
        offset = 0
        for fid, vs, dim in self._item_int_cols:
            ci = self._col_idx.get(f"item_int_feats_{fid}")
            self._item_int_plan.append((ci, dim, offset, vs))
            offset += dim
            self.idx_to_fid[ci] = f"item_int_feats_{fid}"

        # Null indicator plan: (col_idx_in_parquet, offset_in_item_int_buffer)
        self._item_null_plan = []
        for src_fid, vfid in _ITEM_NULL_SOURCE_FIDS:
            ci = self._col_idx.get(f"item_int_feats_{src_fid}")
            null_offset, _ = self.item_int_schema.get_offset_length(vfid)
            self._item_null_plan.append((ci, null_offset))

        self._user_dense_plan = []
        self._pair_dense_plan = []
        offset = 0
        pair_offset = 0
        for fid, dim in self._user_dense_cols:
            ci = self._col_idx.get(f"user_dense_feats_{fid}")
            if fid in PAIR_FEATURES:
                need_log1p = False
                if fid < 89:
                    need_log1p = True
                self._pair_dense_plan.append((ci, dim, pair_offset, need_log1p))
                pair_offset += dim
            else:
                self._user_dense_plan.append((ci, dim, offset))
                offset += dim
            self.idx_to_fid[ci] = f"user_dense_feats_{fid}"

        # Sequence column plan: {domain: ([(col_idx, feat_slot, vocab_size), ...], ts_col_idx)}
        self._seq_plan = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f"{prefix}_{fid}")
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = (
                self._col_idx.get(f"{prefix}_{ts_fid}") if ts_fid is not None else None
            )
            self._seq_plan[domain] = (side_plan, ts_ci)

        # hash
        self.hash_threshold = 600_000

        extra = ""
        if self._split_ts_threshold is not None:
            extra = f", ts_split={self._split_side}@{self._split_ts_threshold}"
        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}{extra}"
        )

    def _filter_batch_by_split_ts(
        self, batch: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Keep rows by comparing ``batch['timestamp']`` to ``_split_ts_threshold``."""
        assert self._split_ts_threshold is not None and self._split_side is not None
        ts = batch["timestamp"]
        if not isinstance(ts, torch.Tensor):
            raise TypeError("batch['timestamp'] must be a torch.Tensor")
        thr = int(self._split_ts_threshold)
        ts_long = ts.long()
        if self._split_side == "train":
            mask = ts_long < thr
        else:
            mask = ts_long >= thr
        if not bool(mask.any().item()):
            return None
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        out: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.index_select(0, idx)
            elif k == "user_id":
                uid = batch["user_id"]
                m_list = mask.tolist()
                out[k] = [uid[i] for i in range(len(uid)) if m_list[i]]
            else:
                out[k] = v
        return out

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        """Populate per-group schema information from ``schema_path``."""
        with open(schema_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # ---- user_int: [[fid, vocab_size, dim], ...] ----
        self._user_int_cols: List[List[int]] = raw["user_int"]
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        self.pair_int_schema = FeatureSchema()
        self.pair_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            if fid in PAIR_FEATURES:
                self.pair_int_schema.add(fid, dim)
                self.pair_int_vocab_sizes.extend([vs] * dim)
            else:
                self.user_int_schema.add(fid, dim)
                self.user_int_vocab_sizes.extend([vs] * dim)

        # ---- item_int ----
        self._item_int_cols: List[List[int]] = raw["item_int"]
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        # Append null-indicator virtual features for item_int_feats 11/83/84/85.
        for _, vfid in _ITEM_NULL_SOURCE_FIDS:
            self.item_int_schema.add(vfid, 1)
            self.item_int_vocab_sizes.append(_ITEM_NULL_VOCAB_SIZE)

        # ---- user_dense: [[fid, dim], ...] ----
        self._user_dense_cols: List[List[int]] = raw["user_dense"]
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        self.pair_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            if fid in PAIR_FEATURES:
                self.pair_dense_schema.add(fid, dim)
            else:
                self.user_dense_schema.add(fid, dim)

        # ---- item_dense (empty) ----
        self.item_dense_schema: FeatureSchema = FeatureSchema()

        # Context category features
        self.context_int_schema: FeatureSchema = FeatureSchema()
        self.context_int_vocab_sizes: List[int] = []

        # ---- sequence domains ----
        self._seq_cfg: Dict[str, Dict[str, Any]] = raw["seq"]
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg["prefix"]
            ts_fid = cfg["ts_fid"]
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg["features"]]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg["features"]}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]

            # max_len: from seq_max_lens arg; unspecified domains fall back to 256.
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    def _load_hist_users(self, hist_users_dir: str) -> None:
        """Memory-map the six per-row hist arrays produced by
        ``build_item_hist_users.py`` and validate that ``meta.json`` matches
        this dataset's parquet layout (sorted basenames + per-rg row counts).
        """
        meta_path = os.path.join(hist_users_dir, "meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"missing {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Cross-check (basename, rg_idx, num_rows) against this dataset's full
        # row-group list (before any row_group_range slice).
        meta_rg = {(e["file"], int(e["rg"])): (int(e["row_start"]), int(e["num_rows"]))
                   for e in meta["rg_layout"]}
        for (bn, rg_idx), base_off in self._rg_global_offsets.items():
            if (bn, rg_idx) not in meta_rg:
                raise ValueError(
                    f"hist_users meta missing entry for ({bn}, rg={rg_idx}); "
                    f"the hist directory was built on a different parquet set."
                )
            m_off, m_n = meta_rg[(bn, rg_idx)]
            if m_off != base_off:
                raise ValueError(
                    f"hist_users meta offset mismatch for ({bn}, rg={rg_idx}): "
                    f"meta={m_off}, dataset={base_off}. File ordering differs."
                )

        self.hist_k_pos = int(meta["k_pos"])
        self.hist_k_neg = int(meta["k_neg"])
        self.hist_dense_dim = int(meta["dense_dim"])
        self.hist_total_rows = int(meta["total_rows"])

        # mmap-load all six arrays (640MB+ in total; mmap keeps RSS small).
        load_mm = lambda name: np.load(
            os.path.join(hist_users_dir, name), mmap_mode="r"
        )
        self._hist_scalars = load_mm("user_lookup_scalars.npy")  # (N, 7) int32
        self._hist_dense = load_mm("user_lookup_dense61.npy")    # (N, 256) f16
        self._hist_pos_idx = load_mm("hist_pos_indices.npy")     # (N, kp) i32
        self._hist_neg_idx = load_mm("hist_neg_indices.npy")     # (N, kn) i32
        # lens are small; load fully (saves a mmap per-batch)
        self._hist_pos_len = np.load(
            os.path.join(hist_users_dir, "hist_pos_lens.npy")
        )  # (N,) int8
        self._hist_neg_len = np.load(
            os.path.join(hist_users_dir, "hist_neg_lens.npy")
        )

        # Sanity shape checks
        N = self.hist_total_rows
        assert self._hist_scalars.shape == (N, 7), self._hist_scalars.shape
        assert self._hist_dense.shape == (N, self.hist_dense_dim)
        assert self._hist_pos_idx.shape == (N, self.hist_k_pos)
        assert self._hist_neg_idx.shape == (N, self.hist_k_neg)

        self._hist_loaded = True
        logging.info(
            f"hist_users loaded from {hist_users_dir}: N={N:,}, "
            f"k_pos={self.hist_k_pos}, k_neg={self.hist_k_neg}, "
            f"dense_dim={self.hist_dense_dim}"
        )

    def _gather_hist(self, global_start: int, B: int) -> Dict[str, torch.Tensor]:
        """Gather the four hist tensors for a contiguous batch.

        Padding rows in hist_*_idx (value -1) are mapped to row 0 in
        user_lookup; the downstream attention uses ``hist_*_lens`` as a
        key_padding_mask so the contents of padded slots never affect the
        output.
        """
        end = global_start + B
        pos_idx = self._hist_pos_idx[global_start:end]   # (B, k_pos)
        neg_idx = self._hist_neg_idx[global_start:end]
        # Replace -1 padding with 0 so fancy-indexing never raises;
        # padding mask is built from lengths downstream.
        pos_safe = np.where(pos_idx >= 0, pos_idx, 0).astype(np.int64)
        neg_safe = np.where(neg_idx >= 0, neg_idx, 0).astype(np.int64)

        pos_scalars = np.asarray(self._hist_scalars[pos_safe], dtype=np.int64)
        pos_dense = np.asarray(self._hist_dense[pos_safe], dtype=np.float32)
        neg_scalars = np.asarray(self._hist_scalars[neg_safe], dtype=np.int64)
        neg_dense = np.asarray(self._hist_dense[neg_safe], dtype=np.float32)
        pos_lens = self._hist_pos_len[global_start:end].astype(np.int32)
        neg_lens = self._hist_neg_len[global_start:end].astype(np.int32)

        return {
            "hist_pos_scalars": torch.from_numpy(pos_scalars),
            "hist_pos_dense": torch.from_numpy(pos_dense),
            "hist_neg_scalars": torch.from_numpy(neg_scalars),
            "hist_neg_dense": torch.from_numpy(neg_dense),
            "hist_pos_lens": torch.from_numpy(pos_lens),
            "hist_neg_lens": torch.from_numpy(neg_lens),
        }

    def __len__(self) -> int:
        # Ceiling per Row Group; this is an upper bound on the true batch count.
        return sum(
            (n + self.batch_size - 1) // self.batch_size for _, _, n in self._rg_list
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [
                rg
                for i, rg in enumerate(rg_list)
                if i % worker_info.num_workers == worker_info.id
            ]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            # base_offset_in_rg: where the next batch starts inside this rg
            base_offset_in_rg = 0
            if self._hist_loaded:
                rg_global_base = self._rg_global_offsets[
                    (os.path.basename(file_path), rg_idx)
                ]
            for batch in pf.iter_batches(
                batch_size=self.batch_size, row_groups=[rg_idx]
            ):
                batch_dict = self._convert_batch(batch)
                if self._hist_loaded:
                    hist = self._gather_hist(
                        rg_global_base + base_offset_in_rg, batch.num_rows
                    )
                    batch_dict.update(hist)
                base_offset_in_rg += batch.num_rows
                if self._split_ts_threshold is not None:
                    batch_dict = self._filter_batch_by_split_ts(batch_dict)
                    if batch_dict is None:
                        continue
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(self, buffer: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """Concatenate the buffered batches, shuffle at the row level, then
        re-slice and yield batch-sized chunks.
        """
        merged: Dict[str, torch.Tensor] = {}
        non_tensor_keys: Dict[str, Any] = {}
        for k in buffer[0].keys():
            if isinstance(buffer[0][k], torch.Tensor):
                merged[k] = torch.cat([b[k] for b in buffer], dim=0)
            else:
                non_tensor_keys[k] = buffer[0][k]
        total_rows = merged["label"].shape[0]
        rand_idx = (
            torch.randperm(total_rows) if self.shuffle else torch.arange(total_rows)
        )
        for i in range(0, total_rows, self.batch_size):
            end = min(i + self.batch_size, total_rows)
            batch: Dict[str, Any] = {k: v[rand_idx[i:end]] for k, v in merged.items()}
            batch.update(non_tensor_keys)
            yield batch
        del merged
        buffer.clear()

    # ---- Helpers ----

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ) -> None:
        """Record out-of-bound indices and (optionally) clip them to 0,
        without printing to the console.
        """
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[oob_mask]
        n = int(oob_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s["count"] += n
            s["max"] = max(s["max"], mx)
            s["min_oob"] = min(s["min_oob"], mn)
        else:
            self._oob_stats[key] = {
                "count": n,
                "max": mx,
                "min_oob": mn,
                "vocab": vocab_size,
            }
        if self.clip_vocab:
            arr[oob_mask] = 0
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values out of range "
                f"[0, {vocab_size}), actual=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json"
            )
            a

    def _clip_vocab_size(
        self,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
    ):
        oob_mask = arr >= vocab_size
        if not oob_mask.any():
            return
        if self.clip_vocab:
            arr[oob_mask] = 0

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        """Dump out-of-bound statistics to a file if ``path`` is provided,
        otherwise to ``logging.info``.
        """
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s["min_oob"] >= s["vocab"] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}"
            )
        msg = "\n".join(lines)
        if path:
            with open(path, "w") as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        """Pad an Arrow ``ListArray`` of ints to shape ``[B, max_len]``.

        Values <= 0 are mapped to 0 (padding). Note: the raw data contains -1
        (missing); currently treated the same way as 0 (padding).

        Returns:
            A tuple ``(padded, lengths)`` where ``padded`` has shape
            ``[B, max_len]`` and ``lengths`` has shape ``[B]``.
        """
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        padded, lengths = pad_varlen_int_jit(offsets, values, max_len)
        return padded, lengths

    # Backwards-compatible alias kept for bench_raw_dataset.py and other
    # external callers that pre-date the rename. New code should call
    # `_pad_varlen_int_column` directly.
    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        """Pad an Arrow ``ListArray<float>`` to shape ``[B, max_dim]``."""
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()

        return pad_varlen_float_jit(offsets, values, max_dim)

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors."""
        B = batch.num_rows

        # ---- meta ----
        timestamps = (
            batch.column(self._col_idx["timestamp"]).to_numpy().astype(np.int64)
        )

        if self.is_training:
            labels = (
                batch.column(self._col_idx["label_type"])
                .fill_null(0)
                .to_numpy(zero_copy_only=False)
                .astype(np.int64)
                == 2
            ).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)
        user_ids = batch.column(self._col_idx["user_id"]).to_pylist()

        # ---- user_int: write into pre-allocated buffer ----
        # Note: null -> 0 (via fill_null), -1 -> 0 (via arr<=0); missing values
        # are treated the same as padding. Features with vs==0 have no vocab
        # information and are forced to 0 on the dataset side so that the
        # model's 1-slot Embedding (created for vs=0) is never indexed out of
        # range.
        user_int = self._buf_user_int[:B]
        user_int[:] = 0
        for ci, dim, offset, vs in self._user_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._clip_vocab_size(arr, vs)
                else:
                    arr[:] = 0
                user_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._clip_vocab_size(padded, vs)
                else:
                    padded[:] = 0
                user_int[:, offset : offset + dim] = padded

        # ---- pair_int ----
        pair_int = self._buf_pair_int[:B]
        pair_int[:] = 0
        for ci, dim, offset, vs in self._pair_int_plan:
            col = batch.column(ci)
            padded, _ = self._pad_varlen_int_column(col, dim, B)
            if vs > 0:
                self._clip_vocab_size(padded, vs)
            else:
                padded[:] = 0
            pair_int[:, offset : offset + dim] = padded

        # ---- item_int ----
        item_int = self._buf_item_int[:B]
        item_int[:] = 0
        for ci, dim, offset, vs in self._item_int_plan:
            col = batch.column(ci)
            if dim == 1:
                arr = col.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
                arr[arr <= 0] = 0
                if vs > 0:
                    self._clip_vocab_size(arr, vs)
                else:
                    arr[:] = 0
                item_int[:, offset] = arr
            else:
                padded, _ = self._pad_varlen_int_column(col, dim, B)
                if vs > 0:
                    self._clip_vocab_size(padded, vs)
                else:
                    padded[:] = 0
                item_int[:, offset : offset + dim] = padded

        # Fill null indicator slots: 1=null/missing, 2=present.
        for ci, null_offset in self._item_null_plan:
            is_null = batch.column(ci).is_null().to_numpy(zero_copy_only=False)[:B]
            item_int[:B, null_offset] = np.where(is_null, 1, 2)

        # ---- user_dense ----
        user_dense = self._buf_user_dense[:B]
        user_dense[:] = 0
        for ci, dim, offset in self._user_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            user_dense[:, offset : offset + dim] = padded

        # ---- pair_dense ----
        pair_dense = self._buf_pair_dense[:B]
        pair_dense[:] = 0
        for ci, dim, offset, needlog1p in self._pair_dense_plan:
            col = batch.column(ci)
            padded = self._pad_varlen_float_column(col, dim, B)
            if needlog1p:
                np.nan_to_num(padded, nan=0.0, copy=False)
                np.maximum(padded, 0, out=padded)
                np.log1p(padded, out=padded)
            pair_dense[:, offset : offset + dim] = padded


        item_dense = self._buf_item_dense[:B]
        item_dense[:] = 0

        result = {
            "user_int_feats": torch.from_numpy(user_int.copy()),
            "user_dense_feats": torch.from_numpy(user_dense.copy()),
            "pair_int_feats": torch.from_numpy(pair_int.copy()),
            "pair_dense_feats": torch.from_numpy(pair_dense.copy()),
            "item_int_feats": torch.from_numpy(item_int.copy()),
            "item_dense_feats": torch.from_numpy(item_dense.copy()),
            "label": torch.from_numpy(labels),
            "user_id": user_ids,
            "_seq_domains": self.seq_domains,
        }

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            # Write directly into the pre-allocated 3D buffer.
            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            # Fused path: first collect (offsets, values, vocab_size, col_idx)
            # for every side-info column, then fill the buffer in a single pass.
            # ---- collect numpy arrays ----
            offsets_list = []
            values_list = []
            vs_list = []
            ci_list = []

            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                offsets_list.append(col.offsets.to_numpy())
                values_list.append(col.values.to_numpy())
                vs_list.append(vs)
                ci_list.append(ci)

            # ---- convert to numba typed list ----
            offsets_tuple = tuple(offsets_list)
            values_tuple = tuple(values_list)
            vocab_sizes_arr = np.array(vs_list, dtype=np.int64)

            # ---- call JIT kernel (clip fused inside) ----
            fill_seq_buffer_jit(out, lengths, offsets_tuple, values_tuple, max_len, vocab_sizes_arr)

            result[domain] = torch.from_numpy(out.copy())
            result[f"{domain}_len"] = torch.from_numpy(lengths.copy())

            # Time bucketing + per-position float time features [B, 5, L]:
            #   0: log1p(sample_ts - event_ts) seconds
            #   1: hours, 2: weeks
            #   3/4: cos/sin of hour-of-day (1 decimal in [0,23.9]), same as user_dense tail
            # Raw event Unix timestamps (aligned with sequence slots), for query stats.
            event_ts_buf = self._buf_event_ts[domain][:B]
            event_ts_buf[:] = 0
            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            ts_float_buf = self._buf_ts_float[domain][:B]
            ts_float_buf[:] = 0
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                ts_padded, _ = pad_varlen_int_jit(ts_offs, ts_vals, max_len)

                event_ts_buf[:] = ts_padded

                time_diff = timestamps[:, None] - ts_padded
                np.maximum(time_diff, 0, out=time_diff)

                raw_buckets = np.searchsorted(BUCKET_BOUNDARIES, time_diff, side="left")
                np.clip(raw_buckets, 0, len(BUCKET_BOUNDARIES) - 1, out=raw_buckets)

                buckets = raw_buckets + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets

                valid = ts_padded != 0
                d = np.where(valid, time_diff.astype(np.float32), 0.0)
                d_days = d / 86400.0

                # ch0: log1p(diff_days) — better discriminability than log1p(seconds)
                #   seq_a/b: [log1p(14), log1p(140)] ≈ [2.7, 4.9]
                #   seq_c:   [log1p(49), log1p(534)] ≈ [3.9, 6.3]
                #   seq_d:   [log1p(2),  log1p(29)]  ≈ [1.1, 3.4]
                ts_float_buf[:, 0, :] = np.log1p(d_days)

                # ch1: domain-specific fine-grained time
                #   seq_c: months (2-year history, hours are meaningless)
                #   seq_d: hours (recent behavior, sub-day resolution matters)
                #   seq_a/b: days (week-scale resolution is appropriate)
                if domain == 'seq_c':
                    ts_float_buf[:, 1, :] = d_days / 30.0
                elif domain == 'seq_d':
                    ts_float_buf[:, 1, :] = d / 3600.0
                else:
                    ts_float_buf[:, 1, :] = d_days

                # ch2: log1p(diff_hours) — finer recency scale than diff_days
                ts_float_buf[:, 2, :] = np.log1p(d / 3600.0)

                # ch3/4: hour-of-day cos/sin
                # seq_c: 2-year-old hod is noise — leave as zero
                if domain != 'seq_c':
                    tc, ts_ = hour_decimal_cos_sin_from_unix_sec(
                        ts_padded, valid_mask=valid
                    )
                    ts_float_buf[:, 3, :] = tc
                    ts_float_buf[:, 4, :] = ts_

                # ch5/6: day-of-week cos/sin
                # seq_c: 2-year-old DoW is noise — leave as zero
                if domain != 'seq_c':
                    dow = (ts_padded // 86400) % 7
                    ts_float_buf[:, 5, :] = np.where(valid, np.cos(2 * np.pi * dow / 7), 0.0)
                    ts_float_buf[:, 6, :] = np.where(valid, np.sin(2 * np.pi * dow / 7), 0.0)

                # ch7: inter-event time gap
                #   seq_c: months (long-term, second-level gaps are meaningless)
                #   seq_d: log1p(hours) — session boundary detection at hour scale
                #   seq_a/b: log1p(days) — avoids redundancy with ch0
                shifted = self._buf_shifted[domain][:B]
                shifted[:] = 0
                shifted[:, :-1] = ts_padded[:, 1:]
                inter_s = np.where(
                    valid & (shifted > 0),
                    np.maximum((ts_padded - shifted).astype(np.float32), 0.0),
                    0.0,
                )
                if domain == 'seq_c':
                    ts_float_buf[:, 7, :] = np.where(valid, inter_s / (86400.0 * 30), 0.0)
                elif domain == 'seq_d':
                    ts_float_buf[:, 7, :] = np.where(valid, np.log1p(inter_s / 3600.0), 0.0)
                else:
                    ts_float_buf[:, 7, :] = np.where(valid, np.log1p(inter_s / 86400.0), 0.0)

            result[f"{domain}_time_bucket"] = torch.from_numpy(time_bucket.copy())
            result[f"{domain}_ts_float_feats"] = torch.from_numpy(ts_float_buf.copy())

            # Sequence-level time stats (6 scalars per sample):
            #   log1p(max_diff), log1p(min_diff), log1p(mean_diff),
            #   count_<=15min, count_<=1h, count_<=1d
            # Precomputed here (CPU) to avoid redundant GPU work in model forward.
            stat_feats = self._buf_ts_stat[domain][:B]
            stat_feats[:] = 0
            if ts_ci is not None:
                idx = self._buf_idx[domain]              # (1, max_len)
                len_mask = idx < lengths[:, None]            # (B, max_len)
                pos_mask = (event_ts_buf > 0) & len_mask    # (B, max_len)
                diff_f = np.where(
                    pos_mask,
                    np.maximum(timestamps[:, None] - event_ts_buf.astype(np.float32), 0.0),
                    0.0,
                )
                n_valid = pos_mask.sum(axis=1)               # (B,)
                has_valid = n_valid > 0

                max_v = np.where(pos_mask, diff_f, -1e9).max(axis=1)
                max_v = np.where(has_valid, max_v, 0.0)

                min_v = np.where(pos_mask, diff_f, 1e9).min(axis=1)
                min_v = np.where(has_valid, min_v, 0.0)

                mean_v = np.where(has_valid, diff_f.sum(axis=1) / np.maximum(n_valid, 1), 0.0)

                stat_feats[:, 0] = np.log1p(np.maximum(max_v, 0.0))
                stat_feats[:, 1] = np.log1p(np.maximum(min_v, 0.0))
                stat_feats[:, 2] = np.log1p(np.maximum(mean_v, 0.0))
                stat_feats[:, 3] = ((diff_f <= 900)  & pos_mask).sum(axis=1).astype(np.float32)
                stat_feats[:, 4] = ((diff_f <= 3600) & pos_mask).sum(axis=1).astype(np.float32)
                stat_feats[:, 5] = ((diff_f <= 86400) & pos_mask).sum(axis=1).astype(np.float32)
            result[f"{domain}_ts_stat_feats"] = torch.from_numpy(stat_feats.copy())

        return result


def collect_all_timestamps_int64(data_dir: str) -> np.ndarray:
    """Concatenate every ``timestamp`` value from all ``*.parquet`` row groups.

    Files are visited in sorted ``glob`` order, matching
    :class:`PCVRParquetDataset` file order. Values are cast to ``int64`` (Unix
    seconds), consistent with ``_convert_batch`` / ``batch['timestamp']``.
    """
    import glob as _glob

    pq_files = sorted(_glob.glob(os.path.join(data_dir, "*.parquet")))
    if not pq_files:
        raise FileNotFoundError(f"No .parquet files in {data_dir}")
    parts: List[np.ndarray] = []
    for fpath in pq_files:
        pf = pq.ParquetFile(fpath)
        names = pf.schema_arrow.names
        if "timestamp" not in names:
            raise ValueError(
                f"Parquet file {fpath} has no 'timestamp' column (required for "
                "split_mode='timestamp'). Expected int64 Unix-second timestamps "
                "aligned with IterableDataset batches."
            )
        for rg in range(pf.metadata.num_row_groups):
            col = pf.read_row_group(rg, columns=["timestamp"]).column(0)
            arr = col.to_numpy(zero_copy_only=False)
            parts.append(np.asarray(arr, dtype=np.int64))
    return np.concatenate(parts, axis=0)


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    split_mode: str = "row_group",
    hist_users_dir: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    **split_mode ``\"row_group\"`` (default):** validation is the last
    ``valid_ratio`` fraction of Row Groups (sorted ``glob`` file order); train
    uses preceding Row Groups. ``train_ratio`` shrinks how many of those train
    Row Groups are used (prefix).

    **split_mode ``\"timestamp\"``:** train/valid are defined **per row** using the
    Parquet ``timestamp`` column (``int64`` Unix seconds), matching
    ``batch['timestamp']`` from :meth:`PCVRParquetDataset._convert_batch`. All Row
    Groups are iterated for both loaders; within each Arrow batch, rows are kept
    or dropped by comparing ``timestamp`` to dataset-wide thresholds from
    quantiles of **all** timestamps in the directory (no tail-RG split).

    Returns:
        A tuple ``(train_loader, valid_loader, train_dataset)``. The third
        element is returned so the caller can access the feature schema
        (``user_int_schema``, ``item_int_schema``, ...) needed to construct
        the model.
    """
    random.seed(seed)

    import glob as _glob

    pq_files = sorted(_glob.glob(os.path.join(data_dir, "*.parquet")))
    if not pq_files:
        raise FileNotFoundError(f"No .parquet files in {data_dir}")

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    use_cuda = torch.cuda.is_available()
    _train_kw: Dict[str, Any] = {}
    if num_workers > 0:
        _train_kw["persistent_workers"] = True
        _train_kw["prefetch_factor"] = 4

    split_mode_l = split_mode.strip().lower()
    if split_mode_l == "timestamp":
        ts_all = collect_all_timestamps_int64(data_dir)
        if ts_all.size == 0:
            raise ValueError("No timestamp values found for split_mode='timestamp'.")
        split_q = 1.0 - float(valid_ratio)
        valid_ts_threshold = int(np.quantile(ts_all.astype(np.float64), split_q))
        train_ts = ts_all[ts_all < valid_ts_threshold]
        if train_ratio < 1.0:
            if train_ts.size == 0:
                logging.warning(
                    "timestamp split: no rows with ts < valid_ts_threshold; "
                    "using train_threshold = valid_ts_threshold (train may be empty)."
                )
                train_threshold = valid_ts_threshold
            else:
                train_threshold = int(
                    np.quantile(train_ts.astype(np.float64), train_ratio)
                )
        else:
            train_threshold = valid_ts_threshold

        logging.info(
            f"timestamp split: rows={ts_all.size}, valid_ratio={valid_ratio}, "
            f"train_ratio={train_ratio}, valid_ts_threshold={valid_ts_threshold}, "
            f"train_ts_threshold={train_threshold} "
            f"(train: ts < {train_threshold}, valid: ts >= {valid_ts_threshold})"
        )

        train_rows_est = int((ts_all < train_threshold).sum())
        valid_rows_est = int((ts_all >= valid_ts_threshold).sum())

        train_dataset = PCVRParquetDataset(
            parquet_path=data_dir,
            schema_path=schema_path,
            batch_size=batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=shuffle_train,
            buffer_batches=buffer_batches,
            row_group_range=None,
            clip_vocab=clip_vocab,
            split_ts_threshold=train_threshold,
            split_side="train",
            hist_users_dir=hist_users_dir,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=use_cuda,
            **_train_kw,
        )

        valid_dataset = PCVRParquetDataset(
            parquet_path=data_dir,
            schema_path=schema_path,
            batch_size=batch_size,
            seq_max_lens=seq_max_lens,
            shuffle=False,
            buffer_batches=0,
            row_group_range=None,
            clip_vocab=clip_vocab,
            split_ts_threshold=valid_ts_threshold,
            split_side="valid",
            hist_users_dir=hist_users_dir,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=use_cuda,
            persistent_workers=False,
        )

        logging.info(
            f"Parquet timestamp split (~rows train<{train_threshold}: "
            f"~{train_rows_est}, valid>={valid_ts_threshold}: ~{valid_rows_est}), "
            f"batch_size={batch_size}, buffer_batches={buffer_batches}, "
            f"all {total_rgs} Row Groups scanned per split"
        )

        return train_loader, valid_loader, train_dataset

    # ---- Default: Row Group order split ----
    if split_mode_l != "row_group":
        raise ValueError(
            f"split_mode must be 'row_group' or 'timestamp', got {split_mode!r}"
        )

    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs

    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])

    logging.info(
        f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
        f"{n_valid_rgs} valid ({valid_rows} rows)"
    )

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        clip_vocab=clip_vocab,
        hist_users_dir=hist_users_dir,
    )

    use_cuda = torch.cuda.is_available()
    _train_kw = {}
    if num_workers > 0:
        _train_kw["persistent_workers"] = True
        _train_kw["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=use_cuda,
        **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs),
        clip_vocab=clip_vocab,
        hist_users_dir=hist_users_dir,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=None,
        num_workers=0,
        pin_memory=use_cuda,
        persistent_workers=False,
    )

    logging.info(
        f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
        f"batch_size={batch_size}, buffer_batches={buffer_batches}"
    )

    return train_loader, valid_loader, train_dataset
