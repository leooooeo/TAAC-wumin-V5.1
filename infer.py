"""PCVRHyFormer inference script (uploaded by the contestant into the
evaluation container).

Model construction mirrors ``train.py``: we rebuild the model from
``schema.json`` + ``ns_groups.json`` + ``train_config.json``. All model
hyperparameters are resolved first from the ckpt directory's
``train_config.json`` (written by ``trainer.py`` when saving a checkpoint),
falling back to ``_FALLBACK_MODEL_CFG`` below (which must stay consistent
with the CLI defaults in ``train.py``).

Only the Parquet data format is supported.

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory (points at the ``global_step``
                       sub-directory containing ``model.pt`` / ``train_config.json``).
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH   Directory for the generated ``predictions.json``.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FeatureSchema, PCVRParquetDataset, NUM_TIME_BUCKETS
from model import PCVRHyFormer, ModelInput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Fallback values used only when ``train_config.json`` is missing from the
# ckpt directory.
#
# These MUST match the argparse defaults in ``train.py``; otherwise once the
# fallback path is actually taken the built model will shape-mismatch the
# saved state_dict.
#
# Special note on ``num_time_buckets``: this value is strictly determined by
# ``dataset.BUCKET_BOUNDARIES`` and is NOT an independent hyperparameter.
# When the feature is enabled we therefore use the constant exposed by the
# dataset module; ``0`` mean
_FALLBACK_MODEL_CFG = {
    "d_model": 64,
    "emb_dim": 64,
    "num_queries": 1,
    "num_hyformer_blocks": 2,
    "num_heads": 4,
    "seq_encoder_type": "transformer",
    "hidden_mult": 4,
    "dropout_rate": 0.01,
    "seq_top_k": 50,
    "seq_causal": False,
    "action_num": 1,
    "num_time_buckets": NUM_TIME_BUCKETS,
    "rank_mixer_mode": "full",
    "use_rope": False,
    "rope_base": 10000.0,
    "emb_skip_threshold": 0,
    "seq_id_threshold": 10000,
    "ns_tokenizer_type": "rankmixer",
    "user_ns_tokens": 0,
    "item_ns_tokens": 0,
    "use_din": False,
    # ── Item-history-user is required at infer time. We don't include
    # ``enable_hist_users`` in the fallback map because there is no "off" mode:
    # main() asserts it and the model is always built with hist on.
    "hist_num_user_ns_tokens": 12,
    "hist_dropout": 0.1,
}

_FALLBACK_SEQ_MAX_LENS = "seq_a:256,seq_b:256,seq_c:512,seq_d:512"
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 16


# Hyperparameter keys used to build the model. Everything else in
# ``train_config.json`` is ignored when constructing ``PCVRHyFormer``.
_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build ``feature_specs = [(vocab_size, offset, length), ...]`` in the
    order of ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset : offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    """Parse a string like ``'seq_a:256,seq_b:256,...'`` into a dict."""
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(","):
        k, v = pair.split(":")
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def load_train_config(model_dir: str) -> Dict[str, Any]:
    """Load ``train_config.json`` from the ckpt directory.

    Returns an empty dict (which triggers fallback resolution) if the file is
    not present.
    """
    train_config_path = os.path.join(model_dir, "train_config.json")
    if os.path.exists(train_config_path):
        with open(train_config_path, "r") as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    logging.warning(
        f"train_config.json not found in {model_dir}, "
        f"falling back to hardcoded defaults. "
        f"Shape mismatch may occur if training used non-default hyperparameters."
    )
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model hyperparameters from ``train_config``; missing keys fall
    back to ``_FALLBACK_MODEL_CFG``.

    Special handling for ``num_time_buckets``: it is not exposed on the CLI
    as an independent hyperparameter; the bucket count is uniquely determined
    by the length of ``dataset.BUCKET_BOUNDARIES``. Resolution order:

      1) ``train_config`` contains ``num_time_buckets`` directly (legacy ckpt)
         -> use that value;
      2) ``train_config`` contains ``use_time_buckets`` (new-style training)
         -> derive as ``NUM_TIME_BUCKETS`` or ``0``;
      3) neither is present -> fall back to ``_FALLBACK_MODEL_CFG[...]``.
    """
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key == "num_time_buckets":
            if "num_time_buckets" in train_config:
                cfg[key] = train_config["num_time_buckets"]
            elif "use_time_buckets" in train_config:
                cfg[key] = NUM_TIME_BUCKETS if train_config["use_time_buckets"] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
                logging.warning(
                    f"train_config missing both 'num_time_buckets' and 'use_time_buckets', "
                    f"using fallback = {cfg[key]}"
                )
            continue

        if key in train_config:
            cfg[key] = train_config[key]
        else:
            cfg[key] = _FALLBACK_MODEL_CFG[key]
            logging.warning(
                f"train_config missing '{key}', using fallback = {cfg[key]}"
            )
    return cfg


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    ns_groups_json: Optional[str] = None,
    device: str = "cpu",
) -> PCVRHyFormer:
    """Construct a ``PCVRHyFormer`` from the dataset schema, an NS-groups JSON,
    and a resolved ``model_cfg`` dict.

    Args:
        dataset: a ``PCVRParquetDataset`` providing the feature schema.
        model_cfg: resolved model hyperparameters, typically the output of
            ``resolve_model_cfg``.
        ns_groups_json: path to the NS-groups JSON file, or ``None`` / empty
            string to disable it (each feature becomes its own singleton group).
        device: torch device.
    """
    # NS grouping. The JSON schema uses *fid* (feature id) values; convert
    # them to positional indices into ``user_int_schema.entries`` /
    # ``item_int_schema.entries`` so ``GroupNSTokenizer`` /
    # ``RankMixerNSTokenizer`` can index ``feature_specs`` directly. This is
    # the same conversion ``train.py`` performs when loading the JSON; doing
    # it here keeps infer.py symmetric with training.
    user_ns_groups: List[List[int]]
    item_ns_groups: List[List[int]]
    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info(f"Loading NS groups from {ns_groups_json}")
        with open(ns_groups_json, "r") as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)
        }
        item_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)
        }
        try:
            user_ns_groups = [
                [user_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg["user_ns_groups"].values()
            ]
            item_ns_groups = [
                [item_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg["item_ns_groups"].values()
            ]
        except KeyError as exc:
            raise KeyError(
                f"NS-groups JSON references fid {exc.args[0]} which is not "
                f"present in the checkpoint's schema.json. The ns_groups.json "
                f"and schema.json must come from the same training run."
            ) from exc
    else:
        logging.info(
            "No NS groups JSON found, using default: each feature as one group"
        )
        user_ns_groups = [[i] for i in range(len(dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(dataset.item_int_schema.entries))]

    # Feature specs.
    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes
    )
    pair_int_feature_specs = build_feature_specs(
        dataset.pair_int_schema, dataset.pair_int_vocab_sizes
    )
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes
    )

    # The hist branch is mandatory at infer time. HistUserEncoder reads the
    # FULL user-side schema (user_int + pair_int + user_dense + pair_dense)
    # so no fid-position resolution is needed — the encoder mirrors the
    # backbone tokenizers and consumes the same dataset buffers.
    model_cfg = {**model_cfg, "enable_hist_users": True}

    logging.info(f"Building PCVRHyFormer with cfg: {model_cfg}")
    model = PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        pair_int_feature_specs=pair_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        **model_cfg,
    ).to(device)

    return model


def load_model_state_strict(
    model: nn.Module,
    ckpt_path: str,
    device: str,
) -> None:
    """Strictly load ``state_dict``; any missing/unexpected key fails fast
    with a diagnostic message.
    """
    state_dict = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        logging.error(
            "Failed to load state_dict in strict mode. This usually means the "
            "model constructed by build_model does NOT match the checkpoint. "
            "Check that train_config.json in the ckpt dir is present and matches "
            "the training hyperparameters."
        )
        raise e


def get_ckpt_path() -> Optional[str]:
    """Locate the first ``*.pt`` file inside the directory pointed at by
    ``$MODEL_OUTPUT_PATH``. Returns ``None`` if no checkpoint is found.
    """
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if not ckpt_path:
        return None
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
    return None


def _batch_to_model_input(
    batch: Dict[str, Any],
    device: str,
) -> ModelInput:
    """Convert a batch dict to ``ModelInput``, handling dynamic seq domains."""
    device_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            device_batch[k] = v.to(device, non_blocking=True)
        else:
            device_batch[k] = v

    seq_domains = device_batch["_seq_domains"]
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    seq_ts_float_feats: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f"{domain}_len"]
        B, _, L = device_batch[domain].shape
        seq_time_buckets[domain] = device_batch.get(
            f"{domain}_time_bucket", torch.zeros(B, L, dtype=torch.long, device=device)
        )
        tf_key = f"{domain}_ts_float_feats"
        if tf_key in device_batch:
            seq_ts_float_feats[domain] = device_batch[tf_key]
        else:
            seq_ts_float_feats[domain] = torch.zeros(
                B, 8, L, dtype=torch.float32, device=device
            )

    seq_ts_stat_feats: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        sk = f"{domain}_ts_stat_feats"
        B, _, L = device_batch[domain].shape

        if sk in device_batch:
            seq_ts_stat_feats[domain] = device_batch[sk]
        else:
            seq_ts_stat_feats[domain] = torch.zeros(B, 6, dtype=torch.float32, device=device)

    return ModelInput(
        user_int_feats=device_batch["user_int_feats"],
        item_int_feats=device_batch["item_int_feats"],
        pair_int_feats=device_batch["pair_int_feats"],
        user_dense_feats=device_batch["user_dense_feats"],
        item_dense_feats=device_batch["item_dense_feats"],
        pair_dense_feats=device_batch["pair_dense_feats"],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
        seq_ts_float_feats=seq_ts_float_feats,
        seq_ts_stat_feats=seq_ts_stat_feats,
        hist_pos_user_int=device_batch.get("hist_pos_user_int"),
        hist_pos_user_dense=device_batch.get("hist_pos_user_dense"),
        hist_pos_pair_int=device_batch.get("hist_pos_pair_int"),
        hist_pos_pair_dense=device_batch.get("hist_pos_pair_dense"),
        hist_neg_user_int=device_batch.get("hist_neg_user_int"),
        hist_neg_user_dense=device_batch.get("hist_neg_user_dense"),
        hist_neg_pair_int=device_batch.get("hist_neg_pair_int"),
        hist_neg_pair_dense=device_batch.get("hist_neg_pair_dense"),
        hist_pos_lens=device_batch.get("hist_pos_lens"),
        hist_neg_lens=device_batch.get("hist_neg_lens"),
    )


def main() -> None:
    # ---- Read environment variables ----
    model_dir = os.environ.get("MODEL_OUTPUT_PATH")
    data_dir = os.environ.get("EVAL_DATA_PATH")
    result_dir = os.environ.get("EVAL_RESULT_PATH")
    # Directory produced by build_item_hist_users.py over the training data.
    # The same file serves training and inference — test rows look up their
    # item_id in it, and because test ts > all train ts the temporal cut at
    # runtime trivially exposes the full training history. REQUIRED.
    hist_users_dir = os.environ.get("EVAL_HIST_USERS_DIR")
    if not hist_users_dir:
        raise ValueError(
            "EVAL_HIST_USERS_DIR is not set. The hist branch is mandatory at "
            "infer time; build the item-history table with build_item_hist_users.py "
            "and point EVAL_HIST_USERS_DIR at its output directory."
        )

    os.makedirs(result_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Schema: prefer the one from model_dir (to exactly match training);
    #      fall back to the one in data_dir if missing. ----
    schema_path = os.path.join(model_dir, "schema.json")
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, "schema.json")
    logging.info(f"Using schema: {schema_path}")

    # ---- Load train_config.json (single source of truth for all hyperparams) ----
    train_config = load_train_config(model_dir)

    # Sampling budget. Defaults inherit from train_config (and fall back to
    # train.py's CLI defaults 16 / 32) so the cross-attention K-axis
    # distribution at infer matches what the model saw during training; a
    # different K would shift the softmax + padding-ratio + empty-token-
    # frequency distribution that the model was fit to. Env vars override.
    #
    # time_gap=0 at infer (vs ~3600 at train) because every test ts is later
    # than every training ts, so there is no leakage to gate against.
    hist_k_pos = int(
        os.environ.get("EVAL_HIST_K_POS", str(train_config.get("hist_k_pos", 16)))
    )
    hist_k_neg = int(
        os.environ.get("EVAL_HIST_K_NEG", str(train_config.get("hist_k_neg", 32)))
    )
    hist_time_gap = int(os.environ.get("EVAL_HIST_TIME_GAP", "0"))

    # ---- Parse seq_max_lens ----
    sml_str = train_config.get("seq_max_lens", _FALLBACK_SEQ_MAX_LENS)
    seq_max_lens = _parse_seq_max_lens(sml_str)
    logging.info(f"seq_max_lens: {seq_max_lens}")

    # ---- Data loading: reuse batch_size / num_workers from training config ----
    batch_size = int(train_config.get("batch_size", _FALLBACK_BATCH_SIZE))
    num_workers = int(train_config.get("num_workers", _FALLBACK_NUM_WORKERS))

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
        hist_users_dir=hist_users_dir,
        hist_k_pos=hist_k_pos,
        hist_k_neg=hist_k_neg,
        hist_time_gap=hist_time_gap,
    )
    total_test_samples = test_dataset.num_rows
    logging.info(f"Total test samples: {total_test_samples}")

    # ---- Build model: every structural hyperparameter is resolved from train_config ----
    model_cfg = resolve_model_cfg(train_config)

    # dataset._load_hist_users already cross-checked each block's total_dim
    # and per-fid entries against the live schema, so a stale/mismatched
    # hist build raises there. Nothing extra to verify here — the model
    # constructor will further fail at strict state_dict load if shapes
    # disagree.

    # ns_groups_json also comes from training config (e.g. run.sh may have
    # passed an empty string to disable it). When trainer.py has copied the
    # JSON into the ckpt dir, train_config records just the basename, so try
    # resolving against ``model_dir`` first before honoring the raw (possibly
    # absolute) path as a fallback.
    ns_groups_json = train_config.get("ns_groups_json", None)
    if ns_groups_json:
        local_candidate = os.path.join(model_dir, os.path.basename(ns_groups_json))
        if os.path.exists(local_candidate):
            ns_groups_json = local_candidate

    model = build_model(
        test_dataset,
        model_cfg=model_cfg,
        ns_groups_json=ns_groups_json,
        device=device,
    )

    # ---- Strictly load weights ----
    ckpt_path = get_ckpt_path()
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}. "
            f"The directory contains: {os.listdir(model_dir) if model_dir and os.path.isdir(model_dir) else 'N/A'}. "
            "This typically means the training job wrote only the sidecar "
            "files (schema.json / train_config.json) for this step but did "
            "not persist model.pt — a symptom of a race between "
            "_remove_old_best_dirs and EarlyStopping.save_checkpoint."
        )
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=min(8, num_workers),
        prefetch_factor=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    all_logits = []
    all_user_ids = []
    logging.info("Starting inference...")
    model.eval()

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch_idx, batch in enumerate(test_loader):
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get("user_id", [])

            logits, _ = model.predict(model_input)
            logits = logits.squeeze(-1)
            all_logits.append(logits.detach().float().cpu())
            all_user_ids.extend(user_ids)

            if (batch_idx + 1) % 200 == 0:
                logging.info(f"  Processed {(batch_idx + 1) * batch_size} samples")

    all_logits = torch.cat(all_logits)
    all_probs = torch.sigmoid(all_logits).numpy()
    logging.info(f"Inference complete: {len(all_probs)} predictions")

    predictions = {
        "predictions": {uid: float(p) for uid, p in zip(all_user_ids, all_probs)}
    }

    # ---- Save predictions.json ----
    output_path = os.path.join(result_dir, "predictions.json")
    with open(output_path, "w") as f:
        json.dump(predictions, f)
    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()
