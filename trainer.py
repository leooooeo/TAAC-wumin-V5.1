"""PCVRHyFormer pointwise trainer (binary-classification, AUC-monitored).

Despite the historical "Ranking" suffix in the class name, the training loop
uses pointwise BCE / Focal loss and evaluates Binary AUC + binary logloss.
"""

import math
import os, time
import glob
import shutil
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from utils import sigmoid_focal_loss, EarlyStopping
from model import ModelInput, PCVRHyFormer
from dataset import TS_STAT_DIM, TS_FLOAT_DIM


class PCVRHyFormerRankingTrainer:
    """PCVRHyFormer trainer for pointwise binary classification.

    Uses PCVR data layout:
    - user_int_feats, user_dense_feats
    - item_int_feats, item_dense_feats
    - seq_a, seq_b, seq_c, seq_d (each with *_len companion)
    - label (binary)

    Loss: BCEWithLogitsLoss or Focal Loss.
    Metrics: BinaryAUROC + binary logloss.
    """

    def __init__(
        self,
        model: PCVRHyFormer,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float,
        num_epochs: int,
        device: str,
        save_dir: str,
        early_stopping: EarlyStopping,
        loss_type: str = "bce",
        focal_alpha: float = 0.1,
        focal_gamma: float = 2.0,
        sparse_lr: float = 0.05,
        sparse_weight_decay: float = 0.0,
        reinit_sparse_after_epoch: int = 1,
        reinit_cardinality_threshold: int = 0,
        ckpt_params: Optional[Dict[str, Any]] = None,
        writer: Optional[Any] = None,
        schema_path: Optional[str] = None,
        ns_groups_path: Optional[str] = None,
        eval_every_n_steps: int = 0,
        train_config: Optional[Dict[str, Any]] = None,
        use_amp: bool = True,
        amp_dtype: str = "bfloat16",
        speed_log_every_n_steps: int = 20,
    ) -> None:
        self.model: PCVRHyFormer = model
        self.train_loader: DataLoader = train_loader
        self.valid_loader: DataLoader = valid_loader
        self.writer = writer
        # schema_path is copied alongside every checkpoint so that infer.py can
        # rebuild the exact same feature schema the model was trained with.
        self.schema_path: Optional[str] = schema_path
        # ns_groups_path is optional; copied next to schema.json when provided
        # and points at an existing file. Keeping the JSON inside the ckpt dir
        # makes the checkpoint self-contained for evaluation environments that
        # do not ship ns_groups.json separately.
        self.ns_groups_path: Optional[str] = ns_groups_path

        # Dual optimizer: Adagrad for sparse Embeddings, AdamW for dense params.
        self.sparse_optimizer: Optional[torch.optim.Optimizer]
        if hasattr(model, "get_sparse_params"):
            sparse_params = model.get_sparse_params()
            dense_params = model.get_dense_params()
            sparse_param_count = sum(p.numel() for p in sparse_params)
            dense_param_count = sum(p.numel() for p in dense_params)
            logging.info(
                f"Sparse params: {len(sparse_params)} tensors, {sparse_param_count:,} parameters (Adagrad lr={sparse_lr})"
            )
            logging.info(
                f"Dense params: {len(dense_params)} tensors, {dense_param_count:,} parameters (AdamW lr={lr})"
            )
            self.sparse_optimizer = torch.optim.Adagrad(
                sparse_params,
                lr=sparse_lr,
                weight_decay=sparse_weight_decay,
                foreach=False,
            )
            self.dense_optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                dense_params,
                lr=lr,
                betas=(0.9, 0.98),
                foreach=False,
            )
        else:
            self.sparse_optimizer = None
            self.dense_optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=(0.9, 0.98),
                foreach=False,
            )

        # Dense LR scheduler: linear warmup (5% of total steps) + cosine decay.
        # Only applied to dense_optimizer; Adagrad is self-adaptive and needs no schedule.
        _warmup_steps = 500
        _decay_steps = max(1, num_epochs * max(1, len(train_loader)) - _warmup_steps)
        self.dense_scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.dense_optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.dense_optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=_warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.dense_optimizer,
                    T_max=_decay_steps,
                    eta_min=lr / 100,
                ),
            ],
            milestones=[_warmup_steps],
        )

        self.num_epochs: int = num_epochs
        self.device: str = device
        self.save_dir: str = save_dir
        self.early_stopping: EarlyStopping = early_stopping
        self.loss_type: str = loss_type
        self.focal_alpha: float = focal_alpha
        self.focal_gamma: float = focal_gamma
        self.reinit_sparse_after_epoch: int = reinit_sparse_after_epoch
        self.reinit_cardinality_threshold: int = reinit_cardinality_threshold
        self.sparse_lr: float = sparse_lr
        self.sparse_weight_decay: float = sparse_weight_decay
        self.ckpt_params: Dict[str, Any] = ckpt_params or {}
        self.eval_every_n_steps: int = eval_every_n_steps
        self.train_config: Optional[Dict[str, Any]] = train_config
        self.speed_log_every_n_steps: int = max(1, speed_log_every_n_steps)

        logging.info(
            f"PCVRHyFormerRankingTrainer loss_type={loss_type}, "
            f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}, "
            f"reinit_sparse_after_epoch={reinit_sparse_after_epoch}"
        )

        self.use_amp = use_amp and torch.cuda.is_available()

        if amp_dtype == "bfloat16":
            self.amp_dtype = torch.bfloat16
            self.scaler = None
        elif amp_dtype == "float16":
            self.amp_dtype = torch.float16
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

        logging.info(
            f"AMP config: dtype={self.amp_dtype}, use_scaler={self.scaler is not None}"
        )

    def _build_step_dir_name(self, global_step: int, is_best: bool = False) -> str:
        """Build a checkpoint sub-directory name such as
        ``global_step2500.layer=2.head=4.hidden=64[.best_model]``.
        """
        parts = [f"global_step{global_step}"]
        for key in ("layer", "head", "hidden"):
            if key in self.ckpt_params:
                parts.append(f"{key}={self.ckpt_params[key]}")
        name = ".".join(parts)
        if is_best:
            name += ".best_model"
        return name

    def _write_sidecar_files(self, ckpt_dir: str) -> None:
        """Write sidecar files next to a ``model.pt``.

        Currently persists up to three files, all overwritten on every call:

        - ``schema.json`` (copied from ``self.schema_path``): feature layout
          metadata needed to rebuild the Parquet dataset.
        - ``ns_groups.json`` (copied from ``self.ns_groups_path`` when set
          and the file exists): NS-token grouping used to construct the
          tokenizer. Making a per-ckpt copy lets evaluation environments
          consume the checkpoint without having to ship the original
          project-level ``ns_groups.json``.
        - ``train_config.json`` (serialized from ``self.train_config``):
          full set of training-time hyperparameters. When ``ns_groups.json``
          is copied into ``ckpt_dir``, the ``ns_groups_json`` field is
          rewritten to the bare filename so that ``infer.py`` resolves it
          against ``ckpt_dir`` rather than the original absolute path on
          the training machine.
        """
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.schema_path and os.path.exists(self.schema_path):
            shutil.copy2(self.schema_path, ckpt_dir)

        ns_groups_copied = False
        if self.ns_groups_path and os.path.exists(self.ns_groups_path):
            shutil.copy2(self.ns_groups_path, ckpt_dir)
            ns_groups_copied = True

        if self.train_config:
            import json

            cfg_to_dump = self.train_config
            if ns_groups_copied:
                # Override the stored path to a filename relative to ckpt_dir;
                # infer.py already falls back to `<ckpt_dir>/<basename>` when
                # the recorded path is not absolute, which keeps the ckpt
                # portable across hosts.
                cfg_to_dump = dict(self.train_config)
                cfg_to_dump["ns_groups_json"] = os.path.basename(self.ns_groups_path)
            with open(os.path.join(ckpt_dir, "train_config.json"), "w") as f:
                json.dump(cfg_to_dump, f, indent=2)

    def _save_step_checkpoint(
        self,
        global_step: int,
        is_best: bool = False,
        skip_model_file: bool = False,
    ) -> str:
        """Save ``model.pt`` plus sidecar files under a ``global_step`` sub-dir.

        Args:
            global_step: current global step used to name the directory.
            is_best: whether this is a new-best checkpoint.
            skip_model_file: if True, skip writing ``model.pt`` (because the
                caller, e.g. EarlyStopping, has already persisted it to the
                same path). Sidecar files are still (re)written.

        Returns:
            The absolute path of the checkpoint directory.
        """
        dir_name = self._build_step_dir_name(global_step, is_best=is_best)
        ckpt_dir = os.path.join(self.save_dir, dir_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        if not skip_model_file:
            torch.save(self.model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
        self._write_sidecar_files(ckpt_dir)
        logging.info(f"Saved checkpoint to {ckpt_dir}/model.pt")
        return ckpt_dir

    def _remove_old_best_dirs(self) -> None:
        """Delete stale ``*.best_model`` directories so that only the latest
        best checkpoint is kept on disk.
        """
        pattern = os.path.join(self.save_dir, "global_step*.best_model")
        for old_dir in glob.glob(pattern):
            shutil.rmtree(old_dir)
            logging.info(f"Removed old best_model dir: {old_dir}")

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move all tensors in ``batch`` to ``self.device`` (``non_blocking=True``,
        to cooperate with ``pin_memory``). Non-tensor values pass through.
        """
        device_batch: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.device, non_blocking=True)
            else:
                device_batch[k] = v
        return device_batch

    def _handle_validation_result(
        self,
        total_step: int,
        val_auc: float,
        val_logloss: float,
    ) -> None:
        """Persist a new-best checkpoint atomically.

        Flow (ordered to avoid leaving empty sidecar-only directories on disk):

        1. Decide whether ``val_auc`` is *likely* to beat the current best
           using the same threshold as ``EarlyStopping._is_not_improved``,
           so our pre-cleanup and EarlyStopping's internal save decision
           stay in sync.
        2. If unlikely, short-circuit: do nothing on disk. We must NOT
           touch ``self.early_stopping.checkpoint_path`` or call
           ``_write_sidecar_files`` because the target directory may not
           exist yet (sidecar-only dirs would otherwise be created here,
           producing checkpoints with missing ``model.pt``).
        3. If likely, point ``EarlyStopping`` at the canonical
           ``global_stepN.best_model/model.pt`` path, remove any stale
           ``*.best_model`` dirs, then run ``EarlyStopping`` (which writes
           ``model.pt`` when it actually confirms a new best).
        4. Only after ``EarlyStopping`` has confirmed a new best
           (``best_score != old_best``) do we write the sidecar files into
           the freshly-created directory; this is guarded so that a
           razor-close score that tripped ``is_likely_new_best`` but not
           ``EarlyStopping``'s own gate does not create a stray dir.
        """
        # Point EarlyStopping at the canonical best-model location for this
        # step. Only done on the likely-new-best branch so that a skipped
        # save never leaks the unused path into EarlyStopping state.
        best_dir = os.path.join(
            self.save_dir,
            self._build_step_dir_name(total_step, is_best=True),
        )
        self.early_stopping.checkpoint_path = os.path.join(best_dir, "model.pt")

        # Remove stale best dirs first so EarlyStopping's write is the only
        # I/O needed when a new best is confirmed.
        # self._remove_old_best_dirs()

        self.early_stopping(
            val_auc,
            self.model,
            {
                "best_val_AUC": val_auc,
                "best_val_logloss": val_logloss,
            },
        )

        # Write sidecar files only when EarlyStopping actually confirmed a
        # new best and wrote model.pt. If the score tripped our heuristic
        # but EarlyStopping internally declined to save, skip to avoid
        # creating an empty (sidecar-only) checkpoint directory.
        if os.path.exists(self.early_stopping.checkpoint_path):
            self._save_step_checkpoint(total_step, is_best=True, skip_model_file=True)

    def train(self) -> None:
        """Main training loop: iterates over epochs, performs step-level and
        epoch-level validation, triggers EarlyStopping and the periodic sparse
        re-initialization strategy.
        """
        print("Start training (PCVRHyFormer)")
        self.model.train()
        total_step = 0
        self._speed_window_t0 = time.perf_counter()

        for epoch in range(1, self.num_epochs + 1):
            train_pbar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                dynamic_ncols=True,
            )
            loss_sum = 0.0

            for step, batch in train_pbar:
                loss, logits, label, grad_norm = self._train_step(batch)
                total_step += 1
                loss_sum += loss

                if self.writer and total_step % self.speed_log_every_n_steps == 0:
                    elapsed = time.perf_counter() - self._speed_window_t0
                    if elapsed > 0:
                        sps = self.speed_log_every_n_steps / elapsed
                        self.writer.add_scalar(
                            "Speed/train_steps_per_sec", sps, total_step
                        )
                    self._speed_window_t0 = time.perf_counter()

                if self.writer and total_step % 20 == 0:
                    self.writer.add_scalar("Loss/train", loss, total_step)
                    self.writer.add_scalar(
                        "LR/dense", self.dense_optimizer.param_groups[0]["lr"], total_step
                    )
                    if self.sparse_optimizer is not None:
                        self.writer.add_scalar(
                            "LR/sparse", self.sparse_optimizer.param_groups[0]["lr"], total_step
                        )
                    # moniter the logits
                    self.writer.add_scalar(
                        "Logits/train_mean", logits.mean(), total_step
                    )
                    self.writer.add_scalar("Logits/train_std", logits.std(), total_step)
                    self.writer.add_scalar("Logits/train_min", logits.min(), total_step)
                    self.writer.add_scalar("Logits/train_max", logits.max(), total_step)
                    # moniter the grad norm
                    self.writer.add_scalar("Grad/train_norm", grad_norm, total_step)
                    seq_weights = getattr(self.model, "last_seq_weights", None)
                    seq_domains = getattr(self.model, "seq_domains", None)
                    if seq_weights is not None and seq_domains is not None:
                        seq_weights_cpu = seq_weights.detach().float().cpu()
                        for i, domain in enumerate(seq_domains):
                            self.writer.add_scalar(
                                f"SeqWeight/{domain}",
                                seq_weights_cpu[:, i].mean(),
                                total_step,
                            )

                    pos_mask = label == 1
                    neg_mask = label == 0

                    if pos_mask.any():
                        pos_logits = logits[pos_mask]
                        self.writer.add_scalar(
                            "Logits/pos_mean", pos_logits.mean(), total_step
                        )
                        self.writer.add_scalar(
                            "Logits/pos_std", pos_logits.std(), total_step
                        )
                    else:
                        self.writer.add_scalar("Logits/pos_mean", 0.0, total_step)

                    if neg_mask.any():
                        neg_logits = logits[neg_mask]
                        self.writer.add_scalar(
                            "Logits/neg_mean", neg_logits.mean(), total_step
                        )
                        self.writer.add_scalar(
                            "Logits/neg_std", neg_logits.std(), total_step
                        )
                    else:
                        self.writer.add_scalar("Logits/neg_mean", 0.0, total_step)

                train_pbar.set_postfix({"loss": f"{loss:.4f}"})

                # Step-level validation (only when eval_every_n_steps > 0).
                if (
                    self.eval_every_n_steps > 0
                    and total_step % self.eval_every_n_steps == 0
                ):
                    logging.info(f"Evaluating at step {total_step}")
                    val_auc, val_logloss = self.evaluate(epoch=epoch)
                    self.model.train()
                    torch.cuda.empty_cache()

                    logging.info(
                        f"Step {total_step} Validation | AUC: {val_auc}, LogLoss: {val_logloss}"
                    )

                    if self.writer:
                        self.writer.add_scalar("AUC/valid", val_auc, total_step)
                        self.writer.add_scalar("LogLoss/valid", val_logloss, total_step)
                        wa = getattr(self, "_last_warm_auc", None)
                        ca = getattr(self, "_last_cold_auc", None)
                        if wa is not None and not math.isnan(wa):
                            self.writer.add_scalar("AUC/valid_warm", wa, total_step)
                        if ca is not None and not math.isnan(ca):
                            self.writer.add_scalar("AUC/valid_cold", ca, total_step)

                    self._handle_validation_result(total_step, val_auc, val_logloss)

                    if self.early_stopping.early_stop:
                        logging.info(f"Early stopping at step {total_step}")
                        return

            logging.info(
                f"Epoch {epoch}, Average Loss: {loss_sum / len(self.train_loader)}"
            )

            val_auc, val_logloss = self.evaluate(epoch=epoch)
            self.model.train()
            torch.cuda.empty_cache()

            logging.info(
                f"Epoch {epoch} Validation | AUC: {val_auc}, LogLoss: {val_logloss}"
            )

            if self.writer:
                self.writer.add_scalar("AUC/valid", val_auc, total_step)
                self.writer.add_scalar("LogLoss/valid", val_logloss, total_step)
                wa = getattr(self, "_last_warm_auc", None)
                ca = getattr(self, "_last_cold_auc", None)
                if wa is not None and not math.isnan(wa):
                    self.writer.add_scalar("AUC/valid_warm", wa, total_step)
                if ca is not None and not math.isnan(ca):
                    self.writer.add_scalar("AUC/valid_cold", ca, total_step)

            self._handle_validation_result(total_step, val_auc, val_logloss)

            if self.early_stopping.early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            # After the configured epoch, reinitialize high-cardinality sparse
            # params (Embeddings) as a form of cold restart to reduce overfit.
            # Reference: KuaiShou Tech., "MultiEpoch: Reusing Training Data
            # for Click-Through Rate Prediction",
            # https://arxiv.org/pdf/2305.19531
            if (
                epoch >= self.reinit_sparse_after_epoch
                and self.sparse_optimizer is not None
            ):
                # Snapshot Adagrad state per parameter via data_ptr, so state
                # of low-cardinality embeddings can be preserved across rebuild.
                old_state: Dict[int, Any] = {}
                for group in self.sparse_optimizer.param_groups:
                    for p in group["params"]:
                        if p.data_ptr() in self.sparse_optimizer.state:
                            old_state[p.data_ptr()] = self.sparse_optimizer.state[p]

                reinit_ptrs = self.model.reinit_high_cardinality_params(
                    self.reinit_cardinality_threshold
                )
                sparse_params = self.model.get_sparse_params()
                self.sparse_optimizer = torch.optim.Adagrad(
                    sparse_params,
                    lr=self.sparse_lr,
                    weight_decay=self.sparse_weight_decay,
                )
                # Restore optimizer state for low-cardinality embeddings only.
                restored = 0
                for p in sparse_params:
                    if p.data_ptr() not in reinit_ptrs and p.data_ptr() in old_state:
                        self.sparse_optimizer.state[p] = old_state[p.data_ptr()]
                        restored += 1
                logging.info(
                    f"Rebuilt Adagrad optimizer after epoch {epoch}, "
                    f"restored optimizer state for {restored} low-cardinality params"
                )

    def _make_model_input(self, device_batch: Dict[str, Any]) -> ModelInput:
        """Construct a ``ModelInput`` NamedTuple from a device_batch dict."""
        seq_domains = device_batch["_seq_domains"]
        seq_data = {d: device_batch[d] for d in seq_domains}
        seq_lens = {d: device_batch[f"{d}_len"] for d in seq_domains}
        seq_time_buckets = {
            d: device_batch.get(
                f"{d}_time_bucket",
                torch.zeros(
                    device_batch[d].shape[0],
                    device_batch[d].shape[2],
                    dtype=torch.long,
                    device=self.device,
                ),
            )
            for d in seq_domains
        }
        seq_ts_float_feats = {}
        for d in seq_domains:
            key = f"{d}_ts_float_feats"
            B, _, L = device_batch[d].shape
            if key in device_batch:
                seq_ts_float_feats[d] = device_batch[key]
            else:
                seq_ts_float_feats[d] = torch.zeros(
                    B, TS_FLOAT_DIM, L, dtype=torch.float32, device=self.device
                )

        seq_ts_stat_feats: Dict[str, torch.Tensor] = {}
        for d in seq_domains:
            sk = f"{d}_ts_stat_feats"
            B, _, L = device_batch[d].shape
            if sk in device_batch:
                seq_ts_stat_feats[d] = device_batch[sk]
            else:
                seq_ts_stat_feats[d] = torch.zeros(B, TS_STAT_DIM, dtype=torch.float32, device=self.device)

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
            # Hist tensors are present only when the dataset was constructed
            # with ``hist_users_dir`` AND the model was built with
            # ``enable_hist_users=True``. Otherwise these stay None and the
            # model's optional hist branch is skipped.
            hist_pos_scalars=device_batch.get("hist_pos_scalars"),
            hist_pos_dense=device_batch.get("hist_pos_dense"),
            hist_neg_scalars=device_batch.get("hist_neg_scalars"),
            hist_neg_dense=device_batch.get("hist_neg_dense"),
            hist_pos_lens=device_batch.get("hist_pos_lens"),
            hist_neg_lens=device_batch.get("hist_neg_lens"),
        )

    def _train_step(self, batch: Dict[str, Any]) -> Tuple[float, torch.Tensor]:
        """Run a single training step and return the scalar loss value."""
        device_batch = self._batch_to_device(batch)
        label = device_batch["label"].float()

        self.dense_optimizer.zero_grad()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.zero_grad()

        with torch.amp.autocast(
            device_type="cuda", enabled=self.use_amp, dtype=self.amp_dtype
        ):
            model_input = self._make_model_input(device_batch)
            logits = self.model(model_input)  # (B, 1)
            logits = logits.squeeze(-1)  # (B,)

            if self.loss_type == "focal":
                loss = sigmoid_focal_loss(
                    logits, label, alpha=self.focal_alpha, gamma=self.focal_gamma
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, label)

        # Backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.sparse_optimizer:
                self.scaler.unscale_(self.sparse_optimizer)
            self.scaler.unscale_(self.dense_optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0, foreach=False
            )

            if self.sparse_optimizer:
                self.scaler.step(self.sparse_optimizer)
            self.scaler.step(self.dense_optimizer)
            self.scaler.update()
            self.dense_scheduler.step()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0, foreach=False
            )
            if self.sparse_optimizer:
                self.sparse_optimizer.step()
            self.dense_optimizer.step()
            self.dense_scheduler.step()

        return loss.item(), logits.detach(), label.detach(), grad_norm

    def evaluate(self, epoch: Optional[int] = None) -> Tuple[float, float]:
        """Run validation over ``self.valid_loader`` and return ``(AUC, logloss)``.

        NaN predictions (which can arise from exploding gradients) are filtered
        out before computing both metrics.
        """
        print("Start Evaluation (PCVRHyFormer) - validation")
        self.model.eval()
        if not epoch:
            epoch = -1

        pbar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))

        all_logits_list = []
        all_labels_list = []
        # has_hist_list[i] tracks whether this row had any hist (pos or neg);
        # used to split AUC into warm/cold groups when the hist branch is on.
        all_has_hist_list: list = []

        with torch.no_grad():
            for step, batch in pbar:
                logits, labels = self._evaluate_step(batch)
                all_logits_list.append(logits.detach().cpu())
                all_labels_list.append(labels.detach().cpu())
                # Read pos/neg lens straight from the CPU batch (avoid round-trip)
                pl = batch.get("hist_pos_lens")
                nl = batch.get("hist_neg_lens")
                if pl is not None and nl is not None:
                    all_has_hist_list.append(((pl > 0) | (nl > 0)))

        all_logits = torch.cat(all_logits_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0).long()

        # Binary AUC via sklearn.
        probs = torch.sigmoid(all_logits).numpy()
        labels_np = all_labels.numpy()

        # Filter NaN predictions (may appear if gradients explode).
        nan_mask = np.isnan(probs)
        if nan_mask.any():
            n_nan = int(nan_mask.sum())
            logging.warning(
                f"[Evaluate] {n_nan}/{len(probs)} predictions are NaN, filtering them out"
            )
            valid_mask = ~nan_mask
            probs = probs[valid_mask]
            labels_np = labels_np[valid_mask]
        else:
            valid_mask = np.ones(len(probs), dtype=bool)

        if len(probs) == 0 or len(np.unique(labels_np)) < 2:
            auc = 0.0
        else:
            auc = float(roc_auc_score(labels_np, probs))

        # Binary logloss (same NaN filtering).
        valid_logits = all_logits[~torch.isnan(all_logits)]
        valid_labels = all_labels[~torch.isnan(all_logits)]
        if len(valid_logits) > 0:
            logloss = F.binary_cross_entropy_with_logits(
                valid_logits, valid_labels.float()
            ).item()
        else:
            logloss = float("inf")

        # Warm / cold AUC split (only meaningful when hist branch is active).
        # cold = both pos and neg pools were empty for this row.
        if all_has_hist_list:
            has_hist_np = torch.cat(all_has_hist_list, dim=0).numpy()
            has_hist_np = has_hist_np[valid_mask]
            n_warm = int(has_hist_np.sum())
            n_cold = int((~has_hist_np).sum())
            warm_auc = cold_auc = float("nan")
            if n_warm > 0 and len(np.unique(labels_np[has_hist_np])) >= 2:
                warm_auc = float(roc_auc_score(
                    labels_np[has_hist_np], probs[has_hist_np]
                ))
            if n_cold > 0 and len(np.unique(labels_np[~has_hist_np])) >= 2:
                cold_auc = float(roc_auc_score(
                    labels_np[~has_hist_np], probs[~has_hist_np]
                ))
            logging.info(
                f"[Evaluate] AUC overall={auc:.4f}  "
                f"warm({n_warm})={warm_auc:.4f}  cold({n_cold})={cold_auc:.4f}  "
                f"gap={warm_auc - cold_auc:+.4f}"
            )
            # Stash for caller (train loop) to write to TensorBoard with the
            # correct global_step.
            self._last_warm_auc = warm_auc
            self._last_cold_auc = cold_auc

        return auc, logloss

    def _evaluate_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single validation step and return ``(logits, labels)``."""
        device_batch = self._batch_to_device(batch)
        label = device_batch["label"]

        model_input = self._make_model_input(device_batch)
        logits = self.model(model_input)  # (B, 1)
        logits = logits.squeeze(-1)  # (B,)

        return logits, label
