#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
VALID_RATIO=0.1

# ---- Debug mode ----
DEBUG_MODE=false
if [ "$1" = "debug" ]; then
    DEBUG_MODE=true
    echo "[run.sh] Running in DEBUG mode"
    shift
fi

if [ "$DEBUG_MODE" = true ]; then
    export TRAIN_DATA_PATH="${SCRIPT_DIR}/../data"
    export TRAIN_CKPT_PATH="${SCRIPT_DIR}/../output/ckpt"
    export TRAIN_LOG_PATH="${SCRIPT_DIR}/../output/log"
    export TRAIN_TF_EVENTS_PATH="${SCRIPT_DIR}/../output/tf_events"
    export USER_CACHE_PATH="${SCRIPT_DIR}/../cache"

    mkdir -p "$TRAIN_CKPT_PATH" "$TRAIN_LOG_PATH" "$TRAIN_TF_EVENTS_PATH"
fi


if [ "$DEBUG_MODE" = true ]; then
    python3 -u "${SCRIPT_DIR}/train.py" \
        --lr 2e-4 \
        --num_epochs 2 \
        --num_workers 1 \
        --buffer_batches 4 \
        --d_model 64 \
        --num_queries 2 \
        --num_hyformer_blocks 1 \
        --emb_skip_threshold 1000000 \
        --ns_groups_json "" \
        --ns_tokenizer_type rankmixer \
        --user_ns_tokens 5 \
        --item_ns_tokens 2 \
        --seq_encoder_type transformer \
        --valid_ratio ${VALID_RATIO} \
        --eval_every_n_steps 100 \
        --schema_path "${SCRIPT_DIR}/schema.json" \
        --reinit_cardinality_threshold 999999 \
        "$@"
else
    python3 -u "${SCRIPT_DIR}/train.py" \
        --seed 7789 \
        --lr 1e-4 \
        --num_epochs 7 \
        --num_workers 8 \
        --buffer_batches 32 \
        --d_model 96 \
        --num_queries 2 \
        --num_hyformer_blocks 2 \
        --ns_groups_json "" \
        --ns_tokenizer_type rankmixer \
        --seq_encoder_type transformer \
        --user_ns_tokens 12 \
        --item_ns_tokens 2 \
        --valid_ratio ${VALID_RATIO} \
        --emb_skip_threshold 1000000 \
        --dropout_rate 0.02 \
        "$@"
fi
