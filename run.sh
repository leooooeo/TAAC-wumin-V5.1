#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
VALID_RATIO=0.1

# ---- Item-history-user (audience matching) add-on ----
# Tag identifies a specific build of the hist lookup tables (k_pos / k_neg /
# time_gap / seed). Bump it whenever those change so train and infer pick up
# the same artifacts. Output lives under $USER_CACHE_PATH so inference can
# share it with training.
HIST_TAG="${HIST_TAG:-kp16_kn32_g3600_v1}"
HIST_K_POS=16
HIST_K_NEG=32
HIST_TIME_GAP=3600
HIST_DROPOUT=0.1

# ---- Debug mode ----
DEBUG_MODE=false
BUILD_ONLY=false
if [ "$1" = "debug" ]; then
    DEBUG_MODE=true
    echo "[run.sh] Running in DEBUG mode"
    shift
fi
if [ "$1" = "build_hist" ]; then
    BUILD_ONLY=true
    echo "[run.sh] BUILD-ONLY mode: only build the item-history-user lookup"
    shift
fi

if [ "$DEBUG_MODE" = true ]; then
    export TRAIN_DATA_PATH="${SCRIPT_DIR}/../data"
    export TRAIN_CKPT_PATH="${SCRIPT_DIR}/../output/ckpt"
    export TRAIN_LOG_PATH="${SCRIPT_DIR}/../output/log"
    export TRAIN_TF_EVENTS_PATH="${SCRIPT_DIR}/../output/tf_events"
    export USER_CACHE_PATH="${SCRIPT_DIR}/../cache"

    mkdir -p "$TRAIN_CKPT_PATH" "$TRAIN_LOG_PATH" "$TRAIN_TF_EVENTS_PATH" "$USER_CACHE_PATH"
fi

# ---- Resolve hist lookup directory under USER_CACHE_PATH ----
HIST_USERS_DIR="${USER_CACHE_PATH}/item_hist_${HIST_TAG}"

# ---- Build hist lookup if missing (or forced via build_hist) ----
if [ "$BUILD_ONLY" = true ] || [ ! -f "${HIST_USERS_DIR}/meta.json" ]; then
    echo "[run.sh] Building item-history-user lookup -> ${HIST_USERS_DIR}"
    if ! python3 -u "${SCRIPT_DIR}/build_item_hist_users.py" \
        --data_dir "${TRAIN_DATA_PATH}" \
        --out_dir "${HIST_USERS_DIR}" \
        --k_pos ${HIST_K_POS} \
        --k_neg ${HIST_K_NEG} \
        --time_gap ${HIST_TIME_GAP}; then
        echo "[run.sh] build_item_hist_users.py FAILED — aborting before train"
        exit 1
    fi
    if [ ! -f "${HIST_USERS_DIR}/meta.json" ]; then
        echo "[run.sh] build script returned 0 but meta.json is missing — aborting"
        exit 1
    fi
else
    echo "[run.sh] Reusing existing hist lookup at ${HIST_USERS_DIR}"
fi
if [ "$BUILD_ONLY" = true ]; then
    exit 0
fi

# 根据 DEBUG_MODE 设置不同的训练参数
if [ "$DEBUG_MODE" = true ]; then
    # Debug 模式参数
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
        --hist_users_dir "${HIST_USERS_DIR}" \
        --hist_dropout ${HIST_DROPOUT} \
        "$@"
else
    # 正常模式参数
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
        --hist_users_dir "${HIST_USERS_DIR}" \
        --hist_dropout ${HIST_DROPOUT} \
        "$@"
fi
