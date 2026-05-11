#!/bin/bash
# Run only the time-bucket EDA script.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Debug mode (uses sibling ../data / ../output paths) ----
DEBUG_MODE=false
if [ "$1" = "debug" ]; then
    DEBUG_MODE=true
    echo "[run.sh] Running EDA in DEBUG mode"
    shift
fi

if [ "$DEBUG_MODE" = true ]; then
    export TRAIN_DATA_PATH="${SCRIPT_DIR}/../data"
    export EDA_SCHEMA_PATH="${SCRIPT_DIR}/schema.json"
    export EDA_MAX_ROWS="${EDA_MAX_ROWS:-50000}"
    export EDA_OUT_JSON="${SCRIPT_DIR}/../output/eda_time_buckets.json"
    mkdir -p "$(dirname "$EDA_OUT_JSON")"
fi

: "${EDA_MAX_ROWS:=200000}"
export EDA_MAX_ROWS

python3 -u "${SCRIPT_DIR}/eda_time_buckets.py" "$@"
