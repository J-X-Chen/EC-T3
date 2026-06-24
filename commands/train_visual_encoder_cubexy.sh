#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -z "${CONDA_PREFIX:-}" && -x "$HOME/miniconda3/envs/hecrl/bin/python" ]]; then
    PYTHON_BIN="$HOME/miniconda3/envs/hecrl/bin/python"
fi

if [[ $# -lt 2 ]]; then
    cat <<'EOF'
Usage:
  bash commands/train_visual_encoder_cubexy.sh X Y
  bash commands/train_visual_encoder_cubexy.sh OBJECT_PREFIX X Y [DATASET_KIND]

Environment overrides:
  DATASET_NAME           Full dataset stem (default: <object>X+Y-<kind>-v0-mv)
  DATASET_KIND           Dataset kind suffix (default: noisy)
  TRAIN_NPZ             Override train npz path
  VAL_NPZ               Override validation npz path
  DATA_DIV              Dataset subsampling factor for encoder training (default: 3)
  BATCH_SIZE            DLP batch size (default: 32)
  NUM_EPOCHS            DLP epochs (default: 250)
  NUM_WORKERS           DLP dataloader workers (default: 4)
  EVAL_EPOCH_FREQ       Full validation/checkpoint frequency in epochs (default: 5)
  N_KP_ENC              Override encoder keypoint count
  N_KP_PRIOR            Override prior keypoint count
  LEARNED_FEATURE_DIM   DLP feature dim (default: 8 for cube, 16 otherwise)
  DATA_ROOT_DIR         Override converted image dataset directory
  CHKPT_DIR             Override exported checkpoint directory
  MAX_STEPS_PER_EPISODE Keep only the first N steps of each episode for a fast smoke test
  FORCE_REBUILD         Set to 1 to rebuild the converted image dataset

Example:
  bash commands/train_visual_encoder_cubexy.sh 2 3
  bash commands/train_visual_encoder_cubexy.sh tableware 1 2
  DATA_DIV=1 NUM_EPOCHS=150 bash commands/train_visual_encoder_cubexy.sh 2 3
EOF
    exit 1
fi

if [[ "$1" =~ ^[0-9]+$ ]]; then
    OBJECT_PREFIX="${OBJECT_PREFIX:-cube}"
    X="$1"
    Y="$2"
    DATASET_KIND="${DATASET_KIND:-${3:-noisy}}"
else
    if [[ $# -lt 3 ]]; then
        echo "Usage for non-cube datasets: bash commands/train_visual_encoder_cubexy.sh OBJECT_PREFIX X Y [DATASET_KIND]" >&2
        exit 1
    fi
    OBJECT_PREFIX="${OBJECT_PREFIX:-$1}"
    X="$2"
    Y="$3"
    DATASET_KIND="${DATASET_KIND:-${4:-noisy}}"
fi

if ! [[ "$X" =~ ^[0-9]+$ && "$Y" =~ ^[0-9]+$ ]]; then
    echo "X and Y must both be non-negative integers." >&2
    exit 1
fi
if ! [[ "$OBJECT_PREFIX" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "OBJECT_PREFIX must contain only letters, numbers, underscores, or dashes." >&2
    exit 1
fi
if ! [[ "$DATASET_KIND" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "DATASET_KIND must contain only letters, numbers, underscores, or dashes." >&2
    exit 1
fi

TOTAL=$((X + Y))
DATASET_STEM="${OBJECT_PREFIX}${X}+${Y}"
DATASET_NAME="${DATASET_NAME:-${DATASET_STEM}-${DATASET_KIND}-v0-mv}"
TRAIN_NPZ="${TRAIN_NPZ:-datasets/data/${DATASET_NAME}.npz}"
VAL_NPZ="${VAL_NPZ:-datasets/data/${DATASET_NAME}-val.npz}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-visual_encoders/data/${DATASET_NAME}}"
CHKPT_DIR="${CHKPT_DIR:-visual_encoders/chkpts/dlp-mv-${DATASET_STEM}-v0}"

recommended_n_kp_enc() {
    local total="$1"
    local object_prefix="$2"
    if [[ "$object_prefix" == "cube" ]]; then
        case "$total" in
            1|2|3) echo 20 ;;
            4) echo 24 ;;
            5) echo 28 ;;
            6) echo 30 ;;
            *) echo $((20 + 2 * (total - 3))) ;;
        esac
    else
        case "$total" in
            1|2) echo 24 ;;
            3) echo 28 ;;
            4) echo 32 ;;
            5) echo 36 ;;
            6) echo 40 ;;
            *) echo $((40 + 2 * (total - 6))) ;;
        esac
    fi
}

recommended_feature_dim() {
    case "$1" in
        cube) echo 8 ;;
        *) echo 16 ;;
    esac
}

DATA_DIV="${DATA_DIV:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-250}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_EPOCH_FREQ="${EVAL_EPOCH_FREQ:-5}"
LEARNED_FEATURE_DIM="${LEARNED_FEATURE_DIM:-$(recommended_feature_dim "$OBJECT_PREFIX")}"
N_KP_ENC="${N_KP_ENC:-$(recommended_n_kp_enc "$TOTAL" "$OBJECT_PREFIX")}"
N_KP_PRIOR="${N_KP_PRIOR:-$((N_KP_ENC + 12))}"
export DLP_CUDNN_BENCHMARK="${DLP_CUDNN_BENCHMARK:-1}"
export DLP_CUDNN_DETERMINISTIC="${DLP_CUDNN_DETERMINISTIC:-0}"
FORCE_FLAG=()
if [[ "${FORCE_REBUILD:-0}" == "1" ]]; then
    FORCE_FLAG+=(--force)
fi
MAX_STEPS_FLAG=()
if [[ -n "${MAX_STEPS_PER_EPISODE:-}" ]]; then
    MAX_STEPS_FLAG+=(--max-steps-per-episode "$MAX_STEPS_PER_EPISODE")
fi

if [[ ! -f "$TRAIN_NPZ" || ! -f "$VAL_NPZ" ]]; then
    echo "Missing ${DATASET_STEM} npz dataset. Expected:" >&2
    echo "  $TRAIN_NPZ" >&2
    echo "  $VAL_NPZ" >&2
    exit 1
fi

if [[ ! -d "${DATA_ROOT_DIR}/train" || ! -d "${DATA_ROOT_DIR}/valid" || "${FORCE_REBUILD:-0}" == "1" ]]; then
    "$PYTHON_BIN" visual_encoders/npz_to_dir_format_cli.py \
        --train-npz "$TRAIN_NPZ" \
        --val-npz "$VAL_NPZ" \
        --data-dir "$DATA_ROOT_DIR" \
        --data-div "$DATA_DIV" \
        "${MAX_STEPS_FLAG[@]}" \
        "${FORCE_FLAG[@]}"
else
    echo "Using existing converted dataset at ${DATA_ROOT_DIR}"
fi

CONFIG_PATH="$(mktemp "/tmp/dlp_${DATASET_STEM}.XXXXXX.yaml")"
trap 'rm -f "$CONFIG_PATH"' EXIT

cat > "$CONFIG_PATH" <<EOF
ds: "hecrl_env"
data_root_dir: "${DATA_ROOT_DIR}"
tasks: null
lr: 0.0002
batch_size: ${BATCH_SIZE}
num_epochs: ${NUM_EPOCHS}
num_workers: ${NUM_WORKERS}
load_model: false
eval_epoch_freq: ${EVAL_EPOCH_FREQ}
n_kp: 1
kp_range: [-1, 1]
weight_decay: 0.0
run_prefix: "_mv_${DATASET_STEM}_${N_KP_ENC}kp_${N_KP_PRIOR}kpp_${LEARNED_FEATURE_DIM}zdim"
pad_mode: "replicate"
sigma: 1.0
dropout: 0.0
kp_activation: "tanh"
warmup_epoch: 1
eval_im_metrics: false
beta_kl: 0.1
beta_rec: 1.0
scale_std: 0.3
offset_std: 0.2
n_kp_enc: ${N_KP_ENC}
n_kp_prior: ${N_KP_PRIOR}
patch_size: 16
learned_feature_dim: ${LEARNED_FEATURE_DIM}
bg_learned_feature_dim: 1
topk: 10
recon_loss_type: "mse"
anchor_s: 0.25
kl_balance: 0.001
EOF

"$PYTHON_BIN" -m visual_encoders.dlp.train_dlp_from_config \
    --config "$CONFIG_PATH" \
    --export-dir "$CHKPT_DIR"

echo
echo "DLP checkpoint is ready:"
echo "  ${CHKPT_DIR}"
echo
echo "For image-based ${DATASET_STEM} RL, use:"
echo "  rep_model_checkpoint=${CHKPT_DIR}"
