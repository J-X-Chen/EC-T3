#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -z "${CONDA_PREFIX:-}" && -x "$HOME/miniconda3/envs/hecrl/bin/python" ]]; then
    PYTHON_BIN="$HOME/miniconda3/envs/hecrl/bin/python"
fi

if [[ $# -lt 2 ]]; then
    cat <<'EOF'
Usage:
  bash commands/train_cubexy.sh X Y [agent] [extra hydra overrides...]

Agents:
  sgiql      EC-SGIQL with diffusion subgoals
  sgiql_dlp  EC-SGIQL with DLP image observations
  sgiql_awr  EC-SGIQL with standard/AWR-style subgoals
  ec_iql     Entity-centric IQL
  ec_iql_dlp Entity-centric IQL with DLP image observations
  hiql       HIQL on state observations
  iql        IQL on state observations

Examples:
  bash commands/train_cubexy.sh 2 3 sgiql
  bash commands/train_cubexy.sh 0 3 sgiql_dlp
  DLP_CHECKPOINT=visual_encoders/chkpts/dlp-mv-cube2+3-v0-smoke2 REP_MODEL_DEVICE=cuda:1 bash commands/train_cubexy.sh 2 3 sgiql_dlp
  bash commands/train_cubexy.sh 2 3 hiql steps=500000 enable_wandb=false
EOF
    exit 1
fi

X="$1"
Y="$2"
shift 2

if ! [[ "$X" =~ ^[0-9]+$ && "$Y" =~ ^[0-9]+$ ]]; then
    echo "X and Y must both be non-negative integers." >&2
    exit 1
fi

AGENT="sgiql"
if [[ $# -gt 0 && "$1" != *=* ]]; then
    AGENT="$1"
    shift 1
fi

TOTAL=$((X + Y))
DATASET_NAME="cube${X}+${Y}-noisy-v0-mv"
TASK_MODE="task_cube${X}+${Y}"
WANDB_SUFFIX="-cube${X}+${Y}"
DLP_CHECKPOINT="${DLP_CHECKPOINT:-visual_encoders/chkpts/dlp-mv-cube${X}+${Y}-v0}"
REP_MODEL_DEVICE="${REP_MODEL_DEVICE:-cuda:0}"

recommended_episode_steps() {
    case "$1" in
        1) echo 500 ;;
        2) echo 800 ;;
        3) echo 1000 ;;
        4) echo 1200 ;;
        5) echo 1500 ;;
        6) echo 2000 ;;
        *) echo $((2000 + ($1 - 6) * 400)) ;;
    esac
}

recommended_num_subgoals() {
    case "$1" in
        1) echo 16 ;;
        2) echo 28 ;;
        3) echo 32 ;;
        4) echo 40 ;;
        5) echo 52 ;;
        6) echo 72 ;;
        *) echo $((12 * $1)) ;;
    esac
}

recommended_steps() {
    echo $((1501000 + 500000 * $1))
}

MAX_EPISODE_STEPS="$(recommended_episode_steps "$TOTAL")"
NUM_SUBGOALS="$(recommended_num_subgoals "$TOTAL")"
STEPS="$(recommended_steps "$TOTAL")"

CMD=(
    "$PYTHON_BIN" train_cubexy.py
    task=manipobj-v0
    train_dataset_name="${DATASET_NAME}"
    steps="${STEPS}"
    max_episode_steps="${MAX_EPISODE_STEPS}"
    env_kwargs.manipobj.mode="${TASK_MODE}"
    env_kwargs.manipobj.num_cubes="${TOTAL}"
    wandb_name_suffix="${WANDB_SUFFIX}"
)

case "$AGENT" in
    sgiql)
        CMD+=(
            obs=ec_state_gen
            agent=sgiql
            sgiql.alpha=0.05
            sgiql.subgoal_policy_type=diffusion
            sgiql.n_diffusion_steps=10
            sgiql.subgoal_k=50
            sgiql.n_diffusion_samples=256
            sgiql.subgoal_steps=25
            sgiql.num_subgoals="${NUM_SUBGOALS}"
            sgiql.value_competence_radius=-30
        )
        ;;
    sgiql_dlp|sgiql-dlp)
        CMD+=(
            obs=dlp
            multiview=true
            rep_model_checkpoint="${DLP_CHECKPOINT}"
            rep_model_device="${REP_MODEL_DEVICE}"
            agent=sgiql
            sgiql.alpha=0.2
            sgiql.subgoal_policy_type=diffusion
            sgiql.n_diffusion_steps=10
            sgiql.subgoal_k=50
            sgiql.n_diffusion_samples=256
            sgiql.subgoal_steps=25
            sgiql.num_subgoals="${NUM_SUBGOALS}"
            sgiql.value_competence_radius=-30
            num_eval_episodes=50
            eval_freq=250000
        )
        ;;
    sgiql_awr|sgiql-awr)
        CMD+=(
            obs=ec_state_gen
            agent=sgiql
            sgiql.alpha=0.05
            sgiql.subgoal_policy_type=standard
            sgiql.beta=3.0
            sgiql.subgoal_k=50
            sgiql.subgoal_steps=25
            sgiql.num_subgoals="${NUM_SUBGOALS}"
        )
        ;;
    ec_iql|eciql)
        CMD+=(
            obs=ec_state_gen
            agent=iql
            iql.alpha=0.05
        )
        ;;
    ec_iql_dlp|eciql_dlp|ec-iql-dlp)
        CMD+=(
            obs=dlp
            multiview=true
            rep_model_checkpoint="${DLP_CHECKPOINT}"
            rep_model_device="${REP_MODEL_DEVICE}"
            agent=iql
            iql.alpha=0.2
            num_eval_episodes=50
            eval_freq=250000
        )
        ;;
    hiql)
        CMD+=(
            obs=state
            agent=hiql
            hiql.alpha=0.2
            hiql.beta=3.0
            hiql.subgoal_k=50
            hiql.subgoal_steps=25
            hiql.num_subgoals="${NUM_SUBGOALS}"
        )
        ;;
    iql)
        CMD+=(
            obs=state
            agent=iql
            iql.alpha=0.05
        )
        ;;
    *)
        echo "Unsupported agent preset: ${AGENT}" >&2
        exit 1
        ;;
esac

if [[ $# -gt 0 ]]; then
    CMD+=("$@")
fi

printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
