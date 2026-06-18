#!/bin/bash
# smoke-test.sh
# Ultralight verification that every case study's env, Docker stack, and
# entry points actually work.
#
# Use this BEFORE running reproduce-one-seed.sh or reproduce-full.sh to
# catch install/setup issues quickly.
#
# Each step runs the minimum needed to confirm the pipeline pieces wire
# up. Numbers in parentheses are the smallest viable knob; outputs are
# meaningless — we are only checking that imports / env / model / data
# can flow end-to-end without error.
#
#   AutoRobust - imports + finetune_dynamic_reports.py --smoke-test
#                (8 train samples, 1 optimizer step, no test eval)
#                + autoattack.py (32 timesteps × 4-step episodes).
#                Assumes the Kaggle dataset is in AutoRobust-Case-Study/data/.
#                Snapshots models/distilbert/ if present and restores on exit.
#   Link       - imports + build/start WAVSEP + train_A2C.py (-t 64)
#                + tear down WAVSEP
#   SQiRL      - imports + bring up SQLiMicrobenchmark + sqirl.py
#                (loss_criteria 3, win_criteria 1, max_timestamp 2)
#                + tear down
#   MiniCAGE   - imports + SB3_blue_training.py (NUM_RUNS=1, 1024 timesteps)
#
# Runtime: ~1–3 min on a warm cache; longer on the first run because the
# WAVSEP and MySQL images need to be pulled/built.
#
# Prerequisites:
#   - Run ./setup-envs.sh first to create the four venvs in .venvs/.
#   - Docker must be running if any selected study uses it (Link, SQiRL).
#
# Usage:
#   ./smoke-test.sh                          # all four case studies
#   ./smoke-test.sh autorobust               # just AutoRobust
#   ./smoke-test.sh link sqirl               # subset
#
# Valid case study names: autorobust, link, sqirl, minicage

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENVS_DIR="$SCRIPT_DIR/.venvs"
# Logs live alongside the script so a failed run is easy to inspect.
# Each run overwrites the previous logs.
LOG_DIR="$SCRIPT_DIR/smoke-test-logs"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

cleanup() {
    # Best-effort teardown so a failed smoke test doesn't leave Docker
    # containers running or modified source files lying around.
    docker stop wavsep-train >/dev/null 2>&1 || true
    (cd "$REPO_ROOT/SQiRL-CaseStudy/Modified-SQiRL/SQLiMicrobenchmark" \
        && docker compose down >/dev/null 2>&1) || true

    local sb3="$REPO_ROOT/MiniCAGE-Case-Study/Modified-MiniCAGE/mini_CAGE/SB3_blue_training.py"
    if [ -f "$sb3.bak" ]; then
        mv "$sb3.bak" "$sb3"
    fi

    # AutoRobust: drop smoke classifier/agents, restore any pre-existing
    # classifier we moved aside.
    local ar="$REPO_ROOT/AutoRobust-Case-Study"
    rm -f "$ar/agents/rew1_99999.pt" "$ar/agents/rew1_pomdp_99999.pt"
    if [ -d "$ar/models/distilbert.smoke-backup" ]; then
        rm -rf "$ar/models/distilbert"
        mv "$ar/models/distilbert.smoke-backup" "$ar/models/distilbert"
    fi
}
trap cleanup EXIT

activate_venv() {
    local name=$1
    local venv_path="$VENVS_DIR/$name"
    if [ ! -d "$venv_path" ]; then
        echo "FAIL: venv '$name' not found at $venv_path." >&2
        echo "      Run ./setup-envs.sh first." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$venv_path/bin/activate"
}

pass() { echo "    ok - $1"; }

section() {
    echo ""
    echo "============================================"
    echo " $1"
    echo "============================================"
}

# Resolve which case studies to run from positional args (default: all).
ALL_STUDIES=(autorobust link sqirl minicage)
TARGETS=()
if [ "$#" -eq 0 ]; then
    TARGETS=("${ALL_STUDIES[@]}")
else
    for arg in "$@"; do
        case " ${ALL_STUDIES[*]} " in
            *" $arg "*) TARGETS+=("$arg") ;;
            *)
                echo "ERROR: unknown case study '$arg'." >&2
                echo "       Valid: ${ALL_STUDIES[*]}" >&2
                exit 1
                ;;
        esac
    done
fi
should_run() {
    case " ${TARGETS[*]} " in
        *" $1 "*) return 0 ;;
        *) return 1 ;;
    esac
}

echo "============================================"
echo " DRL4Sec-Pitfalls: Install Smoke Test"
echo "============================================"
echo "Targets: ${TARGETS[*]}"
echo "Logs:    $LOG_DIR"

# Docker is only required for Link and SQiRL.
if should_run link || should_run sqirl; then
    if ! command -v docker >/dev/null 2>&1; then
        echo "FAIL: docker not found on PATH." >&2
        exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
        echo "FAIL: docker daemon not reachable. Start Docker Desktop / dockerd." >&2
        exit 1
    fi
    pass "docker reachable"
fi

# -----------------------------------------------
# 1. AutoRobust
# -----------------------------------------------
if should_run autorobust; then
section "[1/4] AutoRobust"
activate_venv autorobust
python - <<'PY' >"$LOG_DIR/autorobust-imports.log" 2>&1
import torch, gymnasium, stable_baselines3, transformers
import sentence_transformers, datasets, peft
import pandas, matplotlib, nltk, sklearn
PY
pass "venv imports work"

cd "$REPO_ROOT/AutoRobust-Case-Study"
if [ ! -d data ] || [ -z "$(ls -A data 2>/dev/null)" ]; then
    echo "FAIL: AutoRobust-Case-Study/data/ is empty. Download the Kaggle" >&2
    echo "      malware-and-goodware-dynamic-analysis-reports dataset first." >&2
    exit 1
fi
mkdir -p agents results models

# Snapshot any existing trained classifier so the smoke run does not
# clobber it. The trap restores it on exit (success or failure).
if [ -d models/distilbert ] && [ ! -d models/distilbert.smoke-backup ]; then
    mv models/distilbert models/distilbert.smoke-backup
    pass "snapshotted existing models/distilbert → models/distilbert.smoke-backup"
fi

# Pipe-through verification only: --smoke-test caps each split to 8
# samples, does 1 optimizer step, and skips the post-training eval.
# Results are meaningless; we are just checking that imports, dataset
# load, LoRA setup, forward+backward, and trainer.save_model all work.
python finetune_dynamic_reports.py --smoke-test \
    >"$LOG_DIR/autorobust-finetune.log" 2>&1
pass "classifier finetune pipeline ran (1 step on 8 samples)"

# Minimum-viable attack training. Each env step runs a DistilBERT
# forward pass over the perturbed report, so we cap both the total
# timesteps and the per-episode horizon to keep this under ~10s:
# 32 timesteps / 4-step episodes = ~8 brief rollouts exercising
# reset, sampling, classifier inference, reward, and PPO buffer fill.
# --base-seed 99999 makes the smoke-produced agent unambiguously
# tagged so cleanup can rm it without ambiguity.
python autoattack.py --reward 1 --timesteps 32 --steps 4 \
    --n-agents 1 --base-seed 99999 \
    >"$LOG_DIR/autorobust-attack.log" 2>&1
pass "attack training pipeline ran (32 timesteps × 4-step episodes, seed 99999)"

# Cleanup smoke artifacts (trap also handles this if anything above fails)
rm -f agents/rew1_99999.pt agents/rew1_pomdp_99999.pt
rm -rf models/distilbert
if [ -d models/distilbert.smoke-backup ]; then
    mv models/distilbert.smoke-backup models/distilbert
fi
pass "cleaned up smoke artifacts"

deactivate
fi  # autorobust

# -----------------------------------------------
# 2. Link
# -----------------------------------------------
if should_run link; then
section "[2/4] Link"
activate_venv link
# train_A2C.py (what we run below) only needs stable_baselines3 + gym +
# tensorflow (for tf.compat.v1.logging) + torch. The legacy
# stable_baselines v1 package is in the env for other Link scripts but
# is broken on import under TF 2.x (it pulls tensorflow.contrib which
# was removed in TF 2). Skipping it here so the smoke test reflects
# what the actual training path needs.
python - <<'PY' >"$LOG_DIR/link-imports.log" 2>&1
import tensorflow, torch, gym, stable_baselines3
PY
pass "venv imports work"

cd "$REPO_ROOT/Link-Case-Study/wavsep-docker"
docker build -q -t wavsep-train . >"$LOG_DIR/link-docker-build.log" 2>&1
pass "WAVSEP image built"
docker rm -f wavsep-train >/dev/null 2>&1 || true
docker run -d --rm --name wavsep-train -p 8080:8080 wavsep-train >/dev/null
pass "WAVSEP container started"
cd ..

echo "    ... waiting up to 60s for WAVSEP to come up"
WAVSEP_URL="http://localhost:8080/wavsep/active/Reflected-XSS/RXSS-Detection-Evaluation-GET/index.jsp"
for i in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "$WAVSEP_URL" || echo 000)
    if [ "$code" = "200" ]; then
        pass "WAVSEP reachable (HTTP $code)"
        break
    fi
    sleep 2
    if [ "$i" = "30" ]; then
        echo "FAIL: WAVSEP did not respond within 60s (last code: $code)" >&2
        exit 1
    fi
done

cd Modified-Link
# Minimum-viable: each env step is a HTTP roundtrip to WAVSEP (~100ms),
# so 64 timesteps exercises env reset, action sampling, A2C
# forward/backward, and the WAVSEP HTTP path in ~10s.
python train_A2C.py -u "$WAVSEP_URL" -t 64 >"$LOG_DIR/link-train.log" 2>&1
pass "trained 64 timesteps"
cd ..

docker stop wavsep-train >/dev/null 2>&1 || true
pass "WAVSEP container stopped"
deactivate
fi  # link

# -----------------------------------------------
# 3. SQiRL
# -----------------------------------------------
if should_run sqirl; then
section "[3/4] SQiRL"
activate_venv sqirl
python -c "import torch, sklearn, playwright" >"$LOG_DIR/sqirl-imports.log" 2>&1
pass "venv imports work"

cd "$REPO_ROOT/SQiRL-CaseStudy/Modified-SQiRL/SQLiMicrobenchmark"
docker compose up -d >"$LOG_DIR/sqirl-docker-up.log" 2>&1
pass "SQLiMicrobenchmark stack up"
cd ..


echo "    ... waiting up to 90s for php-apache + MySQL general_log"
SQIRL_LOG="$PWD/SQLiMicrobenchmark/mysql/general.log"
sqirl_ready=0
last_code=000
for i in $(seq 1 45); do
    last_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/training.php || echo 000)
    if [ "$last_code" = "200" ]; then
        : > "$SQIRL_LOG" 2>/dev/null || true
        curl -s -o /dev/null "http://localhost:8000/functions_external/sqli1.php?id=ping_$$" || true
        sleep 1
        if grep -qE 'SELECT|select' "$SQIRL_LOG" 2>/dev/null; then
            pass "services ready (php-apache HTTP 200 + MySQL general_log writing)"
            sqirl_ready=1
            break
        fi
    fi
    sleep 2
done
if [ "$sqirl_ready" != "1" ]; then
    echo "FAIL: SQiRL services not ready within 90s (last training.php code: $last_code)." >&2
    echo "      See $LOG_DIR/sqirl-docker-up.log and 'docker compose ps' in SQLiMicrobenchmark/." >&2
    exit 1
fi

# Minimum-viable: sqirl.py loops until win_criteria wins OR loss_criteria
# losses are reached. Setting them to 1/3 means the loop exits after at
# most ~3 attempted SQL injections. --max_timestamp 2 caps each attempt
# to 2 actions. This exercises env init, agent decision, HTTP roundtrip,
# reward computation, and the natural-exit path in ~5–10s.
python3 sqirl.py -u http://localhost:8000/training.php -v 2 \
    --log_file ./SQLiMicrobenchmark/mysql/general.log \
    --loss_criteria 3 --win_criteria 1 --agent 2 \
    --learning True --episodes 1 --max_timestamp 2 \
    >"$LOG_DIR/sqirl-train.log" 2>&1
pass "sqirl.py ran to completion"

cd SQLiMicrobenchmark
docker compose down >/dev/null 2>&1 || true
cd ..
pass "SQLiMicrobenchmark stack down"
deactivate
fi  # sqirl

# -----------------------------------------------
# 4. MiniCAGE
# -----------------------------------------------
if should_run minicage; then
section "[4/4] MiniCAGE"
activate_venv minicage
python -c "import torch, stable_baselines3, gymnasium, wandb" >"$LOG_DIR/minicage-imports.log" 2>&1
pass "venv imports work"

cd "$REPO_ROOT/MiniCAGE-Case-Study/Modified-MiniCAGE"
# Minimum-viable: 1 agent × 1024 timesteps (one PPO rollout buffer).
# MiniCAGE env is in-process and runs at ~1800 fps, so this completes
# in ~1s and exercises env reset, vec env wrap, PPO rollout, learn(),
# and checkpoint save. The .bak is restored by the trap on failure.
sed -i.bak \
    -e 's/^NUM_RUNS: int = 20/NUM_RUNS: int = 1/' \
    -e 's/^TOTAL_TIMESTEPS: int = 2_500_000/TOTAL_TIMESTEPS: int = 1024/' \
    mini_CAGE/SB3_blue_training.py
# Both the parent dir and mini_CAGE/ must be on PYTHONPATH because the
# training script mixes top-level and package-style imports.
PYTHONPATH="$PWD:$PWD/mini_CAGE" python mini_CAGE/SB3_blue_training.py \
    >"$LOG_DIR/minicage-train.log" 2>&1
pass "trained 1 agent for 1024 timesteps"
mv mini_CAGE/SB3_blue_training.py.bak mini_CAGE/SB3_blue_training.py
deactivate
fi  # minicage

echo ""
echo "============================================"
echo " All smoke checks passed."
echo "============================================"
echo "Logs kept in: $LOG_DIR"
