#!/bin/bash
# reproduce-full.sh
# Reproduces ALL experiments from the paper across all four case studies.
# This runs 20 agents/seeds per configuration to match the paper's results.
#
# Prerequisites:
#   - Run ./setup-envs.sh first to create the four venvs in .venvs/.
#   - Docker must be running (for Link and SQiRL case studies)
#   - CUDA GPU recommended (for AutoRobust case study)
#
# Usage:
#   ./reproduce-full.sh                          # all four case studies
#   ./reproduce-full.sh autorobust               # just AutoRobust
#   ./reproduce-full.sh link sqirl minicage      # subset
#
# Valid case study names: autorobust, link, sqirl, minicage

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENVS_DIR="$SCRIPT_DIR/.venvs"

activate_venv() {
    local name=$1
    local venv_path="$VENVS_DIR/$name"
    if [ ! -d "$venv_path" ]; then
        echo "ERROR: venv '$name' not found at $venv_path." >&2
        echo "  Run ./setup-envs.sh first." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "$venv_path/bin/activate"
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
echo " DRL4Sec-Pitfalls: Full Reproduction"
echo "============================================"
echo "Targets: ${TARGETS[*]}"
echo ""
echo "This will run ALL experiments with 20 agents/seeds per configuration."
echo "This may take several days depending on hardware."
echo ""

# -----------------------------------------------
# 1. AutoRobust Case Study
# -----------------------------------------------
if should_run autorobust; then
echo "============================================"
echo " [1/4] AutoRobust Case Study"
echo "============================================"
activate_venv autorobust
cd "$REPO_ROOT/AutoRobust-Case-Study"

mkdir -p agents results

# The malware classifier is shipped pre-fine-tuned at models/distilbert/.
# Bail early with a useful pointer rather than letting RLattacker fail
# deep inside an env-init traceback.
if [ ! -d models/distilbert ] || [ -z "$(ls -A models/distilbert 2>/dev/null)" ]; then
    echo "ERROR: AutoRobust-Case-Study/models/distilbert/ is missing or empty." >&2
    echo "       The fine-tuned DistilBERT classifier is shipped with the repo;" >&2
    echo "       if you need to regenerate it: python finetune_dynamic_reports.py" >&2
    echo "       (requires the Kaggle malware-and-goodware-dynamic-analysis-reports" >&2
    echo "       dataset in AutoRobust-Case-Study/data/)." >&2
    exit 1
fi

echo "[AutoRobust] Training MDP agents (20 seeds)..."
python autoattack.py --reward 1 --timesteps 100000 --n-agents 20 --base-seed 42

echo "[AutoRobust] Training POMDP agents (20 seeds)..."
python autoattack.py --reward 1 --timesteps 100000 --pomdp --n-agents 20 --base-seed 42

echo "[AutoRobust] Evaluating 20 agents (test_agent() runs MDP + POMDP + Random in one call)..."
python autoattack.py --test --reward 1 --n-runs 20 --offset 200 --n-agents 20 --base-seed 42

echo "[AutoRobust] Done."
deactivate
echo ""
fi  # autorobust

# -----------------------------------------------
# 2. Link Case Study
# -----------------------------------------------
if should_run link; then
echo "============================================"
echo " [2/4] Link Case Study"
echo "============================================"
cd "$REPO_ROOT/Link-Case-Study"

echo "[Link] Building and starting WAVSEP Docker container..."
cd wavsep-docker
docker build -t wavsep-train .
# Clear any leftover container from a previous run.
docker rm -f wavsep-train 2>/dev/null || true
docker run -d --rm --name wavsep-train -p 8080:8080 wavsep-train
cd ..

echo "[Link] Waiting for WAVSEP to start..."
sleep 30

activate_venv link

echo "[Link] Running full training (20 runs x 4 deterministic-mode configurations)..."
WAVSEP_URL="http://localhost:8080/wavsep/active/Reflected-XSS/RXSS-Detection-Evaluation-GET/index.jsp"
cd Modified-Link
for i in $(seq 1 20); do
    echo "[Link] iteration $i/20: -da (deterministic action only)"
    python train_A2C.py -u "$WAVSEP_URL" -t 3500000 -da
    echo "[Link] iteration $i/20: -ds (deterministic state only, Table 1 'Distinct States')"
    python train_A2C.py -u "$WAVSEP_URL" -t 3500000 -ds
    echo "[Link] iteration $i/20: -d  (deterministic both)"
    python train_A2C.py -u "$WAVSEP_URL" -t 3500000 -d
    echo "[Link] iteration $i/20: no flag (Table 1 'Original')"
    python train_A2C.py -u "$WAVSEP_URL" -t 3500000
done
cd ..

echo "[Link] Stopping WAVSEP Docker container..."
docker stop wavsep-train 2>/dev/null || true

echo "[Link] Done."
deactivate
echo ""
fi  # link

# -----------------------------------------------
# 3. SQiRL Case Study
# -----------------------------------------------
if should_run sqirl; then
echo "============================================"
echo " [3/4] SQiRL Case Study"
echo "============================================"
cd "$REPO_ROOT/SQiRL-CaseStudy/Modified-SQiRL"

echo "[SQiRL] Starting SQLi MicroBenchmark Docker container..."
cd SQLiMicrobenchmark
docker compose up -d
cd ..

# Poll until both php-apache and MySQL's general_log are live (matches
# the readiness gate used in reproduce-one-seed.sh).
echo "[SQiRL] Waiting for Docker services to come up..."
SQIRL_LOG="$PWD/SQLiMicrobenchmark/mysql/general.log"
for i in $(seq 1 60); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/training.php 2>/dev/null || echo 000)" = "200" ]; then
        : > "$SQIRL_LOG" 2>/dev/null || true
        curl -s -o /dev/null "http://localhost:8000/functions_external/sqli1.php?id=ping_$$" || true
        sleep 1
        if grep -qE 'SELECT|select' "$SQIRL_LOG" 2>/dev/null; then
            echo "[SQiRL] services ready (php-apache + MySQL general_log)"
            break
        fi
    fi
    sleep 2
    if [ "$i" = "60" ]; then
        echo "[SQiRL] WARNING: services not confirmed ready after 120s; continuing anyway." >&2
    fi
done

activate_venv sqirl

echo "[SQiRL] Training agents (20 runs x 2 conditions)..."
./train_agents.sh

echo "[SQiRL] Evaluating agents..."
./test_agents.sh

echo "[SQiRL] Analyzing results..."
python3 analyze_results.py ./test_results

echo "[SQiRL] Stopping Docker containers..."
cd SQLiMicrobenchmark
docker compose down
cd ..

echo "[SQiRL] Done."
deactivate
echo ""
fi  # sqirl

# -----------------------------------------------
# 4. MiniCAGE Case Study
# -----------------------------------------------
if should_run minicage; then
echo "============================================"
echo " [4/4] MiniCAGE Case Study"
echo "============================================"
activate_venv minicage
cd "$REPO_ROOT/MiniCAGE-Case-Study"

cd Modified-MiniCAGE
# The training script mixes top-level (`from single_agent_gym_wrapper ...`)
# and package-style (`from mini_CAGE.minimal ...`) imports, so both the
# Modified-MiniCAGE/ dir and its mini_CAGE/ subdir must be on PYTHONPATH.
export PYTHONPATH="$PWD:$PWD/mini_CAGE${PYTHONPATH:+:$PYTHONPATH}"

# --- §6.4 (Fig 2): 4 HP configurations x 20 agents against B-line red ---
echo "[MiniCAGE §6.4] Training 4 HP configs (20 agents each) against B-line red..."
for experiment in default alt_alpha alt_gamma alt_eps; do
    echo "[MiniCAGE §6.4] Training EXPERIMENT=$experiment ..."
    EXPERIMENT="$experiment" RED_POLICY="bline" \
        python mini_CAGE/SB3_blue_training.py
done

# --- §8.3 D-UA (Table 3 top): vary action order red↔blue at train time ---
# bline / R2B is already covered above; train the remaining 2 orders.
echo "[MiniCAGE §8.3 D-UA] Training default HPs with B2R and Mixed action orders..."
for order in B2R Mixed; do
    echo "[MiniCAGE §8.3] Training ACTION_ORDER=$order ..."
    EXPERIMENT="default" RED_POLICY="bline" ACTION_ORDER="$order" \
        python mini_CAGE/SB3_blue_training.py
done

# --- §8.3 D-NS (Table 3 bottom): vary red attacker during training ---
echo "[MiniCAGE §8.3 D-NS] Training default HPs against meander and mixed reds..."
for red in meander mixed; do
    echo "[MiniCAGE §8.3] Training RED_POLICY=$red ..."
    EXPERIMENT="default" RED_POLICY="$red" ACTION_ORDER="R2B" \
        python mini_CAGE/SB3_blue_training.py
done

cd ..

echo "[MiniCAGE] Done."
deactivate
fi  # minicage

# --- Extract results --------------------------------------------------
# Always runs at the end so the summary reflects whatever exists on
# disk, regardless of which subset of case studies was executed.
# Matches the paper's per-agent eval budget (1000 episodes).
echo ""
echo "============================================"
echo " Extracting results → reproducibility/results-summary.md"
echo "============================================"
activate_venv minicage
python "$SCRIPT_DIR/extract-results.py" \
    --eval-episodes "${EXTRACT_EVAL_EPISODES:-1000}"
deactivate
echo ""

echo "============================================"
echo " Full reproduction complete."
echo " See reproducibility/results-summary.md"
echo "============================================"
