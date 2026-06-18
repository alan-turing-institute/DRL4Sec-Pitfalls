#!/bin/bash
# reproduce-one-seed.sh
# Reproduces a quick single-seed run for each case study to verify the setup.
# Use this to confirm everything works before running the full reproduction.
#
# Prerequisites:
#   - Run ./setup-envs.sh first to create the four venvs in .venvs/.
#   - Docker must be running (for Link and SQiRL case studies)
#   - CUDA GPU recommended (for AutoRobust case study)
#
# Usage:
#   ./reproduce-one-seed.sh                          # all four case studies
#   ./reproduce-one-seed.sh autorobust               # just AutoRobust
#   ./reproduce-one-seed.sh link sqirl minicage      # subset
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
echo " DRL4Sec-Pitfalls: Single-Seed Reproduction"
echo "============================================"
echo "Targets: ${TARGETS[*]}"
echo ""
echo "This runs one agent/seed per case study for a quick verification."
echo ""

# -----------------------------------------------
# 1. AutoRobust Case Study
# -----------------------------------------------
if should_run autorobust; then
echo "============================================"
echo " [1/4] AutoRobust Case Study (1 seed)"
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

# Paper uses seeds 42-61 (base_seed=42, n_agents=20).
# This runs seed 42, the first seed from the paper results.
echo "[AutoRobust] Training 1 MDP agent (seed 42)..."
python autoattack.py --reward 1 --timesteps 100000 --n-agents 1 --base-seed 42

echo "[AutoRobust] Training 1 POMDP agent (seed 42)..."
python autoattack.py --reward 1 --timesteps 100000 --pomdp --n-agents 1 --base-seed 42

echo "[AutoRobust] Evaluating agent (test_agent runs MDP + POMDP + Random)..."
python autoattack.py --test --reward 1 --n-runs 20 --offset 200 --n-agents 1 --base-seed 42

echo "[AutoRobust] Done."
deactivate
echo ""
fi  # autorobust

# -----------------------------------------------
# 2. Link Case Study
# -----------------------------------------------
if should_run link; then
echo "============================================"
echo " [2/4] Link Case Study (1 run)"
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

WAVSEP_URL="http://localhost:8080/wavsep/active/Reflected-XSS/RXSS-Detection-Evaluation-GET/index.jsp"

cd Modified-Link

# Link does not expose an explicit seed argument; SB3 uses internal random
# seeding. The paper runs 20 iterations per configuration. This runs one
# iteration of each of the two key configurations (Original and Distinct States).
echo "[Link] Training 1 agent (Original formulation)..."
python train_A2C.py -u "$WAVSEP_URL" -t 3500000

echo "[Link] Training 1 agent (Distinct States formulation)..."
python train_A2C.py -u "$WAVSEP_URL" -t 3500000 -ds

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
echo " [3/4] SQiRL Case Study (1 run per condition)"
echo "============================================"
cd "$REPO_ROOT/SQiRL-CaseStudy/Modified-SQiRL"

echo "[SQiRL] Starting SQLi MicroBenchmark Docker container..."
cd SQLiMicrobenchmark
docker compose up -d
cd ..

echo "[SQiRL] Waiting for Docker services to come up..."
SQIRL_LOG="$PWD/SQLiMicrobenchmark/mysql/general.log"
gen_state="?"
for i in $(seq 1 60); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/training.php 2>/dev/null || echo 000)" = "200" ]; then
        gen_state=$(docker exec db mysql -u root -pMYSQL_ROOT_PASSWORD \
            -e "SHOW VARIABLES LIKE 'general_log'" 2>/dev/null \
            | awk '$1 == "general_log" {print $2}')
        if [ "$gen_state" = "ON" ] && [ -f "$SQIRL_LOG" ]; then
            echo "[SQiRL] services ready (php-apache + MySQL general_log=ON)"
            break
        fi
    fi
    if [ $((i % 8)) = 0 ]; then
        echo "[SQiRL] still waiting (iter $i/60, general_log=${gen_state:-?})"
    fi
    sleep 4
    if [ "$i" = "60" ]; then
        echo "[SQiRL] WARNING: services not confirmed ready after 240s; continuing anyway." >&2
    fi
done

activate_venv sqirl

mkdir -p stats_logs stats_logs_sanitized stats_logs_non_san

# Clear any stale `stats_logs/<timestamp>/` dirs left by manual
# sqirl.py invocations (e.g. ad-hoc diagnostics).
rm -rf ./stats_logs/*/ 2>/dev/null || true

echo "[SQiRL] Training 1 agent on combined (sanitized + non-sanitized) endpoint..."
python3 sqirl.py -u http://localhost:8000/training.php -v 2 \
    --log_file ./SQLiMicrobenchmark/mysql/general.log \
    --loss_criteria 200 --win_criteria 14 --agent 2 \
    --learning True --episodes 200 --max_timestamp 30
mv ./stats_logs/*/ stats_logs_sanitized/ 2>/dev/null || true

echo "[SQiRL] Training 1 agent on non-sanitized endpoint..."
python3 sqirl.py -u http://localhost:8000/training_no_san.php -v 2 \
    --log_file ./SQLiMicrobenchmark/mysql/general.log \
    --loss_criteria 200 --win_criteria 14 --agent 2 \
    --learning True --episodes 200 --max_timestamp 30
mv ./stats_logs/*/ stats_logs_non_san/ 2>/dev/null || true

echo "[SQiRL] Cross-evaluating trained agents on no_feedback.php..."
./test_agents.sh

echo "[SQiRL] Analyzing results into Table 2 format..."
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
# The training script uses agent index as seed (seed=idx).
# Set NUM_RUNS=1 to train a single agent with seed=0, matching the first
# agent from the paper's 20-agent runs.
sed -i.bak 's/^NUM_RUNS: int = 20/NUM_RUNS: int = 1/' mini_CAGE/SB3_blue_training.py
# The training script mixes top-level (`from single_agent_gym_wrapper ...`)
# and package-style (`from mini_CAGE.minimal ...`) imports, so both the
# Modified-MiniCAGE/ dir and its mini_CAGE/ subdir must be on PYTHONPATH.
export PYTHONPATH="$PWD:$PWD/mini_CAGE${PYTHONPATH:+:$PYTHONPATH}"

# --- §6.4 (Fig 2): training pitfalls: 4 HP configurations against B-line ---
echo "[MiniCAGE §6.4] Training 4 HP configs (1 agent each) against B-line red..."
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

# --- §8.3 D-NS (Table 3 bottom): vary the red attacker at train time ---
echo "[MiniCAGE §8.3 D-NS] Training default HPs against meander and mixed reds..."
for red in meander mixed; do
    echo "[MiniCAGE §8.3] Training RED_POLICY=$red ..."
    EXPERIMENT="default" RED_POLICY="$red" ACTION_ORDER="R2B" \
        python mini_CAGE/SB3_blue_training.py
done

mv mini_CAGE/SB3_blue_training.py.bak mini_CAGE/SB3_blue_training.py

echo "[MiniCAGE] Checkpoints saved under:"
ls -d ppo_models/SB3_PPO_*_*_* 2>/dev/null | sed 's/^/  /'
cd ../..

echo "[MiniCAGE] Done."
deactivate
fi  # minicage

# --- Extract results --------------------------------------------------
# Always runs at the end so the summary reflects whatever exists on
# disk, regardless of which subset of case studies was executed.
# The minicage venv has torch + SB3 needed for the MiniCAGE eval pass.
echo ""
echo "============================================"
echo " Extracting results → reproducibility/results-summary.md"
echo "============================================"
activate_venv minicage
python "$SCRIPT_DIR/extract-results.py" --eval-episodes 1000
deactivate
echo ""

echo "============================================"
echo " Single-seed reproduction complete."
echo " See reproducibility/results-summary.md"
echo "============================================"
