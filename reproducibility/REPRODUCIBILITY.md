# Reproducibility Guide

This directory contains scripts and instructions to reproduce the experiments from:

> **SoK: The Pitfalls of Deep Reinforcement Learning for Cybersecurity**

## Quick Start

Four shell scripts and one extractor:

| Script | Purpose | Runtime |
|---|---|---|
| `setup-envs.sh` | Create one venv per case study in `.venvs/` (run once) | ~15–30 min |
| `smoke-test.sh` | Ultralight check that every env, Docker stack, and entry point works | ~5–10 min |
| `reproduce-one-seed.sh` | Single-seed run per case study for setup verification | Hours |
| `reproduce-full.sh` | Full reproduction matching the paper (20 seeds/agents) | Days |
| `extract-results.py` | Parses outputs of every case study and writes `results-summary.md` (paper vs reproduced numbers side-by-side). Run automatically at the end of both reproduce scripts. | ~1–10 min |

```bash
cd reproducibility
chmod +x setup-envs.sh smoke-test.sh reproduce-one-seed.sh reproduce-full.sh

# 1. Create the four venvs (one per case study)
./setup-envs.sh

# 2. Lightweight install check (recommended before either reproduce script)
./smoke-test.sh

# 3. Single-seed reproduction
./reproduce-one-seed.sh

# 4. Full paper reproduction
./reproduce-full.sh
```

You can also set up a single venv in isolation:

```bash
./setup-envs.sh autorobust   # or: link | sqirl | minicage
```

## Environments

The four case studies cannot share a Python environment (incompatible Python versions, `numpy 1.x` vs `2.x`, `stable-baselines` v1 vs SB3, etc.), so `setup-envs.sh` creates one venv per study in `reproducibility/.venvs/`:

| venv | Python | Requirements |
|---|---|---|
| `.venvs/autorobust` | 3.10 | `envs/autorobust.txt` |
| `.venvs/link`       | 3.9  | `envs/link.txt` |
| `.venvs/sqirl`      | 3.9  | `envs/sqirl.txt` |
| `.venvs/minicage`   | 3.10 | `envs/minicage.txt` + `rl-reliability-metrics` (cloned and installed automatically) |

Both reproduce scripts `source .venvs/<name>/bin/activate` before each study's section, so the right env is used per case study with no manual switching.

## Prerequisites

### General
- `python3.9` and `python3.10` binaries on `PATH`
  - **macOS:** `brew install python@3.9 python@3.10` (or use [pyenv](https://github.com/pyenv/pyenv): `pyenv install 3.9.20 3.10.15`)
  - **Linux:** use the [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa), pyenv, or your distro's packages
- [Docker](https://www.docker.com/) (for Link and SQiRL case studies)
- CUDA-capable GPU recommended (for AutoRobust case study)

### Per Case Study

| Case Study | Docker | External Data                                                                                                                               |
|---|---|---------------------------------------------------------------------------------------------------------------------------------------------|
| AutoRobust | No  | [Kaggle dataset](https://www.kaggle.com/datasets/greimas/malware-and-goodware-dynamic-analysis-reports) into `AutoRobust-Case-Study/data/`. |
| Link       | Yes (WAVSEP) | None                                                                                                                                        |
| SQiRL      | Yes (SQLi MicroBenchmark) | None                                                                                                                                        |
| MiniCAGE   | No  | [rl-reliability-metrics](https://github.com/google-research/rl-reliability-metrics) (cloned automatically by `setup-envs.sh`)               |

## Case Study Details

### 1. AutoRobust (Evaluation Pitfalls, Section 7.4)

**Setup:**
```bash
source reproducibility/.venvs/autorobust/bin/activate   # created by setup-envs.sh
cd AutoRobust-Case-Study
mkdir -p agents results
```

**Malware classifier:** shipped pre-fine-tuned as a **LoRA adapter** at `AutoRobust-Case-Study/models/distilbert/` (≈5 MB). `envs/RLattacker.py` detects the adapter format and loads via `peft.PeftModel.from_pretrained`, attaching it to the base `distilbert-base-uncased` (fetched from the HuggingFace Hub on first use). A plain standalone checkpoint at the same path also works (loader falls back to `AutoModelForSequenceClassification.from_pretrained`). The reproduce scripts abort with a clear error if the directory is missing. To regenerate the classifier from scratch, download the [Kaggle malware-and-goodware-dynamic-analysis-reports](https://www.kaggle.com/datasets/greimas/malware-and-goodware-dynamic-analysis-reports) dataset into `AutoRobust-Case-Study/data/` and run `python finetune_dynamic_reports.py`.

**Training (paper configuration, 20 agents, seeds 42-61):**
```bash
# MDP agents
python autoattack.py --reward 1 --timesteps 100000 --n-agents 20 --base-seed 42

# POMDP agents
python autoattack.py --reward 1 --timesteps 100000 --pomdp --n-agents 20 --base-seed 42
```

**Evaluation:**
```bash
python autoattack.py --test --reward 1 --n-runs 20 --offset 200
python autoattack.py --test --reward 1 --n-runs 20 --offset 200 --pomdp
```

### 2. Link (Modeling Pitfalls, Section 5.4)

**Setup:**
```bash
source reproducibility/.venvs/link/bin/activate   # created by setup-envs.sh
cd Link-Case-Study

# Start WAVSEP target
cd wavsep-docker && docker build -t wavsep-train . && docker run -d --rm --name wavsep-train -p 8080:8080 wavsep-train && cd ..
```

**Training (paper configuration, 20 runs x 4 configurations):**
```bash
cd Modified-Link
./run_train.sh
```

Or for individual configurations:
```bash
WAVSEP_URL="http://localhost:8080/wavsep/active/Reflected-XSS/RXSS-Detection-Evaluation-GET/index.jsp"

# Original formulation (Table 1)
python train_A2C.py -u "$WAVSEP_URL" -t 3500000

# Distinct States formulation (Table 1)
python train_A2C.py -u "$WAVSEP_URL" -t 3500000 -ds
```

**Cleanup:**
```bash
docker stop wavsep-train
```

### 3. SQiRL (Evaluation Pitfalls, Section 7.4)

**Setup:**
```bash
source reproducibility/.venvs/sqirl/bin/activate   # created by setup-envs.sh
cd SQiRL-CaseStudy/Modified-SQiRL
cd SQLiMicrobenchmark && docker compose up -d && cd ..
```

**Training (paper configuration, 20 runs x 2 conditions):**
```bash
./train_agents.sh
```

**Evaluation:**
```bash
./test_agents.sh
python3 analyze_results.py ./test_results
```

**Cleanup:**
```bash
cd SQLiMicrobenchmark && docker compose down && cd ..
```

**Apple Silicon note:** `SQLiMicrobenchmark/docker-compose.yaml` pins the `db` and `db_seeder` services to `platform: linux/amd64` because `mysql:5.7` has no arm64 image. On Apple Silicon they run under Rosetta/QEMU emulation, which works but is slower; on x86 hosts the platform pin is a no-op.

### 4. MiniCAGE (Training & Deployment Pitfalls, Sections 6.4, 8.3)

**Setup:**
```bash
source reproducibility/.venvs/minicage/bin/activate   # created by setup-envs.sh (also clones + pip-installs rl-reliability-metrics)
cd MiniCAGE-Case-Study
```

`SB3_blue_training.py` is parameterised by three env vars (all defaulting to the §6.4 paper config):

| Env var | Values | Used by |
|---|---|---|
| `EXPERIMENT`   | `default` / `alt_alpha` / `alt_gamma` / `alt_eps` | §6.4 / Fig 2 (HP variation) |
| `RED_POLICY`   | `bline` / `meander` / `mixed`                    | §8.3 / Table 3 bottom (D-NS, vary training adversary) |
| `ACTION_ORDER` | `R2B` / `B2R` / `Mixed`                          | §8.3 / Table 3 top (D-UA, vary action order red↔blue) |

Output is written to `ppo_models/SB3_PPO_<EXPERIMENT>_<RED_POLICY>_<ACTION_ORDER>/`, so different combinations don't overwrite each other.

**Training (paper configuration):**
```bash
cd Modified-MiniCAGE
# Both the parent dir and mini_CAGE/ must be on PYTHONPATH because the
# training script mixes top-level and package-style imports.
export PYTHONPATH="$PWD:$PWD/mini_CAGE"

# §6.4: 4 HP configs × 20 agents against B-line red, R→B order
for experiment in default alt_alpha alt_gamma alt_eps; do
    EXPERIMENT="$experiment" RED_POLICY="bline" ACTION_ORDER="R2B" \
        python mini_CAGE/SB3_blue_training.py
done

# §8.3 D-UA: default HPs × {B2R, Mixed} action orders (R2B already done above)
for order in B2R Mixed; do
    EXPERIMENT="default" RED_POLICY="bline" ACTION_ORDER="$order" \
        python mini_CAGE/SB3_blue_training.py
done

# §8.3 D-NS: default HPs × {meander, mixed} reds (bline already done above)
for red in meander mixed; do
    EXPERIMENT="default" RED_POLICY="$red" ACTION_ORDER="R2B" \
        python mini_CAGE/SB3_blue_training.py
done
```

**§8.3 action ordering:** `mini_CAGE/minimal.py:SimplifiedCAGE` accepts an `action_order` kwarg (`R2B` / `B2R` / `Mixed`). `R2B` is the original behaviour (red moves first, then blue); `B2R` swaps the order; `Mixed` picks one at random per step. The order is threaded from `ACTION_ORDER` through `MiniCageBlue` into `SimplifiedCAGE._process_actions_*`.

**Evaluation:**
```bash
python ../Case-Study-Evaluation/agent_evaluation.py
python ../Case-Study-Evaluation/Conf_intervals_eval.py path/to/risk_eval_bline_final.txt
```

Note: `agent_evaluation.py` has hard-coded `BASE_DIR` and an `EXPERIMENT` list of agent run names. Edit them to point at the directory you want to evaluate (e.g. `ppo_models/SB3_PPO_default_meander/`) before running.

## Seeds and Variance

| Case Study | Seeds | Runs per Config | Notes |
|---|---|---|---|
| AutoRobust | 42-61 (explicit via `--base-seed`) | 20 | `--base-seed 42` with `--n-agents 20` yields seeds 42, 43, ..., 61 |
| Link | Not explicitly set | 20 | SB3 uses internal random seeding; no seed CLI argument exposed |
| SQiRL | Not explicitly set | 20 | No seed argument in the training script |
| MiniCAGE | 0-24 (agent index used as seed) | 25 | `seed=idx` in `SB3_blue_training.py` where `idx` ranges from 0 to `NUM_RUNS-1` |

## Expected Outputs

- **AutoRobust:** PNG plots and CSV results in `AutoRobust-Case-Study/results/`
- **Link:** Models in `Link-Case-Study/Modified-Link/models/`, TensorBoard logs in `tensorboard_log/`
- **SQiRL:** Analysis CSV and LaTeX tables in `SQiRL-CaseStudy/Modified-SQiRL/test_results/`
- **MiniCAGE:** Checkpoints in `Modified-MiniCAGE/ppo_models/`, evaluation text files in `Case-Study-Evaluation/`
