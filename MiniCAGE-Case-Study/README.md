# MiniCAGE Case Study

Extension of the MiniCAGE network defence environment, demonstrating training and deployment pitfalls in DRL-based autonomous cyber defence from the paper: **"SoK: The Pitfalls of Deep Reinforcement Learning for Cybersecurity"**

## Description

The environment is MiniCAGE - a lightweight, fast reimplementation of the CAGE 2 CybORG network defence task (from 
[CybORG++](https://github.com/alan-turing-institute/CybORG_plus_plus)). A blue (defender) PPO agent must prevent a scripted red (attacker) from reaching and impacting the critical server across a 13-host enterprise network. MiniCAGE includes two attacker strategies: **B-Line**, which takes the shortest path to the critical server, and **Meander**, which attempts to compromise the full network en route.

**Case Study 1 - Training Pitfalls (Section 6.4):** Twenty independent PPO blue agents are trained under four 
hyperparameter configurations against a B-Line attacker, with checkpoints saved at eight timestep milestones. This demonstrates the impact of hyperparameter sensitivity (T-HR), the necessity of variance analysis across multiple runs (T-VA), and the importance of demonstrating policy convergence (T-PC).

**Case Study 2 - Deployment Pitfalls (Section 8.3):** Two deployment experiments are run using the same environment:
- **Underlying Assumptions (D-UA):** Agents are trained and evaluated under different action orderings between red and blue (Blue→Red, Red→Blue, and Mixed), showing that a minor environment assumption, if violated at deployment, can severely degrade performance.
- **Non-Stationarity (D-NS):** Agents are trained against a fixed attacker strategy (B-Line, Meander, or Mixed) and evaluated against all three, showing how policies brittle to attacker changes can be mitigated by training against a diverse set of opponents.

The modifications made to the upstream MiniCAGE code are captured in `case_study.diff`.

## Requirements

### 1. Set up the environment

Using conda (recommended):

```bash
conda env create -f Modified-MiniCAGE/environment.yml
conda activate mini_cage
```

Or using pip from the `Modified-MiniCAGE/` directory:

```bash
pip install -r mini_CAGE/requirements.txt
```

### 2. Get the rl-reliability-metrics library

The evaluation scripts depend on Google's [rl-reliability-metrics](https://github.com/google-research/rl-reliability-metrics) library. Clone it into the `Case-Study-Evaluation/` directory and install it:

```bash
cd Case-Study-Evaluation
git clone https://github.com/google-research/rl-reliability-metrics.git
pip install -e rl-reliability-metrics/
```

The directory should be placed at `Case-Study-Evaluation/rl-reliability-metrics/` so that the evaluation scripts can locate it.

## Training Agents

Training is handled by `SB3_blue_training.py`. **Run all training commands from the `Modified-MiniCAGE/` directory** so that `mini_CAGE` is importable as a package.

### Case Study 1: Training Pitfalls

This experiment trains agents under four hyperparameter configurations to demonstrate T-HR, T-VA, and T-PC.

Open `mini_CAGE/SB3_blue_training.py` and configure:

| Variable | Description |
|---|---|
| `EXPERIMENT` | Hyperparameter preset. Choose from `"default"`, `"alt_alpha"`, `"alt_gamma"`, `"alt_eps"` |
| `NUM_RUNS` | Number of parallel agents to train (default: 20) |
| `TOTAL_TIMESTEPS` | Training budget per agent (default: 2,500,000) |
| `WANDB_PROJECT` | Your W&B project name |
| `WANDB_ENTITY` | Your W&B username or team name |
| `USE_WANDB` | Set to `False` to disable W&B logging |
| `USE_TENSORBOARD` | Set to `False` to disable TensorBoard logging |

The four hyperparameter presets correspond to the configurations in Figure 2 of the paper:

| Preset | Learning Rate | Gamma | Clip Range | Paper label |
|---|---|---|---|---|
| `default` | 0.0003 | 0.99 | 0.2 | Default HPs |
| `alt_alpha` | 0.003 | 0.99 | 0.2 | Alt Learning Rate |
| `alt_gamma` | 0.0003 | 0.1 | 0.2 | Alt Discount Factor |
| `alt_eps` | 0.0003 | 0.99 | 0.05 | Alt Clipping |

Run the script once per preset:

```bash
cd Modified-MiniCAGE
python mini_CAGE/SB3_blue_training.py
```

Each run spawns `NUM_RUNS` parallel processes, one per agent, each with a unique seed. Checkpoints are saved to:

```
Modified-MiniCAGE/ppo_models/SB3_PPO_<EXPERIMENT>/agent_XX/
```

at eight timestep milestones (50k, 100k, 250k, 500k, 750k, 1M, 2M, 2.5M steps) plus a final snapshot.

### Case Study 2: Deployment Pitfalls

#### Experiment A: Action Order (D-UA)

This experiment varies the action ordering between red and blue agents - Blue→Red (B→R, the default), Red→Blue (R→B), and Mixed (random each episode) - to show how a fixed assumption about turn order degrades performance if that assumption breaks at deployment (Table 3, top, in the paper).

To reproduce: in `mini_CAGE/SB3_blue_training.py`, modify the `make_env` function to pass the desired action order when instantiating the environment. The action order parameter is set on `SimplifiedCAGE` in `mini_CAGE/minimal.py` - refer to that file for the relevant argument name. Run the training script separately for each of the three action order conditions.

#### Experiment B: Attacker Non-Stationarity (D-NS)

This experiment trains agents against a fixed attacker (B-Line, Meander, or Mixed - a random choice each episode) and evaluates all three trained policies against all three attacker strategies (Table 3, bottom, in the paper).

To reproduce: in `mini_CAGE/SB3_blue_training.py`, change the `red_policy` argument in `make_env` to the desired training attacker:

```python
# In make_env():
env = MiniCageBlue(red_policy="bline", ...)   # B-Line only
env = MiniCageBlue(red_policy="meander", ...) # Meander only
env = MiniCageBlue(red_policy="random", ...)  # Mixed (random each episode)
```

Run the training script once per attacker condition. Cross-attacker evaluation is then handled by `agent_evaluation.py` (see below).

## Evaluation

Evaluation scripts live in `Case-Study-Evaluation/`. **Run them from the `Modified-MiniCAGE/` directory** so that `mini_CAGE` remains importable.

### Agent evaluation

`agent_evaluation.py` loads trained checkpoints, rolls them out for `N_EPISODES` episodes, and computes mean reward and CVaR (lower/upper) per agent and checkpoint.

Open `Case-Study-Evaluation/agent_evaluation.py` and configure:

| Variable | Description |
|---|---|
| `BASE_DIR` | Path to the checkpoint directory, relative to `Modified-MiniCAGE/`. E.g. `"ppo_models/SB3_PPO_default/"` |
| `EXPERIMENT` | List of run name prefixes to evaluate (matched against checkpoint filenames) |
| `SELECTED_INDICES` | Set of agent indices to include |
| `versions` | List of checkpoint stages to evaluate (e.g. `'final'`, `'step_500000'`) |
| `N_EPISODES` | Number of evaluation episodes per agent per checkpoint (default: 1,000) |
| `CROSS_EVAL_RED` | Set to `"meander"` or `"random"` to evaluate against a different red agent than training, or `None` to disable |

For the **D-NS cross-attacker evaluation**, set `CROSS_EVAL_RED` to the attacker you want to evaluate against (e.g. `"meander"` for agents trained on B-Line). This produces a combined output file with both same-attacker and cross-attacker results.

Then run:

```bash
cd Modified-MiniCAGE
python ../Case-Study-Evaluation/agent_evaluation.py
```

Results are written to text files in `BASE_DIR` named `risk_eval_<red>_<stage>.txt`, with per-agent mean reward and CVaR for each checkpoint stage.

### Confidence intervals

To compute 95% t-based confidence intervals over the per-agent mean rewards from an evaluation output file:

```bash
python Case-Study-Evaluation/Conf_intervals_eval.py path/to/risk_eval_bline_final.txt
```

This can be run from any directory. Pass the path to any `risk_eval_*.txt` file produced by `agent_evaluation.py`.
