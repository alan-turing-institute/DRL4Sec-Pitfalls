# AutoRobust Case Study

Reproduction of an RL environment for adversarial malware evasion, demonstrating evaluation pitfalls in DRL-based adversarial malware creation from the paper: **"SoK: The Pitfalls of Deep Reinforcement Learning for Cybersecurity"**

## Description

This case study (Section 7.4 of the paper) implements a Gymnasium environment (`RLattacker`) modeled after AutoRobust, where agents learn to evade ML-based malware detection by modifying dynamic analysis reports. Given a correctly classified malware report, the agent's goal is to cross the classifier's decision boundary by adding entries from a goodware corpus or modifying existing entries, while preserving the report's correctness.

The case study demonstrates two evaluation pitfalls:

- **Gain Attribution (E-GA):** By comparing trained policies (MDP and POMDP) against a random policy baseline, we disentangle the contributions of the state-action space design from the learned policy itself. Figure 3 in the paper shows that while trained policies (MDP & POMDP) have the *potential* to outperform random, the random baseline is already strong. This demonstrates that a large portion of overall performance comes from the action space design rather than the learned policy.
- **Application Motivation (E-AM):** The paper provides theoretical motivation for DRL over gradient-based approaches in adversarial malware creation: DRL operates directly on malware samples in the problem-space, bypassing the need for feature-to-problem-space mapping, and can handle dynamic action spaces.

The environment supports two observability modes:

- **MDP** (fully observable): The agent observes a 768-dimensional DistilBERT `[CLS]` embedding of the current report
- **POMDP** (partially observable): The agent observes a sentence embedding from `all-MiniLM-L6-v2`, without access to the internal classifier state

## Requirements

```bash
pip install gymnasium numpy torch transformers stable-baselines3 pandas matplotlib nltk sentence-transformers
```

A CUDA-capable GPU is recommended. The environment defaults to CUDA if available.

## Training the Malware Detection Model

Before running the RL agents, you need to train the malware detection classifier:

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/greimas/malware-and-goodware-dynamic-analysis-reports)
2. Create and populate the data directory:

```bash
mkdir -p data/
# Place the dataset files in data/
```

3. Run the training script:

```bash
python finetune_dynamic_reports.py
```

This fine-tunes a DistilBERT model on the dynamic analysis reports and saves the trained classifier to `models/distilbert/` for use by the RL environment.

## Training

Create the output directories, then train agents under both observability conditions:

```bash
mkdir -p agents results
```

Train MDP agents (fully observable, 768-dim embedding observation):

```bash
python autoattack.py --reward 1 --timesteps 100000
```

Train POMDP agents (partially observable, sentence embedding observation):

```bash
python autoattack.py --reward 1 --timesteps 100000 --pomdp
```

By default, 5 agents are trained with seeds 42-46. To reproduce the paper's Figure 3 (which uses 20 agents), increase `--n-agents`:

```bash
python autoattack.py --reward 1 --timesteps 100000 --n-agents 20
python autoattack.py --reward 1 --timesteps 100000 --pomdp --n-agents 20
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--reward` | 1 | Reward function (1: delta malware probability, 2: delta + completion bonus) |
| `--timesteps` | 100,000 | Total training timesteps |
| `--steps` | 1,000 | Max steps per episode |
| `--lr` | 0.001 | Learning rate |
| `--arch` | 128 | Policy network hidden layer size (two layers of this size) |
| `--pomdp` | False | Use POMDP mode (sentence embedding observation) |
| `--n-agents` | 5 | Number of agents to train (each with a unique seed) |
| `--base-seed` | 42 | Base random seed (agents use base_seed + i) |

**PPO hyperparameters** (hardcoded in `autoattack.py`):

| Parameter | Value |
|---|---|
| `n_steps` | 1,024 |
| `batch_size` | 32 |
| `gamma` | 0.99 |
| `ent_coef` | 0.001 |
| `clip_range` | 0.2 |
| `max_grad_norm` | 1 |
| `gae_lambda` | 0.95 |
| Network | `[128, 128]` MLP |

Trained models are saved to `agents/` as `rew<N>_<seed>.pt` (MDP) and `rew<N>_pomdp_<seed>.pt` (POMDP).

## Testing

Evaluate all three policy types (MDP, POMDP, and random) side-by-side:

```bash
python autoattack.py --test --reward 1 --n-runs 20 --offset 200
```

This corresponds to the paper's Figure 3, which compares mean malware score with 95/99% confidence intervals across agents. Each agent is evaluated for `--n-runs` episodes starting from report index `--offset` (to use reports unseen during training).

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--test` | - | Run evaluation mode |
| `--n-runs` | 20 | Number of evaluation episodes per agent |
| `--n-agents` | 5 | Number of agents to evaluate |
| `--offset` | 200 | Starting report index (use unseen reports) |

Results are saved to `results/`:
- `test_multi_rew<N>.png`: boxplot comparing MDP, POMDP, and random policies (mean episodic return and mean malware score)
- `test_multi_rew<N>.csv`: per-agent mean and std for all three policy types
