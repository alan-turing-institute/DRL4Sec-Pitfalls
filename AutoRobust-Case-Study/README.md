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

## Malware Detection Model

The fine-tuned DistilBERT classifier this case study uses as its malware detector is **provided pre-trained at `models/distilbert/`** as a **LoRA adapter** (about 5 MB: `adapter_config.json` + `adapter_model.safetensors` + tokenizer files). You do not need to run `finetune_dynamic_reports.py` to reproduce the paper's results, the reproducibility scripts (`reproducibility/reproduce-one-seed.sh`, `reproducibility/reproduce-full.sh`) load `models/distilbert/` directly.

`envs/RLattacker.py` detects the LoRA format (presence of `adapter_config.json`) and loads via `peft.PeftModel.from_pretrained`, attaching the adapter to the base model named in `adapter_config["base_model_name_or_path"]` (`distilbert-base-uncased`, fetched from the HuggingFace Hub on first use). For a plain non-adapter checkpoint the loader falls back to `AutoModelForSequenceClassification.from_pretrained` so either format works.

### How the shipped classifier was produced

It is a LoRA fine-tune of [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) on the [Kaggle malware-and-goodware-dynamic-analysis-reports dataset](https://www.kaggle.com/datasets/greimas/malware-and-goodware-dynamic-analysis-reports), produced by `finetune_dynamic_reports.py` with the defaults in its `CONFIG` block (4 epochs, learning rate 2e-4, LoRA r=16/α=32 over `q_lin,k_lin,v_lin,out_lin`, chunk size 512, max 20 chunks per sample, 80/10/10 train/eval/test split).

### Regenerating the classifier (optional)

If you want to rebuild it from scratch (e.g. to verify the shipped checkpoint, or because you've changed the upstream dataset), download the Kaggle dataset into `data/` and run:

```bash
mkdir -p data/
# Place the Kaggle dataset files in data/ (preserving its directory layout)
python finetune_dynamic_reports.py
```

This will overwrite `models/distilbert/`. Add `--smoke-test` for a fast pipeline-only check (8 samples, 1 optimizer step, no test eval) used by `reproducibility/smoke-test.sh`.

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
