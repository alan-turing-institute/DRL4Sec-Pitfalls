# Link Case Study

Extension of the Link RL environment for XSS payload generation, demonstrating modeling pitfalls in DRL-based web security testing from the paper: **"SoK: The Pitfalls of Deep Reinforcement Learning for Cybersecurity"**

## Description

Link is an RL framework that trains an A2C agent to discover reflected Cross-Site Scripting (XSS) vulnerabilities in web applications in a black-box manner. The agent iteratively builds XSS payloads by selecting HTML tags, attributes, and event handlers through a hand-crafted state-action space built on top of the [Wapiti](https://github.com/wapiti-scanner/wapiti) web scanner.

This case study (Section 5.4 of the paper) demonstrates how environment modeling choices directly impact agent performance. Specifically, the original Link environment introduces unnecessary partial observability (M-PO) and modeling correctness issues (M-MC) through its state representation: multiple distinct XSS payloads map to identical DRL states because a single feature is used to indicate multiple HTML tags (e.g., `img`, `video`, `audio`, `svg`). This conflation means the agent cannot distinguish between payloads that have different real-world effects.

We compare two MDP formulations over 20 training runs:

- **Original**: The state space as published, with merged/overlapping features (47 dimensions)
- **Distinct States**: An unrolled state space where each payload feature maps to a unique state dimension (53 dimensions), removing the artificial partial observability

Table 1 in the paper shows that the *Distinct States* formulation increases the percentage of vulnerabilities found by 10.1% and raises the lower bound of the 95% confidence interval by 17.3%, demonstrating that minor corrections to state-space design can yield significant performance gains.

The modifications made to the upstream Link code are captured in `case_study.diff`.

## Requirements

### 1. Set up the conda environment

```bash
conda env create -f Modified-Link/environment.yaml -n link
conda activate link
```

The environment uses Python 3.8 with Stable-Baselines3, PyTorch, Gym, and TensorFlow.

### 2. Set up the WAVSEP test suite

The training target is [WAVSEP](https://code.google.com/archive/p/wavsep/), a collection of deliberately vulnerable web applications. The `run_train.sh` script will automatically build and launch a WAVSEP Docker container, but Docker must be installed and running on your system.

Alternatively, you can set up WAVSEP manually:

```bash
cd wavsep-docker && docker build -t wavsep-train . && docker run -d --rm --name wavsep-train -p 8080:8080 wavsep-train && cd ..
```

WAVSEP will be accessible at `http://localhost:8080`. The training scripts target the reflected XSS GET evaluation pages at:
```
http://localhost:8080/wavsep/active/Reflected-XSS/RXSS-Detection-Evaluation-GET/index.jsp
```

## Training

All training commands should be run from the `Modified-Link/` directory.

### Running the full experiment

The `run_train.sh` script automates the full case study experiment. It expects a `wavsep-docker/` directory containing a Dockerfile to build the WAVSEP training container. If this directory is not present, set up WAVSEP manually (see Requirements above) and run the training commands directly.

```bash
cd Modified-Link
./run_train.sh
```

The script trains 20 agents under each of four deterministic mode configurations (80 total runs):

| Flag | State dims | Action dims | Mode | Paper reference |
|------|-----------|-------------|------|----------------|
| (none) | 47 | 39 | Non-deterministic | **"Original"** in Table 1 |
| `-d s` | 53 | 39 | Deterministic state only | **"Distinct States"** in Table 1 |

**Table 1 in the paper** reports results from the no-flag (Original) and `-d s` (Distinct States) configurations only. The `-d a` and `-d` configurations are additional ablations that separately control determinism in the action space.

The key difference for Table 1 is the **state space**: the Original formulation (47 dims) conflates multiple HTML tags into a single feature (`HTML_MEDIA_TAG_USED`), while Distinct States (53 dims) unrolls these into separate features (`HTML_IMG_TAG_USED`, `HTML_AUDIO_TAG_USED`, `HTML_VIDEO_TAG_USED`, `HTML_LINK_TAG_USED`, `JS_PROMPT_USED`, `JS_CONFIRM_USED`). See `gym_reflected_xss/envs/observation.py` for the full state definitions.

Each agent trains for 3,500,000 timesteps using A2C with a learning rate of 0.0005 and gamma of 0.95. The network architecture uses three hidden layers of 128 units each with ReLU activation.

### Training a single agent

To reproduce only the Table 1 results, train agents under the two relevant configurations:

```bash
cd Modified-Link

# "Original" (Table 1) -- no -d flag, 47-dim state, 39 actions
python train_A2C.py -u 'http://localhost:8080/wavsep/active/Reflected-XSS/RXSS-Detection-Evaluation-GET/index.jsp' -t 3500000

# "Distinct States" (Table 1) -- deterministic state, 53-dim state, 39 actions
python train_A2C.py -u 'http://localhost:8080/wavsep/active/Reflected-XSS/RXSS-Detection-Evaluation-GET/index.jsp' -t 3500000 -d s
```

**Arguments:**
- `-u`: Target URL for training
- `-t`: Total training timesteps (default: 4,000,000)
- `-d s`: Deterministic state space (53 dims, maps to "Distinct States")
- `-d a`: Deterministic action space (46 actions)

Trained models are saved to `Modified-Link/models/` with timestamped filenames. TensorBoard logs are saved to `Modified-Link/tensorboard_log/`.

## Analysis

### Parsing training logs

`parse_train_log.py` extracts episode statistics from training log files, computing mean and variance metrics incrementally:

```bash
python parse_train_log.py
```

### Compressing log files

Training logs can be large. Use `compress_logs.py` to reduce them to a minimal format while preserving all data needed for analysis:

```bash
python compress_logs.py
```
