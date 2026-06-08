# SoK: The Pitfalls of Deep Reinforcement Learning for Cybersecurity

This repository provides the code and experiments for the paper:

> **SoK: The Pitfalls of Deep Reinforcement Learning for Cybersecurity**
> Shae McFadden, Myles Foley, Elizabeth Bates, Ilias Tsingenopoulos, Sanyam Vyas, Vasilios Mavroudis, Chris Hicks, Fabio Pierazzi

***

## Overview

Deep Reinforcement Learning (DRL) has been increasingly applied to cybersecurity problems, from autonomous network defense to adversarial malware generation and web vulnerability discovery. However, transitioning DRL from laboratory simulations to real-world cyber environments introduces systematic methodological challenges.

This paper identifies and systematizes **11 recurring pitfalls** in the DRL for cybersecurity (DRL4Sec) literature, organized across four stages of agent development:

| Stage | Pitfalls |
|---|---|
| **Environment Modeling** | MDP Specification (M-MS), Modeling Correctness (M-MC), Partial Observability (M-PO) |
| **Agent Training** | Hyperparameter Reporting (T-HR), Variance Analysis (T-VA), Policy Convergence (T-PC) |
| **Performance Evaluation** | Application Motivation (E-AM), Gain Attribution (E-GA), Environment Complexity (E-EC) |
| **System Deployment** | Underlying Assumptions (D-UA), Non-Stationarity (D-NS) |

Through systematic analysis of 66 DRL4Sec papers (2018-2025), we find an average of over five pitfalls per paper. We demonstrate the practical impact of these pitfalls through controlled experiments across three cybersecurity domains.

***

## Pitfalls Review

The breakdown of pitfall prevalence across the 66 DRL4Sec papers reviewed is presented in the figure below. 
For more information about the review process please see: [Review Information](SoK-Review-Information/Review-Information.md).

![DRL4Sec Pitfalls Breakdown](SoK-Review-Information/DRL4Sec-Pitfalls.png)



***

## Case Studies

The repository contains four case study directories, each targeting different pitfalls:

### [AutoRobust Case Study](AutoRobust-Case-Study/README.md): Evaluation Pitfalls (Section 7.4)

**Domain:** Adversarial malware creation

Reproduction of an RL environment where agents learn to evade ML-based malware detection by modifying dynamic analysis reports. Demonstrates the **Gain Attribution (E-GA)** pitfall by disentangling the contributions of state-action space design from the learned policy. Agents are evaluated under fully observable (MDP), partially observable (POMDP), and random policy conditions to show that a large portion of overall performance comes from the action space itself rather than the learned policy.

### [Link Case Study](Link-Case-Study/README.md): Modeling Pitfalls (Section 5.4)

**Domain:** Web security testing (XSS)

Extension of the Link XSS payload generation environment. Demonstrates **Modeling Correctness (M-MC)** and **Partial Observability (M-PO)** pitfalls by comparing two MDP formulations: the original state space (with conflated features) versus a corrected *Distinct States* formulation. Shows that minor state-space corrections increase vulnerability detection by 10.1% with a 17.3% improvement in the lower bound of the 95% confidence interval across 20 training runs.

### [SQiRL Case Study](SQiRL-CaseStudy/README.md): Evaluation Pitfalls (Section 7.4)

**Domain:** Web security testing (SQL injection)

Extension of the SQiRL SQL injection detection environment. Demonstrates the **Environment Complexity (E-EC)** pitfall by training agents with and without experience of sanitized (defended) endpoints. Shows that performance is over-inflated when evaluating in simplified environments: agents trained only on non-sanitized endpoints suffer up to a 47.4% drop in vulnerability detection when tested against sanitized defenses.

### [MiniCAGE Case Study](MiniCAGE-Case-Study/README.md): Training & Deployment Pitfalls (Sections 6.4, 8.3)

**Domain:** Autonomous cyber defense

Extension of the MiniCAGE network defense environment. Used for two sets of experiments:

- **Training Pitfalls (Section 6.4):** Trains 20 PPO blue agents under four hyperparameter configurations to demonstrate Hyperparameter Reporting (T-HR), Variance Analysis (T-VA), and Policy Convergence (T-PC). Shows how a single hyperparameter change can drastically alter performance, and how evaluating over multiple runs with confidence intervals reveals the true distribution of agent behavior.
- **Deployment Pitfalls (Section 8.3):** Demonstrates Underlying Assumptions (D-UA) by varying action ordering between red and blue agents, and Non-Stationarity (D-NS) by training against fixed attacker strategies and evaluating against unseen ones. Shows that minor assumption violations and non-stationary adversaries can severely degrade deployed agent performance.

***

## Repository Structure

```
DRL4Sec-Pitfalls/
├── AutoRobust-Case-Study/     # Adversarial malware evasion (E-GA)
│   ├── autoattack.py          # Training and evaluation script
│   ├── finetune_dynamic_reports.py  # Malware classifier training
│   ├── corpus.json            # Attack corpus
│   ├── sum.json               # Summary data
│   ├── AutoRobust-MDP.md      # MDP specification
│   └── envs/                  # Gymnasium environment
├── Link-Case-Study/           # XSS payload generation (M-MC, M-PO)
│   ├── Modified-Link/         # Extended Link environment
│   │   ├── train_A2C.py       # Agent training
│   │   ├── run_train.sh       # Full experiment automation
│   │   ├── parse_train_log.py # Results analysis
│   │   ├── compress_logs.py   # Log compression utility
│   │   ├── attack_A2C.py      # Attack evaluation
│   │   ├── environment.yaml   # Conda environment spec
│   │   └── gym_reflected_xss/ # Gymnasium environment
│   ├── wavsep-docker/         # WAVSEP Docker target
│   ├── case_study.diff        # Modifications from upstream
│   └── Link-MDP.md            # MDP specification
├── SQiRL-CaseStudy/           # SQL injection detection (E-EC)
│   ├── Modified-SQiRL/        # Extended SQiRL environment
│   │   ├── train_agents.sh    # Full training automation
│   │   ├── test_agents.sh     # Full evaluation automation
│   │   ├── analyze_results.py # Results analysis
│   │   └── SQLiMicrobenchmark/# Docker-based test environment
│   └── case_study.diff        # Modifications from upstream
├── MiniCAGE-Case-Study/       # Autonomous cyber defense (T-HR, T-VA, T-PC, D-UA, D-NS)
│   ├── Modified-MiniCAGE/     # Extended MiniCAGE environment
│   │   ├── mini_CAGE/         # Environment and training code
│   │   ├── Debugged_CybORG/   # Debugged CybORG dependency
│   │   └── environment.yml    # Conda environment spec
│   ├── Case-Study-Evaluation/ # Evaluation and analysis scripts
│   └── MiniCAGE-MDP.md        # MDP specification
├── SoK-Review-Information/    # Paper review methodology
│   ├── Review-Information.md  # Review process details
│   └── DRL4Sec-Pitfalls.png   # Pitfall prevalence figure
└── README.md
```

***

## Important Notice

The Link, SQiRL, and MiniCAGE case studies are extensions of public repositories. The code and READMEs in the `Modified-*` subdirectories contain the modified version of the original environments used to run the experiments in our paper. Each case study directory includes a `case_study.diff` file documenting the modifications made for our experiments.

***
