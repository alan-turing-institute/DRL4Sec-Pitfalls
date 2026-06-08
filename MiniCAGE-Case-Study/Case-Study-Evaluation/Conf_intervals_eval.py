#!/usr/bin/env python3
import re
import math
import sys

# This can be run from the command line for the set of evals done (using agent_evaluation.py)
# It should be super quick. I think it would be easy to automate this to do the set of evals for each training stage so
# you don't have to run them all manually.
# python Conf_intervals_eval.py mini_CAGE/models/SB3_PPO_default/risk_eval_{red_name}_{TRAINING_STAGE}.txt


def extract_mean_rewards_by_block(path):
    """
    Parse the text file and group mean_reward values by each
    'Per-agent results ...' block.

    Returns: list of (label, [mean_rewards])
    """
    blocks = []
    current_block = None
    current_label = None

    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Start of a new per-agent block
            if line.startswith("Per-agent results"):
                # Save previous block if it exists
                if current_block is not None:
                    blocks.append((current_label, current_block))

                current_label = line
                current_block = []
                continue

            # Extract mean_reward values within a block
            if current_block is not None:
                m = re.search(r"mean_reward=([-\d\.]+)", line)
                if m:
                    current_block.append(float(m.group(1)))

        # Add the final block if present
        if current_block is not None:
            blocks.append((current_label, current_block))

    return blocks


def t_critical_95(n):
    """
    Return the 95% two-sided t critical value for sample size n (df = n - 1).

    Tries scipy.stats.t first; if not available, falls back to a small
    hard-coded t-table (sufficient for n <= 25).
    """
    df = n - 1
    try:
        from scipy.stats import t
        return t.ppf(0.975, df)  # 0.975 is for two-sided 95% CI
    except Exception:
        # Hard-coded 95% two-sided t values for df = 1..24
        t_table_95 = {
            1: 12.706,  2: 4.303,  3: 3.182,  4: 2.776,
            5: 2.571,   6: 2.447,  7: 2.365,  8: 2.306,
            9: 2.262,   10: 2.228, 11: 2.201, 12: 2.179,
            13: 2.160,  14: 2.145, 15: 2.131, 16: 2.120,
            17: 2.110,  18: 2.101, 19: 2.093, 20: 2.086,
            21: 2.080,  22: 2.074, 23: 2.069, 24: 2.064,
        }
        if df in t_table_95:
            return t_table_95[df]
        raise RuntimeError(
            f"No hard-coded t critical value for df={df}. "
            "Please install SciPy to support this sample size."
        )


def mean_ci_t(data, confidence=0.95):
    """
    Compute a t-based confidence interval for the mean of data.

    Returns (mean, lower, upper).
    """
    n = len(data)
    if n < 2:
        raise ValueError("Need at least 2 data points to compute a t-based CI.")

    # Sample mean
    mean = sum(data) / n

    # Sample standard deviation with Bessel's correction
    var = sum((x - mean) ** 2 for x in data) / (n - 1)
    s = math.sqrt(var)

    # Standard error
    se = s / math.sqrt(n)

    # t critical for the requested confidence level (95%)
    tcrit = t_critical_95(n)

    margin = tcrit * se
    lower = mean - margin
    upper = mean + margin

    return mean, lower, upper


def main(path):
    blocks = extract_mean_rewards_by_block(path)

    if not blocks:
        print("No 'Per-agent results' blocks with mean_reward found in the file.")
        return

    for label, data in blocks:
        mean, lower, upper = mean_ci_t(data, confidence=0.95)
        print(label)
        print(f"  n = {len(data)}")
        print(f"  mean_reward = {mean:.4f}")
        print(f"  95% t-based CI: [{lower:.4f}, {upper:.4f}]")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_mean_reward_ci.py path/to/risk_eval_file.txt")
    else:
        main(sys.argv[1])

