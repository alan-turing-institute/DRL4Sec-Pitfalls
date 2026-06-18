#!/usr/bin/env python3
"""
extract-results.py: Read outputs from the four case studies' reproduce
steps and write a single human-readable summary at
``reproducibility/results-summary.md``.

Run from a venv that has torch + stable-baselines3 installed
(``reproducibility/.venvs/minicage``) because the MiniCAGE section needs
to load PPO checkpoints and step the env. The Link / SQiRL / AutoRobust
sections only need numpy + pandas.

Usage:
    python reproducibility/extract-results.py             # default eval params
    python reproducibility/extract-results.py --eval-episodes 200
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
REPRO_DIR = REPO_ROOT / "reproducibility"


# ──────────────────────────────────────────────────────────────────────
# Stats helpers
# ──────────────────────────────────────────────────────────────────────

# Hard-coded t critical values for two-sided 95% (df = n - 1, n in 2..30).
# Fallback for envs that don't ship scipy. Matches MiniCAGE's
# Conf_intervals_eval.py table; extended to df 25-29.
_T_TABLE_95 = {
    1: 12.706,  2: 4.303,  3: 3.182,  4: 2.776,  5: 2.571,
    6: 2.447,   7: 2.365,  8: 2.306,  9: 2.262,  10: 2.228,
    11: 2.201,  12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120,  17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080,  22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056,  27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _t_critical_95(n: int) -> float:
    """Two-sided 95% Student-t critical value for df = n-1, with a
    hard-coded fallback so this works in envs without scipy."""
    df = n - 1
    try:
        from scipy.stats import t as _scipy_t  # type: ignore
        return float(_scipy_t.ppf(0.975, df))
    except Exception:
        if df in _T_TABLE_95:
            return _T_TABLE_95[df]
        # Large-sample limit; close enough for df > 30.
        return 1.96


def mean_ci_t_95(values: List[float]) -> Optional[Tuple[float, float, float, int]]:
    """Return (mean, ci_lo, ci_hi, n) for the values, or None if empty.

    Uses a Student-t 95% CI (matches each case study's own analysis
    code). Sample standard deviation uses Bessel's correction. For
    n < 2 the CI collapses to the point estimate.
    """
    if not values:
        return None
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return (m, m, m, n)
    s = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    se = s / math.sqrt(n)
    half = _t_critical_95(n) * se
    return (m, m - half, m + half, n)


# Back-compat shim: keep the old name pointing at the t-based version so
# anything inside the script that still references the old helper picks up
# the new methodology automatically.
mean_ci_95 = mean_ci_t_95


def fmt_ci(stat, fmt: str = "{:.2f}") -> str:
    if stat is None:
        return "_n/a_"
    m, lo, hi, _ = stat
    return f"{fmt.format(m)} ({fmt.format(lo)}, {fmt.format(hi)})"


def fmt_n(stat) -> str:
    return "0" if stat is None else str(stat[3])


# ──────────────────────────────────────────────────────────────────────
# §5.4 Link: Table 1
# ──────────────────────────────────────────────────────────────────────

# train_A2C.py log_title embeds a "deterministic_mode" suffix. Map it back
# to the paper's two formulations. Other suffixes are ablation extras that
# Table 1 does not report.
LINK_MODE_TO_FORMULATION = {
    "non-d": "Original",
    "ds-nda": "Distinct States",
}
LINK_TOTAL_TARGETS = 32  # WAVSEP reflected-XSS targets (paper §5.4)
LINK_SUCCESS_URL_RE = re.compile(
    r"^url:\s+\S+\s+\S*?(Case\d+-[A-Za-z0-9]+\.jsp)", re.MULTILINE
)
LINK_LOG_RE = re.compile(
    r"^train-log-(?P<ts>[\d.\-]+)-+(?P<timesteps>\d+)-A2C-.*?O-(?P<mode>[\w\-]+)\.txt$"
)


def parse_link_log(path: Path) -> Optional[int]:
    """Return the number of distinct WAVSEP Cases for which the agent
    achieved at least one successful XSS detection (i.e. the env wrote
    a `url: <URL>` line, which only happens on done=True). Each Case
    counts once regardless of how many payloads succeeded against it."""
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return None
    unique_cases = set(LINK_SUCCESS_URL_RE.findall(text))
    return len(unique_cases)


def section_link(out: List[str]) -> None:
    out.append("\n## §5.4 Link: Table 1 (Vulnerabilities Found, %)\n")
    log_dir = REPO_ROOT / "Link-Case-Study/Modified-Link"
    logs = sorted(log_dir.glob("train-log-*.txt"))
    if not logs:
        out.append("_No `train-log-*.txt` files found in Modified-Link/ skipping._\n")
        return

    # Methodology mirrors parse_train_log.py:calculate_proportion_ci
    # (lines 324-356): each (run, target) pair is one Bernoulli outcome,
    # and the CI is Student-t on the flat 0/1 list of size
    # n_runs × LINK_TOTAL_TARGETS per formulation.
    by_formulation: Dict[str, List[int]] = {"Original": [], "Distinct States": []}
    n_logs: Dict[str, int] = {"Original": 0, "Distinct States": 0}
    skipped: List[str] = []
    for path in logs:
        m = LINK_LOG_RE.match(path.name)
        if not m:
            skipped.append(path.name)
            continue
        mode = m.group("mode")
        formulation = LINK_MODE_TO_FORMULATION.get(mode)
        if formulation is None:
            # ablation extras like "d" or "nds-da" not in Table 1
            continue
        vf_count = parse_link_log(path)
        if vf_count is None:
            continue
        n_logs[formulation] += 1
        # Append `vf_count` ones (this run hit that target) and
        # `LINK_TOTAL_TARGETS - vf_count` zeros to the flat 0/1 list.
        by_formulation[formulation].extend([1] * vf_count)
        by_formulation[formulation].extend([0] * (LINK_TOTAL_TARGETS - vf_count))

    out.append("| Formulation | VF% (95% CI) | n logs |")
    out.append("|---|---|---|")
    for formulation in ("Original", "Distinct States"):
        stat = mean_ci_t_95([float(b) for b in by_formulation[formulation]])
        if stat is not None:
            m, lo, hi, _ = stat
            stat = (100.0 * m, 100.0 * lo, 100.0 * hi, n_logs[formulation])
        out.append(f"| {formulation} | {fmt_ci(stat, '{:.1f}')} | {n_logs[formulation]} |")
    if skipped:
        out.append(
            f"\n_Skipped {len(skipped)} log file(s) that did not match the expected "
            "filename pattern._\n"
        )



# ──────────────────────────────────────────────────────────────────────
# §7.4 AutoRobust: Figure 3
# ──────────────────────────────────────────────────────────────────────

def section_autorobust(out: List[str]) -> None:
    out.append("\n## §7.4 AutoRobust: Figure 3 (Mean Malware Score across runs)\n")
    csv_path = REPO_ROOT / "AutoRobust-Case-Study/results/test_multi_rew1.csv"
    if not csv_path.exists():
        out.append(
            "_No `AutoRobust-Case-Study/results/test_multi_rew1.csv` skipping._\n"
        )
        return

    try:
        rows = list(csv.DictReader(csv_path.open()))
    except OSError as exc:
        out.append(f"_Error reading {csv_path}: {exc} skipping._\n")
        return

    def col(name: str) -> List[float]:
        return [float(r[name]) for r in rows if r.get(name) not in (None, "")]

    policies = [
        ("MDP",    "mdp_mean_malware_score"),
        ("POMDP",  "pomdp_mean_malware_score"),
        ("Random", "random_mean_malware_score"),
    ]
    out.append("| Policy | Mean malware score (95% CI) | n agents |")
    out.append("|---|---|---|")
    for name, column in policies:
        try:
            stat = mean_ci_t_95(col(column))
        except (KeyError, ValueError):
            stat = None
        out.append(f"| {name} | {fmt_ci(stat, '{:.3f}')} | {fmt_n(stat)} |")


# ──────────────────────────────────────────────────────────────────────
# §7.4 SQiRL: Table 2
# ──────────────────────────────────────────────────────────────────────
#
# The Sanitised vs Non-Sanitised eval split is a per-endpoint property
# of no_feedback.php's sub-targets. The list comes from
# SQiRL-CaseStudy/Modified-SQiRL/analyze_results.py:269.
SQIRL_SANITISED_ENDPOINTS = frozenset([
    "sqli14.php", "sqli15.php", "sqli16.php", "sqli17.php", "sqli18.php",
    "sqli26.php", "sqli27.php", "sqli28.php", "sqli29.php",
])

# Markers in result_stats_*.stats files (mirrors analyze_results.py logic).
SQIRL_INPUT_RE     = re.compile(r"^\[!\] (?:Starting|Changing) to input \w+,(.*)$")
SQIRL_WIN_MARK_RE  = re.compile(r"\[!\] reached win criteria")
SQIRL_LOSS_MARK_RE = re.compile(r"\[!\] reached loss criteria")
SQIRL_ENDPOINT_RE  = re.compile(r"(sqli\d+\.php)")


def parse_sqirl_result_stats(path: Path) -> Optional[Dict[str, bool]]:
    """Return {endpoint: True_if_vuln_found_else_False} for one agent's run,
    or None if the file isn't a recognisable result_stats file. Mirrors
    analyze_results.py:parse_result_stats but trimmed to what we need."""
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return None
    results: Dict[str, bool] = {}
    current_endpoint: Optional[str] = None
    for raw in text.splitlines():
        line = raw.strip()
        m = SQIRL_INPUT_RE.match(line)
        if m:
            ep_match = SQIRL_ENDPOINT_RE.search(m.group(1))
            current_endpoint = ep_match.group(1) if ep_match else None
            continue
        if current_endpoint and SQIRL_WIN_MARK_RE.search(line):
            results[current_endpoint] = True
            current_endpoint = None
            continue
        if current_endpoint and SQIRL_LOSS_MARK_RE.search(line):
            results[current_endpoint] = False
            current_endpoint = None
            continue
    return results if results else None


def section_sqirl(out: List[str]) -> None:
    out.append("\n## §7.4 SQiRL: Table 2 (Vulnerabilities Found, %)\n")
    test_root = REPO_ROOT / "SQiRL-CaseStudy/Modified-SQiRL/test_results"
    if not test_root.exists():
        out.append(
            "_No `test_results/` directory; skipping. Run "
            "`./test_agents.sh` after training to generate it._\n"
        )
        return

    cell_data: Dict[Tuple[str, str], List[int]] = {
        ("sanitized", "san"):     [],
        ("sanitized", "non_san"): [],
        ("non_san",   "san"):     [],
        ("non_san",   "non_san"): [],
    }
    for source_dir in test_root.iterdir():
        if not source_dir.is_dir():
            continue
        m = re.match(r"TEST_(?P<src>sanitized|non_san)_", source_dir.name)
        if not m:
            continue
        training = m.group("src")
        for stat_file in source_dir.glob("result_stats_*.stats"):
            results = parse_sqirl_result_stats(stat_file)
            if not results:
                continue
            for endpoint, won in results.items():
                bucket = "san" if endpoint in SQIRL_SANITISED_ENDPOINTS else "non_san"
                cell_data[(training, bucket)].append(1 if won else 0)

    def _vf_stat(bools: List[int]) -> Optional[Tuple[float, float, float, int]]:
        stat = mean_ci_t_95([float(b) for b in bools])
        if stat is None:
            return None
        m, lo, hi, n = stat
        return (100.0 * m, 100.0 * lo, 100.0 * hi, n)

    out.append(
        "| Training | Eval Sanitized VF% (95% CI) "
        "| Eval Non-Sanitized VF% (95% CI) |"
    )
    out.append("|---|---|---|")
    for training, label in (("sanitized", "Combined"), ("non_san", "Non-Sanitized")):
        san_stat = _vf_stat(cell_data[(training, "san")])
        non_stat = _vf_stat(cell_data[(training, "non_san")])
        out.append(
            f"| {label} | {fmt_ci(san_stat, '{:.1f}')} "
            f"| {fmt_ci(non_stat, '{:.1f}')} |"
        )


# ──────────────────────────────────────────────────────────────────────
# MiniCAGE §6.4 / §8.3: evaluate trained checkpoints
# ──────────────────────────────────────────────────────────────────────

# Import these lazily so the Link/SQiRL/AutoRobust sections can run in any
# venv even if torch/stable-baselines3 aren't available.

MINICAGE_DIR_RE = re.compile(
    r"^SB3_PPO_(?P<exp>[\w]+?)_(?P<red>bline|meander|mixed)_(?P<order>R2B|B2R|Mixed)$"
)


def _import_minicage_env():
    """Return MiniCageBlue or raise."""
    sys.path.insert(0, str(
        REPO_ROOT / "MiniCAGE-Case-Study/Modified-MiniCAGE"
    ))
    sys.path.insert(0, str(
        REPO_ROOT / "MiniCAGE-Case-Study/Modified-MiniCAGE/mini_CAGE"
    ))
    from single_agent_gym_wrapper import MiniCageBlue  # noqa: E402
    from stable_baselines3 import PPO                 # noqa: E402
    return MiniCageBlue, PPO


def _eval_checkpoint(ckpt: Path, red: str, order: str, n_episodes: int, max_steps: int) -> float:
    """Run n_episodes against (red, order); return mean episodic return."""
    MiniCageBlue, PPO = _import_minicage_env()
    env = MiniCageBlue(red_policy=red, max_steps=max_steps, action_order=order)
    model = PPO.load(ckpt, device="cpu")
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(int(action))
            total += reward
        returns.append(total)
    return sum(returns) / len(returns) if returns else float("nan")


def _find_final_checkpoints(exp_dir: Path) -> List[Path]:
    """Return all `*_final.zip` paths under exp_dir/agent_NN/."""
    return sorted(exp_dir.glob("agent_*/*_final.zip"))


def _evaluate_config(exp_dir: Path, eval_combos: List[Tuple[str, str]],
                     n_episodes: int, max_steps: int) -> Dict[Tuple[str, str], Optional[Tuple]]:
    """For one training config, eval each agent against each (red, order)
    combo. Returns {(red, order): Student-t 95% CI over per-agent mean
    episodic returns} matches `Conf_intervals_eval.py:mean_ci_t`."""
    finals = _find_final_checkpoints(exp_dir)
    if not finals:
        return {combo: None for combo in eval_combos}
    out: Dict[Tuple[str, str], Optional[Tuple]] = {}
    for combo in eval_combos:
        red, order = combo
        agent_means = [
            _eval_checkpoint(ckpt, red, order, n_episodes, max_steps)
            for ckpt in finals
        ]
        out[combo] = mean_ci_t_95(agent_means)
    return out


def section_minicage(out: List[str], n_episodes: int, max_steps: int) -> None:
    out.append(
        "\n## §6.4 / §8.3 MiniCAGE: Figure 2 + Table 3 "
        f"({n_episodes} eval episodes per agent)\n"
    )
    ppo_root = REPO_ROOT / "MiniCAGE-Case-Study/Modified-MiniCAGE/ppo_models"
    if not ppo_root.exists():
        out.append("_No `ppo_models/` directory skipping._\n")
        return
    try:
        _import_minicage_env()
    except ImportError as exc:
        out.append(
            f"_Cannot evaluate MiniCAGE missing deps: {exc}. "
            "Run from `reproducibility/.venvs/minicage`._\n"
        )
        return

    # Catalog what training configs we have on disk.
    trained: Dict[Tuple[str, str, str], Path] = {}
    for d in sorted(ppo_root.iterdir()):
        if not d.is_dir():
            continue
        m = MINICAGE_DIR_RE.match(d.name)
        if not m:
            continue
        trained[(m["exp"], m["red"], m["order"])] = d

    if not trained:
        out.append("_No `ppo_models/SB3_PPO_*_*_*/` training dirs skipping._\n")
        return

    out.append(
        f"Found {len(trained)} training configuration(s). "
        "Each agent is evaluated deterministically; results aggregated across agents.\n"
    )

    # --- §6.4 Figure 2: 4 HP configs × bline × R2B (the paper baseline) ---
    out.append("\n### §6.4 Figure 2: vs. B-line, R2B order\n")
    out.append("| HP config | Mean episodic return (95% CI) | n agents |")
    out.append("|---|---|---|")
    for exp in ("default", "alt_alpha", "alt_gamma", "alt_eps"):
        d = trained.get((exp, "bline", "R2B"))
        if d is None:
            out.append(f"| {exp} | _not trained_ | 0 |")
            continue
        stat = _evaluate_config(d, [("bline", "R2B")], n_episodes, max_steps)[("bline", "R2B")]
        out.append(f"| {exp} | {fmt_ci(stat, '{:.2f}')} | {fmt_n(stat)} |")

    # --- §8.3 Table 3 top (D-UA): action order matrix ---
    out.append("\n### §8.3 Table 3 top: D-UA (action order: train × eval)\n")
    eval_combos = [("bline", o) for o in ("R2B", "B2R", "Mixed")]
    out.append("| Training Order | Eval R2B | Eval B2R | Eval Mixed |")
    out.append("|---|---|---|---|")
    for train_order in ("R2B", "B2R", "Mixed"):
        d = trained.get(("default", "bline", train_order))
        if d is None:
            out.append(f"| {train_order} | _not trained_ | _not trained_ | _not trained_ |")
            continue
        stats = _evaluate_config(d, eval_combos, n_episodes, max_steps)
        cells = " | ".join(fmt_ci(stats[combo], '{:.1f}') for combo in eval_combos)
        out.append(f"| {train_order} | {cells} |")

    # --- §8.3 Table 3 bottom (D-NS): red attacker matrix ---
    out.append("\n### §8.3 Table 3 bottom: D-NS (red attacker: train × eval), R2B order\n")
    eval_combos = [(r, "R2B") for r in ("bline", "meander", "mixed")]
    out.append("| Training Red | Eval bline | Eval meander | Eval mixed |")
    out.append("|---|---|---|---|")
    for train_red in ("bline", "meander", "mixed"):
        d = trained.get(("default", train_red, "R2B"))
        if d is None:
            out.append(f"| {train_red} | _not trained_ | _not trained_ | _not trained_ |")
            continue
        stats = _evaluate_config(d, eval_combos, n_episodes, max_steps)
        cells = " | ".join(fmt_ci(stats[combo], '{:.1f}') for combo in eval_combos)
        out.append(f"| {train_red} | {cells} |")
    out.append("")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-episodes", type=int, default=1000,
        help="MiniCAGE evaluation episodes per (agent, eval-config) [default: 100]"
    )
    parser.add_argument(
        "--max-steps", type=int, default=100,
        help="MiniCAGE max steps per evaluation episode [default: 100]"
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPRO_DIR / "results-summary.md",
        help="Output markdown file [default: reproducibility/results-summary.md]"
    )
    parser.add_argument(
        "--skip", choices=["link", "autorobust", "sqirl", "minicage"],
        action="append", default=[],
        help="Skip a case study section (may be repeated)"
    )
    args = parser.parse_args(argv)

    out: List[str] = []
    out.append("# DRL4Sec-Pitfalls Reproduction Results\n")
    out.append(f"**Generated**: {datetime.now().isoformat(timespec='seconds')}\n")
    out.append(textwrap.dedent("""\
        This file is produced by `reproducibility/extract-results.py` from
        whatever case-study outputs currently exist on disk. Sections are
        independent: missing inputs report `_n/a_` or `_not trained_`
        instead of failing.

    """))

    if "link" not in args.skip:
        try:
            section_link(out)
        except Exception as exc:
            out.append(f"\n_Link section errored: {exc}_\n")
    if "autorobust" not in args.skip:
        try:
            section_autorobust(out)
        except Exception as exc:
            out.append(f"\n_AutoRobust section errored: {exc}_\n")
    if "sqirl" not in args.skip:
        try:
            section_sqirl(out)
        except Exception as exc:
            out.append(f"\n_SQiRL section errored: {exc}_\n")
    if "minicage" not in args.skip:
        try:
            section_minicage(out, args.eval_episodes, args.max_steps)
        except Exception as exc:
            out.append(f"\n_MiniCAGE section errored: {exc}_\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out) + "\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
