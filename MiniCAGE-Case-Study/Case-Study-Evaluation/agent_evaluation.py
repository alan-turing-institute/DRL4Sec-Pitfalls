

from __future__ import annotations
from pathlib import Path
import re
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from stable_baselines3 import PPO
from mini_CAGE.single_agent_gym_wrapper import MiniCageBlue

from rl_reliability_metrics.metrics import metrics_offline

BASE_DIR = Path(
    # "models/SB3_PPO_default/"
    "models/SB3_PPO_alt_gamma_granular/"
    # "models/SB3_PPO_alt_eps_0.05/"
    # "models/SB3_PPO_alt_alpha_003/"
)

RED_CONFIGS: Dict[str, Path] = {
    "bline": BASE_DIR,
    # "meander": BASE_DIR / "SB3_PPO_meander",
}

_STEP_FINAL_RE = re.compile(r"_final$")

CROSS_EVAL_RED: Optional[str] = None   # or "meander", "random", or None to disable cross-eval

EXPERIMENT = [
    # 'ppo_mini_cage_bline_20251006_163120',
    # 'ppo_mini_cage_bline_20251007_214908',

    # 'ppo_mini_cage_Red_Blue_bline_20251228_233333',
    # 'ppo_mini_cage_bline_20251005_142732',

    'ppo_mini_cage_Red_Blue_bline_20251229_023510',
    #
    # 'ppo_mini_cage_bline_20251006_163006',
    # 'ppo_mini_cage_bline_20251007_215014',
    # 'ppo_mini_cage_bline_20251007_215015',
    # 'ppo_mini_cage_Red_Blue_bline_20251229_160752',

    # 'ppo_mini_cage_bline_20251013_175139',
    # 'ppo_mini_cage_bline_20251013_122427',
    # 'ppo_mini_cage_Red_Blue_bline_20251229_143628'
]

versions = [
    'final', 'step_50000', 'step_100000', 'step_250000', 'step_500000', 'step_750000', 'step_1000000', 'step_2000000',
    'step_2500000'
]

SELECTED_INDICES = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 20}
N_EPISODES       = 1_000
MAX_STEPS        = 100

def load_agents(version) -> list[Path]:
    """
    Load PPO checkpoints selected by discover_checkpoints().
    """
    agents: list[Path] = []
    for idx in SELECTED_INDICES:
        for exp in EXPERIMENT:
            # if idx > 10:
            #     vidx = idx - 10
            # else:
            #     vidx = idx
            vidx = idx
            ckpt_path = f'{BASE_DIR}/agent_{idx if idx > 9 else f"0{idx}"}/{exp}_{vidx}_{version}.zip'
            try:
                model = PPO.load(ckpt_path, device="cpu")
            except Exception as exc:
                print(f"[WARN] Skipping {ckpt_path}: {exc}")
                continue
            agents.append(Path(ckpt_path))

    if not agents:
        raise RuntimeError(
            f"No checkpoints with suffix in {sorted(SELECTED_INDICES)} "
            f"found under {BASE_DIR}"
        )

    return agents

def evaluate_model(model: PPO, n_episodes: int, red_policy: str):
    """
    Evaluate a single blue model for n_episodes.

    model : PPO
        The blue defender policy.
    n_episodes : int
        Number of evaluation episodes.
    red_policy : str
        - "bline"   : always use bline red
        - "meander" : always use meander red
        - "random"  : for each episode, choose uniformly at random from
                      {"bline", "meander"} and instantiate the env with
                      that red for that episode.
    """
    episode_rewards: List[float] = []

    for _ in range(n_episodes):
        # Choose which red agent to defend against this episode
        if red_policy == "random":
            actual_red = np.random.choice(["bline", "meander"])
        else:
            actual_red = red_policy

        env = MiniCageBlue(red_policy=actual_red, max_steps=MAX_STEPS)

        obs, _ = env.reset()
        done = False
        episode_return = 0.0  # episodic reward

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_return += float(reward)

        episode_rewards.append(episode_return)
        env.close()

    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else float("nan")
    return mean_reward, episode_rewards


def calc_cvar(rollouts: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Lower/Upper CVaR across rollouts.
    """
    lower_metric = metrics_offline.LowerCVaRAcrossRollouts()
    upper_metric = metrics_offline.UpperCVaRAcrossRollouts()
    return lower_metric(rollouts), upper_metric(rollouts)


def evaluate_checkpoints_against_red(
    checkpoints: List[Path],
    eval_red_name: str,
    header: str,
    out_path: Optional[Path],
    write_to_file: bool = True,
):
    """
    Core loop: given a list of checkpoints, evaluate them against
    the specified red policy, print results, and write summary.

    eval_red_name can be "bline", "meander", or "random" (random red each episode).
    """

    print("\n" + "=" * 70)
    print(f" {header}")
    print("=" * 70)

    if not checkpoints:
        print("No checkpoints supplied to evaluation.")
        return {
            "red": eval_red_name,
            "n_agents": 0,
            "agent_names": [],
            "mean_rewards": [],
            "lower_cvars": [],
            "upper_cvars": [],
        }

    mean_rewards: List[float] = []
    lower_cvars_all_agents: List[float] = []
    upper_cvars_all_agents: List[float] = []

    agent_names = [p.parent.name for p in checkpoints]

    for ckpt_path, agent_name in zip(checkpoints, agent_names):
        print(f"\n--- Agent {agent_name} | checkpoint={ckpt_path.name} ---")

        try:
            model = PPO.load(str(ckpt_path), device="cpu")
        except Exception as exc:
            print(f"[WARN] Skipping {ckpt_path.name}: {exc}")
            continue

        mean_rew, episode_rewards = evaluate_model(
            model, N_EPISODES, red_policy=eval_red_name
        )

        if episode_rewards:
            rollout_indices = np.arange(len(episode_rewards), dtype=float)
            agent_rollout = np.vstack(
                [rollout_indices, np.array(episode_rewards, dtype=float)]
            )
            lower_vals, upper_vals = calc_cvar([agent_rollout])
            lower_cvar = float(lower_vals[0])
            upper_cvar = float(upper_vals[0])
        else:
            lower_cvar = float("nan")
            upper_cvar = float("nan")

        mean_rewards.append(mean_rew)
        lower_cvars_all_agents.append(lower_cvar)
        upper_cvars_all_agents.append(upper_cvar)

        print(f"Mean environment reward          = {mean_rew:.3f}")
        if not np.isnan(lower_cvar) and not np.isnan(upper_cvar):
            print(
                "CVaR on episodic rewards (Lower / Upper) = "
                f"{lower_cvar:.3f} / {upper_cvar:.3f}"
            )
        else:
            print("CVaR on episodic rewards (Lower / Upper) = n/a")

    overall_mean_rew  = float(np.nanmean(mean_rewards)) if mean_rewards else float("nan")
    overall_lower_cvr = float(np.nanmean(lower_cvars_all_agents)) if lower_cvars_all_agents else float("nan")
    overall_upper_cvr = float(np.nanmean(upper_cvars_all_agents)) if upper_cvars_all_agents else float("nan")

    print("\n" + "-" * 70)
    print(f"[{eval_red_name}] Overall mean reward                  = {overall_mean_rew:.3f}")
    print(f"[{eval_red_name}] Average Lower CVaR (rewards)         = {overall_lower_cvr:.3f}")
    print(f"[{eval_red_name}] Average Upper CVaR (rewards)         = {overall_upper_cvr:.3f}")


    if write_to_file and out_path is not None:
        with open(out_path, "w") as f:
            f.write(f"{header}\n")
            f.write(f"Output file: {out_path}\n\n")

            f.write("Per-agent results\n")
            f.write("-" * 60 + "\n")
            for agent_name, mrw, l, u in zip(
                agent_names,
                mean_rewards,
                lower_cvars_all_agents,
                upper_cvars_all_agents,
            ):
                f.write(
                    f"{agent_name:>8}: mean_reward={mrw:.3f}, "
                    f"Lower CVaR={l:.3f}, Upper CVaR={u:.3f}\n"
                )

            f.write("\n" + "-" * 60 + "\n")
            f.write(f"Overall mean reward                  = {overall_mean_rew:.3f}\n")
            f.write(f"Average Lower CVaR (rewards)         = {overall_lower_cvr:.3f}\n")
            f.write(f"Average Upper CVaR (rewards)         = {overall_upper_cvr:.3f}\n")

        print(f"\nSaved summary to: {out_path}\n")

    return {
        "red": eval_red_name,
        "n_agents": len(mean_rewards),
        "agent_names": agent_names,
        "mean_rewards": mean_rewards,
        "lower_cvars": lower_cvars_all_agents,
        "upper_cvars": upper_cvars_all_agents,
    }


def evaluate_red_adversary(red_name: str, red_root: Path, training_stage=''):
    """
    Find all checkpoints for a given adversary root
    and evaluate them against the same red policy.
    """
    checkpoints = load_agents(version=training_stage)
    out_path = red_root / f"risk_eval_{red_name}_{training_stage}.txt"
    header = (
        f"Evaluation: blue agents trained vs RED={red_name}, "
        f"evaluated against RED={red_name} | checkpoint={training_stage}"
    )
    return evaluate_checkpoints_against_red(
        checkpoints=checkpoints,
        eval_red_name=red_name,
        header=header,
        out_path=out_path,
        write_to_file=True,
    )


# def cross_evaluate_trained_against_other():
#     """
#     Take defenders trained vs a single RED in RED_CONFIGS and:
#       1) Evaluate them vs their training RED
#       2) Evaluate them vs CROSS_EVAL_RED
#       3) Write BOTH sets of results into
#          risk_eval_<trained>_vs_<other>_<stage>.txt
#
#     Assumes RED_CONFIGS contains exactly one entry:
#       { "<trained_red>": <path> }
#
#     CROSS_EVAL_RED can be "bline", "meander", or "random".
#     """
#     if CROSS_EVAL_RED is None:
#         print("CROSS_EVAL_RED is None; skipping cross-evaluation.")
#         return None
#
#     if len(RED_CONFIGS) != 1:
#         print(f"cross_evaluate_trained_against_other expects one entry in RED_CONFIGS, got {len(RED_CONFIGS)}; skipping cross-eval.")
#         return None
#
#     trained_name, trained_root = next(iter(RED_CONFIGS.items()))
#
#     if trained_name == CROSS_EVAL_RED:
#         print(f"CROSS_EVAL_RED == '{trained_name}', nothing to cross-evaluate; skipping.")
#         return None
#
#     checkpoints = discover_checkpoints(trained_root, TRAINING_STAGE)
#
#     # 1. trained vs trained
#     header_train = (
#         f"Evaluation: blue agents trained vs RED={trained_name}, "
#         f"evaluated against RED={trained_name} | checkpoint={TRAINING_STAGE}"
#     )
#     train_summary = evaluate_checkpoints_against_red(
#         checkpoints=checkpoints,
#         eval_red_name=trained_name,
#         header=header_train,
#         out_path=None,
#         write_to_file=False,
#     )
#
#     # 2. trained vs CROSS_EVAL_RED
#     header_cross = (
#         f"Cross-evaluation: blue agents trained vs RED={trained_name}, "
#         f"evaluated against RED={CROSS_EVAL_RED} | checkpoint={TRAINING_STAGE}"
#     )
#     cross_summary = evaluate_checkpoints_against_red(
#         checkpoints=checkpoints,
#         eval_red_name=CROSS_EVAL_RED,
#         header=header_cross,
#         out_path=None,
#         write_to_file=False,
#     )
#
#     # 3. Write BOTH sections into a single combined file
#     combined_path = trained_root / f"risk_eval_{trained_name}_vs_{CROSS_EVAL_RED}_{TRAINING_STAGE}.txt"
#     with open(combined_path, "w") as f:
#         # Section 1: normal eval (trained vs trained)
#         f.write(header_train + "\n")
#         f.write(f"Output file: {combined_path}\n\n")
#         f.write(f"Per-agent results (trained vs {trained_name}, evaluated vs {trained_name})\n")
#         f.write("-" * 60 + "\n")
#         for agent_name, mrw, l, u in zip(
#             train_summary["agent_names"],
#             train_summary["mean_rewards"],
#             train_summary["lower_cvars"],
#             train_summary["upper_cvars"],
#         ):
#             f.write(
#                 f"{agent_name:>8}: mean_reward={mrw:.3f}, "
#                 f"Lower CVaR={l:.3f}, Upper CVaR={u:.3f}\n"
#             )
#         f.write("\n" + "-" * 60 + "\n")
#         f.write(
#             f"Overall mean reward                  = "
#             f"{float(np.nanmean(train_summary['mean_rewards'])):.3f}\n"
#         )
#         f.write(
#             f"Average Lower CVaR (rewards)         = "
#             f"{float(np.nanmean(train_summary['lower_cvars'])):.3f}\n"
#         )
#         f.write(
#             f"Average Upper CVaR (rewards)         = "
#             f"{float(np.nanmean(train_summary['upper_cvars'])):.3f}\n\n"
#         )
#
#         # Section 2: cross-eval (trained vs other)
#         f.write(header_cross + "\n")
#         f.write(f"Output file: {combined_path}\n\n")
#         f.write(f"Per-agent results (trained vs {trained_name}, evaluated vs {CROSS_EVAL_RED})\n")
#         f.write("-" * 60 + "\n")
#         for agent_name, mrw, l, u in zip(
#             cross_summary["agent_names"],
#             cross_summary["mean_rewards"],
#             cross_summary["lower_cvars"],
#             cross_summary["upper_cvars"],
#         ):
#             f.write(
#                 f"{agent_name:>8}: mean_reward={mrw:.3f}, "
#                 f"Lower CVaR={l:.3f}, Upper CVaR={u:.3f}\n"
#             )
#         f.write("\n" + "-" * 60 + "\n")
#         f.write(
#             f"Overall mean reward                  = "
#             f"{float(np.nanmean(cross_summary['mean_rewards'])):.3f}\n"
#         )
#         f.write(
#             f"Average Lower CVaR (rewards)         = "
#             f"{float(np.nanmean(cross_summary['lower_cvars'])):.3f}\n"
#         )
#         f.write(
#             f"Average Upper CVaR (rewards)         = "
#             f"{float(np.nanmean(cross_summary['upper_cvars'])):.3f}\n"
#         )
#
#     print(f"\n[INFO] Combined normal + cross-eval saved to: {combined_path}\n")
#
#     return cross_summary


def main() -> None:
    all_summaries = []

    # Standard evaluation (e.g. bline-trained agents vs bline, or meander vs meander)
    for red_name, red_root in RED_CONFIGS.items():
        for v in versions:
            summary = evaluate_red_adversary(red_name, red_root, training_stage=v)
            all_summaries.append(summary)

    # # Cross-evaluation: trained vs CROSS_EVAL_RED,
    # # and combined file with both sets of stats
    # cross_summary = cross_evaluate_trained_against_other()
    # if cross_summary is not None:
    #     all_summaries.append(cross_summary)


if __name__ == "__main__":
    main()