import argparse
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from envs.RLattacker import RLattacker
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import sys

class MalwareScoreCallback(BaseCallback):
    """
    Callback that tracks and displays moving average of malware scores.
    Updates on the same line to avoid terminal clutter.
    """
    def __init__(self, window_size=20, update_freq=1024):
        super().__init__()
        self.window_size = window_size
        self.update_freq = update_freq
        self.scores = deque(maxlen=window_size)
        self.episode_count = 0
        self.last_update = 0
        
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones')[0]:
            # Get malware score from info
            info = self.locals.get('infos')[0]
            if 'mal_prob' in info:
                self.scores.append(info['mal_prob'])
                self.episode_count += 1
        
        # Update display periodically
        if self.num_timesteps - self.last_update >= self.update_freq:
            self.last_update = self.num_timesteps
            if len(self.scores) > 0:
                ma = np.mean(self.scores)
                recent = self.scores[-1] if self.scores else 0.0
                min_score = np.min(self.scores) if len(self.scores) > 0 else 1.0
                # Print on same line
                sys.stdout.write(f"\rStep: {self.num_timesteps:>7} | Episodes: {self.episode_count:>4} | "
                               f"Recent: {recent:.4f} | MA({self.window_size}): {ma:.4f} | Min: {min_score:.4f}")
                sys.stdout.flush()
        
        return True
    
    def _on_training_end(self) -> None:
        # Add newline at the end
        sys.stdout.write("\n")
        sys.stdout.flush()

def train_agent(rew, steps, pomdp, lr, arch, ts, n_agents=5, base_seed=42):
    pomdp_suffix = '_pomdp' if pomdp else ''
    
    for agent_idx in range(n_agents):
        seed = base_seed + agent_idx
        print(f"\n=== Training agent with seed {seed} ===")
        
        # Create environment
        env = gym.make("RLattacker-v0", steps=steps, rew=rew, pomdp=pomdp, seed=seed)
        
        model = PPO(
            policy="MlpPolicy",
            env=env,
            n_steps=1024,
            batch_size=32,
            learning_rate=lr,
            gamma=0.99,
            seed=seed,
            ent_coef=0.001,
            clip_range=0.2,
            max_grad_norm=1,
            gae_lambda=0.95,
            policy_kwargs=dict(net_arch=[arch, arch]),
            device='cuda',
            verbose=0
        )
        
        # Create callback for monitoring
        monitor_callback = MalwareScoreCallback(window_size=20, update_freq=1024)
        
        model.learn(total_timesteps=int(ts), callback=monitor_callback, progress_bar=True)
        
        print("\n\n=== Training Complete ===")

        res = env.unwrapped.results()
        mean_score = res[0]
        episodes = res[3]
        
        print(f" Mean malware score: {mean_score:.4f}")
        print(f" Total episodes: {episodes}")
        
        # Save the trained policy
        model_path = f'agents/rew{rew}{pomdp_suffix}_{seed}.pt'
        model.save(model_path)
        print(f"\nModel saved to {model_path}")
        
        env.close()
        del model


def test_agent(rew, steps, n_runs, offset=200, n_agents=5, base_seed=42):
    """Test N agents and aggregate results over M episodes ."""
    
    # Storage for mean and std metrics across agents
    mdp_mean_returns = []  # Mean return for each agent
    mdp_mean_scores = []  # Mean score for each agent
    mdp_std_returns = []  # Std return for each agent
    mdp_std_scores = []  # Std score for each agent
    pomdp_mean_returns = []
    pomdp_mean_scores = []
    pomdp_std_returns = []
    pomdp_std_scores = []
    random_mean_returns = []
    random_mean_scores = []
    random_std_returns = []
    random_std_scores = []
    
    # Test MDP agents
    print(f"Testing {n_agents} MDP agents over {n_runs} episodes each")
    
    for agent_idx in range(n_agents):
        seed = base_seed + agent_idx
        
        env_mdp = gym.make("RLattacker-v0", steps=steps, seed=seed, offset=offset, rew=rew, pomdp=False)
        model_mdp = PPO.load(f'agents/rew{rew}_{seed}.pt', env=env_mdp)
        
        agent_returns = []
        agent_scores = []
        
        for episode in range(n_runs):
            obs, info = env_mdp.reset()
            done = False
            episode_return = 0.0
            
            while not done:
                action, _ = model_mdp.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env_mdp.step(action)
                done = terminated or truncated
                episode_return += reward
            
            agent_returns.append(episode_return)
            agent_scores.append(info['mal_prob'])
        
        mean_return = np.mean(agent_returns)
        mean_score = np.mean(agent_scores)
        std_return = np.std(agent_returns)
        std_score = np.std(agent_scores)
        mdp_mean_returns.append(mean_return)
        mdp_mean_scores.append(mean_score)
        mdp_std_returns.append(std_return)
        mdp_std_scores.append(std_score)
        
        print(f"Agent {agent_idx + 1} - Mean return: {mean_return:.4f}, Mean malware score: {mean_score:.4f}")
        env_mdp.close()
    
    # Test POMDP agents
    print(f"Testing {n_agents} POMDP agents over {n_runs} episodes each")
    
    for agent_idx in range(n_agents):
        seed = base_seed + agent_idx
        
        env_pomdp = gym.make("RLattacker-v0", steps=steps, seed=seed, offset=offset, rew=rew, pomdp=True)
        model_pomdp = PPO.load(f'agents/rew{rew}_pomdp_{seed}.pt', env=env_pomdp)
        
        agent_returns = []
        agent_scores = []
        
        for episode in range(n_runs):
            obs, info = env_pomdp.reset()
            done = False
            episode_return = 0.0
            
            while not done:
                action, _ = model_pomdp.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env_pomdp.step(action)
                done = terminated or truncated
                episode_return += reward
            
            agent_returns.append(episode_return)
            agent_scores.append(info['mal_prob'])
        
        mean_return = np.mean(agent_returns)
        mean_score = np.mean(agent_scores)
        std_return = np.std(agent_returns)
        std_score = np.std(agent_scores)
        pomdp_mean_returns.append(mean_return)
        pomdp_mean_scores.append(mean_score)
        pomdp_std_returns.append(std_return)
        pomdp_std_scores.append(std_score)
        
        print(f"Agent {agent_idx + 1} - Mean return: {mean_return:.4f}, Mean malware score: {mean_score:.4f}")
        env_pomdp.close()
    
    # Test random policy (run N times for consistency)
    print(f"Testing {n_agents} random policies over {n_runs} episodes each")
    
    for agent_idx in range(n_agents):
        seed = base_seed + agent_idx
        
        env_random = gym.make("RLattacker-v0", steps=steps, seed=seed, offset=offset, rew=rew)
        
        agent_returns = []
        agent_scores = []
        
        for episode in range(n_runs):
            obs, info = env_random.reset()
            done = False
            episode_return = 0.0
            
            while not done:
                action = env_random.action_space.sample()
                obs, reward, terminated, truncated, info = env_random.step(action)
                done = terminated or truncated
                episode_return += reward
            
            agent_returns.append(episode_return)
            agent_scores.append(info['mal_prob'])
        
        mean_return = np.mean(agent_returns)
        mean_score = np.mean(agent_scores)
        std_return = np.std(agent_returns)
        std_score = np.std(agent_scores)
        random_mean_returns.append(mean_return)
        random_mean_scores.append(mean_score)
        random_std_returns.append(std_return)
        random_std_scores.append(std_score)
        
        print(f"Run {agent_idx + 1} - Mean return: {mean_return:.4f}, Mean malware score: {mean_score:.4f}")
        env_random.close()
    
    # Print summary statistics
    print("=== Aggregated Test Results (Mean & Std across agents) ===")
    print(f"\nMDP Agent:")
    print(f"  Mean return: {np.mean(mdp_mean_returns):.4f} +- {np.std(mdp_mean_returns):.4f}")
    print(f"  Mean malware score: {np.mean(mdp_mean_scores):.4f} +- {np.std(mdp_mean_scores):.4f}")
    
    print(f"\nPOMDP Agent:")
    print(f"  Mean return: {np.mean(pomdp_mean_returns):.4f} +- {np.std(pomdp_mean_returns):.4f}")
    print(f"  Mean malware score: {np.mean(pomdp_mean_scores):.4f} +- {np.std(pomdp_mean_scores):.4f}")
    
    print(f"\nRandom Policy:")
    print(f"  Mean return: {np.mean(random_mean_returns):.4f} +- {np.std(random_mean_returns):.4f}")
    print(f"  Mean malware score: {np.mean(random_mean_scores):.4f} +- {np.std(random_mean_scores):.4f}")
    
    # Create boxplots of mean values across agents
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Mean return distribution across agents
    ax1 = axes[0]
    bp1 = ax1.boxplot([mdp_mean_returns, pomdp_mean_returns, random_mean_returns], 
                       labels=['MDP', 'POMDP', 'Random'],
                       patch_artist=True)
    for patch, color in zip(bp1['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
        patch.set_facecolor(color)
    ax1.set_ylabel('Mean Episodic Return (↑)', fontsize=18)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Mean malware score distribution across agents
    ax2 = axes[1]
    bp2 = ax2.boxplot([mdp_mean_scores, pomdp_mean_scores, random_mean_scores],
                       labels=['MDP', 'POMDP', 'Random'],
                       patch_artist=True)
    for patch, color in zip(bp2['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
        patch.set_facecolor(color)
    ax2.set_ylabel('Mean Malware Score (↓)', fontsize=18)
    ax2.tick_params(axis='both', labelsize=18)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    plot_path = f'results/test_multi_rew{rew}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nAggregated test comparison plot saved to {plot_path}")
    plt.close()
    
    # Save aggregated results to CSV
    results_df = pd.DataFrame({
        'agent_id': np.arange(1, n_agents + 1),
        'mdp_mean_return': mdp_mean_returns,
        'mdp_std_return': mdp_std_returns,
        'pomdp_mean_return': pomdp_mean_returns,
        'pomdp_std_return': pomdp_std_returns,
        'random_mean_return': random_mean_returns,
        'random_std_return': random_std_returns,
        'mdp_mean_malware_score': mdp_mean_scores,
        'mdp_std_malware_score': mdp_std_scores,
        'pomdp_mean_malware_score': pomdp_mean_scores,
        'pomdp_std_malware_score': pomdp_std_scores,
        'random_mean_malware_score': random_mean_scores,
        'random_std_malware_score': random_std_scores
    })
    csv_path = f'results/test_multi_rew{rew}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Aggregated test results saved to {csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RL Attacker Training and Testing")
    parser.add_argument('--reward', default=1, type=int, help="Reward mode")
    parser.add_argument('--test', action='store_true', help="Test the agent instead of training")
    parser.add_argument('--n-agents', default=5, type=int, help="Number of agents")
    parser.add_argument('--n-runs', default=20, type=int, help="Number of test episodes per agent")
    parser.add_argument('--base-seed', default=42, type=int, help="Base random seed (agents use base_seed + i)")
    parser.add_argument('--offset', default=200, type=int, help="Starting report index for testing")
    parser.add_argument('--steps', default=1000, type=int, help="Max steps per episode")
    parser.add_argument('--lr', default=0.001, type=float, help="Learning rate")
    parser.add_argument('--arch', default=128, type=int, help="Policy network size")
    parser.add_argument('--timesteps', default=1e5, type=float, help="Total training timesteps")
    parser.add_argument('--pomdp', action='store_true', help="Use POMDP mode (scalar observation)")
    
    args = parser.parse_args()
    
    if args.test:
        test_agent(args.reward, args.steps, args.n_runs, args.offset, args.n_agents, args.base_seed)
    else:
        train_agent(args.reward, args.steps, args.pomdp, args.lr, args.arch, args.timesteps, args.n_agents, args.base_seed)