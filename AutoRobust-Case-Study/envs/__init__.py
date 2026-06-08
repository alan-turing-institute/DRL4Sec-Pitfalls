import gymnasium as gym
from gymnasium.envs.registration import register

for env in list(gym.envs.registry.keys()):
     if 'RLattacker-v0' in env:
          del gym.envs.registry[env]

register(
    id='RLattacker-v0',
    entry_point='envs.RLattacker:RLattacker'
)