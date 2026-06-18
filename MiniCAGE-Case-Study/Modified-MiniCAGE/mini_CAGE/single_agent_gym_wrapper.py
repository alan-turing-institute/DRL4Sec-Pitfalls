import gymnasium as gym
import numpy as np
from gymnasium import spaces
import sys, os

sys.path.append(os.path.expanduser("~/PycharmProjects/mini_cage_public")) # adjust to your path

from mini_CAGE.minimal import SimplifiedCAGE, HOSTS
from mini_CAGE.test_agent import Meander_minimal
from mini_CAGE.red_bline_agent import B_line_minimal


class Mixed_minimal:
    """
    Mixed red attacker: at every reset (episode boundary) randomly picks
    between B-line and Meander. The chosen sub-agent persists for the full
    episode so within-episode behaviour stays internally consistent.

    All non-overridden attribute access is delegated to the active sub-agent,
    so the wrapper is transparent to callers (e.g. SimplifiedCAGE._process_actions)
    that inspect red_agent state.
    """

    def __init__(self):
        self._bline = B_line_minimal()
        self._meander = Meander_minimal()
        self._current = self._bline

    def reset(self):
        self._current = np.random.choice([self._bline, self._meander])
        # Meander_minimal has no reset(); only call it if present.
        if hasattr(self._current, "reset"):
            self._current.reset()

    def get_action(self, obs):
        return self._current.get_action(obs)

    def __getattr__(self, name):
        return getattr(self._current, name)


def make_red_agent(name: str, sim: SimplifiedCAGE):
    """
    Return an *already-constructed* red agent that exposes
    `.get_action(observation)` and acts on a *single* env.
    """
    if name.lower() in {"bline", "b_line", "b_line_minimal"}:
        return B_line_minimal()
    if name.lower() in {"meander", "meander_minimal"}:
        return Meander_minimal()
    if name.lower() in {"mixed", "mixed_minimal"}:
        return Mixed_minimal()
    raise ValueError(f"Unknown red agent '{name}'")


class MiniCageBlue(gym.Env):
    """
    Gym-style wrapper that exposes ONLY the Blue side.
    Red is driven by a scripted opponent (B-line or Meander).
    """

    metadata = {"render_modes": []}

    def __init__(
            self,
            red_policy: str = "bline",
            remove_bugs: bool = True,
            max_steps: int = 100,
            action_order: str = "R2B",
    ):
        super().__init__()

        self.sim = SimplifiedCAGE(
            num_envs=1, remove_bugs=remove_bugs, action_order=action_order
        )

        # action / observation spaces
        self.role = "Blue"
        self.action_map = self.sim.action_mapping[self.role]
        self.action_space = spaces.Discrete(len(self.action_map))

        obs_dim = 6 * len(HOSTS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.red_agent = make_red_agent(red_policy, self.sim)
        self._red_obs = None  # will be filled in reset()

        self.max_steps = max_steps
        self.steps_done = 0
        self.last_info = {}

    # helpers
    def _blue_obs(self):
        return self.sim.proc_states["Blue"][0].astype(np.float32)

    # Gym
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # reset the Red agent's internal state (Meander_minimal has no reset())
        if hasattr(self.red_agent, "reset"):
            self.red_agent.reset()

        self.steps_done = 0

        obs_dict, info = self.sim.reset()
        self.last_info = info

        # keep both views for next step
        self._red_obs = obs_dict["Red"][0]
        blue_obs = obs_dict["Blue"][0].astype(np.float32)
        return blue_obs, {}

    def step(self, blue_action):
        self.steps_done += 1

        # Red agent acts
        red_action = self.red_agent.get_action(self._red_obs)
        red_action = red_action.astype(np.int32)
        blue_action = np.array([[blue_action]], dtype=np.int32)


        obs_dict, reward_dict, terminated, info = self.sim.step(
            red_action=red_action, blue_action=blue_action, red_agent=self.red_agent
        )

        self._red_obs = obs_dict["Red"][0]
        info["red_action"] = int(red_action[0, 0])
        info["blue_action"] = int(blue_action[0, 0])

        info["blue_success"] = int(self.sim.blue_success[0, 0])
        info["red_success"] = int(self.sim.red_success[0, 0])

        blue_obs = obs_dict["Blue"][0].astype(np.float32)
        reward = float(reward_dict[self.role][0][0])
        done = self.steps_done >= self.max_steps
        truncated = False
        return blue_obs, reward, done, truncated, info
