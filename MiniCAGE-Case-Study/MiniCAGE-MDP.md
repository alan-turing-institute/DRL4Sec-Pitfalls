# MiniCAGE MDP

***

**State Space**: The blue agent state space consists of: A 78 value vector containing information about each host. 
The first 52 values relate to the red action and level of compromise, split into groups of 4 values per host. 
The first two values indicate if any relevant red activity has occurred, and the next two indicate the level of red compromise:

**Red activity**

| Values | Meaning |
|--------|---------|
| 0, 0 | None |
| 1, 0 | Scan |
| 1, 1 | Exploit |

**Red access level**

| Values | Meaning |
|--------|---------|
| 0, 0 | No access |
| 1, 0 | Unknown access |
| 0, 1 | User level |
| 1, 1 | Privilege level |

The next part of the observation space contains scan information. This is an additional 13 float values, one for each host. 
The value can be: 2 if a host is being scanned in the current timestep, 1 if the host was scanned in a prior timestep, and otherwise 0.
The final part of the observation space is for tracking the decoys placed. 
This is also represented as an additional vector of 13 float values for each host, each value indicating the number of available decoys per host remaining to be placed.

***

**Action Space**: The blue agent action space consists of 53 possible actions and includes the global action `sleep', as well as host specific actions:
- _Analyze_: Reveals further information about the given host to better allow blue to identify if the red agent is present. I.e., Identifying if privileged access has been achieved.
- _Decoy_: Places a decoy service on the given host. There are a set available per host.
- _Remove_: Attempt to remove red access from a host, given privileged access has not been achieved. This could remove red access after a successful red attempt to exploit network services, but not after a successful red escalate action.
- _Restore_: Restores a host to its initial known good state. Whilst this removes all red access, it has consequences for system availability.

The Red action space consists of 56 possible actions, also including the global `sleep' and the Discover Remote systems action that acts upon each subnet: User, Enterprise and Operational. There are host specific actions:
- _Discover Network Services_: This discovers responsive services on the given host.
- _Exploit Network Services_: Attempts to exploit a specific service on the remote system.
- _Escalate_: Escalates the agent's privilege level on the given host. If this succeeds, the host is fully compromised.
- _Impact_: This action can only be applied to the Operational Server. It disrupts the performance of the network and repeating this action is the final goal for the red agents.

***

**Reward Function**: The reward function consists of a series of penalties based upon the level of red access on hosts, red actions and blue actions.
- -0.1: Per user host compromised 
- -1.0: Per enterprise host compromised
- -1.0: Per operational server compromised
- -0.1: Per operational host compromised
- -10: Each successful red impact action
- -1.0: Each time blue restores a host

***

**Transition Function**: Transition dynamics are implicitly defined by the MiniCAGE network environment. 
At each timestep, Red and Blue actions are selected from the same pre-step state (Red first, then Blue). 
The global state evolves via rule-based host-level dynamics which decide environment aspects like red compromise progression, 
service configuration, and Blue defending actions. Stochasticity arises mainly from partial observability, 
though there is some deliberately induced noise (i.e., there is a 5% chance that a red exploit action is not reflected in the immediate blue observation space).

***

**Model Architecture**: StableBaselines3's PPO was implemented using the default MLP actor-critic architecture. 
This includes a shared feature extractor, policy and value heads each with 2 x 64 fully connected layers and ReLU activation functions.

***

**Hyperparameters**: Aside from when these are explicitly changed in Section 6.4, the hyperparameters used throughout the MiniCAGE experiment can be seen in Table 5.

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | $3 \times 10^{-4}$ |
| Discount factor ($\gamma$) | 0.99 |
| Clip Range ($\epsilon$) | 0.2 |
| Epochs | 6 |
| Batch size | 64 |
| n steps | 2048 |
| Lambda ($\lambda$) | 0.95 |
| Entropy coefficient | 0 |
| Value loss coefficient | 0.5 |

***
