# SQiRL MDP
***

**State Space**: A state space vector of 2049 features, 1024 features from the Gated Recurrent Unit (GRU) representation 
of the payload, 1024 from a GRU representation of the SQL statement executed, and a single value that represents an error 
occurring from the injected payload. 

***

**Action Space**: The action space of SQiRL has a base set of 27 tokens to add to the payload. However, it is dynamic as 
these can be added or removed at any location in the payload. Token actions can be split into three categories: 1) Basic Tokens 
(e.g., commas, comments, quote marks); 2) Behavior Changing tokens which include SQL keywords (e.g., OR, AND, IF); 3) 
Sanitization Escape including obfuscation techniques such as capitalization, whitespace and SQL keyword encoding.

***

**Reward Function**: SQiRL uses two rewards, an internal reward based on Random Network Distillation (RND), reward for 
finding new states. It also uses an external reward that penalizes -1 at each timestep a vulnerability is found, and 0 
when it is found. 

***

**Transition Function**: Deterministic $T: \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{S}$. Termination: on 
vulnerability found or $t > 30$.

***

**Model Architecture**: Following from Al Wahaibi et al. we use their pretrained embedding models for SQLi payloads and 
SQL statements, and a DQN architecture of 3 layers, 2048, 1024, 512 nodes per layer.

***

**Hyperparameters**: $\alpha = 5\times 10^{-3}$, $\gamma = 0.99$, rollout $N=1024$, batch $B=512$, episodes per vulnerability $200$.

***