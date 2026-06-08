# AutoRobust MDP

***

**State Space**: $\mathcal{S} \subset \mathbb{R}^{768}$: [CLS] embeddings of a fine-tuned DistilBERT. For reports $>512$ 
tokens, mean-pooled chunk embeddings with 20-token overlap.

***

**Action Space**: Multi-discrete $\mathcal{A} = \{0,1\} \times \{0,\ldots,12\} \times \{0,1,2,3\}$: 
$(a_{\text{op}}, a_{\text{key}}, a_{\text{gen}})$ where $a_{\text{op}}$ selects between adding a goodware entry or 
editing an existing malware entry, $a_{\text{key}}$ selects the dynamic analysis report category to operate on, and 
$a_{\text{gen}}$ specifies the entry replacement strategy (dictionary word, token from goodware corpus, random alphanumeric, 
or random choice between the previous 3).

***

**Reward Functions**: $r_t^{(1)} = p_{t-1} - p_t$, and $r_t^{(2)} = p_{t-1} - p_t + \frac{100}{t}$ if $p_t < 0.5$, 
otherwise $r_t^{(2)} = p_{t-1} - p_t$, where $p_t$ is probability of malware at step $t$.

***

**Transition Function**: Deterministic $T: \mathcal{S} \times \mathcal{A} \rightarrow \mathcal{S}$. 
Given a modified report $R_t \rightarrow R_{t+1}$, the next state is $s_{t+1} = \phi(R_{t+1})$ with 
$\phi: \mathcal{R} \rightarrow \mathbb{R}^{768}$ the [CLS] embedding. Termination: $p_t < 0.5$ or $t \geq 1000$.

***

**Model Architecture**: PPO with MLP policy: $[768 \rightarrow 128 \rightarrow 128 \rightarrow |\mathcal{A}|]$, ReLU activations.

***

**Hyperparameters**: $\alpha = 3 \times 10^{-3}$, $\gamma = 0.99$, rollout $N=1024$, batch $B=32$, timesteps $5 \times 10^4$.

***