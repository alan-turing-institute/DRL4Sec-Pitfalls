# Link MDP

***

**State Space**: The state space of the Link environment consists of a vector of 47 integers that represent:
- The payload input, including information such as the payload appearance and the payload repetitiveness.
- The HTML page response, which encodes: the reflected payload appearance, and payload context information.

***

**Action Space**: Each of the 39 actions of the Link environment can be placed into one of 8 categories:
- _Basic Payload_ (4 actions): Designed to initialize a payload with a simple payload e.g., `<script>alert(1);</script>`.
- _JS Component_ (3 actions): generate a JavaScript snippet e.g., `alert(1);`.
- _Prefix_ (13 actions): Add symbols to the front of the payload e.g., `<, '`.
- _Suffix_ (3 actions): Add symbols to the end of the payload e.g., `<!--`.
- _Tag_ (3 actions): Actions to manipulate the tag: e.g., capitalize the tag.
- _Attribute_ (1 action): Capitalize the payload attribute.
- _JS Snippet_ (5 actions): Manipulate the JavaScript snippet e.g., converting `alert(1)` to  `throw(1)`.
- _Entire String_ (6 actions): Manipulate the entire payload e.g., obfuscate by octet encoding.

***

**Reward Function**: The reward function of this environment first reward if a vulnerability is found, 
reward for the difference between the maximum and current number of steps. 
However, a penalty of -1 is applied: at each timestep or when the payload doesn't change between steps.
Finally, the frequency of the current payload divided by the maximum training steps is subtracted at each step.

***

**Transition Function**: At each timestep the agent selects an action to perform, as described above. 
These are applied to the payload in a rule based manor: actions are added to the payload or substituted when an action artifact is already in the payload, 
e.g., replacing `img` HTML tag with `video` HTML tag. The action for media tags (img, video, audio, svg),  event attributes (onerror, onload, onclick, onmouseover), 
and JS snippets (alert(1), confirm(1), prompt(1)) insert at random from a predefined set. 
Termination of the episode occurs when a vulnerability has been found or $t >= 500$. 

***

**Model Architecture**: Following from Lee et al. we use an actor critic architecture from StableBaselines3 of three layers of 128 neurons. 

***

**Hyperparameters**: $\alpha = 5 \times 10^{-4}$, $\gamma = 0.95$, timesteps $3.5 \times 10^6$.

***