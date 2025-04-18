### From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

"""
1. Core Principle
   -a. Soft Actor‑Critic (SAC) adds a maximum‑entropy objective to continuous‑action actor‑critic learning:
       maximize reward and the policy’s entropy, encouraging wide exploration and robustness.

2. Key Elements
   -a. Stochastic Actor πθ outputs a distribution; actions are sampled.
   -b. Twin Critics (Q1,Q2) take (state, action) and the minimum value is used to curb over‑estimation.
   -c. Entropy Temperature α weights the entropy term; it can be automatically tuned toward a target entropy.
   -d. Off‑policy learning with a replay buffer; soft target updates on critics; no target actor.
   -e. Training losses use
       𝑦 = 𝑟+𝛾[min(𝑄1′,𝑄2′)−𝛼 log𝜋(𝑎′∣𝑠′)]
       for critic targets, and
       
       actor loss = (𝛼 log 𝜋(𝑎∣𝑠)−min(𝑄1,𝑄2))^(𝑚𝑒𝑎𝑛)
       for policy improvement.

3. Network Architectures
   -a. Actor: two 256‑unit hidden layers → mean & log‑std → tanh‑squashed, scaled by max_action.
   -b. Critic: two separate 256‑unit MLPs (Q1 & Q2) on concatenated (state, action).

4. Observed Behaviour (Pendulum example)
   -a. Rewards rise rapidly from ≈ –1500 to ≈ –200 within 40‑50 episodes and remain stable.
   -b. Critic loss increases then levels; actor loss peaks then falls as policy stabilizes.
   -c. Automatically tuned α varies to keep entropy near the target, confirming the mechanism works.
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON = 1e-6 # For numerical stability in log_prob calculation

class ActorNetworkSAC(nn.Module):
    """ Stochastic Gaussian Actor for SAC. """
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(ActorNetworkSAC, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, action_dim)    # Output mean
        self.log_std_layer = nn.Linear(256, action_dim) # Output log std dev
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Outputs squashed action and its log probability. """
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) # Clamp for stability
        std = torch.exp(log_std)

        # Create distribution and sample using reparameterization trick
        normal_dist = Normal(mean, std)
        z = normal_dist.rsample() # Differentiable sample (pre-squashing)
        action_squashed = torch.tanh(z) # Apply tanh squashing

        # Calculate log_prob with correction for tanh
        # log_prob = log_normal(z) - log(1 - tanh(z)^2 + eps)
        log_prob = normal_dist.log_prob(z) - torch.log(1 - action_squashed.pow(2) + EPSILON)
        # Sum log_prob across action dimensions if action_dim > 1
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Scale action to environment bounds
        action_scaled = action_squashed * self.max_action

        return action_scaled, log_prob


# Critic Network (Twin Q) for SAC
class CriticNetworkSAC(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(CriticNetworkSAC, self).__init__()
        # Q1 architecture
        self.l1_q1 = nn.Linear(state_dim + action_dim, 256)
        self.l2_q1 = nn.Linear(256, 256)
        self.l3_q1 = nn.Linear(256, 1)
        # Q2 architecture
        self.l1_q2 = nn.Linear(state_dim + action_dim, 256)
        self.l2_q2 = nn.Linear(256, 256)
        self.l3_q2 = nn.Linear(256, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1_q1(sa))
        q1 = F.relu(self.l2_q1(q1))
        q1 = self.l3_q1(q1)

        q2 = F.relu(self.l1_q2(sa))
        q2 = F.relu(self.l2_q2(q2))
        q2 = self.l3_q2(q2)
        return q1, q2
