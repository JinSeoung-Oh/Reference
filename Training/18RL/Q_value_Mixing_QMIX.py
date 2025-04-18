### From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

"""
1. Purpose
   -a. QMIX targets cooperative multi‑agent tasks with a shared team reward.
   -b. Centralized training + decentralized execution: each agent chooses actions from its own 𝑄_𝑖 but training uses a mixing network 
       that sees the global state.

2. Monotonicity Constraint
   -a. The mixing network is designed so that increasing any local 𝑄_𝑖 cannot decrease the joint value 𝑄_tot
   -b. This guarantees that agents acting greedily w.r.t. their own 𝑄_𝑖 also maximize 𝑄_tot

3. Training Procedure
   -a. Target computation:
       -1. Use target agent networks 𝑄_𝑖′ to pick best next‑step values.
       -2. Feed these into the target mixer with next global state 𝑥′ to get 𝑄′_tot
       -3. TD target 𝑦=𝑟+𝛾𝑄′_tot
   -b. Current estimate: feed executed actions’ 𝑄_𝑖 into the main mixer with current state 𝑥x to get 𝑄_tot
   -c. Loss: MSE between 𝑦 and 𝑄_tot; one optimizer updates all agent networks and the mixer.
   -d. Soft target updates maintain stability.

4. Components
   -a. Agent networks 𝑄_𝑖(𝑜_𝑖) (local observation → action values).
   -b. QMIX mixer: combines {𝑄_𝑖} using hypernetworks that generate mixing weights/biases from the global state.

5. Empirical Outcome
   -a. Shared reward rises markedly (though still noisy).
   -b. TD loss drops by ~100× in 150 episodes and stabilizes.
   -c. Epsilon decays slowly to preserve exploration.
"""

# Agent Network (similar to DQN's Q-network)
class AgentQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super(AgentQNetwork, self).__init__()
        # Could be MLP or RNN (DRQN) depending on observability needs
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim) # Q-values for each action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

# Simplified QMIX Mixing Network
class QMixer(nn.Module):
    def __init__(self, num_agents: int, global_state_dim: int, mixing_embed_dim: int = 32):
        super(QMixer, self).__init__()
        self.num_agents = num_agents
        self.state_dim = global_state_dim
        self.embed_dim = mixing_embed_dim

        # Hypernetwork for W1 (generates positive weights)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, 64), nn.ReLU(),
            nn.Linear(64, self.num_agents * self.embed_dim)
        )
        # Hypernetwork for b1
        self.hyper_b1 = nn.Linear(self.state_dim, self.embed_dim)

        # Hypernetwork for W2 (generates positive weights)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_dim, 64), nn.ReLU(),
            nn.Linear(64, self.embed_dim) # Output size embed_dim -> reshape to (embed_dim, 1)
        )
        # Hypernetwork for b2 (scalar bias)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, agent_qs: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        # agent_qs shape: (batch_size, num_agents)
        # global_state shape: (batch_size, global_state_dim)
        batch_size = agent_qs.size(0)
        agent_qs_reshaped = agent_qs.view(batch_size, 1, self.num_agents)

        # Generate weights/biases from global state
        w1 = torch.abs(self.hyper_w1(global_state)).view(batch_size, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(global_state).view(batch_size, 1, self.embed_dim)

        w2 = torch.abs(self.hyper_w2(global_state)).view(batch_size, self.embed_dim, 1)
        b2 = self.hyper_b2(global_state).view(batch_size, 1, 1)

        # Mixing layers (ensure non-linearity like ELU/ReLU after first layer)
        hidden = F.elu(torch.bmm(agent_qs_reshaped, w1) + b1) # (batch, 1, embed_dim)
        q_tot = torch.bmm(hidden, w2) + b2 # (batch, 1, 1)

        return q_tot.view(batch_size, 1) # Return shape (batch, 1)

