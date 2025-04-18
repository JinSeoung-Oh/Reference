### From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

"""
1. Core Idea
   -a. PlaNet (Deep Planning Network) learns a world model in a compact latent space, not in raw observation space, 
       and performs planning inside that latent space.

2. Operational Loop
   -a. Encode the current observation 𝑜_𝑡 to a latent state 𝑠_𝑡
   -b. Plan with the Cross‑Entropy Method (CEM) in latent space to find the best action sequence; execute only the first action 
       𝑎_𝑡\* in the real environment.
   -c. Store (𝑜_𝑡,𝑎_𝑡\*,𝑟_𝑡,𝑜_(𝑡+1),done) in a replay buffer.
   -d. Train the world model off‑line on sampled sequences to predict next latent state and reward, improving future planning.

3. World Model (simplified for vector states)
   -a. A feed‑forward DynamicsModel jointly predicts next state vector and immediate reward from current state + action.

4. CEM Planner (outline)
   -a. Maintains a Gaussian distribution over action sequences, iteratively:
       -1. Sample many candidate sequences.
       -2. Roll out each through the learned model, accumulate discounted rewards.
       -3. Select elites with highest returns.
       -4. Refit mean and std to elites; after several iterations, output the first action of the final mean sequence.

5. Model Training (outline)
   -a. Sample sequences → predict next state & reward → minimize MSE losses → back‑propagate and update parameters.

6. Observed Behaviour (Pendulum example)
   -a. Model loss drops by ~100× in ≈ 25 iterations; episode rewards rise from ≈ –800 to ≈ –100 … –200, though with variability—showing 
       that a good world model quickly enables productive planning.
"""

# Simplified Dynamics Model (Predicts next state vector and reward)
class DynamicsModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 200):
        super(DynamicsModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_next_state = nn.Linear(hidden_dim, state_dim)
        self.fc_reward = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pred_next_state = self.fc_next_state(x)
        pred_reward = self.fc_reward(x)
        return pred_next_state, pred_reward

# CEM Planner Outline
def cem_planner(model: DynamicsModel, initial_state: torch.Tensor, horizon: int,
                num_candidates: int, num_elites: int, num_iterations: int,
                gamma: float, action_low, action_high, action_dim, device) -> torch.Tensor:

    # Initialize action distribution (e.g., Gaussian mean=0, std=high)
    action_mean = torch.zeros(horizon, action_dim, device=device)
    action_std = torch.ones(horizon, action_dim, device=device)  # Start with high variance

    for _ in range(num_iterations):
        # 1. Sample candidate action sequences (batch, horizon, action_dim)
        action_dist = Normal(action_mean, action_std)
        candidate_actions = action_dist.sample((num_candidates,))
        candidate_actions = torch.clamp(candidate_actions,
                                        torch.tensor(action_low, device=device),
                                        torch.tensor(action_high, device=device))

        # 2. Evaluate sequences using the model
        total_rewards = torch.zeros(num_candidates, device=device)
        current_states = initial_state.repeat(num_candidates, 1)
        with torch.no_grad():
            for t in range(horizon):
                actions_t = candidate_actions[:, t, :]
                next_states, rewards = model(current_states, actions_t)
                total_rewards += (gamma ** t) * rewards.squeeze(-1)
                current_states = next_states

        # 3. Select elite action sequences
        _, elite_indices = torch.topk(total_rewards, num_elites)
        elite_actions = candidate_actions[elite_indices]

        # 4. Refit the action distribution
        action_mean = elite_actions.mean(dim=0)
        action_std = elite_actions.std(dim=0) + 1e-6  # Add epsilon for stability

    # Return the first action of the final mean sequence
    return action_mean[0]

# --- Model Training Outline ---
# (Assumes model, optimizer, sequence_buffer are defined)

for _ in range(num_train_steps):
    1. Sample batch of sequences from sequence_buffer
    2. For each step in the sequence (or a subset):
       - Get state_t, action_t, reward_t, next_state_t from batch
       - Predict next_state_pred, reward_pred = model(state_t, action_t)
       - Calculate loss = MSE(next_state_pred, next_state_t) + MSE(reward_pred, reward_t)
    3. Average loss over batch/steps
    4. optimizer.zero_grad()
    5. loss.backward()
    6. optimizer.step()



