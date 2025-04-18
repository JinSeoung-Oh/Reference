### From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

""""
1. Asynchronous Parallelism
   -a. A3C runs many independent workers in parallel, each with its own environment copy.
   -b. A worker periodically copies parameters from a shared Global Network, interacts for n steps, computes gradients locally, 
       and pushes those gradients back without waiting for other workers.
   -c. This asynchronicity decorrelates data, supplies a continuous stream of diverse updates, and removes the need for experience‑replay buffers.

2. N‑Step Returns & Advantage
   -a. Each worker forms an n‑step return
       𝑅_𝑡 = 𝑟_𝑡 +𝛾𝑟_(𝑡+1)+⋯+𝛾^(𝑛−1)𝑟_(𝑡+𝑛−1)+𝛾^𝑛𝑉(𝑠_𝑡+𝑛)
   -b. Advantage: 𝐴^_𝑡=𝑅_𝑡−𝑉(𝑠_𝑡), giving lower‑variance policy updates.

3. Shared Actor–Critic Network
   -a. A single neural net with shared layers outputs action logits (actor head) and state value (critic head).

4. Worker Update Mechanics
   -a. For its rollout, a worker collects: log‑probs, values, entropies, rewards, dones.
   -b. It computes policy loss, value loss, and entropy bonus from these n‑step sequences.
   -c. Gradients are taken on the local copy, then asynchronously applied to the global network via a shared optimizer.

5. Training Behaviour
   -a. Rewards rise from negative to high positive; episode lengths fall sharply, stabilizing around episode 250‑300, 
       indicating convergence to an efficient policy.
""""

# Shared Actor-Critic Network (Same structure as A2C)
class ActorCriticNetwork(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super(ActorCriticNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.actor_head = nn.Linear(128, n_actions) # Action logits
        self.critic_head = nn.Linear(128, 1)      # State value

    def forward(self, x: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        if not isinstance(x, torch.Tensor):
             x = torch.tensor(x, dtype=torch.float32, device=x.device)
        elif x.dtype != torch.float32:
             x = x.to(dtype=torch.float32)
        if x.dim() == 1: x = x.unsqueeze(0)

        x = F.relu(self.layer1(x))
        shared_features = F.relu(self.layer2(x))
        action_logits = self.actor_head(shared_features)
        state_value = self.critic_head(shared_features)
        # Ensure value is squeezed if batch dim was added
        if x.shape[0] == 1 and state_value.dim() > 0: 
            state_value = state_value.squeeze(0)
        
        return Categorical(logits=action_logits.to(x.device)), state_value

# Calculate N-Step Returns and Advantages (used within each worker)
def compute_n_step_returns_advantages(rewards: List[float],
                                      values: List[torch.Tensor], # V(s_t) predictions from network
                                      bootstrap_value: torch.Tensor, # V(s_{t+n}) prediction, detached
                                      dones: List[float], # Done flags (0.0 or 1.0)
                                      gamma: float
                                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Computes n-step returns (critic targets) and advantages (actor guidance). """
    n_steps = len(rewards)
    returns = torch.zeros(n_steps, dtype=torch.float32) # Store results on CPU
    advantages = torch.zeros(n_steps, dtype=torch.float32)
    
    # Detach values used for advantage calculation (as they form part of the baseline)
    values_detached = torch.cat([v.detach() for v in values]).squeeze().cpu()
    R = bootstrap_value.detach().cpu() # Start with bootstrapped value

    for t in reversed(range(n_steps)):
        R = rewards[t] + gamma * R * (1.0 - dones[t]) # Calculate n-step return
        returns[t] = R
        
        # Ensure values_detached has correct shape for advantage calculation
        value_t = values_detached if values_detached.dim() == 0 else values_detached[t]
        advantages[t] = R - value_t # Advantage = N-step Return - V(s_t)

    # Optional: standardize advantages
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages

# --- Worker Loss Calculation (Conceptual - performed by each worker) ---
# Assuming 'log_probs_tensor', 'values_pred_tensor', 'entropies_tensor' contain
# the outputs from the network for the n-step rollout,
# and 'returns_tensor', 'advantages_tensor' contain the calculated targets.

policy_loss = -(log_probs_tensor * advantages_tensor.detach()).mean()
value_loss = F.mse_loss(values_pred_tensor, returns_tensor.detach())
entropy_loss = -entropies_tensor.mean()

total_loss = policy_loss + value_loss_coeff * value_loss + entropy_coeff * entropy_loss

# --- Gradient Application (Conceptual) ---
global_optimizer.zero_grad()
total_loss.backward() # Calculates grads on local model
# Transfer gradients from local_model.parameters() to global_model.parameters()
global_optimizer.step() # Applies gradients to global model
