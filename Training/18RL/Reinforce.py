### From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

"""
1. Shift to Policy‑Based Learning
   -a. REINFORCE directly parameterizes a policy 𝜋(𝑎∣𝑠;𝜃) and adjusts 𝜃 so that actions leading to higher total return become more probable.

2. Monte‑Carlo Episode Processing
   -a. The agent samples an entire episode using its current policy.
   -b. For each time step 𝑡, it stores log 𝜋𝜃(𝑎_𝑡∣𝑠_𝑡) and the reward 𝑟_𝑡
   -c. After the episode ends, it computes the discounted return 𝐺_𝑡 for every step.

3. Policy Gradient Update
   -a. The update rule increases log 𝜋𝜃 for actions followed by high 𝐺_𝑡 and decreases it for actions followed by low 
       𝐺_𝑡
   -b. Gradient ascent is applied via an optimizer to maximize expected return.

4. Components Mentioned
   -a. Policy network: three fully connected layers (128‑unit hidden layers) producing action logits, 
                       interpreted with a Categorical distribution.
   -b. Action selection: sampling from the distribution inherently provides exploration (no ε‑greedy).
   -c. Return computation: discounted backward sum; optional standardization stabilizes training.
   -d. Loss: −∑_𝑡 𝐺_𝑡 log𝜋𝜃(𝑎_𝑡∣𝑠_𝑡); the optimizer updates 𝜃

5. Behavior Observed
   -a. Episode lengths drop sharply then stabilize at a low value.
   -b. Total rewards move from negative to high positive.
   -c. Loss shows large fluctuations, illustrating high variance typical of basic REINFORCE.
"""

# Define the Policy Network architecture
class PolicyNetwork(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions) # Outputs logits

    def forward(self, x: torch.Tensor) -> Categorical:
        """ Forward pass, returns a Categorical distribution over actions. """
        if not isinstance(x, torch.Tensor):
             x = torch.tensor(x, dtype=torch.float32, device=device) # Assuming 'device' is defined
        elif x.dtype != torch.float32:
             x = x.to(dtype=torch.float32)
        if x.dim() == 1: x = x.unsqueeze(0)

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        action_logits = self.layer3(x)
        # Use Categorical distribution directly from logits
        return Categorical(logits=action_logits)

# Action Selection by Sampling
def select_action_reinforce(state: torch.Tensor, policy_net: PolicyNetwork) -> Tuple[int, torch.Tensor]:
    """ Selects action by sampling from the policy network's output distribution. """
    action_dist = policy_net(state)
    action = action_dist.sample()      # Sample action
    log_prob = action_dist.log_prob(action) # Get log probability of the chosen action
    return action.item(), log_prob

# Calculate Discounted Returns
def calculate_discounted_returns(rewards: List[float], gamma: float, standardize: bool = True) -> torch.Tensor:
    """ Calculates discounted returns G_t for each step t, optionally standardizes. """
    n_steps = len(rewards)
    returns = torch.zeros(n_steps, dtype=torch.float32) # Keep on CPU for calculation
    discounted_return = 0.0
    # Iterate backwards
    for t in reversed(range(n_steps)):
        discounted_return = rewards[t] + gamma * discounted_return
        returns[t] = discounted_return

    if standardize:
        mean_return = torch.mean(returns)
        std_return = torch.std(returns) + 1e-8 # Add epsilon for stability
        returns = (returns - mean_return) / std_return

    return returns.to(device) # Move to appropriate device at the end

# Policy Update Step
def optimize_policy(log_probs: List[torch.Tensor],
                      returns: torch.Tensor,
                      optimizer: optim.Optimizer) -> float:
    """ Performs one REINFORCE policy gradient update. """
    # Stack log probabilities and ensure returns have correct shape
    log_probs_tensor = torch.stack(log_probs)
    returns = returns.detach() # Treat returns as fixed targets for this update

    # Calculate loss: - Sum(G_t * log_pi(a_t|s_t))
    # We minimize the negative objective
    loss = -torch.sum(returns * log_probs_tensor)

    # Perform optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

