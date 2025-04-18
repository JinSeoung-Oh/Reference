### From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

"""
1. Purpose
   -a. DQN replaces the Q‑table with a neural network 𝑄𝜃(𝑠,𝑎) to handle large or continuous state spaces.

2. Key Stabilizers
   -a. Replay Buffer: stores (𝑠,𝑎,𝑟,𝑠′,done) tuples; random mini‑batch sampling breaks correlation.
   -b. Target Network 𝑄𝜃− : a slow‑moving copy of the main Q‑network that supplies stable TD targets.

3. Training Loop
   -a. Agent gathers experience with an ε‑greedy policy using the main network.
   -b. Samples a batch from replay memory.
   -c. TD target: 
       𝑦 = 𝑟+𝛾 max_𝑎′ 𝑄𝜃−(𝑠′,𝑎′)
   -d. Update main network by minimizing MSE/Huber loss between 𝑄𝜃(𝑠,𝑎) and 𝑦
   -e. Periodically copy weights: 𝜃^−←𝜃

4. Empirical Behaviour
   -a. Rewards climb from negative to positive by ≈ 200–250 episodes; episode lengths shrink.
   -b. Epsilon decays, letting the agent exploit learned Q‑values; final grid policy directs toward the goal.
"""

# DQN Network (MLP)
class DQN(nn.Module):
    def __init__(self, n_observations: int, n_actions: int):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions) # Outputs Q-value for each action

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass to get Q-values. """
        # Ensure input is float tensor on correct device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=x.device)
        elif x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)  # Raw Q-values

# Structure for storing transitions
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

# Replay Memory Buffer
class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args: Any) -> None:
        """ Saves a transition tuple (s, a, s', r, done). """
        # Ensure tensors are stored on CPU to avoid GPU memory issues with buffer
        processed_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                processed_args.append(arg.cpu())
            elif isinstance(arg, bool):  # Store done flag as tensor for consistency
                processed_args.append(torch.tensor([arg], dtype=torch.bool))
            else:
                processed_args.append(arg)
        self.memory.append(Transition(*processed_args))

    def sample(self, batch_size: int) -> Optional[List[Transition]]:
        """ Samples a random batch of transitions. """
        if len(self.memory) < batch_size:
            return None
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

# Action Selection (Epsilon-Greedy using DQN)
def select_action_dqn(state: torch.Tensor,
                        policy_net: nn.Module,
                        epsilon: float,
                        n_actions: int,
                        device: torch.device) -> torch.Tensor:
    """ Selects action epsilon-greedily using the policy Q-network. """
    if random.random() < epsilon:
        # Explore: choose a random action
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    else:
        # Exploit: choose the best action based on Q-network
        with torch.no_grad():
            # Add batch dim if needed, ensure tensor is on correct device
            state = state.unsqueeze(0) if state.dim() == 1 else state
            state = state.to(device)
            # Get Q-values and select action with max Q
            action = policy_net(state).max(1)[1].view(1, 1)
    return action

# DQN Optimization Step Outline
def optimize_model_dqn(memory: ReplayMemory,
                         policy_net: DQN,
                         target_net: DQN,
                         optimizer: optim.Optimizer,
                         batch_size: int,
                         gamma: float,
                         device: torch.device):
    """ Performs one step of optimization on the DQN policy network. """
    # 1. Sample batch from memory
    # 2. Prepare batch tensors (states, actions, rewards, next_states, dones) on 'device'
    # 3. Compute Q(s_t, a_t) using policy_net for the actions taken
           state_action_values = policy_net(state_batch).gather(1, action_batch)
    # 4. Compute V(s_{t+1}) = max_{a'} Q(s_{t+1}, a'; θ⁻) using target_net
           with torch.no_grad(): next_state_values = target_net(non_final_next_states).max(1)[0]
    # 5. Compute TD Target y = reward + gamma * V(s_{t+1}) (handle terminal states)
              expected_state_action_values = (next_state_values * gamma) + reward_batch
    # 6. Compute loss (e.g., Huber loss) between Q(s_t, a_t) and TD Target y
              loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # 7. Optimize the policy_net
              optimizer.zero_grad()
              loss.backward()
              torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
              optimizer.step()
