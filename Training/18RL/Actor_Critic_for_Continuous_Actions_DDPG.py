### From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

"""
1. Purpose
   -a. DDPG adapts actor‑critic methods to continuous action spaces (e.g., torques, velocities).

2. Main Components
   -a. Deterministic Actor outputs a single continuous action 𝑎=𝜇𝜃(𝑠) instead of a probability distribution.
   -b. Q‑Critic learns 𝑄_𝜙(𝑠,𝑎), evaluating how good the actor’s action is.
   -c. Exploration is added manually by injecting noise (Gaussian or Ornstein‑Uhlenbeck) into the actor’s action during training.

3. Off‑Policy Mechanisms (from DQN)
   -a. Replay Buffer stores (𝑠,𝑎,𝑟,𝑠′,done)) transitions and supplies mini‑batches for updates.
   -b. Target Networks (for both actor and critic) are soft‑updated toward the main networks to stabilize TD targets.

4. Training Loop
   -a. Critic update:
       𝑦=𝑟+𝛾(1−done)𝑄_𝜙′(𝑠′,𝜇𝜃′(𝑠′)), loss=(𝑄_𝜙(𝑠,𝑎)−𝑦)^2
       and apply a gradient step on 𝜙
   -b. Actor update: minimize −𝑄_𝜙(𝑠,𝜇𝜃(𝑠)) with respect to 𝜃
   -c. Soft update targets: 𝜙′ ← 𝜏𝜙+(1−𝜏)𝜙′, same for 𝜃′

5. Networks
   -a. Actor: 256‑unit hidden layers, tanh output scaled by max_action.
   -b. Critic: concatenates state and action, outputs a scalar Q‑value.

6. Empirical Behaviour (Pendulum example)
   -a. Episode rewards rise from ≈ –1500 to ≈ –250 by episode 100.
   -b. Actor’s average Q‑value climbs; critic loss grows because target Q‑values increase as the policy improves.
"""

# Simplified Actor Network for DDPG (outputs continuous action)
class ActorNetworkDDPG(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(ActorNetworkDDPG, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)
        self.max_action = max_action # To scale the output

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        # Use tanh to output values between -1 and 1, then scale
        action = self.max_action * torch.tanh(self.layer3(x))
        return action

# Simplified Critic Network for DDPG (takes state and action)
class CriticNetworkDDPG(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(CriticNetworkDDPG, self).__init__()
        # Process state and action separately or together
        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1) # Outputs a single Q-value

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        q_value = self.layer3(x)
        return q_value

# Replay Buffer (Concept - use deque or list)
# Target Networks (Concept - create copies of actor/critic)
target_actor = ActorNetworkDDPG(...)
target_critic = CriticNetworkDDPG(...)
target_actor.load_state_dict(actor.state_dict()) # Initialize
target_critic.load_state_dict(critic.state_dict())

# --- DDPG Update Logic Outline ---
# (Assumes optimizers: actor_optimizer, critic_optimizer are defined)
# (Assumes tau: soft update rate, gamma: discount factor are defined)

# 1. Sample a batch from replay_buffer: states, actions, rewards, next_states, dones

# --- Critic Update ---
# Get next actions from target actor: next_actions = target_actor(next_states)
# Get target Q-value from target critic: target_q = target_critic(next_states, next_actions)
# Calculate TD Target: td_target = rewards + gamma * (1 - dones) * target_q
# Get current Q-value estimate: current_q = critic(states, actions)
# Calculate Critic Loss (MSE): critic_loss = F.mse_loss(current_q, td_target.detach())
# Update Critic:
critic_optimizer.zero_grad()
critic_loss.backward()
critic_optimizer.step()

# --- Actor Update ---
# Get actions for current states from main actor: actor_actions = actor(states)
# Calculate Actor Loss (negative Q-value from main critic): actor_loss = -critic(states, actor_actions).mean()
# Update Actor:
actor_optimizer.zero_grad()
actor_loss.backward()
actor_optimizer.step()

# --- Soft Update Target Networks ---
soft_update(target_critic, critic, tau)
soft_update(target_actor, actor, tau)




