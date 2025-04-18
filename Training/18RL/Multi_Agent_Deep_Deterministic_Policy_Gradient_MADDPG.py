### From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

"""
1. Motivation
   -a. Multiple agents updating simultaneously make the environment non‑stationary from any individual’s viewpoint.
   -b. MADDPG counters this by using centralized training (critics see all agents’ info) while keeping decentralized execution 
       (each actor uses only its own observation).

2. Architecture
   -a. Decentralized Actors μθi: each agent i maps its local observation 𝑜_𝑖 to a deterministic action 
                                 𝑎_𝑖
   -b. Centralized Critic 𝑄_(𝜙_𝑖) for each agent: inputs the joint observations and joint actions of all agents to estimate agent i’s value.
   -c. Replay Buffer stores joint transitions (𝑜,𝑎,𝑟,𝑜′,𝑑𝑜𝑛𝑒)
   -d. Target networks (actor & critic) are updated softly to stabilize TD targets.

3. Training Steps (per agent i)
   -a. Critic update:
       𝑦_𝑖 =𝑟_𝑖 + 𝛾(1−done_𝑖)𝑄_(𝜙′_𝑖)(𝑜′,𝑎′), loss=(𝑄_(𝜙𝑖)(𝑜,𝑎)−𝑦_𝑖)^2
       where 𝑎′are actions from all target actors.
   -b. Actor update: maximize 𝑄_(𝜙_𝑖) w.r.t. agent i’s action while other agents’ actions are from their current actors.
   -c. Soft update: 𝜃′_𝑖 ← 𝜏𝜃_𝑖+(1−𝜏)𝜃′_𝑖, same for 𝜙_𝑖

4. Observed Behaviour (example)
   -a. Critic and actor losses improve steadily, but shared reward rises slowly with high variance, suggesting coordination remains 
       challenging within 200 episodes.
"""

# Actor Network (similar to DDPG's, one per agent)
class ActorNetworkMADDPG(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, max_action: float):
        super(ActorNetworkMADDPG, self).__init__()
        self.layer1 = nn.Linear(obs_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(obs))
        x = F.relu(self.layer2(x))
        action = self.max_action * torch.tanh(self.layer3(x)) # Scaled continuous action
        return action

# Centralized Critic Network (one per agent)
class CentralizedCriticMADDPG(nn.Module):
    def __init__(self, joint_obs_dim: int, joint_action_dim: int):
        super(CentralizedCriticMADDPG, self).__init__()
        self.layer1 = nn.Linear(joint_obs_dim + joint_action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1) # Outputs single Q-value for this agent

    def forward(self, joint_obs: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([joint_obs, joint_actions], dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        q_value = self.layer3(x)
        return q_value

# --- MADDPG Update Logic Outline (for agent i) ---
# 1. Sample batch of joint transitions from buffer:
batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones

# --- Critic_i Update ---
with torch.no_grad():
    target_next_actions = [target_actor_j(batch_next_obs_j) for j in range(num_agents)]
    target_next_actions_cat = torch.cat(target_next_actions, dim=1)
    joint_next_obs_cat = batch_next_obs.view(batch_size, -1)
    q_target_next = target_critic_i(joint_next_obs_cat, target_next_actions_cat)
    td_target_i = batch_rewards_i + gamma * (1 - batch_dones_i) * q_target_next

joint_obs_cat = batch_obs.view(batch_size, -1)
joint_actions_cat = batch_actions.view(batch_size, -1)
q_current_i = critic_i(joint_obs_cat, joint_actions_cat)

critic_loss_i = F.mse_loss(q_current_i, td_target_i.detach())
# Optimize critic_i

# --- Actor_i Update ---
current_actions = [actor_j(batch_obs_j) for j in range(num_agents)]
current_actions[i] = actor_i(batch_obs_i)
current_actions_cat = torch.cat(current_actions, dim=1)

actor_loss_i = -critic_i(joint_obs_cat, current_actions_cat).mean()
# Optimize actor_i

# --- Soft Update Target Networks ---
soft_update(target_critic_i, critic_i, tau)
soft_update(target_actor_i, actor_i, tau)
