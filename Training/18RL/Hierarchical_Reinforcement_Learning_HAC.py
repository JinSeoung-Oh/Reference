### From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

"""
1. Purpose & Structure
   -a. Hierarchical Actor‑Critic (HAC) tackles long‑horizon or sparse‑reward tasks by stacking two levels of policies.
       -1. High level (L1): observes current state s and final goal G; outputs a subgoal g₀ for the lower level.
       -2. Low level (L0): receives s plus subgoal g₀; selects primitive actions for H steps, earning an intrinsic reward based on achieving g₀.

2. Learning Signals
   -a. L0 updates from intrinsic rewards and hindsight relabeling of goals.
   -b. L1 updates from the environment reward accumulated during the lower‑level rollout plus its own state transition.

3. Goal‑Conditioned Networks
   -a. Both levels use goal‑conditioned Q‑networks that take the concatenation of state and goal as input and output Q‑values:
       -1. L0: Q for primitive actions.
       -2. L1: Q for discrete subgoals.

4. Hindsight Experience Replay
   -a. A Hindsight Replay Buffer stores transitions and, with probability p, relabels the intended goal with the achieved goal, 
       recomputing reward/done, to boost learning from failures.

5. Training Loop (conceptual)
   -a. High level sets subgoal → Low level attempts it for H steps → intrinsic & extrinsic rewards collected → both levels store transitions
       (with achieved goals) → off‑policy updates using the hindsight buffer.

6. Observed Outcome (GridWorld example)
   -a. Environment rewards decrease; episode lengths stay high; high‑level loss rises—indicating HAC failed to learn effectively in this implementation.
"""

# Goal-Conditioned Q-Network (Can be used for Actor/Critic proxy in HAC)
# Assumes state and goal are concatenated as input
class GoalConditionedQNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super(GoalConditionedQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        # Output depends on level:
        # L0: Q-values for primitive actions
        # L1: Q-values for selecting discrete subgoals
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state_goal_concat: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state_goal_concat))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

# --- Network Dimensions ---
# state_dim = env.state_dim
# goal_dim = env.goal_dim # Usually same as state_dim if goals are states
# primitive_action_dim = env.action_dim
# subgoal_action_dim = env.rows * env.cols # If subgoals are discrete grid cells

# low_level_net = GoalConditionedQNetwork(state_dim + goal_dim, primitive_action_dim)
# high_level_net = GoalConditionedQNetwork(state_dim + goal_dim, subgoal_action_dim)

# Simplified Hindsight Replay Buffer Concept
class HindsightReplayBuffer:
    def __init__(self, capacity: int, hindsight_prob: float = 0.8):
        self.memory = deque([], maxlen=capacity)
        self.hindsight_prob = hindsight_prob

    def push(self, state, action, reward, next_state, goal, done, level, achieved_goal):
        # Store the full transition including the intended goal and what was achieved
        self.memory.append({
            'state': state, 'action': action, 'reward': reward,
            'next_state': next_state, 'goal': goal, 'done': done,
            'level': level, 'achieved_goal': achieved_goal
        })

    def sample(self, batch_size: int, level: int):
        # 1. Filter buffer for transitions of the correct 'level'
        # 2. Sample a batch of these transitions
        # 3. For each transition in the batch:
        #    - Keep the original transition (state, action, reward, next_state, goal, done)
        #    - With probability 'hindsight_prob':
        #        - Create a *new* hindsight transition:
        #            - Use the same state, action, next_state.
        #            - Replace 'goal' with 'achieved_goal'.
        #            - Recalculate 'reward' based on whether next_state matches the *new* hindsight goal
        #              (e.g., 0 if matched, -1 if not for L0 intrinsic reward).
        #            - Recalculate 'done' based on whether the hindsight goal was achieved.
        #        - Add this hindsight transition to the batch being prepared.
        # 4. Convert the final batch (originals + hindsight) into tensors.

