## From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

""""
1. Agent‑Environment Loop
   -a. An RL cycle consists of: observe current state s, choose action a via the current policy, execute a, environment transitions to state s′ 
       and returns an immediate reward r.
   -b. The tuple (𝑠,𝑎,𝑟,𝑠′) is the feedback used to adjust later decisions; the loop then restarts from s′.

2. Purpose and Capability of the First (Naïve) Agent
   -a. Objective: from any state, select the action that has produced the highest average immediate reward so far.
   -b. Limitation: considers only one‑step outcomes, ignoring long‑term consequences. True “learning” (credit assignment across multiple steps)
                   is deferred to more advanced algorithms.

3. Memory Design
   -a. Implemented as a nested dictionary:
       memory[(state)][action]→[list of immediate rewards]
   -b. Every time the agent acts, it appends the received reward to the list for that 
       (state,action) pair.

4. Policy: ε‑Greedy on Average Immediate Reward
   -a. Exploration: with probability ε, pick a random action.
   -b. Exploitation: otherwise, compute the mean of stored rewards for each action in the current state; choose the action(s) with the highest mean.
   -c. Tie‑breaking / unseen actions:
       -1. If several actions share the best mean, select uniformly at random among them.
       -2. If the state has no recorded rewards yet, fall back to a random action.

5. Learning Update Rule
   -a. After each action, call update_simple_memory to append the new reward to the corresponding list.
   -b. No discounting, no value estimates beyond the single step.

6. Visualization Metrics Explained
   -a. Reward Trend (line plot): episode totals plus a moving average illustrate that the agent starts poorly (negative totals) but gradually improves.
   -b. State‑Visit Heatmap: intensity shows how often each grid cell is visited; bright near the start indicates heavy early exploration, 
                            while the goal cell is reached less frequently.
   -c. Best‑Action Reward Heatmap: for every state, displays the agent’s current estimate of the best immediate reward; 
                                   only the goal cell attains a distinctly high value.
   -d. Action Histogram from (0,0): counts of chosen actions reveal a preference for “right” and “down,” matching the most direct path toward the goal.
""""


# Memory Structure: memory[(state_tuple)][action_index] -> [list_of_rewards]
agent_memory: DefaultDict[Tuple[int, int], DefaultDict[int, List[float]]] = \
    defaultdict(lambda: defaultdict(list))

# Example: Store reward for taking action 1 in state (0,0)
# agent_memory[(0,0)][1].append(-0.1)

def choose_simple_action(state: Tuple[int, int],
                         memory: DefaultDict[Tuple[int, int], DefaultDict[int, List[float]]],
                         epsilon: float,
                         n_actions: int) -> int:
    """Chooses action epsilon‑greedily based on average immediate reward."""
    if random.random() < epsilon:
        return random.randrange(n_actions)  # Explore
    else:
        # Exploit: find action with best average immediate reward
        state_action_memory = memory[state]
        best_avg_reward = -float('inf')
        best_actions = []

        for action_idx in range(n_actions):
            rewards = state_action_memory[action_idx]
            # If action never tried from this state, assume avg reward 0
            avg_reward = np.mean(rewards) if rewards else 0.0
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_actions = [action_idx]
            elif avg_reward == best_avg_reward:
                best_actions.append(action_idx)
        
        # If no best action yet (state unseen) pick randomly
        if not best_actions:
            return random.randrange(n_actions)
        # Break ties randomly
        return random.choice(best_actions)

def update_simple_memory(memory: DefaultDict[Tuple[int, int], DefaultDict[int, List[float]]],
                         state: Tuple[int, int],
                         action: int,
                         reward: float) -> None:
    """Adds the received reward to the memory for the state‑action pair."""
    memory[state][action].append(reward)
