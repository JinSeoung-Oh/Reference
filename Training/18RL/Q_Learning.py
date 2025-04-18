### From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

"""
1. Motivation
   -a. Unlike the previous “MemoryBot,” which optimizes only immediate rewards, Q‑Learning estimates the total discounted future return for each 
       (state, action) pair, producing the action‑value function 
       𝑄∗(𝑠,𝑎)

2. Core Idea
   -a. After executing action a in state s, receiving reward r, and landing in state s′, the algorithm forms a target value
       TD target=𝑟+𝛾max_𝑎′ 𝑄(𝑠′,𝑎′)
       where 𝛾 discounts future rewards.
   -b. The update moves the old Q‑value toward this target by a fraction 𝛼 (learning rate):
       𝑄(𝑠,𝑎)←𝑄(𝑠,𝑎)+𝛼[TD target−𝑄(𝑠,𝑎)]
       – the bracketed term is the temporal‑difference (TD) error.

3. Data Structure
   -a. A Q‑table stores estimates:
       q_table[(state)][action]→𝑄(𝑠,𝑎)
       − unvisited entries default to 0.

4. Policy: ε‑Greedy Using Q‑Values
   -a. Explore with probability ε (random action).
   -b. Exploit otherwise by selecting an action with the highest Q‑value in the current state (ties broken at random).
   -c. If a state has no stored values yet, choose randomly.

5. Learning Step (update_q_value)
   -a. Retrieves the current 𝑄(𝑠,𝑎)
   -b. Obtains max_𝑎′ 𝑄(𝑠′,𝑎′) for the next state (or 0 if terminal / unseen).
   -c. Computes the TD target and TD error, then updates 𝑄(𝑠,𝑎) by 𝛼×TD error

6. Interpretation of the Visualization
   -a. Reward & Episode Length plots: episodes shorten and rewards improve as the agent learns.
   -b. Four Q‑value heatmaps (up, down, left, right): brighter cells near the goal show preferred actions.
   -c. Learned‑policy plot: arrows depict the optimal path discovered toward the goal.
"""

# Q-Table: q_table[(state_tuple)][action_index] -> q_value
q_table: DefaultDict[Tuple[int, int], Dict[int, float]] = \
    defaultdict(lambda: defaultdict(float))

# Example: Accessing Q((0,0), 0=Up) will return 0.0 initially
# print(q_table[(0,0)][0])

# Epsilon-Greedy Action Selection based on Q-values
def choose_q_learning_action(state: Tuple[int, int],
                             q_table: DefaultDict[Tuple[int, int], Dict[int, float]],
                             epsilon: float,
                             n_actions: int) -> int:
    """ Chooses action epsilon-greedily based on Q-table values. """
    if random.random() < epsilon:
        return random.randrange(n_actions)  # Explore
    else:
        # Exploit: Choose action with highest Q-value for this state
        q_values_for_state = q_table[state]
        if not q_values_for_state:           # State unvisited
            return random.randrange(n_actions)

        max_q = -float('inf')
        best_actions = []
        for action_idx in range(n_actions):
            q_val = q_values_for_state[action_idx]  # defaultdict → 0 if missing
            if q_val > max_q:
                max_q = q_val
                best_actions = [action_idx]
            elif q_val == max_q:
                best_actions.append(action_idx)

        return random.choice(best_actions)          # Break ties randomly

# Q-Learning Update Rule
def update_q_value(q_table: DefaultDict[Tuple[int, int], Dict[int, float]],
                   state: Tuple[int, int],
                   action: int,
                   reward: float,
                   next_state: Tuple[int, int],
                   alpha: float,        # Learning Rate
                   gamma: float,        # Discount Factor
                   n_actions: int,
                   is_done: bool) -> None:
    """ Performs a single Q-Learning update step. """
    # Current estimate
    current_q = q_table[state][action]

    # Best future value from next state
    if not is_done and q_table[next_state]:
        max_next_q = max(q_table[next_state].values())
    else:
        max_next_q = 0.0                    # 0 if terminal or unseen

    # TD target and error
    td_target = reward + gamma * max_next_q
    td_error  = td_target - current_q

    # Update rule
    q_table[state][action] = current_q + alpha * td_error
