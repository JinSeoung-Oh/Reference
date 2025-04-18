### From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

"""
1. On‑policy Nature
   -a. SARSA (State–Action–Reward–State–Action) updates the value of the policy it is actually executing, including exploratory moves.
   -b. By contrast, Q‑Learning is off‑policy: it updates toward the optimal action even when a different exploratory action was taken.

2. Update Target
   -a. Transition:
          𝑎,𝑟
        𝑠→    𝑠′
   -b. The agent chooses its next action 𝑎′ in 𝑠′ using the same ε‑greedy policy.
   -c. Target = 𝑟+𝛾𝑄(𝑠′,𝑎′)
   -d. Update 𝑄(𝑠,𝑎)←𝑄(𝑠,𝑎)+𝛼[target−𝑄(𝑠,𝑎)]
   -e. Quintuple involved: (𝑠,𝑎,𝑟,𝑠′,𝑎′)

3. Data Structure
   -a. Shares the same Q‑table format as Q‑Learning:
       q_table[(state)][action]→𝑄(𝑠,𝑎)

4. ε‑Greedy Policy (unchanged from Q‑Learning)
   -a. Explore with probability ε (random action).
   -b. Exploit otherwise by selecting an action with the highest stored Q‑value in the current state (ties broken randomly).

5. Consequence of Using 𝑄(𝑠′,𝑎′)
   -a. Because the update includes the Q‑value of the action actually chosen—which might be exploratory—SARSA often learns a more 
       conservative policy in risky environments (e.g., cliffs), as it evaluates the policy being followed.

6. Visualization Interpretation
   -a. Episode lengths shrink over time, indicating faster routes to the goal.
   -b. Reward plot shows high but variable returns due to continued exploration.
   -c. Policy grid (arrows) confirms a route toward goal states “T,” though some variability remains.
"""

# Epsilon-Greedy Action Selection (Same function as Q-Learning)
def choose_sarsa_action(state: Tuple[int, int],
                          q_table: DefaultDict[Tuple[int, int], Dict[int, float]],
                          epsilon: float,
                          n_actions: int) -> int:
    """ Chooses action epsilon-greedily based on Q-table values. """
    if random.random() < epsilon:
        return random.randrange(n_actions)  # Explore
    else:
        # Exploit: Choose action with highest Q-value
        q_values_for_state = q_table[state]
        if not q_values_for_state:
            return random.randrange(n_actions)
        max_q = -float('inf')
        best_actions = []
        for action_idx in range(n_actions):
            q_val = q_values_for_state[action_idx]  # Defaults to 0.0
            if q_val > max_q:
                max_q = q_val
                best_actions = [action_idx]
            elif q_val == max_q:
                best_actions.append(action_idx)
        return random.choice(best_actions) if best_actions else random.randrange(n_actions)

# SARSA Update Rule
def update_sarsa_value(q_table: DefaultDict[Tuple[int, int], Dict[int, float]],
                       state: Tuple[int, int],
                       action: int,
                       reward: float,
                       next_state: Tuple[int, int],
                       next_action: int,  # The action ACTUALLY chosen for the next step
                       alpha: float,
                       gamma: float,
                       is_done: bool) -> None:
    """ Performs a single SARSA update step. """
    current_q = q_table[state][action]

    # Q-value for the next state–action pair chosen by policy
    q_next_state_action = 0.0
    if not is_done:
        q_next_state_action = q_table[next_state][next_action]

    td_target = reward + gamma * q_next_state_action
    td_error  = td_target - current_q

    q_table[state][action] = current_q + alpha * td_error
