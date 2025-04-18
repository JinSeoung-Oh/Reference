### From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

"""
1. On‑policy Foundation
   -a. Expected SARSA preserves SARSA’s on‑policy character: it updates values for the policy actually being followed.

2. Target Construction
   -a. After (𝑠,𝑎,𝑟,𝑠′), the algorithm examines every possible action 𝑎′ in 𝑠′
   -b. It weights each 𝑄(𝑠′,𝑎′) by the probability 𝜋(𝑎′∣𝑠′) given by the current ε‑greedy policy.
   -c. The expected next value is
       𝐸_(𝑎′∼𝜋)[𝑄(𝑠′,𝑎′)] = ∑_𝑎′ 𝜋(𝑎′∣𝑠′)𝑄(𝑠′,𝑎′)
   -d. Target = 𝑟 + 𝛾𝐸[𝑄(𝑠′,𝑎′)]
   -e. Update 𝑄(𝑠,𝑎)←𝑄(𝑠,𝑎)+𝛼[target−𝑄(𝑠,𝑎)]

3. Variance Reduction
   -a. Averaging over all actions (instead of using a single sampled 𝑎′) lowers update variance and often yields smoother, faster learning.

4. Structures and Policy
   -a. Q‑table format and ε‑greedy action selection are identical to those in SARSA.
   -b. Only the update rule changes to incorporate the expectation.

5. Visualization Notes
   -a. Episode lengths fall rapidly, indicating quick acquisition of efficient paths.
   -b. Rewards reach high values consistently after early learning, reflecting a stable policy.
   -c. Policy grid shows sensible routes toward terminal states “T”.
"""

# Expected SARSA Update Rule
def update_expected_sarsa_value(
    q_table: DefaultDict[Tuple[int, int], Dict[str, float]],
    state: Tuple[int, int],
    action: int,  # Use integer action index now
    reward: float,
    next_state: Tuple[int, int],
    alpha: float,
    gamma: float,
    epsilon: float,  # Current epsilon needed for expectation
    n_actions: int,
    is_done: bool
) -> None:
    """ Performs a single Expected SARSA update step. """

    # Get the current Q-value estimate
    current_q = q_table[state][action]

    # Calculate the expected Q-value for the next state
    expected_q_next = 0.0
    if not is_done and q_table[next_state]:  # Check if next state exists and has entries
        q_values_next_state = q_table[next_state]
        if q_values_next_state:  # Check if dictionary is not empty
            max_q_next = max(q_values_next_state.values())
            # Find all best actions (handle ties)
            best_actions = [a for a, q in q_values_next_state.items() if q == max_q_next]

            # Calculate probabilities under epsilon-greedy
            prob_greedy = (1.0 - epsilon) / len(best_actions)  # Split greedy prob among best actions
            prob_explore = epsilon / n_actions

            # Expected value E[Q(s', A')]
            for a_prime in range(n_actions):
                prob_a_prime = 0.0
                if a_prime in best_actions:
                    prob_a_prime += prob_greedy
                prob_a_prime += prob_explore
                expected_q_next += prob_a_prime * q_values_next_state.get(a_prime, 0.0)

    # TD Target: R + gamma * E[Q(s', A')]
    td_target = reward + gamma * expected_q_next

    # TD Error: TD Target - Q(s, a)
    td_error = td_target - current_q

    # Update Q-value
    q_table[state][action] = current_q + alpha * td_error
