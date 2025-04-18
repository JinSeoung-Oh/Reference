### From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

"""
1. Hybrid Idea
   -a. Dyna‑Q augments model‑free Q‑learning with a simple model‑based component to gain sample efficiency.
   -b. Each real experience (𝑠,𝑎,𝑟,𝑠′) is used three ways:
       -1. Direct Q‑learning update on 𝑄(𝑠,𝑎)
       -2. Model learning: store that (𝑠,𝑎) leads to (𝑟,𝑠′)
       -3. Planning: perform k simulated updates by sampling previously seen 
                     (𝑠_𝑝,𝑎_𝑝), retrieving (𝑟_𝑝,𝑠′_𝑝) from the model, and applying another Q‑learning update.

2. Data Structures
   -a. Q‑table: 𝑞_𝑡𝑎𝑏𝑙𝑒_𝑑𝑦𝑛𝑎𝑞[(state)][action]→𝑄(𝑠,𝑎)
   -b. Model: 𝑚𝑜𝑑𝑒𝑙_𝑑𝑦𝑛𝑎𝑞[(state,action)]→(𝑟,𝑠′)
   -c. observed_state_actions list stores all encountered state–action pairs for random sampling during planning.

3. Model Update
   -a. update_model records (𝑟,𝑠′) for a state–action pair and adds the pair to the sampling list if new.

4. Planning Step
  -a. planning_steps iterates k times:
      – randomly picks a stored (𝑠_𝑝,𝑎_𝑝),
      – looks up its predicted (𝑟_𝑝,𝑠′_𝑝),
      – calls the standard update_q_value (Q‑learning) with is_done=False.

5. Behavior Observed (visualization)
   -a. Episode lengths drop rapidly; rewards rise steeply—evidence of fast convergence.
   -b. Efficiency gain is attributed to planning with k=50 simulated updates per real step.
"""

# Q-Table: q_table[(state_tuple)][action_index] -> q_value (same as before)
q_table_dynaq: DefaultDict[Tuple[int, int], Dict[int, float]] = \
    defaultdict(lambda: defaultdict(float))

# Model: model[(state_tuple, action_index)] -> (reward, next_state_tuple)
model_dynaq: Dict[Tuple[Tuple[int, int], int], Tuple[float, Tuple[int, int]]] = {}

# Track observed state-action pairs for sampling during planning
observed_state_actions: List[Tuple[Tuple[int, int], int]] = []

# Model Update Function
def update_model(model: Dict[Tuple[Tuple[int, int], int], Tuple[float, Tuple[int, int]]],
                 observed_pairs: List[Tuple[Tuple[int, int], int]],
                 state: Tuple[int, int],
                 action: int,
                 reward: float,
                 next_state: Tuple[int, int]) -> None:
    """ Updates the deterministic tabular model and observed pairs list. """
    state_action = (state, action)
    model[state_action] = (reward, next_state)  # Store outcome

    # Add to list if this pair hasn't been seen before
    if state_action not in observed_pairs:
        observed_pairs.append(state_action)

# Planning Step Function
def planning_steps(k: int,  # Number of planning steps
                   q_table: DefaultDict[Tuple[int, int], Dict[int, float]],
                   model: Dict[Tuple[Tuple[int, int], int], Tuple[float, Tuple[int, int]]],
                   observed_pairs: List[Tuple[Tuple[int, int], int]],
                   alpha: float, gamma: float, n_actions: int) -> None:
    """ Performs 'k' simulated Q-learning updates using the model. """
    if not observed_pairs:  # Can't plan without observations
        return

    for _ in range(k):
        # 1. Sample a random previously observed state-action pair
        state_p, action_p = random.choice(observed_pairs)

        # 2. Query the model for the simulated outcome
        reward_p, next_state_p = model[(state_p, action_p)]

        # 3. Apply Q-learning update using the simulated experience
        #    (Assuming simulated steps don't end the episode unless model says so)
        #    Need the update_q_value function from Q-Learning section here.
        update_q_value(q_table, state_p, action_p, reward_p, next_state_p,
                       alpha, gamma, n_actions, is_done=False)  # Assume not done in simulation
                       # (A more complex model could predict 'done')


