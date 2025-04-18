### From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

"""
1. Purpose
   -a. Monte Carlo Tree Search (MCTS) performs online planning from the current state by simulating many futures in a search tree 
       instead of learning global value/policy functions.

2. Four MCTS Phases
   -a. Selection: follow tree edges from the root using a strategy such as UCT (balances exploitation & exploration) until reaching a leaf 
                  that is not fully expanded.
   -b. Expansion: if the leaf is non‑terminal, choose one untried action, simulate it, and create a new child node.
   -c. Simulation (Rollout): from that node, run a fast rollout policy (often random) to the end of the episode or a depth limit and record 
                             the total reward R.
   -d. Backpropagation: update every node on the visited path with R by incrementing its visit count N and adding to its total value W.

3. Core Data & Formulas
   -a. Each node tracks state, children, untried actions, visit count N, total value W; average value 𝑄=𝑊/𝑁
   -b. UCT score (child selection):
       𝑄(𝑠,𝑎) + 𝑐 np.root(ln 𝑁(𝑠) / 𝑁(𝑠,𝑎))
       where c is an exploration constant.

4. Requirements
   -a. Access to an environment model / simulator to step from state and action.
   -b. A rollout policy (simple, fast).
   -c. The algorithm builds a tree online and does not retain a learning curve across episodes; performance depends on simulations per move.

5. Empirical Outcome (GridWorld example)
   -a. Achieves stable positive rewards (~ +6 – +7) from the beginning; episode lengths ~ 30–40 steps; 
       path visualization shows a direct route from start to goal.

"""
class MCTSNode:
    """ Represents a node in the Monte Carlo Tree Search. """
    def __init__(self, state: Tuple[int, int], parent: Optional['MCTSNode'] = None, action: Optional[int] = None):
        self.state = state
        self.parent = parent
        self.action_that_led_here = action # Action parent took to get here

        self.children: Dict[int, MCTSNode] = {} # map action -> child node
        # Function to get possible actions (depends on the environment)
        self.untried_actions: List[int] = self._get_possible_actions(state)

        self.visit_count: int = 0
        self.total_value: float = 0.0 # Sum of rewards from rollouts

    def _get_possible_actions(self, state):
        # Placeholder: Replace with actual environment call
        # Example for grid world (assuming env is accessible or passed):
        if env.is_terminal(state): return []
        return list(range(env.get_action_space_size())) # Or env.get_valid_actions(state)

    def is_fully_expanded(self) -> bool:
        return not self.untried_actions

    def is_terminal(self) -> bool:
        # Placeholder: Replace with actual environment call
        return env.is_terminal(self.state)

    def get_average_value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

    def select_best_child_uct(self, exploration_constant: float) -> 'MCTSNode':
        """ Selects child node with the highest UCT score. """
        best_score = -float('inf')
        best_child = None
        for action, child in self.children.items():
            if child.visit_count == 0:
                score = float('inf') # Prioritize unvisited nodes
            else:
                exploit = child.get_average_value()
                explore = exploration_constant * math.sqrt(math.log(self.visit_count) / child.visit_count)
                score = exploit + explore

            if score > best_score:
                best_score = score
                best_child = child
        if best_child is None: # Should only happen if node has no children yet
            return self # Should ideally not be called on a node with no children
        return best_child



