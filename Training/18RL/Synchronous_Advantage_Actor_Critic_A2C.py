### From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

"""
1. Synchronous Actor‑Critic
   -a. A2C uses an Actor 𝜋_𝜃(𝑎∣𝑠) for action selection and a Critic 𝑉_𝜙(𝑠) for state evaluation, updating both synchronously on each batch.

2. Advantage Estimate
   -a. Advantage for time step 𝑡:
       𝐴^_𝑡 = 𝑅_𝑡 − 𝑉_𝜙(𝑠_𝑡),
       where 𝑅_𝑡 is an 𝑛-step or GAE return. Lower variance than REINFORCE.

3. Rollout → Update Cycle
   -a. Collect 𝑁 steps of (𝑠,𝑎,𝑟,𝑠′) plus value estimates 𝑉(𝑠)
   -b. Compute returns‑to‑go and advantages for the batch.
   -c. Policy loss (actor): −𝐸[log 𝜋𝜃(𝑎_𝑡∣𝑠_𝑡)𝐴^_𝑡]− 𝛽 entropy
   -d. Value loss (critic): MSE between 𝑉_𝜙(𝑠_𝑡) and 𝑅_𝑡
   -e. Apply gradients to actor and critic simultaneously.

4. Behaviour Observed
   -a. Rewards rise quickly to near‑maximum; episode lengths drop rapidly.
   -b. Value loss peaks early, then declines as the critic stabilizes.
   -c. Policy loss is noisy but trends downward; final policy grid shows a consistent path toward the goal.
"""

# A2C Update Step
def update_a2c(actor: PolicyNetwork,
               critic: ValueNetwork,
               actor_optimizer: optim.Optimizer,
               critic_optimizer: optim.Optimizer,
               states: torch.Tensor,
               actions: torch.Tensor,
               advantages: torch.Tensor,    # Calculated GAE advantages
               returns_to_go: torch.Tensor, # Target for value function (Adv + V_old)
               value_loss_coeff: float,     # Weight for critic loss
               entropy_coeff: float         # Weight for entropy bonus
               ) -> Tuple[float, float, float]: # Avg losses

    # --- Evaluate current networks ---
    policy_dist = actor(states)
    log_probs = policy_dist.log_prob(actions)     # Log prob of actions taken
    entropy = policy_dist.entropy().mean()        # Average entropy
    values_pred = critic(states).squeeze()        # Critic's value prediction

    # --- Calculate Losses ---
    # Policy Loss (Actor): - E[log_pi * A_detached] - entropy_bonus
    policy_loss = -(log_probs * advantages.detach()).mean() - entropy_coeff * entropy

    # Value Loss (Critic): MSE(V_pred, Returns_detached)
    value_loss = F.mse_loss(values_pred, returns_to_go.detach())

    # --- Optimize Actor ---
    actor_optimizer.zero_grad()
    policy_loss.backward()        # Calculate actor gradients
    actor_optimizer.step()        # Update actor weights

    # --- Optimize Critic ---
    critic_optimizer.zero_grad()
    (value_loss_coeff * value_loss).backward()  # Scale loss before backward
    critic_optimizer.step()       # Update critic weights

    # Return losses for logging (policy objective part, value loss, entropy)
    return policy_loss.item() + entropy_coeff * entropy.item(), value_loss.item(), entropy.item()


