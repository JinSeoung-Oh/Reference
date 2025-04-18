### From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

"""
1. Motivation & Concept
   -a. Policy‑gradient methods can shift the policy too far in one noisy update.
   -b. PPO stabilizes learning by optimizing a clipped surrogate objective that limits how much the new policy  
       𝜋_new can diverge from the old policy 𝜋_old

2. Framework
   -a. Actor‑critic, on‑policy: collects a batch of (𝑠,𝑎,𝑟,𝑠′) using 𝜋_old plus its log‑probs; a critic 
       𝑉_old supplies state‑value estimates.
   -b. Advantages 𝐴^ are computed (typically with Generalized Advantage Estimation).

3. Update Process (repeated for several epochs on the same batch)
   -a. Evaluate 𝜋_new on the batch states to get new log‑probs.
   -b. Form the probability ratio 𝑟=exp(log 𝜋_new − log 𝜋_old)
   -c. Compute two terms:
       surr1 = 𝑟𝐴^ , surr2 = clip(𝑟,1−𝜖,1+𝜖)
   -d. Policy loss = −min(surr1, surr2) minus an entropy bonus.
   -e. Value loss = MSE between critic predictions and returns‑to‑go.
   -f. Update actor and critic via gradient descent; repeat for the preset number of epochs.

4. Observed Behaviour
   -a. Episode length and reward curves show very rapid, stable convergence.
   -b. Policy loss and entropy fall smoothly; value loss stays bounded.
   -c. Final policy grid displays a deterministic, sensible path to the goal.
"""

# PPO Update Step (Simplified view, assumes data is batched)
def update_ppo(actor: PolicyNetwork,
               critic: ValueNetwork,
               actor_optimizer: optim.Optimizer,
               critic_optimizer: optim.Optimizer,
               states: torch.Tensor,
               actions: torch.Tensor,
               log_probs_old: torch.Tensor, # Log probs from the policy used for rollout
               advantages: torch.Tensor,    # Calculated GAE advantages
               returns_to_go: torch.Tensor, # Target for value function (Adv + V_old)
               ppo_epochs: int,             # Number of updates per batch
               ppo_clip_epsilon: float,     # Clipping parameter ε
               value_loss_coeff: float,   # Weight for critic loss
               entropy_coeff: float) -> Tuple[float, float, float]: # Avg losses

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0

    # Data should be detached before entering the loop
    advantages = advantages.detach()
    log_probs_old = log_probs_old.detach()
    returns_to_go = returns_to_go.detach()

    # Perform multiple epochs of updates on the same batch
    for _ in range(ppo_epochs):
        # --- Actor (Policy) Update ---
        policy_dist = actor(states)              # Get current policy distribution
        log_probs_new = policy_dist.log_prob(actions) # Log prob of actions under *new* policy
        entropy = policy_dist.entropy().mean()    # Encourage exploration

        # Calculate the ratio r(θ) = π_new / π_old
        ratio = torch.exp(log_probs_new - log_probs_old)

        # Calculate clipped surrogate objective parts
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - ppo_clip_epsilon, 1.0 + ppo_clip_epsilon) * advantages

        # Policy Loss: -(min(surr1, surr2)) - entropy_bonus
        policy_loss = -torch.min(surr1, surr2).mean() - entropy_coeff * entropy

        # Optimize actor
        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()

        # --- Critic (Value) Update ---
        values_pred = critic(states).squeeze() # Predict V(s) with current critic
        value_loss = F.mse_loss(values_pred, returns_to_go) # Compare to calculated returns

        # Optimize critic
        critic_optimizer.zero_grad()
        (value_loss_coeff * value_loss).backward() # Scale loss before backward
        critic_optimizer.step()

        # Accumulate stats
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy.item()

    # Return average losses over the epochs
    return total_policy_loss / ppo_epochs, total_value_loss / ppo_epochs, total_entropy / ppo_epochs

