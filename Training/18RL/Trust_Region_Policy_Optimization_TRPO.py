### From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

"""
1. Goal
   -a. TRPO seeks monotonic policy improvement by limiting each update’s size via a KL‑divergence constraint: 
       𝐷_(KL)(𝜋_old∥𝜋_new)≤𝛿

2. Batch Collection & Advantage
   -a. Gather on‑policy experience with 𝜋_old; compute advantages 𝐴 and the standard policy gradient 𝑔

3. Constrained Update Computation
   -a. Solve 𝐹_𝑠 ≈ 𝑔 where 𝐹 is the Fisher Information Matrix (FIM).
   -b. Use the Conjugate Gradient (CG) method, requiring only Fisher‑Vector Products (FVPs)—no explicit FIM inversion.

4. Line Search & Trust Region
   -a. Scale the direction 𝑠 to satisfy the KL limit; perform a back‑tracking line search to ensure the step both respects
       the constraint and improves the surrogate objective.

5. Critic Role
   -a. The critic supplies value estimates for advantage computation but is not part of the KL‑constrained optimization; 
       it is updated afterward with an MSE loss.

6. Observed Behaviour
   -a. Rewards and episode lengths converge very rapidly (≈ 20‑30 iterations).
   -b. Critic loss stabilizes; KL divergence plot should stay below the max‑KL threshold.
"""

# Computes the Fisher vector product using Hessian-vector product approximation
def fisher_vector_product(actor, states, vector, cg_damping):
    log_probs = actor.get_log_probs(states).detach()
    kl = (log_probs.exp() * (log_probs - log_probs.detach())).sum()
    grads = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
    flat_grads = torch.cat([g.view(-1) for g in grads])
    
    gv = torch.dot(flat_grads, vector)
    hv = torch.autograd.grad(gv, actor.parameters())
    flat_hv = torch.cat([h.view(-1) for h in hv])
    
    # Adds a damping term to improve numerical stability
    return flat_hv + cg_damping * vector

# Implements the conjugate gradient method to solve Ax = b
def conjugate_gradient(fvp_func, b, cg_iters=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rs_old = torch.dot(r, r)
    
    for _ in range(cg_iters):
        Ap = fvp_func(p)
        alpha = rs_old / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = torch.dot(r, r)
        if rs_new < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x

# Performs backtracking line search to find an acceptable step size
def backtracking_line_search(actor, states, actions, advantages, old_log_probs,
                             step_direction, initial_step_size, max_kl, decay=0.8, max_iters=10):
    theta_old = {name: param.clone() for name, param in actor.named_parameters()}
    for i in range(max_iters):
        step_size = initial_step_size * (decay ** i)
        for param, step in zip(actor.parameters(), step_size * step_direction):
            param.data.add_(step)
        kl = actor.kl_divergence(states, old_log_probs)
        surrogate = actor.surrogate_loss(states, actions, advantages, old_log_probs)
        if kl <= max_kl and surrogate >= 0:
            return step_size * step_direction, True
        for name, param in actor.named_parameters():
            param.data.copy_(theta_old[name])
    return None, False

# Updates the actor and critic using the TRPO algorithm
def update_trpo(actor, critic, actor_optimizer, critic_optimizer,
                states, actions, advantages, returns_to_go, log_probs_old,
                max_kl=0.01, cg_iters=10, cg_damping=0.1, line_search_decay=0.8,
                value_loss_coeff=0.5, entropy_coeff=0.01):
    
    policy_loss = actor.surrogate_loss(states, actions, advantages, log_probs_old)
    grads = torch.autograd.grad(policy_loss, actor.parameters())
    g = torch.cat([grad.view(-1) for grad in grads])
    
    fvp_func = lambda v: fisher_vector_product(actor, states, v, cg_damping)
    step_direction = conjugate_gradient(fvp_func, g, cg_iters)
    
    sAs = torch.dot(step_direction, fvp_func(step_direction))
    step_size = torch.sqrt(2 * max_kl / (sAs + 1e-8))
    
    step, success = backtracking_line_search(actor, states, actions, advantages, log_probs_old,
                                             step_direction, step_size, max_kl, line_search_decay)
    if success:
        with torch.no_grad():
            for param, step_val in zip(actor.parameters(), step):
                param.data.add_(step_val)
    
    value_loss = nn.MSELoss()(critic(states), returns_to_go)
    critic_optimizer.zero_grad()
    value_loss.backward()
    critic_optimizer.step()
    
    return policy_loss.item(), value_loss.item()




