### From https://medium.com/@kyeg/building-the-muon-optimizer-in-pytorch-a-geometric-approach-to-neural-network-optimization-17f4601be548

"""
1. Muon’s Innovative Perspective
   Traditional optimizers—SGD variants, Adam, RMSProp—treat weight updates as purely mathematical steps aimed at minimizing the loss surface. 
   Muon, by contrast, asks a fundamentally different question: “How do changes in the weight matrices actually alter the network’s input→output mapping?”
   -a. Function‑centric View
       -1. Instead of seeing Δ𝑊 as an abstract vector in parameter space, Muon interprets it as a transformation on the network’s function 
           𝑓𝑊(𝑥). This shift makes the optimizer explicitly aware of how weight adjustments translate to real behavioral changes.
       -2. By measuring updates in terms of their operator norm impact—that is, the guaranteed bound on output changes—Muon enforces 
           that each optimization step yields controlled, predictable modifications to the model’s function.
   -b. Geometry over Geometryless Gradients
       -1. Muon leverages RMS‑to‑RMS operator norms to metrize layers: it understands “size” not merely as Euclidean length of gradient, 
           but as maximal average amplification of activations.
       -2. This geometric framing reveals that standard gradient steps often produce wildly varying functional effects across layers of different 
           dimensions. 
           Muon’s perspective guides updates so that their functional magnitude remains consistent, regardless of layer size or architecture.
   -c. Scale‑Invariance and Model Agnosticism
       -1. Because Muon measures updates by their true impact on outputs rather than raw gradient magnitude, 
           it naturally adapts learning rates across networks of different widths. This makes the same optimizer hyperparameters effective 
           whether your model has 128, 512, or 4096 hidden units—an ability rooted in its function‑focused design.

2. Key Benefits of Muon
   Building on its novel perspective, Muon delivers several concrete advantages:
   -a. Automatic Learning Rate Transfer
       -1. What it is: Muon’s update rule normalizes for layer dimensions np.root(𝑑𝑥𝑑𝑦) and gradient scale (∥𝐺∥_𝐹), 
                       so the effective step size in function space stays the same when you widen or narrow a network.
       -2. Why it matters: You no longer need to hunt for a new learning rate whenever you change the number of neurons 
                           or attention heads—hyperparameter tuning becomes dramatically simpler.
   -b. Faster Convergence
       -1. What it is: By orthogonalizing gradient updates (via Newton–Schulz) and directly targeting the most impactful functional directions, 
                       Muon traverses the loss landscape more efficiently.
       -2. Empirical evidence: Benchmarks on NanoGPT and CIFAR‑10 show Muon reaching target accuracies in fewer epochs than Adam or AdamW,
                               translating to real wall‑clock training speedups.
   -c. Reduced Hyperparameter Tuning
       -1. What it is: The only key hyperparameters are the base learning rate 𝛼 and the number of Newton–Schulz iterations. 
                       Because the update size self‑scales with gradient norms and layer size, default settings work robustly across a wide range of models.
       -2. Why it matters: Teams spend less time on grid searches and more time iterating on model design and data curation.
   -d. Predictable and Stable Training Behavior
       -1. What it is: Enforcing an operator‑norm constraint on Δ𝑊 guarantees an upper bound on output changes ∥Δ𝑦∥_RMS, reducing catastrophic parameter jumps and oscillations.
       -2. Why it matters: Training curves become smoother, divergence is less likely, and checkpoint averaging yields more reliable improvements.
   -e. Improved Numerical Properties
       -1. What it is: Orthogonalized updates help the optimizer escape the “linearized regime” near initialization more effectively, 
                       promoting exploration of richer functional behaviors.
       -2. Why it matters: Models are less prone to get stuck in narrow valleys, and generalization properties can improve as a result.

3. Theoretical Components
   -a. Metrizing the Linear Layer
       -1. Vector RMS norm: ∣𝑥∣_RMS=(1/np.root(𝑑𝑥))∥𝑥∥2
       -2. Matrix RMS‑to‑RMS operator norm: ∣𝑊∣_op,RMS=∣𝑊∣_sp/np.root(𝑑𝑥𝑑𝑦)
   -b. Perturbation Bound
       ∥Δ𝑦∥_RMS ≤∣ Δ𝑊∣_op,RMS∥𝑥∥_RMS,
       allowing direct control of output change via operator‑norm regularization.
   -c. Gradient Dualization
       min_Δ𝑊 ⟨𝐺,Δ𝑊⟩  s.t.∣Δ𝑊∣_op,RMS ≤ 𝛽 ⟹ Δ𝑊∗ = −𝛼 𝛽 𝑈 𝑉^𝑇
   -d. Newton–Schulz Orthogonalization
       Iterative map 𝑋↦(3𝑋−𝑋^3)/2 applied to normalized gradient 𝐺/∥𝐺∥_𝐹 yields 𝑈𝑉^𝑇 without full SVD.
   -e. Final Update Rule
       𝑊←𝑊−𝛼 np.root(𝑑𝑥𝑑𝑦)/∥𝐺∥_𝐹 NS(𝐺)
       
4. Conclusion and Future Outlook
   -a. Geometric Optimization Emergence: Muon demonstrates that grounding updates in their functional impact yields faster, more stable training.
   -b. Layer‑wise Modular Extensions: The same geometric approach can be adapted to convolutional, attention, and normalization layers.
   -c. Towards “Set‑and‑Forget” Optimizers: As models scale, Muon’s automatic learning rate adaptation paves the way for optimizers 
       that require minimal tuning.
   -d. Community Invitation: A PyTorch implementation is available—researchers and practitioners are encouraged to experiment, 
       benchmark, and extend geometric optimization techniques.
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import List, Dict, Optional, Tuple, Union, Callable, Any, Iterator

class Muon(Optimizer):
    """
    Implements the Muon optimization algorithm for linear layers.
    
    Muon uses a geometric approach to optimization, specifically addressing
    how changes in weight matrices affect neural network behavior.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        ns_iters (int, optional): number of Newton-Schulz iterations (default: 5)
        momentum (float, optional): momentum factor (default: 0.9)
        weight_decay (float, optional): weight decay coefficient (default: 0)
    """
    
    def __init__(self, 
                 params: Iterator[torch.nn.Parameter], 
                 lr: float = 1e-3, 
                 ns_iters: int = 5, 
                 momentum: float = 0.9, 
                 weight_decay: float = 0):
        
        defaults = dict(lr=lr, ns_iters=ns_iters, momentum=momentum, weight_decay=weight_decay)
        super(Muon, self).__init__(params, defaults)
    
    def newton_schulz_orthogonalize(self, X: torch.Tensor, num_iters: int) -> torch.Tensor:
        """
        Apply Newton-Schulz iterations to approximate orthogonalization.
        
        This function applies the polynomial f(X) = (3X - X^3)/2 repeatedly to a normalized matrix,
        which gradually forces all singular values to 1 while preserving singular vectors.
        
        Args:
            X (torch.Tensor): Input matrix to orthogonalize
            num_iters (int): Number of Newton-Schulz iterations
            
        Returns:
            torch.Tensor: Orthogonalized matrix
        """
        # First, normalize the input matrix to get spectral norm close to 1
        # We use Frobenius norm as a simple approximation for initialization
        norm = torch.norm(X, p='fro')
        if norm < 1e-8:
            return X  # Avoid division by zero
        
        X = X / norm
        
        # Apply Newton-Schulz iterations
        for _ in range(num_iters):
            X = (3 * X - torch.matmul(torch.matmul(X, X), X)) / 2
            
        return X
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss
            
        Returns:
            Optional[float]: Loss value if closure is provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            ns_iters = group['ns_iters']
            momentum_factor = group['momentum']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Handle weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                state = self.state[p]
                
                # Initialize momentum buffer if needed
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                # Get momentum buffer
                momentum_buffer = state['momentum_buffer']
                
                # Update momentum buffer with current gradient
                momentum_buffer.mul_(momentum_factor).add_(grad, alpha=1 - momentum_factor)
                
                # Only apply Muon updates to matrices (linear layers)
                if len(p.shape) == 2:
                    # Get input and output dimensions for normalization
                    d_in, d_out = p.shape
                    
                    # Use the momentum buffer for orthogonalization
                    ortho_grad = self.newton_schulz_orthogonalize(momentum_buffer, ns_iters)
                    
                    # Scale by sqrt(d_in * d_out) / |G|_F as per Muon's formula
                    grad_norm = torch.norm(momentum_buffer, p='fro')
                    if grad_norm > 1e-8:  # Avoid division by zero
                        scaling = (d_in * d_out)**0.5 / grad_norm
                        update = ortho_grad * scaling
                        
                        # Apply the update
                        p.add_(update, alpha=-lr)
                    
                else:
                    # For non-matrix parameters (biases, etc.), use standard update with momentum
                    p.add_(momentum_buffer, alpha=-lr)
        
        return loss

---------------------------------------------------------------------------------------------------------------
class EnhancedMuon(Optimizer):
    """
    Enhanced implementation of the Muon optimization algorithm.
    
    This version includes additional features like adaptive momentum, gradient clipping,
    learning rate scheduling support, and detailed parameter tracking.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        ns_iters (int, optional): number of Newton-Schulz iterations (default: 5)
        momentum (float, optional): momentum factor (default: 0.9)
        weight_decay (float, optional): weight decay coefficient (default: 0)
        grad_clip_norm (float, optional): gradient clipping norm (default: None)
        track_stats (bool, optional): whether to track optimization statistics (default: False)
    """
    
    def __init__(self, 
                 params: Iterator[torch.nn.Parameter], 
                 lr: float = 1e-3, 
                 ns_iters: int = 5, 
                 momentum: float = 0.9, 
                 weight_decay: float = 0,
                 grad_clip_norm: Optional[float] = None,
                 track_stats: bool = False):
        
        defaults = dict(
            lr=lr, 
            ns_iters=ns_iters, 
            momentum=momentum, 
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            track_stats=track_stats
        )
        super(EnhancedMuon, self).__init__(params, defaults)
        
        self.global_stats = {
            'update_magnitudes': [],
            'gradient_norms': []
        }
    
    def newton_schulz_orthogonalize(self, X: torch.Tensor, num_iters: int) -> torch.Tensor:
        """
        Enhanced Newton-Schulz orthogonalization with stability checks.
        """
        # Check for numerical stability
        if torch.isnan(X).any() or torch.isinf(X).any():
            # Return identity matrix of appropriate size as fallback
            return torch.eye(X.shape[0], X.shape[1], device=X.device)
        
        # Normalize the input matrix
        norm = torch.norm(X, p='fro')
        if norm < 1e-8:
            return X
        
        X = X / norm
        
        # Apply Newton-Schulz iterations with stability checks
        for i in range(num_iters):
            X_squared = torch.matmul(X, X)
            X_cubed = torch.matmul(X_squared, X)
            
            # Check for numerical issues
            if torch.isnan(X_cubed).any() or torch.isinf(X_cubed).any():
                # Revert to previous iteration
                break
            
            X_new = (3 * X - X_cubed) / 2
            
            # Early stopping if convergence is reached
            if torch.norm(X_new - X, p='fro') < 1e-6:
                X = X_new
                break
                
            X = X_new
            
        return X
    
    def get_dimension_scaling(self, shape: Tuple[int, ...]) -> float:
        """
        Calculate the appropriate dimension scaling factor for different parameter shapes.
        
        For matrices (linear layers), this is sqrt(d_in * d_out).
        For other parameter types, we use appropriate heuristics.
        
        Args:
            shape (tuple): Shape of the parameter tensor
            
        Returns:
            float: Scaling factor
        """
        if len(shape) == 2:  # Linear layer weights
            d_in, d_out = shape
            return (d_in * d_out) ** 0.5
        elif len(shape) == 1:  # Bias vectors
            return shape[0] ** 0.5
        elif len(shape) == 4:  # Conv layer weights
            # For convolutions, the scaling considers channels and kernel size
            c_out, c_in, k_h, k_w = shape
            return (c_in * c_out * k_h * k_w) ** 0.5
        else:
            # Default scaling for other parameter types
            return torch.prod(torch.tensor(shape)).float() ** 0.5
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step with enhanced capabilities.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            ns_iters = group['ns_iters']
            momentum_factor = group['momentum']
            weight_decay = group['weight_decay']
            grad_clip_norm = group['grad_clip_norm']
            track_stats = group['track_stats']
            
            # Gradient clipping if enabled
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(group['params'], grad_clip_norm)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                state = self.state[p]
                
                # Initialize momentum buffer and other state if needed
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                    state['step'] = 0
                    state['update_history'] = [] if track_stats else None
                
                # Increment step count
                state['step'] += 1
                
                # Update momentum buffer
                momentum_buffer = state['momentum_buffer']
                momentum_buffer.mul_(momentum_factor).add_(grad, alpha=1 - momentum_factor)
                
                # Store gradient norm for tracking
                grad_norm = torch.norm(grad, p='fro').item()
                if track_stats:
                    self.global_stats['gradient_norms'].append(grad_norm)
                
                # Apply Muon update based on parameter type
                if len(p.shape) >= 2:  # For matrices and higher-dimensional tensors
                    # Reshape to matrix for higher dimensions
                    original_shape = p.shape
                    if len(p.shape) > 2:
                        p_flat = p.reshape(p.shape[0], -1)
                        momentum_flat = momentum_buffer.reshape(momentum_buffer.shape[0], -1)
                    else:
                        p_flat = p
                        momentum_flat = momentum_buffer
                    
                    # Apply Newton-Schulz orthogonalization
                    ortho_grad = self.newton_schulz_orthogonalize(momentum_flat, ns_iters)
                    
                    # Get dimension scaling
                    dim_scaling = self.get_dimension_scaling(original_shape)
                    
                    # Calculate update
                    buffer_norm = torch.norm(momentum_flat, p='fro')
                    if buffer_norm > 1e-8:
                        scaling = dim_scaling / buffer_norm
                        update = ortho_grad * scaling
                        
                        # Reshape back if needed
                        if len(p.shape) > 2:
                            update = update.reshape(original_shape)
                        
                        # Apply the update
                        p.add_(update, alpha=-lr)
                        
                        # Track update magnitude
                        if track_stats:
                            update_mag = torch.norm(update, p='fro').item() * lr
                            state['update_history'].append(update_mag)
                            self.global_stats['update_magnitudes'].append(update_mag)
                    
                else:
                    # For non-matrix parameters, use standard update
                    p.add_(momentum_buffer, alpha=-lr)
        
        return loss
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return optimization statistics for analysis.
        
        Returns:
            Dict[str, Any]: Dictionary containing tracked statistics
        """
        stats = {
            'global': self.global_stats,
            'parameters': {}
        }
        
        for group in self.param_groups:
            if group['track_stats']:
                for p in group['params']:
                    if p in self.state and 'update_history' in self.state[p]:
                        state = self.state[p]
                        stats['parameters'][id(p)] = {
                            'shape': p.shape,
                            'updates': state['update_history'],
                            'steps': state['step']
                        }
        
        return stats
