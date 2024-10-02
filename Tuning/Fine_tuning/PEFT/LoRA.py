## From https://lightning.ai/lightning-ai/studios/code-lora-from-scratch

## sudo
input_dim = 768  # e.g., the hidden size of the pre-trained model
output_dim = 768  # e.g., the output size of the layer
rank = 8  # The rank 'r' for the low-rank adaptation

W = ... # from pretrained network with shape input_dim x output_dim

W_A = nn.Parameter(torch.empty(input_dim, rank)) # LoRA weight A
W_B = nn.Parameter(torch.empty(rank, output_dim)) # LoRA weight B

# Initialization of LoRA weights
nn.init.kaiming_uniform_(W_A, a=math.sqrt(5))
nn.init.zeros_(W_B)

def regular_forward_matmul(x, W):
    h = x @ W
return h

def lora_forward_matmul(x, W, W_A, W_B):
    h = x @ W  # regular matrix multiplication
    h += x @ (W_A @ W_B)*alpha # use scaled LoRA weights
return h


## coding LoRA from Scratch
import torch
import torch.nn.Functional as F

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

# Note that LoRA is usually applied to a neural network's linear (feedforward) layers. 
# For example, suppose we have a simple PyTorch model or module with two linear layers (e.g., this could 
# be the feedforward module of a transformer block). And suppose this module's forward method looks like as follows:
def forward(self, x):
    x = self.linear_1(x)
    x = F.relu(x)
    x = self.linear_2(x)
    return x
# If we use LoRA, we'd add the LoRA updates to these linear layer outputs as follows:
def forward(self, x):
    x = self.linear_1(x) + self.lora_1(x)
    x = F.relu(x)
    x = self.linear_2(x) + self.lora_2(x)
    return logits
## So, In code, when implementing LoRA by modifying existing PyTorch models, an easy way to implement 
## such a LoRA-modification of the linear layers is to replace each Linear layer with a LinearWithLoRA layer that combines 
## the Linear layer with our previous LoRALayer implementation:

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


## Example of LoRA
from transformers import AutoModelForSequenceClassification


model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)

for param in model.parameters():
    param.requires_grad = False

# At first, we have to find out all Linear layer using print(model)
# Find all Linear layer in pre-trained model, then

from functools import partial

# default hyperparameter choices <-- to get more good result, we have to optimize this part. (find out efficient method for this)
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False

layers = []

assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha)

for layer in model.distilbert.transformer.layer:
    if lora_query:
        layer.attention.q_lin = assign_lora(layer.attention.q_lin)
    if lora_key:
        layer.attention.k_lin = assign_lora(layer.attention.k_lin)
    if lora_value:
        layer.attention.v_lin = assign_lora(layer.attention.v_lin)
    if lora_projection:
        layer.attention.out_lin = assign_lora(layer.attention.out_lin)
    if lora_mlp:
        layer.ffn.lin1 = assign_lora(layer.ffn.lin1)
        layer.ffn.lin2 = assign_lora(layer.ffn.lin2)
if lora_head:
    model.pre_classifier = assign_lora(model.pre_classifier)
    model.classifier = assign_lora(model.classifier)
