### From https://levelup.gitconnected.com/quantization-aware-training-with-pytorch-38d0bdb0f873

"""
* Introduction & Motivation
  Neural networks at edge deployment must be both small and accurate—a difficult balance once 
  you’ve exhausted architectural tricks, multi-layer fusion, and compiler optimizations. 
  Three broad strategies prevail for shrinking models without sacrificing much accuracy:
  -a. Quantization
  -b. Pruning
  -c. Knowledge Distillation
  This summary focuses on quantization: what it is, why it helps, the two main workflows (PTQ vs. QAT), 
  and how to implement QAT in PyTorch—including its underlying math and gradient tricks.

1. What Is Quantization?
   Quantization converts network parameters and activations from higher-precision floats 
   (e.g. FP32 or FP16) into lower-precision integers (e.g. INT8 or even INT4).
   -a. Why?
       -1. Compute efficiency: 8-bit integer arithmetic uses cheaper, higher-throughput hardware units 
                               (e.g. NVIDIA Tensor Cores).
       -2. Bandwidth reduction: Layers that are memory-bound benefit most, as moving 2× less data cuts memory stalls.
       -3. Smaller footprint: Less storage, smaller updates, better cache utilization.
       -4. Energy savings: Moving fewer bits from memory to compute units saves power.

2. High-Level Quantization Techniques
   -a. Zero-point / Affine quantization
   -b. Absmax quantization, etc.
   But in practice, you choose between two workflows:

   2.1 Post-Training Quantization (PTQ)
       Apply quantization after finishing training, using a small calibration set to collect activation statistics
       and choose quantization parameters.
       -a. Pros: Fast to deploy, no retraining needed—ideal for quick prototyping.
       -b. Cons: Often lower accuracy; demands careful calibration.
       
       Two flavors:
       -a. Dynamic PTQ
           -1. Compute activation ranges on the fly at inference time, adapting to data distribution as you go.
       -b. Static PTQ
           -1. Pre-compute activation ranges from a representative calibration dataset.
           -2. Run full-precision forward passes to measure min/max values, then fix quantization scales 
               before deployment.
               
   2.2 Quantization-Aware Training (QAT)
       Simulate lower-precision arithmetic during training by inserting “fake quantization” operations
       into the forward pass. The model learns to compensate for quantization noise.
       -a. Workflow in PyTorch
           -1. Preparation
               -1) Wrap sensitive layers (Conv, Linear, Activations) with simulated quantizers via prepare_qat
                   or prepare_qat_fx.
           -2. Training
               -1) Forward pass applies “fake” INT8 rounding and clamping to weights/activations.
               -2) Backward pass uses the Straight-Through Estimator (STE) so that gradients ignore the 
                   non-differentiable rounding step.
           -3. Conversion
               -1) Replace fake-quant modules with real INT8 kernels using convert or convert_fx.
               -2) Result: a true quantized model ready for highly efficient inference.
       -b. Pros: Highest accuracy among quantized models—model adapts during training.
       -c. Cons: Requires retraining, more compute, and more complex setup.

3. Mathematics of Fake Quantization
   Uniform affine quantization maps a float 𝑥_float to an integer representation via:
   -a. scale=(𝑥_max−𝑥_min)/(𝑞_max−𝑞_min),
   -b. zeroPt=round(𝑞_min−(𝑥_min/scale)),
   -c. 𝑥_𝑞=clamp(round(𝑥_float/scale)+zeroPt,𝑞_min,𝑞_max),
   -d. 𝑥_deq=(𝑥_𝑞−zeroPt)×scale

   Fake quantization during QAT uses the same rounding/clamping but keeps values in floating point:
   𝑥_fake=(round(𝑥_float/scale)+zeroPt−zeroPt)×scale

4. Gradient Flow & the Straight-Through Estimator
   -a. Rounding is non-differentiable.
   -b. QAT treats the fake-quant op as identity during backprop:
       ∂𝐿/∂𝑥_float ≈ ∂𝐿/∂𝑥_fake,
       letting gradients flow through unimpeded so the model adjusts weights to naturally align with quantized values.

5. Key Takeaways
   -a. Quantization is the most straightforward way to shrink models for edge devices, offering compute, memory, 
       and energy savings.
   -b. PTQ is quick but less precise; QAT embeds quantization into training for maximal accuracy.
   -c. Fake quantization math plus the STE enable end-to-end training under quantization constraints.
   -d. PyTorch’s prepare_qat/convert API makes implementing QAT a structured three-step process: prepare, train, convert.
"""

### -1. Eager Mode Quantization
import os, torch, torch.nn as nn, torch.optim as optim

# 1. Model definition with QuantStub/DeQuantStub
class QATCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant   = torch.quantization.QuantStub()
        self.conv1   = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1   = nn.ReLU()
        self.pool    = nn.MaxPool2d(2)
        self.conv2   = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2   = nn.ReLU()
        self.fc      = nn.Linear(32*14*14, 10)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.relu2(self.conv2(x))
        x = x.flatten(1)
        x = self.fc(x)
        return self.dequant(x)

# 2. QAT preparation
model = QATCNN()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)

# 3. Tiny training loop
opt = optim.SGD(model.parameters(), lr=1e-2)
crit = nn.CrossEntropyLoss()
for _ in range(3):
    inp = torch.randn(16,1,28,28)
    tgt = torch.randint(0,10,(16,))
    opt.zero_grad(); crit(model(inp), tgt).backward(); opt.step()

# 4. Convert to real int8
model.eval()
int8_model = torch.quantization.convert(model)

# 5. Storage benefit
torch.save(model.state_dict(), "fp32.pth")
torch.save(int8_model.state_dict(), "int8.pth")
mb = lambda p: os.path.getsize(p)/1e6
print(f"FP32: {mb('fp32.pth'):.2f} MB  vs  INT8: {mb('int8.pth'):.2f} MB")
----------------------------------------------------------------------------------------
### -2. FX Graph Mode Quantization
import torch, torchvision.models as models
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization import prepare_qat_fx, convert_fx

model = models.resnet18(weights=None)     # or pretrained=True
model.train()

# 1-liner qconfig mapping
qmap = get_default_qat_qconfig_mapping("fbgemm")
# Graph rewrite
model_prepared = prepare_qat_fx(model, qmap)

# Fine-tune for a few epochs
model_prepared.eval()
int8_resnet = convert_fx(model_prepared)
-----------------------------------------------------------------------------------------
### -3. PyTorch 2 Export Quantization
import torch
from torch import nn
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
    prepare_qat_pt2e, convert_pt2e)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer, get_symmetric_quantization_config)

class Tiny(nn.Module):
    def __init__(self): super().__init__(); self.fc=nn.Linear(8,4)
    def forward(self,x): return self.fc(x)

ex_in = (torch.randn(2,8),)
exported = torch.export.export_for_training(Tiny(), ex_in).module()
quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
qat_mod = prepare_qat_pt2e(exported, quantizer)

# Fine-tune the model ...
int8_mod = convert_pt2e(qat_mod)
torch.ao.quantization.move_exported_model_to_eval(int8_mod)
---------------------------------------------------------------------------------------------
### -4. Large-Language-Model Int4/Int8 Hybrid Demo
import torch
from torchtune.models.llama3 import llama3
from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer

model = llama3(vocab_size=4096, num_layers=16,
               num_heads=16, num_kv_heads=4,
               embed_dim=2048, max_seq_len=2048).cuda()

qat_quant = Int8DynActInt4WeightQATQuantizer()
model = qat_quant.prepare(model).train()

#  ––– Kathy-like micro-fine-tune –––
optim = torch.optim.AdamW(model.parameters(), 1e-4)
lossf = torch.nn.CrossEntropyLoss()
for _ in range(100):
    ids   = torch.randint(0,4096,(2,128)).cuda()
    label = torch.randint(0,4096,(2,128)).cuda()
    loss  = lossf(model(ids), label)
    optim.zero_grad(); loss.backward(); optim.step()

model_quant = qat_quant.convert(model)
torch.save(model_quant.state_dict(),"llama3_int4int8.pth")
