## https://medium.com/@tamangmilan/want-to-learn-quantization-in-the-large-language-model-57f062d2ec17
"""
1. What is Quantization?
   Quantization is a technique used to reduce the size of a large machine learning model by compressing its weight parameters and activations. 
   This is achieved by converting these values from a higher precision (e.g., FP32) to a lower precision (e.g., INT8, FP16).

   For example, a base model like Llama 3 8B, which is originally 32GB in size, can be reduced to 8GB using INT8 quantization (a 75% reduction) 
   and even further to 4GB using INT4 quantization (a 90% reduction).

2. Why Quantization?
   Quantization is essential for enabling model fine-tuning and inference on devices with limited memory and processing power, 
   such as personal computers or mobile devices. Without quantization, these tasks often require expensive cloud resources. 
   Quantization helps reduce the model size while maintaining similar accuracy, making it feasible to perform complex AI tasks on less powerful hardware.

3. How Does Quantization Work?
   Quantization maps the model's weight values from higher precision to lower precision using linear quantization methods, 
   which can be either asymmetric or symmetric.

   -1. Asymmetric Linear Quantization
     In asymmetric quantization, values from the original tensor range (𝑊_𝑚𝑖𝑛,𝑊_𝑚𝑎𝑥)are mapped 
     to the quantized tensor range (𝑄_𝑚𝑖𝑛,𝑄_𝑚𝑎𝑥).

     Key Components
      - Scale Value (S): Scales down the values of the original tensor to the quantized tensor range.
      - Zero Point (Z): A non-zero value in the quantized tensor range that maps to zero in the original tensor range.

     Quantization Formula
      - 𝑄 = clamp(round(𝑊/𝑆 + 𝑍),𝑄_𝑚𝑖𝑛,𝑄_𝑚𝑎𝑥)

     De-Quantization Formula
      - 𝑊 = 𝑆×(𝑄−𝑍)

   -2. B. Symmetric Linear Quantization
       In symmetric quantization, zero in the original tensor maps to zero in the quantized tensor.
       The mapping happens between [−𝑊_𝑚𝑎𝑥,𝑊_𝑚𝑎𝑥] of the original tensor range to [−𝑄_𝑚𝑎𝑥,𝑄_𝑚𝑎𝑥]
       of the quantized tensor range. There is no zero point in symmetric quantization.

4. Quantization and De-Quantization:
   The quantization and de-quantization processes are simpler as zero is mapped directly to zero without needing a zero point.
"""
#Coding Example in PyTorch
#Let's walk through the steps of implementing asymmetric quantization in PyTorch.

## Step 1: Assign Random Values to the Original Weight Tensor

import torch

original_weight = torch.randn((4, 4), dtype=torch.float32)
print(original_weight)

## Step 2: Define Functions for Quantization and De-Quantization

def asymmetric_quantization(original_weight):
    quantized_data_type = torch.int8
    Wmax = original_weight.max().item()
    Wmin = original_weight.min().item()
    Qmax = torch.iinfo(quantized_data_type).max
    Qmin = torch.iinfo(quantized_data_type).min

    S = (Wmax - Wmin) / (Qmax - Qmin)
    Z = Qmin - (Wmin / S)
    
    if Z < Qmin:
        Z = Qmin
    elif Z > Qmax:
        Z = Qmax
    else:
        Z = int(round(Z))
    
    quantized_weight = (original_weight / S) + Z
    quantized_weight = torch.clamp(torch.round(quantized_weight), Qmin, Qmax)
    quantized_weight = quantized_weight.to(quantized_data_type)
    
    return quantized_weight, S, Z

def asymmetric_dequantization(quantized_weight, scale, zero_point):
    dequantized_weight = scale * (quantized_weight.to(torch.float32) - zero_point)
    return dequantized_weight

## Step 3: Calculate Quantized Weight, Scale, and Zero Point

quantized_weight, scale, zero_point = asymmetric_quantization(original_weight)
print(f"Quantized weight: {quantized_weight}\n")
print(f"Scale: {scale}\n")
print(f"Zero point: {zero_point}\n")

## Step 4: De-Quantize the Quantized Weight

dequantized_weight = asymmetric_dequantization(quantized_weight, scale, zero_point)
print(f"Dequantized weight: {dequantized_weight}")

## Step 5: Calculate the Quantization Error

quantization_error = (dequantized_weight - original_weight).square().mean()
print(f"Quantization error: {quantization_error}")

### Symmetric Quantization Code
#   For symmetric quantization, the key difference is that the zero point is always zero. 
#   The quantization and de-quantization process remains similar, but without the need for a zero point.

def symmetric_quantization(original_weight):
    quantized_data_type = torch.int8
    Wmax = original_weight.abs().max().item()
    Qmax = torch.iinfo(quantized_data_type).max
    Qmin = torch.iinfo(quantized_data_type).min

    S = Wmax / max(Qmax, -Qmin)
    quantized_weight = original_weight / S
    quantized_weight = torch.clamp(torch.round(quantized_weight), Qmin, Qmax)
    quantized_weight = quantized_weight.to(quantized_data_type)
    
    return quantized_weight, S

def symmetric_dequantization(quantized_weight, scale):
    dequantized_weight = scale * quantized_weight.to(torch.float32)
    return dequantized_weight

# Example usage
quantized_weight_sym, scale_sym = symmetric_quantization(original_weight)
dequantized_weight_sym = symmetric_dequantization(quantized_weight_sym, scale_sym)

quantization_error_sym = (dequantized_weight_sym - original_weight).square().mean()
print(f"Symmetric quantization error: {quantization_error_sym}")

# Quantization techniques are powerful tools for reducing model sizes and enabling more efficient use of resources 
# without significant loss in accuracy. By understanding and implementing these techniques,
# one can deploy large models in resource-constrained environments effectively.
