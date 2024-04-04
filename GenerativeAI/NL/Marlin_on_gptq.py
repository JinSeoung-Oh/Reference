## From https://towardsdatascience.com/marlin-nearly-ideal-inference-speed-for-4-bit-large-language-models-feb0b610dd8e
"""
Large language models (LLMs) are often too large for consumer hardware use,
prompting the need for size reduction techniques like quantization to lower memory consumption. 
Despite recent advancements in 4-bit quantization algorithms and optimized CUDA kernels, quantized LLMs still lack optimal inference throughput.
In particular, inference with 4-bit models, utilizing INT4 data type, involves slow INT4xFP16 operations, 
necessitating optimized CUDA kernels. The Institute of Science and Technology Austria (ISTA) proposes Marlin, an optimized INT4xFP16 matmul kernel,
to achieve close to ideal (4x) inference speed. Marlin maximizes GPU usage for INT4 LLMs by efficiently utilizing GPU capabilities, 
including memory systems and cores, with optimizations such as efficient data fetching from L2 cache, double buffering, 
and strategic order of dequantization and computation during inference. Moreover, Marlin introduces optimizations for multi-GPU settings, 
enabling increased parallel processing without loading more data at once, resulting in nearly optimal GPU resource utilization.
Remarkably, even with a batch size of 1, 
Marlin outperforms existing frameworks like ExLlamaV2 and AWQ, while at a batch size of 8, 
these frameworks are slower than FP16 inference, while Marlin remains almost 4 times faster.
"""

! pip install --upgrade transformers auto-gptq accelerate optimum

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

GPTQ_MODEL = "kaitchup/Mistral-7B-v0.1-gptq-4bit"
marlin_model = AutoGPTQForCausalLM.from_quantized(
      GPTQ_MODEL,
      use_marlin=True,
      device_map='auto')

save_dir = "Mistral-7B-v0.1-gptq-marlin-4bit"
marlin_model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)


"""
Marlin is indeed faster but vLLM only benefits from it for batch sizes larger than 8.
The gap between Marlin and vanilla GPTQ increases with larger batch sizes.
(not related to Marlin but interesting) vLLM is already extremely well-optimized for decoding without batching (batch size = 1). 
Decoding with a batch size of 2 is 2x slower than without batching.
If you only need small batch sizes, then it might not be worth converting your models to Marlin, yet
."""

