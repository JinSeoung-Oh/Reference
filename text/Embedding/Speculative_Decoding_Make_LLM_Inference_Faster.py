"""
From https://medium.com/ai-science/speculative-decoding-make-llm-inference-faster-c004501af120

# Understanding Speculative Decoding for LLM Inference Speedup
  Speculative Decoding is a method to accelerate the inference process of large language models (LLMs) by leveraging a smaller, 
  auxiliary model alongside the main, larger model. 
  This technique can significantly reduce the time required for generating text without compromising the accuracy of the output.

1. Basic Concept
   -1. Speculative Decoding involves using two models
       -1) Target Model (Main LLM): This is the primary, large model used for generating high-quality text.
       -2) Small Draft Model: A smaller and faster model that generates initial predictions or "draft tokens."

2. Process
   -1. Drafting Phase
       The small draft model generates a sequence of potential tokens in parallel.
       These tokens are speculative and are generated quickly due to the efficiency of the smaller model.

   -2. Verification Phase
       The target model then verifies the tokens produced by the draft model.
       If the speculative tokens are correct, they are accepted, which speeds up the overall process.
       If the tokens are incorrect or not optimal, the target model corrects them.

3. How It Speeds Up Inference
   -1. Parallel Processing
       Unlike traditional autoregressive methods where each token is generated sequentially, speculative decoding allows for multiple tokens to be processed in parallel.
       This reduces the number of sequential steps required.
   -2. Efficiency
       The smaller draft model can quickly generate easy-to-predict tokens, which the larger model only needs to verify, rather than generating all tokens from scratch.
       This division of labor reduces the computational load on the larger model.
   -3. Token Acceptance Rate (TAR)
       The effectiveness of speculative decoding heavily depends on the draft model's ability to generate tokens that the target model will accept.
       A high TAR means that fewer corrections are needed, leading to greater speedups.

In practice, speculative decoding has shown to improve throughput (tokens generated per second) significantly. For instance, when using a smaller draft model (e.g., OPT-125M)
with a larger target model (e.g., OPT-66B)

4. Key Insights
   -1. Selection of Draft Model
       The draft model should be significantly smaller and efficient enough to provide a high TAR without adding significant latency.
   -2. Verification Overhead
       The target model’s role in verification ensures that the final output quality remains high, despite the speculative nature of the initial token generation.

By using speculative decoding, it's possible to achieve substantial speedups in LLM inference, making real-time applications more feasible without sacrificing the quality of the generated text.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

prompt = "Alice and Bob"
checkpoint = "EleutherAI/pythia-1.4b-deduped"
assistant_checkpoint = "EleutherAI/pythia-160m-deduped"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint).to(device)
outputs = model.generate(**inputs, assistant_model=assistant_model)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
# ['Alice and Bob are sitting in a bar. Alice is drinking a beer and Bob is drinking a']





