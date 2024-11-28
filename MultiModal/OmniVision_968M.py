### From https://levelup.gitconnected.com/omnivision-968m-the-worlds-most-compact-and-smallest-multimodal-vision-language-model-for-edge-ai-4ccd66082bfb

"""
In the rapidly advancing realm of artificial intelligence—particularly multimodal AI—the emergence 
of OmniVision-968M represents a significant breakthrough. 
Designed by Nexa AI, this vision-language model builds upon the well-regarded LLaVA (Large Language and Vision Assistant) 
framework while introducing critical enhancements.
Its compact and efficient design addresses the unique challenges of edge computing, 
making it a key enabler for resource-constrained applications.

1. The Need for Compact Vision-Language Models in Edge AI
   Edge computing—processing data directly on devices like IoT sensors, smartphones, 
   and cameras—presents challenges such as:

   -1. Limited computational resources.
   -2. Latency sensitivity.
   -3. Stringent power constraints.
  
   Traditional multimodal models, while powerful, demand substantial computational power, 
   making them unsuitable for edge deployment. OmniVision-968M bridges this gap by:

   -1. Reducing model size.
   -2. Optimizing token usage and computation.
   -3. Retaining state-of-the-art performance, even in constrained environments.

   This model is ideal for autonomous vehicles, smart homes, AR/VR systems, and mobile assistants, 
   where high performance and efficiency are paramount.

2. Key Innovations in OmniVision-968M
   -1. Efficient Token Compression: 9x Reduction
       -a. Challenge: Models like LLaVA generate large numbers of tokens for visual data (e.g., 729 tokens for a 27×27 image grid).
       -b. Solution: OmniVision reduces tokens to 81 using a novel projection layer that:
           - Reshapes embeddings from [batch_size, 729, hidden_size] to [batch_size, 81, hidden_size*9].
           - Achieves a ninefold reduction in tokens, drastically cutting latency and computational costs.
       -c. Outcome: Faster processing and lower resource usage without compromising information integrity.
   -2. Enhanced Accuracy with Direct Preference Optimization (DPO)
       -a. Challenge: Multimodal models often hallucinate or generate irrelevant responses.
       -b. Solution: OmniVision employs minimal-edit DPO, which:
           - Trains the model using teacher-generated corrected outputs with paired chosen-rejected samples.
           - Ensures high accuracy while maintaining the model’s natural response style.
       -c. Outcome: Robust, reliable outputs, critical for real-time edge applications.

3. State-of-the-Art Architecture
   OmniVision combines three cutting-edge components:

   -a. Base Language Model: Qwen2.5–0.5B-Instruct, optimized for text processing efficiency.
   -b. Vision Encoder: SigLIP-400M, featuring a 14×14 patch size at 384 resolution for detailed embeddings.
   -c. Projection Layer: A Multi-Layer Perceptron (MLP) that aligns visual and textual inputs seamlessly.

4. Training Methodology
   The model undergoes a rigorous three-stage training pipeline:

   -a. Pretraining:
       - Uses a large corpus of image-caption pairs to establish visual-linguistic alignment.
       - Fine-tunes only the projection layer, ensuring efficient specialization without altering the core language model.
   -b. Supervised Fine-Tuning (SFT):
       - Refines the model with image-based question-answering tasks, enhancing its ability to handle real-world scenarios.
   -c. Direct Preference Optimization (DPO):
       - Generates and optimizes responses to image prompts using teacher-guided minimal corrections.
       - Focuses on enhancing accuracy without overhauling natural output tendencies.

5. Benchmark Performance
   OmniVision-968M consistently outperforms competing models like nanoLLAVA and Qwen2-VL-2B across multiple benchmarks.
   Key highlights include:

   -1. Superior accuracy in multimodal tasks.
   -2. Remarkable token efficiency for edge deployment.
   -3. Faster response times due to compressed token sequences.

6. Optimized for Edge Deployment
   -1. Token Compression for Efficiency
       By reducing token counts, OmniVision enables:

      -a. Faster processing on resource-constrained devices.
      -b. Lower power consumption, critical for edge scenarios.

   -2. Minimal-Edit DPO
      This innovative training approach:

      -a. Ensures contextually relevant outputs.
      -b. Enhances real-time interactions for applications like:
          - Virtual assistants.
          - AR/VR systems.
          - Mobile apps.
"""

# Install core packages
pip install torch torchvision torchaudio einops timm pillow
pip install git+https://github.com/huggingface/transformers
pip install git+https://github.com/huggingface/accelerate
pip install git+https://github.com/huggingface/diffusers
pip install huggingface_hub
pip install sentencepiece bitsandbytes protobuf record

# Install Nexa SDK with GPU support
CMAKE_ARGS="-DGGML_CUDA=ON -DSD_CUBLAS=ON" pip install nexaai --prefer-binary \
  --index-url https://nexaai.github.io/nexa-sdk/whl/cu124 \
  --extra-index-url https://pypi.org/simple --no-cache-dir

nexa run omnivision
nexa run omnivision -st



