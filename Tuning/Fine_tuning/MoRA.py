## From https://arxiv.org/pdf/2405.12130v1
## From https://medium.com/gitconnected/mora-the-next-level-of-lora-for-superior-fine-tuning-in-llms-a8b136c87aef


"""
The increasing complexity and size of large language models (LLMs) have driven the need for efficient
fine-tuning methods that do not compromise performance while reducing computational and memory overhead. 
Traditional fine-tuning methods can be expensive, prompting the development of parameter-efficient fine-tuning (PEFT) methods
like Low-Rank Adaptation (LoRA) and now, Matrix-based Rank Adaptation (MoRA).

1. Limitations of LoRA
   -1. LoRA (Low-Rank Adaptation)
       - Advantages
         LoRA efficiently adapts LLMs by introducing low-rank updates to the model's weight matrices,
         which reduces the number of trainable parameters and thus computational cost.
       - Limitations
         While effective for many tasks, LoRA's low-rank updates can limit performance in memory-intensive tasks.
         These tasks require the model to store and retrieve substantial new information, where low-rank updates might not capture the complexity needed.

2. Introduction to MoRA (Matrix-based Rank Adaptation)
   -1. MoRA Overview
      - Core Concept
        MoRA builds on the foundation of LoRA by introducing high-rank updates using a square matrix. 
        This approach maintains parameter efficiency while enhancing the model's capacity for learning and memory retention.
   -2. Key Innovation
       Unlike LoRA's use of two low-rank matrices (A and B), MoRA employs a single square matrix (M) to facilitate high-rank updates,
       ensuring more comprehensive updates to the pretrained weight matrix (W0).

3. MoRA vs. LoRA: Key Differences
   | Feature |	LoRA	| MoRA | 
   |  Update Mechanism |	Low-rank matrices (A, B)	| Single square matrix (M) | 
   | Learning Capacity	| Limited for memory-intensive tasks	| Enhanced through high-rank updates | 
   | Parameter Efficiency	| High	| High | 
   | Computational Cost	| Low	| Comparable to LoRA, but with improved performance | 

4. Detailed Steps in MoRA
   -1. Compression Function (fcomp)
       - Reduces the input dimension from k to r.
       - Techniques: Truncation, sharing columns, decoupling, rotation.

   -2. Matrix Multiplication (M)
       - The compressed input is multiplied by the square matrix M (size r x r).
  
   -3. Decompression Function (fdecomp)
       - Increases the output dimension from r back to d.
       - Ensures that the high-rank updates are effectively applied.
   
   -4. Weight Update Integration
       - The updated weight matrix (ΔW) is integrated into the original model, enhancing learning capacity.

5. Compression and Decompression Strategies
   -1. Truncation
       Directly truncates the input dimension and appends zeros, which may lead to information loss.
   -2. Sharing Rows and Columns
       Preserves more information by sharing parts of the matrix, effective for larger ranks.
   -3. Decoupling
       For smaller ranks, decouples input vectors into smaller subvectors before applying M, mitigating information loss.
   -4. Rotation
       Uses rotation operators to enhance expressiveness and distinguish between input segments.

6. Experimental Results
   -1. Performance Evaluation
       - Instruction Tuning
         MoRA matches LoRA's performance, efficiently adapting models to specific instructions.
       - Mathematical Reasoning
         Significant improvement due to high-rank updates, enhancing complex reasoning abilities.
       - Continual Pretraining 
         Outperforms LoRA in tasks requiring retention of new domain-specific knowledge.

   2. Quantitative Analysis
      - Memory-Intensive Tasks
        MoRA demonstrates superior memory capabilities, as shown in experiments involving the memorization of Universally Unique Identifiers (UUIDs),
        achieving higher accuracy with fewer training steps.

7. Conclusion
   MoRA represents a significant advancement in parameter-efficient fine-tuning of LLMs, addressing the limitations of LoRA by introducing high-rank updates. 
   This innovation enhances the model's learning capacity, making it particularly effective for memory-intensive tasks without compromising on parameter efficiency.



"""
