### From https://medium.com/@zilliz_learn/matryoshka-representation-learning-explained-the-method-behind-openais-efficient-text-embeddings-a600dfe85ff8

"""
1. Motivation for MRL
   -a. Challenge: Trade-offs between cost and performance during training and inference.
       -1. Training: Larger models and datasets improve capability but at higher computational and time costs.
       -2. Inference: Larger models generate high-dimensional embeddings that require more memory and increase retrieval complexity.
   -b. Example: Using BERT for information retrieval:
       -1. Base model (768 dimensions): 30.72 GB for 10M embeddings.
       -2. Large model (1024 dimensions): 40.96 GB for 10M embeddings.
       -3. Larger embeddings improve retrieval precision but at the cost of efficiency.

2. Introduction to Matryoshka Representation Learning (MRL)
   -a. Concept: Inspired by Matryoshka dolls, MRL allows models to generate embeddings of various sizes in a single forward pass.
   -b. Benefits:
       -1. Flexibility to adapt embedding dimensions based on task requirements.
       -2. Efficient trade-offs between computational costs and model performance.
   -c. Applications: Semantic search, information retrieval, multilingual processing, etc.

3. How MRL Works
   -a. Embedding Dimensions:
       -1. MRL-trained models can generate embeddings of different sizes, e.g., 16, 32, 64, 128, and 1024 dimensions.
       -2. Each dimension level optimizes a specific loss function during training.
   -b. Training with MRL:
       The overall loss function is the sum of individual loss functions for each defined dimension:
        𝐿_MRL = (∑ 𝑖=1 to 𝑛) 𝐿_𝑖

        -1. Models can produce smaller or larger embeddings without retraining.
        -2. The first dimensions carry more significant information, while later dimensions add granularity.
        
    -c. Optimization:
        -1. MRL is model-agnostic and can be applied to any architecture, including fine-tuning pre-trained models like BERT.
        -2. Shorter embeddings are not truncated versions of longer embeddings but are specifically optimized during training.

4. Experimental Results and Applications
   -a. Classification
       -1. ResNet50 on ImageNet-1K:
           - Comparable top-1 accuracy between MRL-trained and fixed-size models.
           - 1-Nearest Neighbor (1-NN): Up to 2% improvement in accuracy for smaller feature sizes.
       -2. ViT on JFT-300M:
           - Performance matches fixed-size models across representation sizes.
           - Better results at lower dimensions due to optimized features.
       -3. Adaptive Classification:
           - MRL-trained ResNet50 achieves comparable accuracy to fixed-size models with significantly reduced feature sizes.

   -b. Retrieval
       -1. Retrieval Quality:
           - mAP improvement of up to 3% for MRL-trained models over fixed-size models.
           - Adaptive retrieval uses smaller embeddings for shortlisting and larger embeddings for reranking:
             Example:
             ImageNet-1K: 16 dimensions for shortlisting, 2048 dimensions for reranking.
             Theoretical speedup: 128x.
             Real-world speedup: 14x with HNSW.

           - ImageNet-4K: Similar results with 64 dimensions for shortlisting.

      -2. Speed-Accuracy Trade-off: 
          -1. Adaptive retrieval achieves better efficiency without significant loss in mAP.

5. Key Advantages of MRL
   -1. Flexibility:
       - Models trained with MRL can adapt to various embedding sizes without additional retraining.
   -2. Efficiency:
       - Reduces computational and memory overhead during inference.
       - Speeds up retrieval tasks significantly.
   -3. Scalability:
       - Applicable to models across different modalities (text, vision, vision-text).
   -4. Performance:
       - Comparable or better accuracy than fixed-size models in both classification and retrieval tasks.

6. Use Cases and Future Potential
   -1. Information Retrieval:
       Efficient embedding generation for scalable vector databases.
   -2. Semantic Search:
       Adaptive embeddings tailored to task requirements.
   -3. Multimodal Models:
       Fine-tuning vision-text models for tasks requiring nuanced representations.
   -4. General ML Models:
       Application to downstream tasks with varying dimensional requirements.

In summary, MRL offers a powerful and flexible approach to train machine learning models capable of producing multi-scale embeddings,
enabling efficient trade-offs between cost and performance across a wide range of applications.
"""

from sentence_transformers import SentenceTransformer

matryoshka_dim_short = 64
matryoshka_dim_long = 768
text = ["The weather is so nice!"]

short_embedding = SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka", truncate_dim=matryoshka_dim_short).encode(text)
long_embedding = SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka", truncate_dim=matryoshka_dim_long).encode(text)

print(f"Shape: {short_embedding.shape, long_embedding.shape}")
print(short_embedding[0][0:10])
print(long_embedding[0][0:10])

from sentence_transformers.util import cos_sim

similarities = cos_sim(short_embedding[0], long_embedding[0][:matryoshka_dim_short])
print(similarities)
# tensor([[1.]])
