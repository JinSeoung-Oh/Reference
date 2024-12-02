### From https://pub.towardsai.net/multimodal-rag-unveiled-a-deep-dive-into-cutting-edge-advancements-0eeb514c3ac4

"""
1. Classification of Multimodal RAG
   Multimodal RAG can be classified based on the storage and retrieval modality:

   -1. Image-based RAG: Focuses on using image embeddings for retrieval, useful for tasks requiring visual information.
       -a. Example: Handling single-page documents or long documents retaining their visual structure.
   -2. Text-based RAG: Relies on textual embeddings for retrieval, excluding visual context.
   -3. Cross-modality RAG: Integrates text and image modalities into a single framework.
       -a. Two Approaches:
           - Separate Vector Stores: Text and image embeddings are stored separately.
           - Unified Vector Store: Textual data and image-generated text summaries share the same vector space.

   - Text-only retrieval employs text embedding models (e.g., OpenAI's text-embedding-3-small).
   - Image-only retrieval uses models like CLIP for embeddings.
   - Cross-modality RAG pipelines utilize a combination of text and image processing components.

2. Highlighted Implementations
   -1. ColPali
       -a. Open Source: GitHub
       -b. Core Idea: ColPali enhances document retrieval by directly processing document images using Vision-Language Models (VLMs). 
                      It generates high-quality contextual embeddings, avoiding traditional text extraction and layout analysis.

       -c. Key Features:
           - Direct Embedding: Converts document pages into embeddings without preprocessing.
           - Late Interaction Mechanism: Aggregates similarity scores from multiple vectors to determine document relevance.
           - Dataset: Trained on 127,460 query-page pairs from academic datasets and synthetic web-crawled PDFs.
           - Performance:
             1) Latency: ~30ms per query using NVIDIA L4 GPUs.
             2) Memory: ~256KB per page after dimensionality reduction.
           
       -d. Challenges:
           - Real-time latency (~30ms) may not suffice for applications like autonomous driving.
           - Limited accuracy with unstructured or handwritten documents.
"""
## Example code

def process_pdfs_with_colpali(pdf_files, output_dir, model, processor):
    all_embeddings = []
    all_page_info = []

    for pdf_file in pdf_files:
        pdf_images = convert_pdf_to_images(pdf_file, os.path.join(output_dir, "pdf_images"))

        for page_num, image in enumerate(pdf_images):
            image_input = processor.process_image(image).to(model.device)
            with torch.no_grad():
                page_embedding = model(**vars(image_input))

            if len(page_embedding.shape) == 3:
                page_embedding = page_embedding.mean(dim=1)

            all_embeddings.append(page_embedding.cpu().numpy().squeeze())
            all_page_info.append({"pdf": pdf_file, "page": page_num})

    embeddings_array = np.array(all_embeddings)
    np.save(Path(output_dir) / "embeddings.npy", embeddings_array)
    np.save(Path(output_dir) / "page_info.npy", all_page_info)

    return embeddings_array, all_page_info

## Query Answering:
def answer_query_with_colpali(query, embeddings_array, page_info, model, processor):
    query_input = processor.process_text(query).to(model.device)
    with torch.no_grad():
        query_embedding = model(**vars(query_input))

    if len(embeddings_array.shape) == 3:
        embeddings_array = embeddings_array.mean(axis=1)
    if len(query_embedding.shape) == 3:
        query_embedding = query_embedding.mean(axis=1)

    embeddings_array = embeddings_array.squeeze()
    query_embedding = query_embedding.cpu().numpy().squeeze()

    similarity_scores = np.dot(embeddings_array, query_embedding.T)
    top_k_indices = np.argsort(similarity_scores.flatten())[-5:][::-1]

    top_results = [
        {"score": similarity_scores.flatten()[i], "info": page_info[i]}
        for i in top_k_indices
    ]
    return top_results
"""
   -2. M3DOCRAG
      -a. Core Idea: M3DOCRAG expands multimodal RAG capabilities by integrating text and image data to handle cross-page, 
                     multi-document scenarios.

      -b. Pipeline:
          - Document Embedding: Extracts embeddings for each page using ColPali.
          - Page Retrieval: Finds top-K relevant pages using Faiss for similarity search.
          - Question Answering: A multimodal LLM (e.g., Qwen2-VL) generates answers based on retrieved pages.

      -c.Challenges:
         - Limited by the scalability of retrieval systems.
         - Current models struggle with multilingual and complex visual semantics.

   -3. VisRAG
       -a. Open Source: GitHub
       -b. Core Idea: VisRAG processes document images directly using VLMs, eliminating text parsing while retaining comprehensive visual 
                      and textual information.
       -c. Key Features:
           - Dual-Encoder Paradigm: Maps queries and document images to a shared embedding space.
           - Single/Multi-Image Generation: Combines retrieved pages using concatenation or weighted selection.
       -d. Challenges:
           - High computational resource demands.
           - Dependency on large-scale pretraining datasets and models.
           
   -4. OmniSearch
       -a. Open Source: GitHub 
       -b. Core Idea: OmniSearch introduces dynamic retrieval by breaking queries into sub-questions and adjusting retrieval actions iteratively.
       -c. Framework:
           - Planning Agent: Decomposes queries into actionable steps.
           - Retriever: Executes multimodal retrievals (text, image).
           - Sub-Question Solver: Summarizes content and integrates it into answers.

       -d. Strengths:
           - Adaptive retrieval paths.
           - Avoids error propagation through iterative refinement.

       -e. Challenges:
           - High token and computational costs.
           - Generalizability across varied domains is still limited.
"""
## Example prompt
sys_prompt_1 = '''
You are a helpful multimodal question answering assistant. Decompose the original question into sub-questions and solve them step by step.

<Thought>
Analyze questions and answer sub-questions, think about what is next.
<Sub-Question>
Sub-question solved in one step.
<Search>
Method: Image Retrieval with Input Image. Text Retrieval: xxx. Image Retrieval with Text Query: xxx.

<End>
Final Answer: Provide the precise answer.
'''

"""
3. Comparison of Techniques
   Technique	Strengths	Limitations
   
   -a. ColPali	High accuracy, efficient embeddings	Latency issues, struggles with unstructured data
   -b. M3DOCRAG	Multi-page/document support, cross-modality integration	Scaling and multilingual challenges
   -c. VisRAG	Preserves all visual/textual info, OCR-free	Resource-intensive, high dependency on large models/datasets
   -d. OmniSearch	Dynamic query decomposition, iterative retrieval	Token/computation costs, requires significant fine-tuning

4. Future Directions
   -a. Dynamic Retrieval: Adopting adaptive pipelines like OmniSearch.
   -b. Multimodal Alignment: Enhancing cross-modal embeddings for semantic consistency.
   -c. Real-Time Systems: Reducing latency for applications in real-time environments.
   -d. Integration with Agents: Using agents for modular and interactive retrieval strategies.
   -e. Productization: Addressing scalability and hardware efficiency for deployment.

This article sets the foundation for future research in multimodal RAG and underscores the need for tailored approaches 
to tackle specific challenges.
"""
