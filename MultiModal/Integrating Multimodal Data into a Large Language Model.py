### From https://towardsdatascience.com/integrating-multimodal-data-into-a-large-language-model-d1965b8ab00c

!pip install llama-index ipython cohere rank-bm25 pydantic nest-asyncio python-dotenv openai llama-parse

from llama_parse import LlamaParse
import os

###### Function to read all files from a specified directory
def read_docs(data_dir) -> List[str]:
    files = []
    for f in os.listdir(data_dir):
        fname = os.path.join(data_dir, f)
        if os.path.isfile(fname):
            files.append(fname)
    return files

parser = LlamaParse(
    result_type="markdown",
    premium_mode=True,
    api_key=os.getenv("LLAMA_CLOUD_API_KEY")
)

files = read_docs(data_dir = DATA_DIR) 
print("Parsing...")
json_results = parser.get_json_result(files)
print("Getting image dictionaries...")
images = parser.get_images(json_results, download_path=image_dir)
print("Retrieving nodes...")

###### Contextual Retrieval
# Function to get page number of images using regex on file names
def get_img_page_number(file_name):
    match = re.search(r"-page-(\d+)\.jpg$", str(file_name))
    if match:
        return int(match.group(1))
    return 0

# Function to get image files sorted by page
def _get_sorted_image_files(image_dir):
    raw_files = [f for f in list(Path(image_dir).iterdir()) if f.is_file()]
    sorted_files = sorted(raw_files, key=get_img_page_number)
    return sorted_files

# Context prompt template for contextual chunking
CONTEXT_PROMPT_TMPL = """
You are an AI assistant specializing in document analysis. Your task is to provide brief, relevant context for a chunk of text from the given document.
Here is the document:
<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Provide a concise context (2-3 sentences) for this chunk, considering the following guidelines:
1. Identify the main topic or concept discussed in the chunk.
2. Mention any relevant information or comparisons from the broader document context.
3. If applicable, note how this information relates to the overall theme or purpose of the document.
4. Include any key figures, dates, or percentages that provide important context.
5. Do not use phrases like "This chunk discusses" or "This section provides". Instead, directly state the context.

Please give a short succinct context to situate this chunk within the overall document to improve search retrieval of the chunk. 
Answer only with the succinct context and nothing else.

Context:
"""

CONTEXT_PROMPT = PromptTemplate(CONTEXT_PROMPT_TMPL)

# Function to generate context for each chunk
def _assign_context(document: str, chunk: str, llm) -> str:
    prompt = CONTEXT_PROMPT.format(document=document, chunk=chunk)
    response = llm.complete(prompt)
    context = response.text.strip()
    return context

# Function to create text nodes with context
def retrieve_nodes(json_results, image_dir, llm) -> List[TextNode]:
    nodes = []
    for result in json_results:
        json_dicts = result["pages"]
        document_name = result["file_path"].split('/')[-1]
        docs = [doc["md"] for doc in json_dicts]  # Extract text
        image_files = _get_sorted_image_files(image_dir)  # Extract images
        # Join all docs to create the full document text
        document_text = "\n\n".join(docs)
        for idx, doc in enumerate(docs):
            # Generate context for each chunk (page)
            context = _assign_context(document_text, doc, llm)
            # Combine context with the original chunk
            contextualized_content = f"{context}\n\n{doc}"
            # Create the text node with the contextualized content
            chunk_metadata = {"page_num": idx + 1}
            chunk_metadata["image_path"] = str(image_files[idx])
            chunk_metadata["parsed_text_markdown"] = docs[idx]
        
            node = TextNode(
                text=contextualized_content,
                metadata=chunk_metadata,
            )
            nodes.append(node)
    return nodes
# Get text nodes
text_node_with_context = retrieve_nodes(json_results, image_dir, llm)First page of the report (image by author)First page of the report (image by author)

###### Enhancing Contextual Retrieval with BM25 and Re-ranking

    # Create the vector store index
    index = VectorStoreIndex(text_node_with_context, embed_model=embed_model)
    index.storage_context.persist(persist_dir=output_dir)
    # Build BM25 index
    documents = [node.text for node in text_node_with_context]
    tokenized_documents = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_documents)
    # Save bm25 and text_node_with_context
    with open(os.path.join(output_dir, 'tokenized_documents.pkl'), 'wb') as f:
        pickle.dump(tokenized_documents, f)
    with open(os.path.join(output_dir, 'text_node_with_context.pkl'), 'wb') as f:
        pickle.dump(text_node_with_context, f)

########
# Define the QA prompt template
RAG_PROMPT = """\
Below we give parsed text from documents in two different formats, as well as the image.

---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query. Generate the answer by analyzing parsed markdown, raw text and the related
image. Especially, carefully analyze the images to look for the required information.
Format the answer in proper format as deems suitable (bulleted lists, sections/sub-sections, tables, etc.)
Give the page's number and the document name where you find the response based on the Context.

Query: {query_str}
Answer: """

PROMPT = PromptTemplate(RAG_PROMPT)

# Initialize the multimodal LLM
MM_LLM = OpenAIMultiModal(model="gpt-4o-mini", temperature=0.0, max_tokens=16000)

######### query_engin
# DeFfine the QueryEngine integrating all methods
class QueryEngine(CustomQueryEngine):
    # Public fields
    qa_prompt: PromptTemplate
    multi_modal_llm: OpenAIMultiModal
    node_postprocessors: Optional[List[BaseNodePostprocessor]] = None

    # Private attributes using PrivateAttr
    _bm25: BM25Okapi = PrivateAttr()
    _llm: OpenAI = PrivateAttr()
    _text_node_with_context: List[TextNode] = PrivateAttr()
    _vector_index: VectorStoreIndex = PrivateAttr()

    def __init__(
        self,
        qa_prompt: PromptTemplate,
        bm25: BM25Okapi,
        multi_modal_llm: OpenAIMultiModal,
        vector_index: VectorStoreIndex,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        llm: OpenAI = None,
        text_node_with_context: List[TextNode] = None,
    ):
        super().__init__(
            qa_prompt=qa_prompt,
            retriever=None,
            multi_modal_llm=multi_modal_llm,
            node_postprocessors=node_postprocessors
        )
        self._bm25 = bm25
        self._llm = llm
        self._text_node_with_context = text_node_with_context
        self._vector_index = vector_index

    def custom_query(self, query_str: str):
        # Prepare the query bundle
        query_bundle = QueryBundle(query_str)

        bm25_nodes = []
        if best_match_25 == 1:  # if BM25 search is selected
            # Retrieve nodes using BM25
            query_tokens = query_str.split()
            bm25_scores = self._bm25.get_scores(query_tokens)
            top_n_bm25 = 5  # Adjust the number of top nodes to retrieve
            # Get indices of top BM25 scores
            top_indices_bm25 = bm25_scores.argsort()[-top_n_bm25:][::-1]
            bm25_nodes = [self._text_node_with_context[i] for i in top_indices_bm25]
            logging.info(f"BM25 nodes retrieved: {len(bm25_nodes)}")
        else:
            logging.info("BM25 not selected.")

        # Retrieve nodes using vector-based retrieval from the vector store
        vector_retriever = self._vector_index.as_query_engine().retriever
        vector_nodes_with_scores = vector_retriever.retrieve(query_bundle)
        # Specify the number of top vectors you want
        top_n_vectors = 5  # Adjust this value as needed
        # Get only the top 'n' nodes
        top_vector_nodes_with_scores = vector_nodes_with_scores[:top_n_vectors]
        vector_nodes = [node.node for node in top_vector_nodes_with_scores]
        logging.info(f"Vector nodes retrieved: {len(vector_nodes)}")

        # Combine nodes and remove duplicates
        all_nodes = vector_nodes + bm25_nodes
        unique_nodes_dict = {node.node_id: node for node in all_nodes}
        unique_nodes = list(unique_nodes_dict.values())
        logging.info(f"Unique nodes after deduplication: {len(unique_nodes)}")

        nodes = unique_nodes

        if re_ranking == 1:  # if re-ranking is selected
            # Apply Cohere Re-ranking to rerank the combined results
            documents = [node.get_content() for node in nodes]
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    reranked = cohere_client.rerank(
                        model="rerank-english-v2.0",
                        query=query_str,
                        documents=documents,
                        top_n=3  # top-3 re-ranked nodes
                    )
                    break
                except CohereError as e:
                    if attempt < max_retries - 1:
                        logging.warning(f"Error occurred: {str(e)}. Waiting for 60 seconds before retry {attempt + 1}/{max_retries}")
                        time.sleep(60)  # Wait before retrying
                    else:
                        logging.error("Error occurred. Max retries reached. Proceeding without re-ranking.")
                        reranked = None
                        break

            if reranked:
                reranked_indices = [result.index for result in reranked.results]
                nodes = [nodes[i] for i in reranked_indices]
            else:
                nodes = nodes[:3]  # Fallback to top 3 nodes
            logging.info(f"Nodes after re-ranking: {len(nodes)}")
        else:
            logging.info("Re-ranking not selected.")

        # Limit and filter node content for context string
        max_context_length = 16000  # Adjust as required
        current_length = 0
        filtered_nodes = []

        # Initialize tokenizer
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        for node in nodes:
            content = node.get_content(metadata_mode=MetadataMode.LLM).strip()
            node_length = len(tokenizer.encode(content))
            logging.info(f"Node ID: {node.node_id}, Content Length (tokens): {node_length}")
            if not content:
                logging.warning(f"Node ID: {node.node_id} has empty content. Skipping.")
                continue
            if current_length + node_length <= max_context_length:
                filtered_nodes.append(node)
                current_length += node_length
            else:
                logging.info(f"Reached max context length with Node ID: {node.node_id}")
                break
        logging.info(f"Filtered nodes for context: {len(filtered_nodes)}")

        # Create context string
        ctx_str = "\n\n".join(
            [n.get_content(metadata_mode=MetadataMode.LLM).strip() for n in filtered_nodes]
        )

        # Create image nodes from the images associated with the nodes
        image_nodes = []
        for n in filtered_nodes:
            if "image_path" in n.metadata:
                image_nodes.append(
                    NodeWithScore(node=ImageNode(image_path=n.metadata["image_path"]))
                )
            else:
                logging.warning(f"Node ID: {n.node_id} lacks 'image_path' metadata.")
        logging.info(f"Image nodes created: {len(image_nodes)}")

        # Prepare prompt for the LLM
        fmt_prompt = self.qa_prompt.format(context_str=ctx_str, query_str=query_str)

        # Use the multimodal LLM to interpret images and generate a response
        llm_response = self.multi_modal_llm.complete(
            prompt=fmt_prompt,
            image_documents=[image_node.node for image_node in image_nodes],
            max_tokens=16000
        )

        logging.info(f"LLM response generated.")

        # Return the final response
        return Response(
            response=str(llm_response),
            source_nodes=filtered_nodes,
            metadata={
                "text_node_with_context": self._text_node_with_context,
                "image_nodes": image_nodes,
            },
        )

# Initialize the query engine with BM25, Cohere Re-ranking, and Query Expansion
query_engine = QueryEngine(
    qa_prompt=PROMPT,
    bm25=bm25,
    multi_modal_llm=MM_LLM,
    vector_index=index,
    node_postprocessors=[],
    llm=llm,
    text_node_with_context=text_node_with_context
)
print("All done")

############# run the query inference
original_query = """What are the top countries to whose citizens the Finnish Immigration Service issued the highest number of first residence permits in 2023?
Which of these countries received the highest number of first residence permits?"""
response = query_engine.query(original_query)
display(Markdown(str(response)))

