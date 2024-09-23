## From https://ai.gopubby.com/kotaemon-unveiled-innovations-in-rag-framework-for-document-qa-0b6d67e4b9b7
"""
The article presents Kotaemon, an innovative open-source Retrieval-Augmented Generation (RAG) framework
that is designed to enhance question-answering (QA) systems and dialogue models, particularly for document-based AI applications. 
As of September 13, 2024, Kotaemon has gained considerable attention with 11.5K stars on GitHub.

Key Features and Architecture:
1. PDF Parsing:
   - Kotaemon handles unstructured documents, focusing on PDF parsing via the PDFThumbnailReader class.
   - The pipeline extracts text, generates thumbnails for each page using PyMuPDF, and stores the data with metadata like page numbers in a list called documents.
   - OCRReader is used for extracting text from tables and images using Optical Character Recognition (OCR), relying on an endpoint for full OCR pipeline processing.

2. Document Chunking:
   - Kotaemon uses the TokenSplitter class for chunking parsed documents, utilizing parameters like chunk size, overlap, 
     and separators to ensure efficient tokenization and compatibility with LLMs.

3. Re-ranking:
   Kotaemon offers four re-ranking algorithms:
   CohereReranking uses the Cohere API.
   LLMReranking and LLMTrulensScoring employ LLMs and TruLens prompts for re-ranking results.
   LLMScoring filters documents based on their relevance to the query.

4. GraphRAG:
   A standout feature, GraphRAG organizes knowledge into structured graphs rather than plain text, improving reasoning over complex documents.
   The pipeline uses GraphRAGIndexingPipeline for indexing and GraphRAGRetrieverPipeline for retrieving documents, 
   which enhances the retrieval process by building a knowledge graph.

5. Reasoning Framework:
   Kotaemon supports multiple reasoning methods:
   - Simple: Uses FullQAPipeline for direct QA tasks.
   - Complex: Uses DecomposeQuestionPipeline to break down complex queries into smaller sub-questions.
   - Agent-based: Uses ReAct and ReWOO to handle dynamic conversations step-by-step.

6. LLMs and Embedding Models:
   Kotaemon supports popular LLMs such as OpenAI, Ollma, Claude, and Groq, along with embedding models for efficient document retrieval, including FastEmbed and Cohere.

7. Web Service and Storage:
   The system uses Gradio for its frontend, providing an interactive UI for document uploads and QA interactions.
   Docstores and Vectorstores manage content and embeddings for document storage and retrieval, respectively.

8. Insights and Future Improvements:
   - Document Parsing: While Kotaemon effectively parses standard PDFs, handling complex data such as images or high-resolution documents remains a challenge. 
                       Integrating advanced models could enhance this capability.
   - GraphRAG: The structured reasoning and knowledge graphs provided by GraphRAG offer potential for high-performance document-based reasoning. 
               Further development in this area could make Kotaemon a leader in knowledge-intensive queries.
   - Performance Optimization: Improving the efficiency of indexing and refining query processing methods could enhance Kotaemon’s performance in real-world applications.

9. Conclusion:
   Kotaemon is a powerful framework for building document-focused AI applications, combining advanced parsing, chunking, reasoning,
   and retrieval techniques with the flexibility of open-source development.
   It holds significant potential for organizations relying on intelligent document-based QA systems and continues to grow in popularity due to its innovative features.
"""

##### PDF Parsing
class PDFThumbnailReader(PDFReader):
    """PDF parser with thumbnail for each page."""

    def __init__(self) -> None:
        """
        Initialize PDFReader.
        """
        super().__init__(return_full_document=False)

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        documents = super().load_data(file, extra_info, fs)

        page_numbers_str = []
        filtered_docs = []
        is_int_page_number: dict[str, bool] = {}

        for doc in documents:
            if "page_label" in doc.metadata:
                page_num_str = doc.metadata["page_label"]
                page_numbers_str.append(page_num_str)
                try:
                    _ = int(page_num_str)
                    is_int_page_number[page_num_str] = True
                    filtered_docs.append(doc)
                except ValueError:
                    is_int_page_number[page_num_str] = False
                    continue

        documents = filtered_docs
        page_numbers = list(range(len(page_numbers_str)))

        print("Page numbers:", len(page_numbers))
        page_thumbnails = get_page_thumbnails(file, page_numbers)

        documents.extend(
            [
                Document(
                    text="Page thumbnail",
                    metadata={
                        "image_origin": page_thumbnail,
                        "type": "thumbnail",
                        "page_label": page_number,
                        **(extra_info if extra_info is not None else {}),
                    },
                )
                for (page_thumbnail, page_number) in zip(
                    page_thumbnails, page_numbers_str
                )
                if is_int_page_number[page_number]
            ]
        )

        return documents

##### OCRReader
class OCRReader(BaseReader):
    """Read PDF using OCR, with high focus on table extraction

    Example:
        ```python
        >> from kotaemon.loaders import OCRReader
        >> reader = OCRReader()
        >> documents = reader.load_data("path/to/pdf")
        ```

    Args:
        endpoint: URL to FullOCR endpoint. If not provided, will look for
            environment variable `OCR_READER_ENDPOINT` or use the default
            `kotaemon.loaders.ocr_loader.DEFAULT_OCR_ENDPOINT`
            (http://127.0.0.1:8000/v2/ai/infer/)
        use_ocr: whether to use OCR to read text (e.g: from images, tables) in the PDF
            If False, only the table and text within table cells will be extracted.
    """

    def __init__(self, endpoint: Optional[str] = None, use_ocr=True):
        """Init the OCR reader with OCR endpoint (FullOCR pipeline)"""
        super().__init__()
        self.ocr_endpoint = endpoint or os.getenv(
            "OCR_READER_ENDPOINT", DEFAULT_OCR_ENDPOINT
        )
        self.use_ocr = use_ocr

    def load_data(
        self, file_path: Path, extra_info: Optional[dict] = None, **kwargs
    ) -> List[Document]:
        """Load data using OCR reader

        Args:
            file_path (Path): Path to PDF file
            debug_path (Path): Path to store debug image output
            artifact_path (Path): Path to OCR endpoints artifacts directory

        Returns:
            List[Document]: list of documents extracted from the PDF file
        """
        file_path = Path(file_path).resolve()

        # call the API from FullOCR endpoint
        if "response_content" in kwargs:
            # overriding response content if specified
            ocr_results = kwargs["response_content"]
        else:
            # call original API
            resp = tenacious_api_post(
                url=self.ocr_endpoint, file_path=file_path, table_only=not self.use_ocr
            )
            ocr_results = resp.json()["result"]

        debug_path = kwargs.pop("debug_path", None)
        artifact_path = kwargs.pop("artifact_path", None)

        # read PDF through normal reader (unstructured)
        pdf_page_items = read_pdf_unstructured(file_path)
        # merge PDF text output with OCR output
        tables, texts = parse_ocr_output(
            ocr_results,
            pdf_page_items,
            debug_path=debug_path,
            artifact_path=artifact_path,
        )
        extra_info = extra_info or {}

        # create output Document with metadata from table
        documents = [
            Document(
                text=strip_special_chars_markdown(table_text),
                metadata={
                    "table_origin": table_text,
                    "type": "table",
                    "page_label": page_id + 1,
                    **extra_info,
                },
                metadata_template="",
                metadata_seperator="",
            )
            for page_id, table_text in tables
        ]
        # create Document from non-table text
        documents.extend(
            [
                Document(
                    text=non_table_text,
                    metadata={"page_label": page_id + 1, **extra_info},
                )
                for page_id, non_table_text in texts
            ]
        )

        return documents

##### Document Chunking
class TokenSplitter(LlamaIndexDocTransformerMixin, BaseSplitter):
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 20,
        separator: str = " ",
        **params,
    ):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            **params,
        )

    def _get_li_class(self):
        from llama_index.core.text_splitter import TokenTextSplitter

        return TokenTextSplitter

##### Re-ranking
RERANK_PROMPT_TEMPLATE = """Given the following question and context,
return YES if the context is relevant to the question and NO if it isn't.

> Question: {question}
> Context:
>>>
{context}
>>>
> Relevant (YES / NO):"""

##### class LLMTrulensScoring
SYSTEM_PROMPT_TEMPLATE = PromptTemplate(
    """You are a RELEVANCE grader; providing the relevance of the given CONTEXT to the given QUESTION.
        Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant.

        A few additional scoring guidelines:

        - Long CONTEXTS should score equally well as short CONTEXTS.

        - RELEVANCE score should increase as the CONTEXTS provides more RELEVANT context to the QUESTION.

        - RELEVANCE score should increase as the CONTEXTS provides RELEVANT context to more parts of the QUESTION.

        - CONTEXT that is RELEVANT to some of the QUESTION should score of 2, 3 or 4. Higher score indicates more RELEVANCE.

        - CONTEXT that is RELEVANT to most of the QUESTION should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

        - CONTEXT that is RELEVANT to the entire QUESTION should get a score of 9 or 10. Higher score indicates more RELEVANCE.

        - CONTEXT must be relevant and helpful for answering the entire QUESTION to get a score of 10.

        - Never elaborate."""  # noqa: E501
)

##### GraphRAGIndexingPipeline
class GraphRAGIndexingPipeline(IndexDocumentPipeline):
    """GraphRAG specific indexing pipeline"""
    ...
    ...
    def call_graphrag_index(self, input_path: str):
        # Construct the command
        command = [
            "python",
            "-m",
            "graphrag.index",
            "--root",
            input_path,
            "--reporter",
            "rich",
            "--init",
        ]

        # Run the command
        yield Document(
            channel="debug",
            text="[GraphRAG] Creating index... This can take a long time.",
        )
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        command = command[:-1]

        # Run the command and stream stdout
        with subprocess.Popen(command, stdout=subprocess.PIPE, text=True) as process:
            if process.stdout:
                for line in process.stdout:
                    yield Document(channel="debug", text=line)

    def stream(
        self, file_paths: str | Path | list[str | Path], reindex: bool = False, **kwargs
    ) -> Generator[
        Document, None, tuple[list[str | None], list[str | None], list[Document]]
    ]:
        file_ids, errors, all_docs = yield from super().stream(
            file_paths, reindex=reindex, **kwargs
        )

        # assign graph_id to file_ids
        graph_id = self.store_file_id_with_graph_id(file_ids)
        # call GraphRAG index with docs and graph_id
        graph_index_path = self.write_docs_to_files(graph_id, all_docs)
        yield from self.call_graphrag_index(graph_index_path)

        return file_ids, errors, all_docs

##### Retrieval Pipeline
class GraphRAGRetrieverPipeline(BaseFileIndexRetriever):
    """GraphRAG specific retriever pipeline"""    
    ...
    ...

    def run(
        self,
        text: str,
    ) -> list[RetrievedDocument]:
        if not self.file_ids:
            return []
        context_builder = self._build_graph_search()

        local_context_params = {
            "text_unit_prop": 0.5,
            "community_prop": 0.1,
            "conversation_history_max_turns": 5,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": 10,
            "top_k_relationships": 10,
            "include_entity_rank": False,
            "include_relationship_weight": False,
            "include_community_rank": False,
            "return_candidate_context": False,
            "embedding_vectorstore_key": EntityVectorStoreKey.ID,
            # set this to EntityVectorStoreKey.TITLE i
            # f the vectorstore uses entity title as ids
            "max_tokens": 12_000,
            # change this based on the token limit you have on your model
            # (if you are using a model with 8k limit, a good setting could be 5000)
        }

        context_text, context_records = context_builder.build_context(
            query=text,
            conversation_history=None,
            **local_context_params,
        )
        documents = self.format_context_records(context_records)
        plot = self.plot_graph(context_records)

        return documents + [
            RetrievedDocument(
                text="",
                metadata={
                    "file_name": "GraphRAG",
                    "type": "plot",
                    "data": plot,
                },
            ),
        ]

##### Reasoning
# Setup your new reasoning pipeline or modify existing one.
KH_REASONINGS = [
    "ktem.reasoning.simple.FullQAPipeline",
    "ktem.reasoning.simple.FullDecomposeQAPipeline",
    "ktem.reasoning.react.ReactAgentPipeline",
    "ktem.reasoning.rewoo.RewooAgentPipeline",
]

DEFAULT_REWRITE_PROMPT = (
    "Given the following question, rephrase and expand it "
    "to help you do better answering. Maintain all information "
    "in the original question. Keep the question as concise as possible. "
    "Give answer in {lang}\n"
    "Original question: {question}\n"
    "Rephrased question: "
)

###### Complex Reasoning Pipeline
DECOMPOSE_SYSTEM_PROMPT_TEMPLATE = (
        "You are an expert at converting user complex questions into sub questions. "
        "Perform query decomposition using provided function_call. "
        "Given a user question, break it down into the most specific sub"
        " questions you can (at most 3) "
        "which will help you answer the original question. "
        "Each sub question should be about a single concept/fact/idea. "
        "If there are acronyms or words you are not familiar with, "
        "do not try to rephrase them."
    )


##### Agent-Based Reasoning
DEFAULT_QA_PROMPT = (
    "Answer the following questions as best you can. Give answer in {lang}. "
    "You have access to the following tools:\n"
    "{tool_description}\n"
    "Use the following format:\n\n"
    "Question: the input question you must answer\n"
    "Thought: you should always think about what to do\n\n"
    "Action: the action to take, should be one of [{tool_names}]\n\n"
    "Action Input: the input to the action, should be different from the action input "
    "of the same action in previous steps.\n\n"
    "Observation: the result of the action\n\n"
    "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
    "#Thought: I now know the final answer\n"
    "Final Answer: the final answer to the original input question\n\n"
    "Begin! After each Action Input.\n\n"
    "Question: {instruction}\n"
    "Thought: {agent_scratchpad}\n"
)
