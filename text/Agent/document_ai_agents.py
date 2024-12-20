### From https://towardsdatascience.com/build-a-document-ai-pipeline-for-any-type-of-pdf-with-gemini-9221c8e143db

from document_ai_agents.document_utils import extract_images_from_pdf
from document_ai_agents.image_utils import pil_image_to_base64_jpeg
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal
import json
import google.generativeai as genai
from langchain_core.documents import Document
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END

class DocumentParsingAgent:
    @classmethod
    def get_images(cls, state):
        """
        Extract pages of a PDF as Base64-encoded JPEG images.
        """
        assert Path(state.document_path).is_file(), "File does not exist"
        # Extract images from PDF
        images = extract_images_from_pdf(state.document_path)
        assert images, "No images extracted"
        # Convert images to Base64-encoded JPEG
        pages_as_base64_jpeg_images = [pil_image_to_base64_jpeg(x) for x in images]
        return {"pages_as_base64_jpeg_images": pages_as_base64_jpeg_images}


class DetectedLayoutItem(BaseModel):
    """
    Schema for each detected layout element on a page.
    """
    element_type: Literal["Table", "Figure", "Image", "Text-block"] = Field(
        ..., 
        description="Type of detected item. Examples: Table, Figure, Image, Text-block."
    )
    summary: str = Field(..., description="A detailed description of the layout item.")

class LayoutElements(BaseModel):
    """
    Schema for the list of layout elements on a page.
    """
    layout_items: list[DetectedLayoutItem] = []

class FindLayoutItemsInput(BaseModel):
    """
    Input schema for processing a single page.
    """
    document_path: str
    base64_jpeg: str
    page_number: int

class DocumentParsingAgent:
    def __init__(self, model_name="gemini-1.5-flash-002"):
        """
        Initialize the LLM with the appropriate schema.
        """
        layout_elements_schema = prepare_schema_for_gemini(LayoutElements)
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": layout_elements_schema,
            },
        )
    def find_layout_items(self, state: FindLayoutItemsInput):
        """
        Send a page image to the LLM for segmentation and summarization.
        """
        messages = [
            f"Find and summarize all the relevant layout elements in this PDF page in the following format: "
            f"{LayoutElements.schema_json()}. "
            f"Tables should have at least two columns and at least two rows. "
            f"The coordinates should overlap with each layout item.",
            {"mime_type": "image/jpeg", "data": state.base64_jpeg},
        ]
        # Send the prompt to the LLM
        result = self.model.generate_content(messages)
        data = json.loads(result.text)
        
        # Convert the JSON output into documents
        documents = [
            Document(
                page_content=item["summary"],
                metadata={
                    "page_number": state.page_number,
                    "element_type": item["element_type"],
                    "document_path": state.document_path,
                },
            )
            for item in data["layout_items"]
        ]
        return {"documents": documents}

class DocumentParsingAgent:
    @classmethod
    def continue_to_find_layout_items(cls, state):
        """
        Generate tasks to process each page in parallel.
        """
        return [
            Send(
                "find_layout_items",
                FindLayoutItemsInput(
                    base64_jpeg=base64_jpeg,
                    page_number=i,
                    document_path=state.document_path,
                ),
            )
            for i, base64_jpeg in enumerate(state.pages_as_base64_jpeg_images)
        ]

class DocumentParsingAgent:
    def build_agent(self):
        """
        Build the agent workflow using a state graph.
        """
        builder = StateGraph(DocumentLayoutParsingState)
        
        # Add nodes for image extraction and layout item detection
        builder.add_node("get_images", self.get_images)
        builder.add_node("find_layout_items", self.find_layout_items)
        # Define the flow of the graph
        builder.add_edge(START, "get_images")
        builder.add_conditional_edges("get_images", self.continue_to_find_layout_items)
        builder.add_edge("find_layout_items", END)
        
        self.graph = builder.compile()


if __name__ == "__main__":
    _state = DocumentLayoutParsingState(
        document_path="path/to/document.pdf"
    )
    agent = DocumentParsingAgent()
    
    # Step 1: Extract images from PDF
    result_images = agent.get_images(_state)
    _state.pages_as_base64_jpeg_images = result_images["pages_as_base64_jpeg_images"]
    
    # Step 2: Process the first page (as an example)
    result_layout = agent.find_layout_items(
        FindLayoutItemsInput(
            base64_jpeg=_state.pages_as_base64_jpeg_images[0],
            page_number=0,
            document_path=_state.document_path,
        )
    )
    # Display the results
    for item in result_layout["documents"]:
        print(item.page_content)
        print(item.metadata["element_type"])

---------------------------------------------------------------------------------------------------------------
### RAG Agent

class DocumentRAGAgent:
    def index_documents(self, state: DocumentRAGState):
        """
        Index the parsed documents into the vector store.
        """
        assert state.documents, "Documents should have at least one element"
        # Check if the document is already indexed
        if self.vector_store.get(where={"document_path": state.document_path})["ids"]:
            logger.info(
                "Documents for this file are already indexed, exiting this node"
            )
            return  # Skip indexing if already done
        # Add parsed documents to the vector store
        self.vector_store.add_documents(state.documents)
        logger.info(f"Indexed {len(state.documents)} documents for {state.document_path}")

    def answer_question(self, state: DocumentRAGState):
        """
        Retrieve relevant chunks and generate a response to the user's question.
        """
        # Retrieve the top-k relevant documents based on the query
        relevant_documents: list[Document] = self.retriever.invoke(state.question)

        # Retrieve corresponding page images (avoid duplicates)
        images = list(
            set(
                [
                    state.pages_as_base64_jpeg_images[doc.metadata["page_number"]]
                    for doc in relevant_documents
                ]
            )
        )
        logger.info(f"Responding to question: {state.question}")
        # Construct the prompt: Combine images, relevant summaries, and the question
        messages = (
            [{"mime_type": "image/jpeg", "data": base64_jpeg} for base64_jpeg in images]
            + [doc.page_content for doc in relevant_documents]
            + [
                f"Answer this question using the context images and text elements only: {state.question}",
            ]
        )
        # Generate the response using the LLM
        response = self.model.generate_content(messages)
        return {"response": response.text, "relevant_documents": relevant_documents}

      def build_agent(self):
        """
        Build the RAG agent workflow.
        """
        builder = StateGraph(DocumentRAGState)
        # Add nodes for indexing and answering questions
        builder.add_node("index_documents", self.index_documents)
        builder.add_node("answer_question", self.answer_question)
        # Define the workflow
        builder.add_edge(START, "index_documents")
        builder.add_edge("index_documents", "answer_question")
        builder.add_edge("answer_question", END)
        self.graph = builder.compile()

if __name__ == "__main__":
    from pathlib import Path

  # Import the first agent to parse the document
    from document_ai_agents.document_parsing_agent import (
        DocumentLayoutParsingState,
        DocumentParsingAgent,
    )
    # Step 1: Parse the document using the first agent
    state1 = DocumentLayoutParsingState(
        document_path=str(Path(__file__).parents[1] / "data" / "docs.pdf")
    )
    agent1 = DocumentParsingAgent()
    result1 = agent1.graph.invoke(state1)
    # Step 2: Set up the second agent for retrieval and answering
    state2 = DocumentRAGState(
        question="Who was acknowledged in this paper?",
        document_path=str(Path(__file__).parents[1] / "data" / "docs.pdf"),
        pages_as_base64_jpeg_images=result1["pages_as_base64_jpeg_images"],
        documents=result1["documents"],
    )
    agent2 = DocumentRAGAgent()
    # Index the documents
    agent2.graph.invoke(state2)
    # Answer the first question
    result2 = agent2.graph.invoke(state2)
    print(result2["response"])
    # Answer a second question
    state3 = DocumentRAGState(
        question="What is the macro average when fine-tuning on PubLayNet using M-RCNN?",
        document_path=str(Path(__file__).parents[1] / "data" / "docs.pdf"),
        pages_as_base64_jpeg_images=result1["pages_as_base64_jpeg_images"],
        documents=result1["documents"],
    )
    result3 = agent2.graph.invoke(state3)
    print(result3["response"])
