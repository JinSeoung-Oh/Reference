### From https://pub.towardsai.net/building-an-on-premise-document-intelligence-stack-with-docling-ollama-phi-4-extractthinker-6ab60b495751

!pip install extract-thinker

from extract_thinker import DocumentLoaderMarkItDown, DocumentLoaderDocling
from extract_thinker.models.contract import Contract
from pydantic import Field
from extract_thinker import Classification

TEST_CLASSIFICATIONS = [
    Classification(
        name="Invoice",
        description="This is an invoice document",
        contract=InvoiceContract
    ),
    Classification(
        name="Driver License",
        description="This is a driver license document",
        contract=DriverLicense
    )
]

class InvoiceContract(Contract):
    invoice_number: str = Field(description="Unique invoice identifier")
    invoice_date: str = Field(description="Date of the invoice")
    total_amount: float = Field(description="Overall total amount")

class DriverLicense(Contract):
    name: str = Field(description="Full name on the license")
    age: int = Field(description="Age of the license holder")
    license_number: str = Field(description="License number")

# DocumentLoaderDocling or DocumentLoaderMarkItDown
document_loader = DocumentLoaderDocling()

-------------------------------------------------------------------------------------
### Local Extraction Process
import os
from dotenv import load_dotenv

from extract_thinker import (
    Extractor,
    Process,
    Classification,
    SplittingStrategy,
    ImageSplitter,
    TextSplitter
)

# Load environment variables (if you store LLM endpoints/API_BASE, etc. in .env)
load_dotenv()

# Example path to a multi-page document
MULTI_PAGE_DOC_PATH = "path/to/your/multi_page_doc.pdf"

def setup_local_process():
    """
    Helper function to set up an ExtractThinker process
    using local LLM endpoints (e.g., Ollama, LocalAI, OnPrem.LLM, etc.)
    """

    # 1) Create an Extractor
    extractor = Extractor()

    # 2) Attach our chosen DocumentLoader (Docling or MarkItDown)
    extractor.load_document_loader(document_loader)

    # 3) Configure your local LLM
    #    For Ollama, you might do:
    os.environ["API_BASE"] = "http://localhost:11434"  # Replace with your local endpoint
    extractor.load_llm("ollama/phi4")  # or "ollama/llama3.3" or your local model
    
    # 4) Attach extractor to each classification
    TEST_CLASSIFICATIONS[0].extractor = extractor
    TEST_CLASSIFICATIONS[1].extractor = extractor

    # 5) Build the Process
    process = Process()
    process.load_document_loader(document_loader)
    return process

def run_local_idp_workflow():
    """
    Demonstrates loading, classifying, splitting, and extracting
    a multi-page document with a local LLM.
    """
    # Initialize the process
    process = setup_local_process()

    # (Optional) You can use ImageSplitter(model="ollama/moondream:v2") for the split
    process.load_splitter(TextSplitter(model="ollama/phi4"))

    # 1) Load the file
    # 2) Split into sub-documents with EAGER strategy
    # 3) Classify each sub-document with our TEST_CLASSIFICATIONS
    # 4) Extract fields based on the matched contract (Invoice or DriverLicense)
    result = (
        process
        .load_file(MULTI_PAGE_DOC_PATH)
        .split(TEST_CLASSIFICATIONS, strategy=SplittingStrategy.LAZY)
        .extract(vision=False, completion_strategy=CompletionStrategy.PAGINATE)
    )

    # 'result' is a list of extracted objects (InvoiceContract or DriverLicense)
    for item in result:
        # Print or store each extracted data model
        if isinstance(item, InvoiceContract):
            print("[Extracted Invoice]")
            print(f"Number: {item.invoice_number}")
            print(f"Date: {item.invoice_date}")
            print(f"Total: {item.total_amount}")
        elif isinstance(item, DriverLicense):
            print("[Extracted Driver License]")
            print(f"Name: {item.name}, Age: {item.age}")
            print(f"License #: {item.license_number}")

# For a quick test, just call run_local_idp_workflow()
if __name__ == "__main__":
    run_local_idp_workflow()
