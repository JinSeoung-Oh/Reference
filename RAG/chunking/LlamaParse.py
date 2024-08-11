## From https://generativeai.pub/llamaparse-revolutionizing-pdf-document-parsing-with-genai-e3192c075d2c

!pip install -U llama-index --upgrade --no-cache-dir --force-reinstall --user
!pip install llama-parse

from llama_parse import LlamaParse
import nest_asyncio
nest_asyncio.apply()
import os
from llama_index.core import SimpleDirectoryReader
os.environ["LLAMA_CLOUD_API_KEY"] = "your api key from llama cloud"

parser = LlamaParse(
    result_type="markdown",
    verbose=True,
    language="en",
    num_workers=2,
)
file_extractor = {".pdf": parser}
pdf_documents = SimpleDirectoryReader(
    "./data", file_extractor=file_extractor
).load_data()

