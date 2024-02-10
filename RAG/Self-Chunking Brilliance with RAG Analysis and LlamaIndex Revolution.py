## From https://ai.gopubby.com/self-chunking-brilliance-with-rag-analysis-and-llamaindex-revolution-dd590d734484

!pip install -qU llama_index llama_hub sentence-transformers accelerate "huggingface_hub[inference]"
!pip install transformers arize-phoenix --upgrade --quiet
!pip install -qU chromadb  pypdf wikipedia

!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install  llama-cpp-python --no-cache-dir

# Download the Model
!wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q8_0.gguf -O ./mistral-7b-instruct-v0.1.Q8_0.gguf

#Download Data
!mkdir data

from llama_index.llama_dataset import download_llama_dataset
from llama_index.llama_pack import download_llama_pack

rag_dataset, documents = download_llama_dataset(
  "Uber10KDataset2021", "./data"
)

# Phoenix can display in real time the traces automatically
# collected from your LlamaIndex application.
import phoenix as px

# Look for a URL in the output to open the App in a browser.
px.launch_app()
# The App is initially empty, but as you proceed with the steps below,
# traces will appear automatically as your LlamaIndex application runs.

import llama_index

llama_index.set_global_handler("arize_phoenix")

import os,re
import logging, sys
import torch

import nest_asyncio

nest_asyncio.apply()

from llama_index.llms import LlamaCPP
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
# Embeddings
from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding, CohereEmbedding

# Tokenizer must match the model we're using

from llama_index import set_global_tokenizer

# huggingface
from transformers import AutoTokenizer
set_global_tokenizer(
  AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1").encode
)

#Setup  OPEN API Key
os.environ["OPENAI_API_KEY"] = ""

llm = LlamaCPP(
    model_url=None, # We'll load locally.
    model_path='/workspace/mistral-7b-instruct-v0.1.Q8_0.gguf', # 6-bit model
    temperature=0.1,
    max_new_tokens=1024, # Increasing to support longer responses
    context_window=8192, # Mistral7B has an 8K context-window
    generate_kwargs={},
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 33}, # 33 was all that was needed for this model and the RTX 3090
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True
)

# Self Chunking: We create a function to split text into paragraphs but keep numbered sections and bullet points together. 

# Define the regular expression pattern for splitting paragraphs
para_split_pattern = re.compile(r'\n\n\n')

# Splits a document's text into paragraphs but if it has numbered or bulleted points, they will be included with the paragraph before it.
def split_text_into_paragraphs(text):


    # Use the pattern to split the text into paragraphs
    paragraphs = para_split_pattern.split(text)

    # Combine paragraphs that should not be split
    combined_paragraphs = [paragraphs[0]]

    for p in paragraphs[1:]:
        # Check if the paragraph starts with a number or a dash and, if so, concatenate it to the previous paragraph so we keep them all in one chunk

        # Strip out any leading new lines
        p = p.lstrip('\n')

        if p and (p[0].isdigit() or p[0] == '-'):
            combined_paragraphs[-1] += '\n\n\n' + p
        else:
            combined_paragraphs.append(p)

    # Remove empty strings from the result
    combined_paragraphs = [p.strip() for p in combined_paragraphs if p.strip()]

    return combined_paragraphs


# Importing necessary modules
from llama_index.utilities.token_counting import TokenCounter
from llama_index.schema import TextNode
import uuid

# Initializing the TokenCounter using the global tokenizer set in LLM
token_counter = TokenCounter()

# Defining the paragraph separator
paragraph_separator = "\n\n\n"

# Variables to store maximum and total paragraph tokens
max_paragraph_tokens = 0
total_paragraph_tokens = 0

# List to hold TextNode objects representing paragraphs
paragraph_nodes = []

# Loop through the documents, splitting each into paragraphs and checking the number of tokens per paragraph
for document in documents:

    # List to store token lengths for each paragraph
    paragraph_token_lens = []

    # Splitting document into paragraphs
    paragraphs = split_text_into_paragraphs(document.text)

    # Displaying document information
    print(f"Document {document.metadata['file_name']} has {len(paragraphs)} paragraphs, token lengths:")

    for paragraph in paragraphs:
        # Counting tokens for each paragraph
        token_count = token_counter.get_string_tokens(paragraph)
        paragraph_token_lens.append(token_count)

        # Updating maximum paragraph tokens
        if token_count > max_paragraph_tokens:
            max_paragraph_tokens = token_count

        # Updating total paragraph tokens
        total_paragraph_tokens += token_count

        # Creating and adding a TextNode for each paragraph
        node = TextNode(text=paragraph, id=uuid.uuid4())
        node.metadata['document_name'] = document.metadata['file_name']
        node.metadata['token_count'] = token_count
        paragraph_nodes.append(node)

    # Displaying token lengths for each paragraph in the document
    print(paragraph_token_lens)

# Displaying the maximum paragraph tokens
print(f"\n** The maximum paragraph tokens is {max_paragraph_tokens} **")

# Calculating and displaying the average paragraph token count
average_paragraph_tokens = int(total_paragraph_tokens / len(paragraph_nodes))

from llama_index import ServiceContext
embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large", cache_folder=None)
service_context = ServiceContext.from_defaults(
    # chunk_size=500, # The token chunk size for each chunk. ** we are creating chunks automatically so no need to set this **
    # chunk_overlap=5, # The token overlap of each chunk when splitting. ** we are creating chunks automatically so no need to set this **
    llm=llm, # Our LLM, whichever we wnat to use
    embed_model=embed_model, # Our embeddings
    num_output=1000, # Let's allow up to 1000 tokens to be output
)

index_from_nodes = VectorStoreIndex(paragraph_nodes, show_progress=True, service_context=service_context)

for i, document_id in enumerate(index_from_nodes.docstore.docs):
    document = index_from_nodes.docstore.get_document(document_id)
    print(f"--- {i} ---\n{document.extra_info['document_name']}")
    print(f"id: {document.node_id}")
    print(f"characters: {len(document.text)}")
    print(f"tokens: {document.extra_info['token_count']}")
    print(f"[Text Start]\n{document.text}\n[Text End]\n")
