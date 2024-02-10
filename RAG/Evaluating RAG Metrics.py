# More detail have to see : https://medium.com/towards-artificial-intelligence/evaluating-rag-metrics-across-different-retrieval-methods-770aa01380c8

%%capture
!pip install -U langchain==0.1
!pip install -U langchain_openai
!pip install -U openai
!pip install -U ragas
!pip install -U arxiv
!pip install -U pymupdf
!pip install -U chromadb
!pip install -U tiktoken
!pip install -U accelerate
!pip install -U bitsandbytes
!pip install -U datasets
!pip install -U sentence_transformers
!pip install -U FlagEmbedding
!pip install -U ninja
!pip install -U flash_attn --no-build-isolation
!pip install -U tqdm
!pip install -U rank_bm25
!pip install -U transformers

import os
import openai
from getpass import getpass

openai.api_key = getpass("Please provide your OpenAI Key: ")
os.environ["OPENAI_API_KEY"] = openai.api_key

from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders.merge import MergedDataLoader

papers = ["2310.13800", "2307.03109", "2304.08637", "2310.05657", "2305.13091", "2311.09476", "2308.10633", "2309.01431", "2311.04348"]

docs_to_merge = []

for paper in papers:
    loader = ArxivLoader(query=paper)
    docs_to_merge.append(loader)

all_loaders = MergedDataLoader(loaders=docs_to_merge)

all_docs = all_loaders.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-large-en-v1.5"

encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

hf_bge_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'},
    encode_kwargs=encode_kwargs
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                               chunk_overlap = 16,
                                               length_function=len)

docs = text_splitter.split_documents(all_docs)
vectorstore = Chroma.from_documents(docs, hf_bge_embeddings)
base_retriever = vectorstore.as_retriever()

#relevant_docs = base_retriever.get_relevant_documents("What are the challenges in evaluating Retrieval Augmented Generation pipelines?")

from langchain.prompts import ChatPromptTemplate

template = """<human>: Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':

### CONTEXT
{context}

### QUESTION
Question: {question}

\n

<bot>:
"""

prompt = ChatPromptTemplate.from_template(template)

from operator import itemgetter
import torch
from langchain_openai import ChatOpenAI
from langchain_community.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig, pipeline
from langchain_core.output_parser import StrOutputParser
from langchain_core.runnable import RunnableLambda, RunnablePassthrough

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
# Note: this tutorial is running on an A100m if you're not using an A100
# or ampere supported GPU then bfloat won't work for you
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained("llmware/dragon-deci-7b-v0",
                                             quantization_config = quantization_config,
                                             trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained("llmware/dragon-deci-7b-v0",
                                          trust_remote_code=True)

generation_config = GenerationConfig(
    max_length=4096,
    temperature=1e-3,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)

pipeline = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=4096,
                    temperature=1e-3,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                    )

deci_dragon = HuggingFacePipeline(pipeline=pipeline)

retrieval_augmented_qa_chain = (
    {"context": itemgetter("question") | base_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": prompt | deci_dragon, "context": itemgetter("context")}
)

#############################################################
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

question_schema = ResponseSchema(
    name="question",
    description="a question about the context."
)

question_response_schemas = [
    question_schema,
]
question_output_parser = StructuredOutputParser.from_response_schemas(question_response_schemas)
format_instructions = question_output_parser.get_format_instructions()
question_generation_llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
bare_prompt_template = "{content}"
bare_template = ChatPromptTemplate.from_template(template=bare_prompt_template)

## Generating Questions
from langchain.prompts import ChatPromptTemplate
qa_template = """\
<human>: You are a University Professor creating a test for advanced students. For each context, create a question that is specific to the context. Avoid creating generic or general questions.
question: a question about the context.
Format the output as JSON with the following keys:
question
context: {context}
<bot>:
"""

prompt_template = ChatPromptTemplate.from_template(template=qa_template)
messages = prompt_template.format_messages(
    context=docs[0],
    format_instructions=format_instructions
)

question_generation_chain = bare_template | question_generation_llm
response = question_generation_chain.invoke({"content" : messages})
output_dict = question_output_parser.parse(response.content)

## Generating Context
from tqdm import tqdm
import random

random.seed(42)

qac_triples = []

# randomly select 100 chunks from the ~1300 chunks
for text in tqdm(random.sample(docs, 100)):
  messages = prompt_template.format_messages(
      context=text,
      format_instructions=format_instructions
  )
  response = question_generation_chain.invoke({"content" : messages})
  try:
    output_dict = question_output_parser.parse(response.content)
  except Exception as e:
    continue
  output_dict["context"] = text
  qac_triples.append(output_dict)

## Generate “Ground Truth” Answers using GPT-4
answer_generation_llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
answer_schema = ResponseSchema(
    name="answer",
    description="an answer to the question"
)
answer_response_schemas = [
    answer_schema,
]
answer_output_parser = StructuredOutputParser.from_response_schemas(answer_response_schemas)
format_instructions = answer_output_parser.get_format_instructions()
qa_template = """\
<human>: You are a University Professor creating a test for advanced students. For each question and context, create an answer.
answer: a answer about the context.
Format the output as JSON with the following keys:
answer
question: {question}
context: {context}
<bot>:
"""

prompt_template = ChatPromptTemplate.from_template(template=qa_template)
messages = prompt_template.format_messages(
    context=qac_triples[0]["context"],
    question=qac_triples[0]["question"],
    format_instructions=format_instructions
)
answer_generation_chain = bare_template | answer_generation_llm
# Let's inspect the result of one
response = answer_generation_chain.invoke({"content" : messages})
output_dict = answer_output_parser.parse(response.content)

for triple in tqdm(qac_triples):
  messages = prompt_template.format_messages(
      context=triple["context"],
      question=triple["question"],
      format_instructions=format_instructions
  )
  response = answer_generation_chain.invoke({"content" : messages})
  try:
    output_dict = answer_output_parser.parse(response.content)
  except Exception as e:
    continue
  triple["answer"] = output_dict["answer"]

##### RAG Evaluation Using ragas
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    answer_correctness,
    answer_similarity
)

from ragas.metrics.critique import harmfulness
from ragas import evaluate

def create_ragas_dataset(rag_pipeline, eval_dataset):
  rag_dataset = []
  for row in tqdm(eval_dataset):
    answer = rag_pipeline.invoke({"question" : row["question"]})
    rag_dataset.append(
        {"question" : row["question"],
         "answer" : answer["response"],
         "contexts" : [context.page_content for context in answer["context"]],
         "ground_truths" : [row["ground_truth"]]
         }
    )
  rag_df = pd.DataFrame(rag_dataset)
  rag_eval_dataset = Dataset.from_pandas(rag_df)
  return rag_eval_dataset

def evaluate_ragas_dataset(ragas_dataset):
  result = evaluate(
    ragas_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        context_relevancy,
        answer_correctness,
        answer_similarity
    ],
  )
  return result

from tqdm import tqdm
import pandas as pd

basic_qa_ragas_dataset = create_ragas_dataset(retrieval_augmented_qa_chain, eval_dataset)
basic_qa_result = evaluate_ragas_dataset(basic_qa_ragas_dataset)
