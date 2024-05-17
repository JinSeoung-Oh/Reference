# https://pub.towardsai.net/how-to-optimize-chunk-sizes-for-rag-in-production-fae9019796b6

!pip install -qqq nest-asyncio llama-index openai Cython torch torchvision

# Getting lamma to work with us
from llama_index.core  import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator
)
from llama_index.llms.openai import OpenAI
import openai
import time

# importing openAI API key from googcle colab secrets 
from google.colab import userdata
openai.api_key = userdata.get('openai_key')

# Confirm torch is working by using following code
import torch
print(torch.__version__)

# Some async tasks need to be done
import nest_asyncio
nest_asyncio.apply()

# Load Data
documents = SimpleDirectoryReader("./data").load_data()
print(len(documents))
print(documents[0])
eval_documents = documents[0:10]

# Load Data
import random 
documents = SimpleDirectoryReader("./data").load_data()
eval_documents = [documents[random.randint(0, len(documents)-1)] for _ in range(10)]


import time
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

eval_questions_all = []
num_questions_per_chunk = 1

data_generator = RagDatasetGenerator.from_documents(eval_documents)

eval_questions = data_generator.generate_questions_from_nodes()
eval_questions_all.append(eval_questions.to_pandas()['query'].to_list())

# Define Faithfulness and Relevancy Evaluators which are based on GPT-4
faithfulness_gpt3_5_t = FaithfulnessEvaluator()
relevancy_gpt3_5_t = RelevancyEvaluator()


questions = eval_questions.to_pandas()['query'].to_list() 
display(questions)

# Tuning question generation to fit your business case
q_gen_query = f"You are a scientific researcher. \
            Your task is to setup {num_questions_per_chunk} questions. \
            The questions must be related to following \
            1. my interest 1 2.My interest 2 3. My interest 3 \
            Restrict the questions to the context information provided."
data_generator = RagDatasetGenerator.from_documents(eval_documents,  
                  question_gen_query=q_gen_query)

# Define function to calculate average response time, average faithfulness and average relevancy metrics for given chunk size
def evaluate_response_time_and_accuracy(chunk_size, eval_questions):
    total_response_time = 0
    total_faithfulness = 0
    total_relevancy = 0

    response_vectors = []
    # create vector index
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0, chunk_size = chunk_size )
    # Update settings during each run 
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = round(chunk_size/10,0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=100)
    
    vector_index = VectorStoreIndex.from_documents(eval_documents)

    query_engine = vector_index.as_query_engine()
    num_questions = len(eval_questions)

    for question in eval_questions:
        start_time = time.time() 
        # Generate a response vector
        response_vector = query_engine.query(question)

        elapsed_time = time.time() - start_time
        
        # Evaluate the quality of response 
        faithfulness_result = faithfulness_gpt3_5_t.evaluate_response( response=response_vector ).passing

        relevancy_result = relevancy_gpt3_5_t.evaluate_response( query=question, response=response_vector ).passing
        
        # Document the quality of resposne
        response_vectors.append({"chunk_size" : chunk_size,
                                 "question" : question,
                                 "response_vector" : response_vector,
                                 "faithfulness_result" : faithfulness_result,
                                 "relevancy_result" : relevancy_result})

        total_response_time += elapsed_time
        total_faithfulness += faithfulness_result
        total_relevancy += relevancy_result
    
    # Get average score over all questions
    average_response_time = total_response_time / num_questions
    average_faithfulness = total_faithfulness / num_questions
    average_relevancy = total_relevancy / num_questions

    return average_response_time, average_faithfulness, average_relevancy, response_vectors

# Iterate over different chunk sizes to evaluate the metrics to help fix the chunk size.
response_vectors_all = []
for chunk_size in [128, 256, 512, 1024, 2048]:
  avg_time, avg_faithfulness, avg_relevancy, response_vectors = evaluate_response_time_and_accuracy(chunk_size, all_questions)
  [response_vectors_all.append(i) for i in response_vectors]
  print(f"Chunk size {chunk_size} - Average Response time: {avg_time:.2f}s, Average Faithfulness: {avg_faithfulness:.2f}, Average Relevancy: {avg_relevancy:.2f}")
  time.sleep(20)
