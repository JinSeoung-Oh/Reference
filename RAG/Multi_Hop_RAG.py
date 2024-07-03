## From https://blog.stackademic.com/multi-hop-retrieval-and-reasoning-for-complex-questions-using-dspy-qdrant-and-llama3-841580138a81
# A multi-hop question is a type of question that requires multiple steps or “hops” to arrive at an answer. 
# Rather than being a straightforward query with a single answer,
# a multi-hop question involves breaking down the original question into smaller parts, each of which requires its own analysis.

"""
requirements.txt file.

dspy-ai
qdrant-client
ollama
wikipedia
fastembed
langchain
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader
from qdrant_client import QdrantClient
from dspy.retrieve.qdrant_rm import QdrantRM
import dspy
from dsp.utils import deduplicate

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""


    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField()

class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""


    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()


        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops


    def forward(self, question):
        context = []


        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)


        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=512,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

docs = WikipediaLoader(query="Leonardo DiCaprio").load_and_split(text_splitter = text_splitter )
# List to hold the content of each document
doc_contents = [doc.page_content for doc in docs]


# List to hold the IDs for each document
doc_ids = list(range(1, len(docs) + 1))

# Initialize the client
client = QdrantClient(":memory:")


client.add(
    collection_name="leo_collection",
    documents=doc_contents,
    ids=doc_ids,
)

qdrant_retriever_model = QdrantRM("leo_collection", client, k=10)


ollama_model = dspy.OllamaLocal(model="llama3",model_type='text',
                                max_tokens=350,
                                temperature=0.1,
                                top_p=0.8, frequency_penalty=1.17, top_k=40)


dspy.settings.configure(lm= ollama_model, rm=qdrant_retriever_model)

# Ask any question you like to this simple RAG program.
my_question = "Give me all the co-actors of Leonardo DiCaprio in the movie in which one of his co-stars was Robert De Niro?"


# Get the prediction. This contains `pred.context` and `pred.answer`.
uncompiled_baleen = SimplifiedBaleen()  # uncompiled (i.e., zero-shot) program
pred = uncompiled_baleen(my_question)


## If you want to see all history
ollama_model.inspect_history(n=3)




