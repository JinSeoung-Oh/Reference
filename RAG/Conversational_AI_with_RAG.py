## From https://medium.com/@praveenveera92/building-conversational-ai-with-rag-a-practical-guide-61bf449bef67

from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
import glob
from itertools import groupby

load_dotenv('../.env')
data_source = '../dataset/docs'
persist_path = '../dataset/db'
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

extension = '*.pdf'
file_paths = glob.glob(f'{data_source}/{extension}')

def get_document_splits_with_ids(docs):
    """
    This function is used to assign unique IDs to each document chunk for further processing.
    The IDs are generated in the format: {source_file_name}_{page_number}_{chunk_index}
    Args:
        docs (list): A list of document chunks.
    Returns:
        list: A list of document IDs.
    """
    document_ids = []  # Initialize an empty list to store document IDs

    # Group the document chunks by page number
    for page, chunks in groupby(docs, lambda chunk: chunk.metadata['page']):
        # Generate document IDs for each chunk in the current page group
        document_ids.extend([f"{chunk.metadata['source'].split('/')[-1]}_{page}_{chunk_id}" for chunk_id, chunk in enumerate(chunks)])

    return document_ids  # Return the list of document IDs

##### Chroma DB Client for Ingestion (db.py)
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_chroma import Chroma  # Import the Chroma class for creating and managing a vector database
from langchain_openai import OpenAIEmbeddings  
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  

try:
    for file_path in file_paths:
        print('[INFO] (Started) Processing file:', os.path.basename(file_path)) 
        loader = PyPDFLoader(file_path)  
        docs = loader.load() 
        print('[INFO] File Processed in Loader:', file_path)  
        docs_splits = text_splitter.split_documents(docs)  
        ids = get_document_splits_with_ids(docs_splits) 
        print(f'[INFO] File Processed in Spliter: {len(docs_splits)}') 

        # load it into Chroma
        db = Chroma.from_documents(
            documents=docs_splits,  
            embedding=embeddings, 
            persist_directory=persist_path, 
            ids=ids  # The list of unique IDs for each document chunk
        )
        print(f'[INFO] (Complete) File ingested in ChromaDB: {file_path}') 

    print(f'[INFO] (Complete) All {extension} files ingested succesfully in path {data_source}')
    print(f"[INFO] ChromaDB collection Name: {db._LANGCHAIN_DEFAULT_COLLECTION_NAME}, with collection count {db._collection.count()}")  # Print the name and count of the Chroma vector database collection

except Exception as e:
    print('[ERROR] Failed to process file:', file_path, e)

persistent_client = chromadb.PersistentClient(
                        path=persist_path
                    )
LANGCHAIN_DEFAULT_COLLECTION_NAME = 'langchain' 
collection = persistent_client.get_or_create_collection(LANGCHAIN_DEFAULT_COLLECTION_NAME)
persistent_collection = persistent_client.get_collection(LANGCHAIN_DEFAULT_COLLECTION_NAME)
embedded_query = embeddings.embed_query('What is few shot learning?')
query_result = persistent_collection.query(embedded_query)

db = Chroma(
    client=persistent_client,  # Pass the persistent_client instance created earlier
    collection_name=LANGCHAIN_DEFAULT_COLLECTION_NAME,  # Use the default collection name defined earlier
    embedding_function=embeddings,  # Pass the embeddings instance for generating text embeddings
)
results = db.similarity_search_with_relevance_scores('What is LLM?') 

for i in results:
    print(i)  # Print the search result (a tuple containing the document and relevance score)
    print(f"[INFO] Page : {i[0].metadata['page']}, Source:{i[0].metadata['source']}, relevance source: {i[1]}")  # Print additional information about the search result, including the page number, source file, and relevance score


##### Building a Basic RAG Chain
retriever = db.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 3} 
)

retriever.invoke("what is LLM?") 

##### 
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = hub.pull("rlm/rag-prompt")
prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:")

def format_docs(docs):
    """
    Format a list of documents by joining their page_content with newline separators.
    Args:
        docs (list): A list of document objects.
    Returns:
        str: A string containing the concatenated page_content of all documents, separated by newlines.
    """
    return "\n\n".join(doc.page_content for doc in docs)# Define the Retrieval-Augmented Generation (RAG) chain

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}  # Retrieve and format the relevant documents, and pass the question as-is
    | prompt  # Apply the prompt to the context and question
    | llm  # Pass the prompted input to the language model
    | StrOutputParser()  # Parse the output of the language model using the StrOutputParser
)

for chunk in rag_chain.stream("What is Few shot learning?"):
    print(chunk, end="", flush=True) 


##### RAG with customised prompt
from langchain_core.prompts import PromptTemplate  # Import the PromptTemplate class from the langchain_core.prompts module

# Customizing the prompt
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}  # This placeholder will be replaced with the retrieved context (relevant documents)

Question: {question}  

Helpful Answer:"""  

# Create a PromptTemplate instance from the template string
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}  # Retrieve and format relevant documents, pass the question as-is
    | custom_rag_prompt  # Apply the custom prompt to the context and question
    | llm  # Pass the prompted input to the language model
    | StrOutputParser()  # Parse the output of the language model as a string
)

# Stream the output of the RAG chain for the question "What is LLM?"
for chunk in rag_chain.stream("What is Few shot learning?"):
    print(chunk, end="", flush=True)  # Print each chunk of the output without newlines and flush the output buffer


###################### RAG with Chat History ##########################
from langchain.chains import create_history_aware_retriever  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),  # Set the system prompt
        MessagesPlaceholder("chat_history"),  # Placeholder for the chat history
        ("human", "{input}"),  # Placeholder for the user's input question
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm,  # Pass the language model instance
    retriever,  # Pass the retriever instance
    contextualize_q_prompt  # Pass the prompt for contextualizing the question
)


### Building the Full QA Chain
from langchain.chains import create_retrieval_chain  
from langchain.chains.combine_documents import create_stuff_documents_chain  

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),  # Set the system prompt
        MessagesPlaceholder("chat_history"),  # Placeholder for the chat history
        ("human", "{input}"),  # Placeholder for the user's input question
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

#### usage
from langchain_core.messages import HumanMessage
chat_history = []

first_question = "What is LLM?"
ai_response_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})  
print('user query:', first_question) 
print('ai response:', ai_response_1["answer"])  
chat_history.extend([HumanMessage(content=first_question), ai_response_1["answer"]])  

# Ask the second question
second_question = "What are the different types of it?"
ai_response_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})  
chat_history.extend([HumanMessage(content=second_question), ai_response_2["answer"]]) 
print('user query:', (second_question)) 
print('ai response:', ai_response_2["answer"])

# Ask the third question
third_question = "Can you translate your previous response to French?"
ai_response_3 = rag_chain.invoke({"input": third_question, "chat_history": chat_history})
print('user query:', (third_question)) 
print('ai response:', ai_response_3["answer"]) 
chat_history.extend([HumanMessage(content=third_question), ai_response_3["answer"]])  
