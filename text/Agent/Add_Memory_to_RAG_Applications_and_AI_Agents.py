### From https://medium.com/@3rdSon/how-to-add-memory-to-rag-applications-and-ai-agents-0066fe068755

"""
After building a RAG application, the need for persistent memory for production readiness became evident, 
as existing resources mostly covered in-memory storage, which isn’t sufficient for scalable applications. 
To add memory to a RAG application or AI agent, here’s the approach:

1. Understanding Memory’s Role:
   Memory enables the agent to relate past interactions to new queries, as seen in ChatGPT’s ability to recall previous questions in a conversation, 
   providing relevant follow-up answers.
   For RAG, memory allows the agent to infer and respond based on prior questions, enhancing its contextual continuity.

2. Components Needed for Memory:
    -1. Database: A persistent storage system (e.g., MongoDB) to store user queries, agent responses, 
                  and associated metadata like chat IDs and user information.
    -2. Retrieval Function: A function to retrieve past questions or context each time a new query is received.
    -3. Relation Check Function: A function using a language model (LLM) to determine if the current question relates to previous ones. 
                                 If related, it combines the current and past queries to form a coherent question, 
                                 embedding it and sending it to a vector database or AI agent. If unrelated, it proceeds with the question as-is.

3. Implementation Tools:
   Technologies such as Langchain for LLM interaction, OpenAI GPT-3.5 for LLM capabilities, and pymongo for MongoDB integration 
   support this memory integration.
"""

from pymongo import MongoClient
from datetime import datetime
from bson.objectid import ObjectId

# Connect to MongoDB (modify the URI to match your setup)
client = MongoClient("mongodb://localhost:27017/")
db = client["your_database_name"] # The name of your database
collection = db["my_ai_application"] # The name of the collection

# Sample document to be inserted
document = {
    "_id": ObjectId("66c990f566416e871fdd0b43"),  # you can omit this to auto-generate
    "question": "Who is the President of America?",
    "email": "nnajivictorious@gmail.com",
    "response": "The current president of the United States is Joe Biden.",
    "chatId": "52ded9ebd9ac912c8433b699455eb655",
    "userId": "6682632b88c6b314ce887716",
    "isActive": True,
    "isDeleted": False,
    "createdAt": datetime(2024, 8, 24, 7, 51, 17, 503000),
    "updatedAt": datetime(2024, 8, 24, 7, 51, 17, 503000)
}

# Insert the document into the collection
result = collection.insert_one(document)
print(f"Inserted document with _id: {result.inserted_id}")
-----------------------------------------------------------------------------------------
from typing import List

client = MongoClient("mongodb://localhost:27017/")
db = client.your_database_name
collection = db.my_ai_application
# no need to initialize this connection if you had already done it

def get_last_three_questions(email: str, chat_id: str) -> List[str]:
    """
    Retrieves the last three questions asked by a user in a specific chat session.

    Args:
        email (str): The user's email address used to filter results.
        chat_id (str): The unique identifier for the chat session.

    Returns:
        List[str]: A list containing the last three questions asked by the user,
                   ordered from most recent to oldest.
    """
    query = {"email": email, "chatId": chat_id}
    results = collection.find(query).sort("createdAt", -1).limit(3)
    questions = [result["question"] for result in results]
    return questions


# Call the function
past_questions = get_last_three_questions("nnajivictorious@gmail.com", "52ded9ebd9ac912c8433b699455eb655")
-----------------------------------------------------------------------------------------
from langchain_openai import OpenAI 
from dotenv import load_dotenv

# Load your OpenAI API key from .env file
load_dotenv()
CHAT_LLM = OpenAI()

new_question_modifier = """
Your primary task is to determine if the latest question requires context from the chat history to be understood.

IMPORTANT: If the latest question is standalone and can be fully understood without any context from the chat history or is not related to the chat history, you MUST return it completely unchanged. Do not modify standalone questions in any way.

Only if the latest question clearly references or depends on the chat history should you reformulate it as a complete, standalone legal question. When reformulating:

"""

def modify_question_with_memory(new_question: str, past_questions: List[str]) -> str:
    """
    Modifies a new question by incorporating past questions as context.

    This function takes a new question and a list of past questions, combining them
    into a single prompt for the language model (LLM) to generate a standalone question
    with sufficient context. If there are no past questions, the new question is returned as-is.

    Args:
        new_question (str): The latest question asked.
        past_questions (List[str]): A list of past questions for context.

    Returns:
        str: A standalone question that includes necessary context from past questions.
    """
    if past_questions:
        past_questions_text = " ".join(past_questions)
        # Combine the system prompt with the past questions and the new question
        system_prompt = f"{new_question_modifier}\nChat history: {past_questions_text}\nLatest question: {new_question}"
        # Get the standalone question using the LLM
        standalone_question = CHAT_LLM.invoke(system_prompt)
    else:
        standalone_question = new_question

    return standalone_question


modified_question = modify_question_with_memory(new_question="your new question here", past_questions=past_questions)# Load your OpenAI API key from .env file



