## From https://towardsdatascience.com/advanced-retrieval-techniques-for-better-rags-c53e1b03c183

import os
from dotenv import load_dotenv

load_dotenv()
pyt

import bs4
from operator import itemgetter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# 1. Index the documents
# 1.1 Load the documents
web_loader = WebBaseLoader(
    web_paths=("https://docs.djangoproject.com/en/5.1/topics/security/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(id=("docs-content"))),
)
docs = web_loader.load()

# 1.2 Split the documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=100
)
splits = text_splitter.split_documents(docs)

# 1.3 Index the documents
vector_store = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
retriever = vector_store.as_retriever()


# 2. Helper functions
def output_parser(x):
    "This helper function parses the LLM output, prints it, and returns it."
    parsed_output = StrOutputParser().invoke(x)
    print("\n" + parsed_output + "\n")

    return parsed_output

def qa_constructor(questions: list[str]) -> str:
    qa_pairs = []
    for q in questions:
        r = chain.invoke(q)
        qa_pairs.append((q, r))

    qa_pairs_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs]).strip()

    print("\n" + "Generated QA pairs:")
    print(qa_pairs_str)

    return qa_pairs_str


# 3. Create a basic RAG chain
# 3.1 Define the prompt template
rag_template = """
    Answer the question in the following context:
    {context}
    
    Question: {question}
    """

prompt_template = ChatPromptTemplate.from_template(rag_template)

# 3.2 Define the model
llm = ChatOpenAI(temperature=0.5)

# 3.3 Define the chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | output_parser
)


# 4. Create a decomposition chain
decomposition_template = """
Break the following user question into smaller, more specific questions.
Provide these subquestions separated by newlines. 
Do not rephrase if you see unknown terms.
Question: {question}
subquestions:
"""

decomposition_prompt_template = ChatPromptTemplate.from_template(decomposition_template)


decomposition_chain = (
    {"question": RunnablePassthrough()}
    | decomposition_prompt_template
    | llm
    | output_parser
    | (lambda x: x.split("\n"))
    | qa_constructor
)



# 5. Create the final RAG chain
decompsition_rag_template = """
    Answer the question in the following context:
    {context}

    Here are some background questions and answers that will help you answer the question:
    {qa_pairs}

    Question: {question}
    """

decomposition_prompt_template = ChatPromptTemplate.from_template(
    decompsition_rag_template
)

decomposition_rag_chain = (
    {
        "context": itemgetter("question") | retriever,
        "qa_pairs": itemgetter("question") | decomposition_chain,
        "question": RunnablePassthrough(),
    }
    | decomposition_prompt_template
    | llm
    | output_parser
)


# 6. Invoke the chain
decomposition_rag_chain.invoke(
    {"question": "Can SSL certification prevent SQL injection attacks?"}
)


##### Query decomposition and Recursive answering
# 1. Query decomposition
# ---------------------------------------------------
# Decomposition chain
decomposition_prompt = """
Break the following user question into smaller, more specific questions.
Provide these subquestions separated by newlines. 
Do not rephrase if you see unknown terms.
Question: {question}
subquestions:
"""

decomposition_prompt_template = ChatPromptTemplate.from_template(decomposition_prompt)

# Answer chain
decompositon_chain = (
    decomposition_prompt_template | llm | output_parser | (lambda x: x.split("\n"))
)


questions = decompositon_chain.invoke(
    "Can SSL certification prevent SQL injection attacks?"
)

# 2. Retrieval and Recursive answering
# ---------------------------------------------------
recursive_answering_prompt = """
You need to answer the questions below in the following context.
Question: {question}
Context: {context}

Here are any prior questions and your answers:
{qa_pairs}

Answer:
"""

recursive_answering_prompt_template = ChatPromptTemplate.from_template(
    recursive_answering_prompt
)

recursive_answering_prompt_chain = (
    {
        "question": itemgetter(
            "question",
        ),
        "context": itemgetter(
            "context",
        ),
        "qa_pairs": itemgetter(
            "qa_pairs",
        ),
    }
    | recursive_answering_prompt_template
    | llm
    | output_parser
)


def recursively_answer_questions(questions: list[str]) -> str:

    qa_pairs = ""
    for question in questions:
        docs = retriever.invoke(question)
        context = " ".join([doc.page_content for doc in docs])

        answer = recursive_answering_prompt_chain.invoke(
            {
                "question": question,
                "context": context,
                "qa_pairs": qa_pairs,
            }
        )

        qa_pairs += f"Q: {question}\nA: {answer}\n"

    return qa_pairs


# 3. Final answering
# ---------------------------------------------------
final_prompt = """
Provide a comprehensive answer to the following question based on the subquestions you answered.
Question: {question}

Here are the subquestions and answers you provided:
{qa_pairs}

Answer: 
"""

final_prompt_template = ChatPromptTemplate.from_template(final_prompt)

final_prompt_chain = (
    {
        "question": itemgetter(
            "question",
        ),
        "qa_pairs": itemgetter(
            "qa_pairs",
        ),
    }
    | final_prompt_template
    | llm
    | output_parser
)

final_answer = final_prompt_chain.invoke(
    {
        "question": "Can SSL certification prevent SQL injection attacks?",
        "qa_pairs": qa_pairs,
    }
)

print(final_answer)


#### Generating followup questions and answers
# 1. Chain for answering the question and generate followup quesion
answer_and_followup_prompt = """
You need to answer the question below in the following context and generate a followup question.
Context: {context}
Question: {question}

Also, consider any prior questions and answers you've generated:
Prior questions and answers: {prior_qa}

Provide output as a dictionary with the following keys:
Answer,
Followup
"""

answer_and_followup_prompt_template = ChatPromptTemplate.from_template(
    answer_and_followup_prompt
)


answer_and_followup_chain = (
    {
        "context": itemgetter("context"),
        "question": itemgetter("question"),
        "prior_qa": itemgetter("prior_qa"),
    }
    | answer_and_followup_prompt_template
    | llm
    | output_parser
    | loads
)

# 2. Recursive function to dive deep into the domain
def recursively_ask(question, prior_qa="", n=3):
    context = retriever.invoke(question)

    response = answer_and_followup_chain.invoke(
        {
            "context": context,
            "question": question,
            "prior_qa": prior_qa,
        }
    )

    answer, followup = response["Answer"], response["Followup"]

    prior_qa += f"Q:{question}\nA:{answer}\n\n"
    n -= 1
    
    if n == 0:
        return prior_qa
    else:
        return recursively_ask(followup, prior_qa, n)


question = "Can SSL certification prevent SQL injection attacks?"
question_and_answers = recursively_ask(question, n=3)


# 3. Final RAG execution
final_prompt = """
Provide a comprehensive answer to the following question based on the subquestions you answered.
Question: {question}

Here are the subquestions and answers you provided:
{qa_pairs}

Answer: 
"""

final_prompt_template = ChatPromptTemplate.from_template(final_prompt)

final_prompt_chain = (
    {
        "question": itemgetter(
            "question",
        ),
        "qa_pairs": itemgetter(
            "qa_pairs",
        ),
    }
    | final_prompt_template
    | llm
    | output_parser
)

final_answer = final_prompt_chain.invoke(
    {
        "question": question,
        "qa_pairs": question_and_answers,
    }
)




