# https://angelina-yang.medium.com/code-implementation-for-storm-27d31a96bc4b

from litellm import completion, acompletion
import json
from tavily import TavilyClient
import anthropic
import json

from fastembed import TextEmbedding
from typing import List
from qdrant_client import QdrantClient

tavily = TavilyClient(api_key=TAVILY_API_KEY)

def generate_related_topics(topic: str) -> list[str]:
    system_prompt = """I want to write a long-form article about a topic. I will give you the topic and I want you to suggest 3 related sub-topics to expand the content."""

    user_prompt = """Here's the topic:\n\nTOPIC:{topic}"""
    try:
        response = completion(
            api_key=CLAUDE_API_KEY,
            model="claude-3-opus-20240229",
            messages=[{"content": system_prompt, "role": "system"}, {"content": user_prompt.format(topic=topic), "role": "user"}],
            temperature=0.5,
            max_tokens=200,
        )
        response = response.choices[0].message.content

        return response
    except Exception as e:
        print(f"Error in translation: {e}")

def generate_perspectives(topic: str, related_topics: str) -> list[str]:
    system_prompt = """You need to select a group of 3 writers who will work together to write a comprehensive article on the topic. Each of them represents a different perspective , role , or affiliation related to this topic .
    You can use other related topics for inspiration. For each role, add description of what they will focus on. Give your answer strictly in the following format without adding anything additional:1. short summary of writer one: description \n 2. short summary of writer two: description \n...\n\n"""

    user_prompt = """Here's the topic:\n\nTOPIC:{topic}\n\nRelated topics: {related_topics}"""
    try:

        response = completion(
            api_key = CLAUDE_API_KEY,
            model="claude-3-opus-20240229",
            messages=[{ "content": system_prompt,"role": "system"},{ "content": user_prompt.format(topic=topic, related_topics=related_topics),"role": "user"}],
            temperature=0.5,
            max_tokens=500,
        )
        response = response.choices[0].message.content

        return response
    except Exception as e:
        print(f"Error in translation: {e}")

def generate_question(topic: str, perspective: str, history: list[str]):
    system_prompt = """You are an experienced writer and want to edit a long-form article about a given topic . Besides your identity as a writer, you have a specific focus when researching the topic .
Now , you are chatting with an expert to get information . Ask good questions to get more useful information .
Please ask no more than one question at a time and don 't ask what you have asked before. Other than generating the question, don't adding anything additional.
Your questions should be related to the topic you want to write.\n\nConversation history: {history}\n\n"""

    user_prompt = """Here's the topic:\n\nTOPIC:{topic}\n\nYour specific focus: {perspective}\n\nQuestion:"""

    context = "\n".join(history)
    try:

        response = completion(
            api_key = CLAUDE_API_KEY,
            model="claude-3-opus-20240229",
            messages=[{ "content": system_prompt.format(history=context),"role": "system"},{ "content": user_prompt.format(topic=topic, perspective=perspective),"role": "user"}],
            temperature=0.5,
            max_tokens=200,
        )
        response = response.choices[0].message.content

        return response
    except Exception as e:
        print(f"Error in translation: {e}")

def generate_answer(topic: str, question: str, context: str):
    system_prompt = """You are an expert who can use information effectively . You are chatting with a
writer who wants to write an article on topic you know . You
have gathered the related information and will now use the information to form a response.
Make your response as informative as possible and make sure every sentence is supported by the gathered information.\n\nRelated information: {context}\n\n"""

    user_prompt = """Here's the topic:\n\nTOPIC:{topic}\n\nQuestion: {question}"""
    try:

        response = completion(
            api_key = CLAUDE_API_KEY,
            model="claude-3-opus-20240229",
            messages=[{ "content": system_prompt.format(context=context),"role": "system"},{ "content": user_prompt.format(topic=topic, question=question),"role": "user"}],
            temperature=0.5,
            max_tokens=600,
        )
        response = response.choices[0].message.content

        return response
    except Exception as e:
        print(f"Error in translation: {e}")

def generate_outline(topic: str) -> str:
    system_prompt = """Write an outline for an article about a given topic.
Here is the format of your writing:
Use "#" Title " to indicate section title , "##" Title " to indicate subsection title , "###" Title " to indicate subsubsection title , and so on.
Do not include other information.\n\n"""

    user_prompt = """Here's the topic:\n\nTOPIC:{topic}"""

    try:

        response = completion(
            api_key = CLAUDE_API_KEY,
            model="claude-3-opus-20240229",
            messages=[{ "content": system_prompt,"role": "system"},{ "content": user_prompt.format(topic=topic),"role": "user"}],
            temperature=0.5,
            max_tokens=500,
        )
        response = response.choices[0].message.content

        return response
    except Exception as e:
        print(f"Error in translation: {e}")
      
def refine_outline(topic: str, outline: str, conversation: list[list[str]]) -> str:
    system_prompt = """I want you to improve an outline of an article about {topic} topic. You already have a draft outline given below that
covers the general information. Now you want to improve it based on the given
information learned from an information - seeking conversation to make it more
comprehensive. Here is the format of your writing:
Use "#" Title " to indicate section title , "##" Title " to indicate
subsection title , "###" Title " to indicate subsubsection title , and so on. Do not include other information.\n\ndraft outline: {outline}\n\n"""

    user_prompt = """learned information: {conversation}"""
    flattened_list = [item for sublist in conversation for item in sublist]
    context = ''.join(flattened_list)
    try:
        response = completion(
            api_key = CLAUDE_API_KEY,
            model="claude-3-opus-20240229",
            messages=[{ "content": system_prompt.format(topic=topic, outline=outline),"role": "system"},{ "content": user_prompt.format(conversation=context),"role": "user"}],
            temperature=0.5,
            max_tokens=800,
        )
        response = response.choices[0].message.content

        return response
    except Exception as e:
        print(f"Error in translation: {e}")

def write_section(section: str) -> str:

    search_result = client.query(
        collection_name="demo_collection",
        query_text=section,
        limit=5
    )
    references = generate_references_string(search_result)


    system_prompt = """You are an expert in writing. I will give you an outline of
    a section of a blog and several references. You will generate the article of the section using the provided refrences.
    You MUST cite your writing using the given sources. Do not include other information. Include 'reference id' for each sentence in this format: [ref_id]. Your response MUST be in markdown format.\n\nREFERENCES: {references}\n\n"""


    user_prompt = """SECTION OUTLINE: {section}"""

    try:
        response = completion(
            api_key = CLAUDE_API_KEY,
            model="claude-3-opus-20240229",
            messages=[{ "content": system_prompt.format(references=references),"role": "system"},{ "content": user_prompt.format(section=section),"role": "user"}],
            temperature=0.5,
            max_tokens=1500,
        )
        response = response.choices[0].message.content

        return response
    except Exception as e:
        print(f"Error in translation: {e}")

def generate_references_string(references):
    output = []
    for ref in references:
        ref_id = ref.id
        ref_url = ref.metadata.get('source', '')
        ref_content = ref.metadata.get('document', '')

        # Construct a formatted string for each reference
        reference_str = f"Reference ID:\n {ref_id}\nURL: {ref_url}\nContent: {ref_content}\n"

        output.append(reference_str)

    return '\n'.join(output)

topic = 'Building A Powerful LinkedIn Presence'
related_topics = generate_related_topics(topic)
related_topics_json = json.dumps(related_topics)

perspectives = generate_perspectives(topic, related_topics)
perspectives = perspectives.split('\n\n')

res = generate_question(topic, perspectives[0], [])
all_conversations = []
references = []
duplicate_references = set()
total_questions = 3
for p in perspectives[:1]:
    history = []
    for i in range(total_questions):
        question = generate_question(topic, p, history)
        print(f"QUESTION: {question}")
        history.append(question)
        tavily_response = tavily.search(query=question)
        results = tavily_response['results']
        all_context = ""
        for result in results:
            all_context += result['content'] + "\n"
            if result['url'] in duplicate_references:
                continue
            duplicate_references.add(result['url'])
            references.append({"title": result['title'], "source": result['url'], "content": result['content']})
        answer = generate_answer(topic, question, all_context)
        history.append(answer)
    all_conversations.append(history)
print("DONE.")

outline = generate_outline(topic)
refined_outline = refine_outline(topic, outline, all_conversations)
rr = refined_outline.split("\n\n")

documents = []
metadata = []
ids = []
for i in range(len(references)):
    documents.append(references[i]['title'] + " " + references[i]["content"])
    metadata.append({"source":references[i]["source"]})
    ids.append(i)

client = QdrantClient(":memory:")
client.delete_collection(collection_name="demo_collection")
ids = client.add(
    collection_name="demo_collection",
    documents=documents,
    metadata=metadata,
    ids=ids
)

search_result = client.query(
    collection_name="demo_collection",
    query_text=rr[0],
    limit=5
)

for s in search_result:
    print(s.metadata['source'])
    print(s.id)

article = ""
for section_outline in rr[1::]:
    sec = write_section(section_outline)
    article += sec + "\n\n"

print("article DONE!")

print(article)


