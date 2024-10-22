## From https://generativeai.pub/building-an-ai-powered-web-search-assistant-using-gpt-4-and-streamlit-0687afb15265
## This is just reference. Based on this, can build web_search engine with AI

!pip install streamlit phi openai duckduckgo-search

import streamlit as st
st.title("AI Web Search Assistant 🤖")
st.caption("This app allows you to search the web using GPT-4o")
openai_access_token = st.text_input("OpenAI API Key", type="password")
query = st.text_input("Enter the Search Query")
if query and openai_access_token:
    st.write("Search results will appear here.")

from phi.assistant import Assistant
from phi.tools.duckduckgo import DuckDuckGo
from phi.llm.openai import OpenAIChat

# Create the assistant with DuckDuckGo and GPT-4
if openai_access_token:
    assistant = Assistant(
        llm=OpenAIChat(
            model="gpt-4o",
            max_tokens=1024,
            temperature=0.9,
            api_key=openai_access_token), tools=[DuckDuckGo()], show_tool_calls=True
    )
    # Process the query
    if query:
        response = assistant.run(query, stream=False)
        st.write(response)

  with st.sidebar:
    st.image("path_to_your_picture.jpg", width=100)
    st.header("About Me")
    st.write("")
    st.write("")
    st.write("[LinkedIn](https://www.linkedin.com/in/your-linkedin-id)")


#####
streamlit run eb_search_ai_assistant.py


