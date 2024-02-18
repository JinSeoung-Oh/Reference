## From https://medium.com/mlearning-ai/go-open-go-lean-llmware-now-can-boost-your-ai-powered-enterprise-953fd1223a34

### What is LLMWare?
### LLMWare is a company that has developed a series of specialized AI models called SLIMs (Structured Language Instruction Models) to address the challenges faced by businesses 
### in implementing AI for complex tasks. These SLIMs are designed to seamlessly orchestrate multi-step workflows, generate structured data, 
### and operate within secure private clouds.
### They are budget-friendly and open-source, allowing businesses to customize them to their unique needs. SLIMs are tailored to specific tasks, 
### helping businesses automate their workflows and improve efficiency. 
### LLMWare aims to provide a solution that addresses the three major roadblocks to AI adoption in complex tasks: siloed operations, cryptic outputs, and privacy concerns.

### Step-by-step guide
### 1. Understand Your Needs 
###    Before you dive into using SLIMs, take some time to understand your business needs. What are the specific challenges you're facing?
###    What kind of AI tasks are you looking to automate? This will help you identify which SLIMs are most suitable for your needs.
### 2. Explore SLIMs
###    Visit the LLMWare.ai website and explore the different SLIMs available. Each SLIM is designed for a specific task, such as NER (Named Entity Recognition), 
###    summarization, and more. Take a look at the documentation and resources provided to get a better understanding of what each SLIM does.
### 3. Choose the Right SLIMs
###    Once you understand your needs and the available SLIMs, choose the SLIMs that best fit your requirements. 
###    Consider factors such as functionality, performance, and compatibility with your existing systems.
### 4. Install and Set Up SLIMs
###    Follow the installation instructions provided by LLMWare.ai to set up the chosen SLIMs on your systems. 
###    Depending on the SLIMs you choose, you may need to install certain dependencies or libraries.
### 5. Integrate SLIMs into Your Workflow
###    Once the SLIMs are set up, integrate them into your existing workflows. 
###    This may involve modifying your code to call the SLIMs for specific tasks or creating new scripts to handle the integration.
### 6. Test and Iterate
###    Test the SLIMs in your workflows to ensure they're functioning as expected. 
###    If you encounter any issues or limitations, reach out to LLMWare.ai for support or consider customizing the SLIMs to better fit your needs.
### 7. Scale Up
###    Once you're satisfied with the performance of the SLIMs, consider scaling up their usage across your organization. 
###    This may involve deploying the SLIMs on more machines or integrating them into additional workflows.
### 8. Stay Updated
###    Keep an eye on new releases and updates from LLMWare.ai to stay up-to-date with the latest improvements and features. 
###    This will help you continuously optimize and improve your AI workflows.
### 9. Provide Feedback
###    If you encounter any issues or have suggestions for improvement, provide feedback to LLMWare.ai. 
###    This will help them enhance the SLIMs and provide better support to their users.

## Build LLMware powered App
mkdir llmware
cd llmware
python -m venv venv

source venv/bin/activate  #activate the venv on Mac/Linux
venv\Scripts\activate  #activate the venv on Windows

pip install llmware
pip install streamlit

import streamlit as st
from llmware.models import ModelCatalog
from llmware.prompts import Prompt

def perform_sum(text):
    #nli_model = ModelCatalog().load_model("slim-nli-tool")
    prompter = Prompt().load_model("llmware/bling-tiny-llama-v0")
    instruction = "What is a brief summary?"
    response_sum = prompter.prompt_main(instruction, context=text)
    return response_sum

def tags(text):
    tags_model = ModelCatalog().load_model("slim-tags-tool")
    response_tags = tags_model.function_call(text, get_logits=False)
    return response_tags

def topics(text):
    topics_model = ModelCatalog().load_model("slim-topics-tool")
    response_topics = topics_model.function_call(text, get_logits=False)
    return response_topics

def intent(text):
    intent_model = ModelCatalog().load_model("slim-intent-tool")
    response_intent = intent_model.function_call(text, get_logits=False)
    return response_intent

def category(text):
    category_model = ModelCatalog().load_model("slim-category-tool")
    response_category = category_model.function_call(text, get_logits=False)
    return response_category

def ner(text):
    ner_model = ModelCatalog().load_model("slim-ner-tool")
    response_ner = ner_model.function_call(text, get_logits=False)
    return response_ner

# Streamlit app layout
st.image('logo.png',use_column_width="auto")
st.title("Intensive Enterprise NLP Tasks")
st.markdown("### using only CPU resources")

# Text input
text = st.text_area("Enter text here:")

# Analysis tools selection
analysis_tools = st.multiselect(
    "Select the analysis tools to use:",
    ["Generate Tags", "Identify Topics",
     "Perform Intent", "Get Category",
     "Perform NER", "Perform Summarization"],
    ["Generate Tags"]  # Default selection
)

# Run the selected TASKS/Agents and display results in plain json format
if st.button("Analyze"):
    results = {}
    
    if "Generate Tags" in analysis_tools:
        results["Generate Tags"] = tags(text)
    if "Identify Topics" in analysis_tools:
        results["Identify Topics"] = topics(text)
    if "Perform Intent" in analysis_tools:
        results["Perform Intent"] = intent(text)
    if "Get Category" in analysis_tools:
        results["Get Category"] = category(text)
    if "Perform NER" in analysis_tools:
        results["Perform NER"] = ner(text)
    if "Perform Summarization" in analysis_tools:
        results["Perform Summarization"] = perform_sum(text)
    
    for tool, response in results.items():
        st.subheader(tool)
        st.json(response)

streamlit run myapp.py

Monitor and Optimize: Monitor the performance of the SLIMs over time and make any necessary optimizations. This may involve adjusting parameters, retraining models, or incorporating feedback from users.

