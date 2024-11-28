### From https://pub.towardsai.net/building-multi-agent-ai-systems-from-scratch-openai-vs-ollama-8ec0dae98639

## Ollama
#- app.py
import streamlit as st
from agents import AgentManager
from utils.logger import logger
import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

def main():
    st.set_page_config(page_title="Multi-Agent AI System", layout="wide")
    st.title("Multi-Agent AI System with Collaboration and Validation")

    st.sidebar.title("Select Task")
    task = st.sidebar.selectbox("Choose a task:", [
        "Summarize Medical Text",
        "Write and Refine Research Article",
        "Sanitize Medical Data (PHI)"
    ])

    agent_manager = AgentManager(max_retries=2, verbose=True)

    if task == "Summarize Medical Text":
        summarize_section(agent_manager)
    elif task == "Write and Refine Research Article":
        write_and_refine_article_section(agent_manager)
    elif task == "Sanitize Medical Data (PHI)":
        sanitize_data_section(agent_manager)

def summarize_section(agent_manager):
    st.header("Summarize Medical Text")
    text = st.text_area("Enter medical text to summarize:", height=200)
    if st.button("Summarize"):
        if text:
            main_agent = agent_manager.get_agent("summarize")
            validator_agent = agent_manager.get_agent("summarize_validator")
            with st.spinner("Summarizing..."):
                try:
                    summary = main_agent.execute(text)
                    st.subheader("Summary:")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"SummarizeAgent Error: {e}")
                    return

            with st.spinner("Validating summary..."):
                try:
                    validation = validator_agent.execute(original_text=text, summary=summary)
                    st.subheader("Validation:")
                    st.write(validation)
                except Exception as e:
                    st.error(f"Validation Error: {e}")
                    logger.error(f"SummarizeValidatorAgent Error: {e}")
        else:
            st.warning("Please enter some text to summarize.")

def write_and_refine_article_section(agent_manager):
    st.header("Write and Refine Research Article")
    topic = st.text_input("Enter the topic for the research article:")
    outline = st.text_area("Enter an outline (optional):", height=150)
    if st.button("Write and Refine Article"):
        if topic:
            writer_agent = agent_manager.get_agent("write_article")
            refiner_agent = agent_manager.get_agent("refiner")
            validator_agent = agent_manager.get_agent("validator")
            with st.spinner("Writing article..."):
                try:
                    draft = writer_agent.execute(topic, outline)
                    st.subheader("Draft Article:")
                    st.write(draft)
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"WriteArticleAgent Error: {e}")
                    return

            with st.spinner("Refining article..."):
                try:
                    refined_article = refiner_agent.execute(draft)
                    st.subheader("Refined Article:")
                    st.write(refined_article)
                except Exception as e:
                    st.error(f"Refinement Error: {e}")
                    logger.error(f"RefinerAgent Error: {e}")
                    return

            with st.spinner("Validating article..."):
                try:
                    validation = validator_agent.execute(topic=topic, article=refined_article)
                    st.subheader("Validation:")
                    st.write(validation)
                except Exception as e:
                    st.error(f"Validation Error: {e}")
                    logger.error(f"ValidatorAgent Error: {e}")
        else:
            st.warning("Please enter a topic for the research article.")

def sanitize_data_section(agent_manager):
    st.header("Sanitize Medical Data (PHI)")
    medical_data = st.text_area("Enter medical data to sanitize:", height=200)
    if st.button("Sanitize Data"):
        if medical_data:
            main_agent = agent_manager.get_agent("sanitize_data")
            validator_agent = agent_manager.get_agent("sanitize_data_validator")
            with st.spinner("Sanitizing data..."):
                try:
                    sanitized_data = main_agent.execute(medical_data)
                    st.subheader("Sanitized Data:")
                    st.write(sanitized_data)
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.error(f"SanitizeDataAgent Error: {e}")
                    return

            with st.spinner("Validating sanitized data..."):
                try:
                    validation = validator_agent.execute(original_data=medical_data, sanitized_data=sanitized_data)
                    st.subheader("Validation:")
                    st.write(validation)
                except Exception as e:
                    st.error(f"Validation Error: {e}")
                    logger.error(f"SanitizeDataValidatorAgent Error: {e}")
        else:
            st.warning("Please enter medical data to sanitize.")

if __name__ == "__main__":
    main()

#- agents/__init__.py
from .summarize_tool import SummarizeTool
from .write_article_tool import WriteArticleTool
from .sanitize_data_tool import SanitizeDataTool
from .summarize_validator_agent import SummarizeValidatorAgent
from .write_article_validator_agent import WriteArticleValidatorAgent
from .sanitize_data_validator_agent import SanitizeDataValidatorAgent
from .refiner_agent import RefinerAgent  # New import
from .validator_agent import ValidatorAgent  # New import

class AgentManager:
    def __init__(self, max_retries=2, verbose=True):
        self.agents = {
            "summarize": SummarizeTool(max_retries=max_retries, verbose=verbose),
            "write_article": WriteArticleTool(max_retries=max_retries, verbose=verbose),
            "sanitize_data": SanitizeDataTool(max_retries=max_retries, verbose=verbose),
            "summarize_validator": SummarizeValidatorAgent(max_retries=max_retries, verbose=verbose),
            "write_article_validator": WriteArticleValidatorAgent(max_retries=max_retries, verbose=verbose),
            "sanitize_data_validator": SanitizeDataValidatorAgent(max_retries=max_retries, verbose=verbose),
            "refiner": RefinerAgent(max_retries=max_retries, verbose=verbose),      # New agent
            "validator": ValidatorAgent(max_retries=max_retries, verbose=verbose)   # New agent
        }

    def get_agent(self, agent_name):
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found.")
        return agent

#- agents/agent_base.py
import ollama
from abc import ABC, abstractmethod
from loguru import logger
import os

class AgentBase(ABC):
    def __init__(self, name, max_retries=2, verbose=True):
        self.name = name
        self.max_retries = max_retries
        self.verbose = verbose

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

    def call_llama(self, messages, temperature=0.7, max_tokens=150):
        """
        Calls the Llama model via Ollama and retrieves the response.

        Args:
            messages (list): A list of message dictionaries.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum number of tokens in the response.

        Returns:
            str: The content of the model's response.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                if self.verbose:
                    logger.info(f"[{self.name}] Sending messages to Ollama:")
                    for msg in messages:
                        logger.debug(f"  {msg['role']}: {msg['content']}")
                
                # Call the Ollama chat API
                response = ollama.chat(
                    model='llama3.2:3b',  # Updated model name
                    messages=messages
                )
                
                # Parse the response to extract the text content
                reply = response['message']['content']
                
                if self.verbose:
                    logger.info(f"[{self.name}] Received response: {reply}")
                
                return reply
            except Exception as e:
                retries += 1
                logger.error(f"[{self.name}] Error during Ollama call: {e}. Retry {retries}/{self.max_retries}")
        raise Exception(f"[{self.name}] Failed to get response from Ollama after {self.max_retries} retries.")

#- agents/refiner_agent.py
from .agent_base import AgentBase

class RefinerAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="RefinerAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, draft):
        messages = [
            {
                "role": "system",
                "content": "You are an expert editor who refines and enhances research articles for clarity, coherence, and academic quality."
            },
            {
                "role": "user",
                "content": (
                    "Please refine the following research article draft to improve its language, coherence, and overall quality:\n\n"
                    f"{draft}\n\nRefined Article:"
                )
            }
        ]
        refined_article = self.call_llama(
            messages=messages,
            temperature=0.5,
            max_tokens=2048
        )
        return refined_article

#- agents/sanitize_data_tool.py
from .agent_base import AgentBase

class SanitizeDataTool(AgentBase):
    def __init__(self, max_retries=3, verbose=True):
        super().__init__(name="SanitizeDataTool", max_retries=max_retries, verbose=verbose)

    def execute(self, medical_data):
        messages = [
            {"role": "system", "content": "You are an AI assistant that sanitizes medical data by removing Protected Health Information (PHI)."},
            {
                "role": "user",
                "content": (
                    "Remove all PHI from the following data:\n\n"
                    f"{medical_data}\n\nSanitized Data:"
                )
            }
        ]
        sanitized_data = self.call_llama(messages, max_tokens=500)
        return sanitized_data

#- agents/sanitize_data_validator_agent.py 

class SanitizeDataValidatorAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="SanitizeDataValidatorAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, original_data, sanitized_data):
        system_message = "You are an AI assistant that validates the sanitization of medical data by checking for the removal of Protected Health Information (PHI)."
        user_content = (
            "Given the original data and the sanitized data, verify that all PHI has been removed.\n"
            "List any remaining PHI in the sanitized data and rate the sanitization process on a scale of 1 to 5, where 5 indicates complete sanitization.\n\n"
            f"Original Data:\n{original_data}\n\n"
            f"Sanitized Data:\n{sanitized_data}\n\n"
            "Validation:"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        validation = self.call_llama(messages, max_tokens=512)
        return validation

#- agents/summarize_tool.py
from .agent_base import AgentBase

class SummarizeTool(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="SummarizeTool", max_retries=max_retries, verbose=verbose)

    def execute(self, text):
        messages = [
            {"role": "system", "content": "You are an AI assistant that summarizes medical texts."},
            {
                "role": "user",
                "content": (
                    "Please provide a concise summary of the following medical text:\n\n"
                    f"{text}\n\nSummary:"
                )
            }
        ]
        summary = self.call_llama(messages, max_tokens=300)
        return summary

#- agents/summarize_validator_agent.py
from .agent_base import AgentBase

class SummarizeValidatorAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="SummarizeValidatorAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, original_text, summary):
        system_message = "You are an AI assistant that validates summaries of medical texts."
        user_content = (
            "Given the original text and its summary, assess whether the summary accurately and concisely captures the key points of the original text.\n"
            "Provide a brief analysis and rate the summary on a scale of 1 to 5, where 5 indicates excellent quality.\n\n"
            f"Original Text:\n{original_text}\n\n"
            f"Summary:\n{summary}\n\n"
            "Validation:"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        validation = self.call_llama(messages, max_tokens=512)
        return validation

#- agents/validator_agent.py
from .agent_base import AgentBase

class ValidatorAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="ValidatorAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, topic, article):
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that validates research articles for accuracy, completeness, and adherence to academic standards."
            },
            {
                "role": "user",
                "content": (
                    "Given the topic and the research article below, assess whether the article comprehensively covers the topic, follows a logical structure, and maintains academic standards.\n"
                    "Provide a brief analysis and rate the article on a scale of 1 to 5, where 5 indicates excellent quality.\n\n"
                    f"Topic: {topic}\n\n"
                    f"Article:\n{article}\n\n"
                    "Validation:"
                )
            }
        ]
        validation = self.call_llama(
            messages=messages,
            temperature=0.3,         # Lower temperature for more deterministic output
            max_tokens=500
        )
        return validation

#- agents/write_article_tool.py
from .agent_base import AgentBase

class WriteArticleTool(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="WriteArticleTool", max_retries=max_retries, verbose=verbose)

    def execute(self, topic, outline=None):
        system_message = "You are an expert academic writer."
        user_content = f"Write a research article on the following topic:\nTopic: {topic}\n\n"
        if outline:
            user_content += f"Outline:\n{outline}\n\n"
        user_content += "Article:\n"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        article = self.call_llama(messages, max_tokens=1000)
        return article

#- agents/write_article_validator_agent.py
from .agent_base import AgentBase

class WriteArticleValidatorAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="WriteArticleValidatorAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, topic, article):
        system_message = "You are an AI assistant that validates research articles."
        user_content = (
            "Given the topic and the article, assess whether the article comprehensively covers the topic, follows a logical structure, and maintains academic standards.\n"
            "Provide a brief analysis and rate the article on a scale of 1 to 5, where 5 indicates excellent quality.\n\n"
            f"Topic: {topic}\n\n"
            f"Article:\n{article}\n\n"
            "Validation:"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        validation = self.call_llama(messages, max_tokens=512)
        return validation

#- requirements.txt
ollama
streamlit
pandas
loguru
python-dotenv

