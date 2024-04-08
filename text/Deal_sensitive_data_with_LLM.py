# From https://pub.towardsai.net/all-you-need-to-know-about-sensitive-data-handling-using-large-language-models-1a39b6752ced

###### with Ollama
! pip install langchain
! pip install langchain-community

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

llm = Ollama(model="llama2")

template = """
You are a sensitive data identifier and masker. 
You are capable of identifying sensitive information in text and applying a mask using "****". 
Sensitive data can also be embedded in the context of the text and is not always explicitly mentioned (such as topics about health, financials, addresses, .. )
Ensure that Personall identifiable data is detected and masked.
Ensure that the detection takes into account the data protection laws and regulations like GDPR, CCPA, and HIPA
Ensure that the input text is not altered or changed in any way and just mask the detected sensitive information.
Ensure high confidence in the information you mask.
The content returned should not include anything other than the text input with the required masking applied.
If no sensitive text is detected, return the input as is with no additional content.

The sentence:
{sentence}
"""

output_parser = StrOutputParser()

# Setup the prompt
prompt = PromptTemplate.from_template(template)

# Create the Chain
chain = prompt | llm | output_parser

sentence = """
Last week, while discussing our summer plans, 
Mike hinted he's finally taking that solo trip to Bali
he saved up for, after his bonus came through.
He received more than 10K as bonus
"""

# Run the detection
response = chain.invoke({'sentence':sentence})

##### Azure OpenAI Setup
! pip install langchain-openai

import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Add the required environmental variables
os.environ["OPENAI_API_VERSION"] = <API VERSION>
os.environ["AZURE_OPENAI_API_KEY"] = <Your AZURE OPENAI KEY>
os.environ["AZURE_OPENAI_ENDPOINT"] = <Your AZURE  OPENAI ENDPOINT>

# Define the prompt template
template = """
You are a sensitive data identifier and masker. 
You are capable of identifying sensitive information in text and applying a mask using "****". 
Sensitive data can also be embedded in the context of the text and is not always explicitly mentioned (such as topics about health, financials, addresses, .. )
Ensure that Personal identifiable data is detected and masked.
Ensure that the detection takes into account the data protection laws and regulations like GDPR, CCPA, and HIPA
Ensure that the input text is not altered or changed in any way and just mask the detected sensitive information.
Ensure high confidence in the information you mask.
The content returned should not include anything other than the text input with the required masking applied.
If no sensitive text is detected, return the input as is with no additional content.

The sentence:
{sentence}
"""

prompt = ChatPromptTemplate.from_template(template)

# Select the model
llm = AzureChatOpenAI(
    azure_deployment="gpt4",
)

# Setup the Chain
chain = prompt | llm

sentence = """
Last week, while discussing our summer plans, 
Mike hinted he's finally taking that solo trip to Bali
he saved up for, after his bonus came through.
He received more than 10K as bonus
"""

# Run the LLM
response = chain.invoke({'sentence':sentence})

print(response.content)
