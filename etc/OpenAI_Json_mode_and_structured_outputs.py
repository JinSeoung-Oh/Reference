## From https://medium.com/@ingridwickstevens/transform-unstructured-legal-text-into-organized-data-with-openais-structured-outputs-0a6a6aaf553b

# JSON Mode
from openai import OpenAI
import json

client = OpenAI()

# Define the contract text
contract_text = """
Title: Service Agreement between ABC Corp and XYZ Inc.

Introduction: This Service Agreement ("Agreement") is made and entered into on the 15th day of January 2024, 
by and between ABC Corp, a corporation located in California, and XYZ Inc., a corporation located in Texas.

Scope of Services: XYZ Inc. will provide software development and maintenance services to ABC Corp as outlined in Appendix A.

Term: The Agreement shall commence on January 15, 2024, and shall continue for a period of one year, 
with an option for renewal upon mutual agreement.

Confidentiality: Both parties agree to keep all exchanged information confidential.

Termination: Either party may terminate the Agreement with a 30-day written notice under conditions outlined in Section 9.

Jurisdiction: This Agreement shall be governed by the laws of the State of California.

Signatures: This Agreement is signed by representatives of ABC Corp and XYZ Inc.
"""

# Set up the JSON mode request
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a legal assistant who summarizes contracts in JSON format."},
        {
            "role": "user",
            "content": f"""Analyze the following contract and return in JSON format the title, summary, 
            key terms, jurisdiction, and legal implications. Ensure all text is valid JSON.

            Contract text:
            {contract_text}
            """
        }
    ],
    response_format={"type": "json_object"}
)

# Parse and print the JSON response
json_response = completion.choices[0].message.content
parsed_response = json.loads(json_response)
print(json.dumps(parsed_response, indent=4))

---------------------------------------------------------------------------------------------------------
# Structured Outputs
import json
from openai import OpenAI

client = OpenAI()

# Contract summarization prompt and schema setup
contract_summarizer_prompt = '''
    You are an AI legal assistant. Given a legal contract, summarize its key points in a structured JSON format. 
    Include the title, a brief summary, a list of key legal terms, jurisdiction, and any legal implications or important clauses.
'''

MODEL = "gpt-4o"

def get_contract_summary(contract_text):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system", 
                "content": contract_summarizer_prompt
            },
            {
                "role": "user", 
                "content": contract_text
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "legal_summary",
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "key_terms": {"type": "array", "items": {"type": "string"}},
                        "jurisdiction": {"type": "string"},
                        "implications": {"type": "string"}
                    },
                    "required": ["title", "summary", "key_terms", "jurisdiction", "implications"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    )
    return response.choices[0].message

# Example contract text
contract_text = """
Title: Service Agreement between ABC Corp and XYZ Inc.

Introduction: This Service Agreement ("Agreement") is made and entered into on the 15th day of January 2024, 
by and between ABC Corp, a corporation located in California, and XYZ Inc., a corporation located in Texas.

Scope of Services: XYZ Inc. will provide software development and maintenance services to ABC Corp as outlined in Appendix A.

Term: The Agreement shall commence on January 15, 2024, and shall continue for a period of one year, 
with an option for renewal upon mutual agreement.

Confidentiality: Both parties agree to keep all exchanged information confidential.

Termination: Either party may terminate the Agreement with a 30-day written notice under conditions outlined in Section 9.

Jurisdiction: This Agreement shall be governed by the laws of the State of California.

Signatures: This Agreement is signed by representatives of ABC Corp and XYZ Inc.
"""

# Run the function and print the structured output
result = get_contract_summary(contract_text)

# Parse the JSON content
parsed_content = json.loads(result.content)

# Print the parsed content with indentation for readability
print(json.dumps(parsed_content, indent=4))

