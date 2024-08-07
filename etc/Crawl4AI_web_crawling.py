## From https://medium.com/@honeyricky1m3/crawl4ai-automating-web-crawling-and-data-extraction-for-ai-agents-33c9c7ecfa26

! pip install “crawl4ai @ git+https://github.com/unclecode/crawl4ai.git" transformers torch nltk

from crawl4ai import WebCrawler
import os
from crawl4ai import WebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field

# Create an instance of WebCrawler
crawler = WebCrawler()

# Warm up the crawler (load necessary models)
crawler.warmup()

# Run the crawler on a URL
result = crawler.run(url="https://openai.com/api/pricing/")

### Data Structuring using LLM
class OpenAIModelFee(BaseModel):
    model_name: str = Field(..., description="Name of the OpenAI model.")
    input_fee: str = Field(..., description="Fee for input token for the OpenAI model.")
    output_fee: str = Field(..., description="Fee for output token ßfor the OpenAI model.")

url = 'https://openai.com/api/pricing/'
crawler = WebCrawler()
crawler.warmup()

result = crawler.run(
        url=url,
        word_count_threshold=1,
        extraction_strategy= LLMExtractionStrategy(
            provider= "openai/gpt-4o", api_token = os.getenv('OPENAI_API_KEY'), 
            schema=OpenAIModelFee.schema(),
            extraction_type="schema",
            instruction="""From the crawled content, extract all mentioned model names along with their fees for input and output tokens. 
            Do not miss any models in the entire content. One extracted model JSON format should look like this: 
            {"model_name": "GPT-4", "input_fee": "US$10.00 / 1M tokens", "output_fee": "US$30.00 / 1M tokens"}."""
        ),            
        bypass_cache=True,
    )

print(result.extracted_content)

### tools.py
import os
from crawl4ai import WebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field
from praisonai_tools import BaseTool

class ModelFee(BaseModel):
    llm_model_name: str = Field(..., description="Name of the model.")
    input_fee: str = Field(..., description="Fee for input token for the model.")
    output_fee: str = Field(..., description="Fee for output token for the model.")

class ModelFeeTool(BaseTool):
    name: str = "ModelFeeTool"
    description: str = "Extracts model fees for input and output tokens from the given pricing page."

    def _run(self, url: str):
        crawler = WebCrawler()
        crawler.warmup()

        result = crawler.run(
            url=url,
            word_count_threshold=1,
            extraction_strategy= LLMExtractionStrategy(
                provider="openai/gpt-4o",
                api_token=os.getenv('OPENAI_API_KEY'), 
                schema=ModelFee.schema(),
                extraction_type="schema",
                instruction="""From the crawled content, extract all mentioned model names along with their fees for input and output tokens. 
                Do not miss any models in the entire content. One extracted model JSON format should look like this: 
                {"model_name": "GPT-4", "input_fee": "US$10.00 / 1M tokens", "output_fee": "US$30.00 / 1M tokens"}."""
            ),            
            bypass_cache=True,
        )
        return result.extracted_content

if __name__ == "__main__":
    # Test the ModelFeeTool
    tool = ModelFeeTool()
    url = "https://www.openai.com/pricing"
    result = tool.run(url)
    print(result)

