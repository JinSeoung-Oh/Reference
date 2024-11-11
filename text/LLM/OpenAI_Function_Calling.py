## From https://www.pragnakalp.com/simplifying-openai-function-calling-with-structured-output-a-2024-guide/

# Install required packages
!pip install openai==1.47.1
!pip install finnhub-python==2.4.20

# Import required library
import json
import requests
import finnhub
import openai
from openai import OpenAI
from pydantic import BaseModel

# Define variables
GPT_MODEL = "gpt-4o-mini"
finnhub_api_key = "your_finnhub_api_key"
openai_api_key = "your_openai_api_key"
alphavantage_api_key = "your_alphavantage_api_key"

# Init OpenAI and finnhub client
client = OpenAI(api_key=openai_api_key)
finnhub_client = finnhub.Client(api_key = finnhub_api_key)

# Call `finnhub` API and get the current price of the stock
# Call `finnhub` API and get the current price of the stock
def get_current_stock_price(arguments):
   try:
       arguments = json.loads(arguments)['ticker_symbol']
       price_data=finnhub_client.quote(arguments)
       stock_price = price_data.get('c', None)
       print(stock_price)
       if stock_price == 0:
           return "This company is not listed within USA, please provide another name."
       else:
           return stock_price
   except:
       return "This company is not listed within USA, please provide another name."

# Call `alphavantage` API and exchange the currency rate
def currency_exchange_rate(arguments):
   try:
       from_country_currency = json.loads(arguments)['from_country_currency']
       to_country_currency = json.loads(arguments)['to_country_currency']
       url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_country_currency}&to_currency={to_country_currency}&apikey={alphavantage_api_key}'
       r = requests.get(url)
       data = r.json()
       return data['Realtime Currency Exchange Rate']['5. Exchange Rate']
   except:
       return "I am unable to parse this, please try something new."

-------------------------------------------------------------------------------------------------------------
  ##### define function
  {
    "name": "get_current_stock_price",
    "description": "It will get the current stock price of the US company.",
    "parameters": {
        "type": "object",
        "properties": {
            "ticker_symbol": {
                "type": "string",
                "description": "This is the symbol of the company.",
            }
        },
        "required": ["ticker_symbol"],
    },
}
-------------------------------------------------------------------------------------------------------------

# Define pydantic classes, which specify the required input for the functions
class stockPriceData(BaseModel):
   ticker_symbol: str

class currencyExchangeRate(BaseModel):
   from_country_currency: str
   to_country_currency: str

# Convert pydantic class to function call definition, which we were writing manually
tools = [openai.pydantic_function_tool(stockPriceData), openai.pydantic_function_tool(currencyExchangeRate)]

# Below code combines the whole flow to generate output

user_input = input("Please enter your question here: (if you want to exit then write 'exit' or 'bye'.) ")

# We will continue QNA till user says `exit` or `bye`
while user_input.strip().lower() != "exit" and user_input.strip().lower() != "bye":
   # Prepare messages to send to OpenAI
   messages = [
       {
           "role": "system",
           "content": "You are a helpful customer support assistant. Use the supplied tools to assist the user.",
       },
       {
           "role": "user",
           "content": user_input,
       }
   ]

   # Call OpenAI with prompt instruction, User Input and function definition (tools)
   response = client.chat.completions.create(
       model=GPT_MODEL, messages=messages, tools=tools
   )

   # Check if model has detected the tool call
   if response.choices[0].finish_reason == "tool_calls":
       # Fetch tool name and arguments
       tool_name = response.choices[0].message.tool_calls[0].function.name
       fn_argument = response.choices[0].message.tool_calls[0].function.arguments
       print("Detected tool", tool_name)
       print("Extracted function arguments:", fn_argument)

       # call the function associated with the particular tool
       if tool_name == "stockPriceData":
           result = get_current_stock_price(fn_argument)
           print(f"Stock price of {json.loads(fn_argument)['ticker_symbol']}:", result)
       elif tool_name == "currencyExchangeRate":
           result = currency_exchange_rate(fn_argument)
           print(f"Currency exchange rate from {json.loads(fn_argument)['from_country_currency']} to {json.loads(fn_argument)['to_country_currency']} is {result}.")

   # Check for the normal replies
   elif response.choices[0].finish_reason == "stop":
       print("response:", response.choices[0].message.content)

   # Check if OpeAI is identifying our content as restricted one
   elif response.choices[0].finish_reason == "content_filter":
       print("Your request or response may include restricted content.")

   # Check if we are exceeding maximum context window
   elif response.choices[0].finish_reason == "length":
       print(f"Your input token exceed the maximum input window of the model `{GPT_MODEL}`.")

   # Continue next iteration
   user_input = input("Please enter your question here: ")



