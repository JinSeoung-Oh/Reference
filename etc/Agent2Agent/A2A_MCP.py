### From https://medium.com/@the_manoj_desai/the-power-duo-how-a2a-mcp-let-you-build-practical-ai-systems-today-9c19064b027b

"""
1. Architecture Overview
   -a. MCP Servers expose discrete capabilities (tools) over HTTP with typed interfaces.
   -b. A2A Agents consume those tools via simple RPC calls, adding natural‑language logic.
   -c. Orchestrator Agent (often backed by an LLM) routes user queries to the right specialized agents.
   -d. Client sends user messages into the orchestrator over A2A.
   This separation makes each component modular, replaceable, and reusable.
"""
----------------------------------------------------------------------------------------------------------------------
### 2. Building a Calculator MCP Server
# calculator_mcp_server.py
from python_a2a.mcp import FastMCP, text_response

# 1) Create the MCP server
calculator_mcp = FastMCP(
    name="Calculator MCP",
    version="1.0.0",
    description="Provides mathematical calculation functions"
)

# 2) Define tools with type hints
@calculator_mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@calculator_mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b

@calculator_mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@calculator_mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b, with zero‑check."""
    if b == 0:
        return text_response("Cannot divide by zero")
    return a / b

# 3) Run the server (non‑blocking in Jupyter; use plain run() in scripts)
import threading
def run_server():
    calculator_mcp.run(host="0.0.0.0", port=5001)

threading.Thread(target=run_server, daemon=True).start()
print("Calculator MCP server running at http://0.0.0.0:5001")

----------------------------------------------------------------------------------------------------------------------
### 3. A2A Agent That Uses the Calculator MCP
# calculator_agent.py
from python_a2a import A2AServer, Message, TextContent, MessageRole, run_server
from python_a2a.mcp import FastMCPAgent
import re

class CalculatorAgent(A2AServer, FastMCPAgent):
    def __init__(self):
        A2AServer.__init__(self)
        FastMCPAgent.__init__(
            self,
            mcp_servers={"calc": "http://localhost:5001"}  # link to MCP
        )

    async def handle_message_async(self, message):
        text = message.content.text.lower()
        # Extract numbers
        nums = [float(n) for n in re.findall(r"[-+]?\d*\.?\d+", text)]
        if len(nums) >= 2:
            if "add" in text or "+" in text:
                r = await self.call_mcp_tool("calc", "add", a=nums[0], b=nums[1])
                response = f"The sum of {nums[0]} and {nums[1]} is {r}"
            elif "subtract" in text or "-" in text:
                r = await self.call_mcp_tool("calc", "subtract", a=nums[0], b=nums[1])
                response = f"The difference between {nums[0]} and {nums[1]} is {r}"
            elif "multiply" in text or "*" in text:
                r = await self.call_mcp_tool("calc", "multiply", a=nums[0], b=nums[1])
                response = f"{nums[0]} × {nums[1]} = {r}"
            elif "divide" in text or "/" in text:
                r = await self.call_mcp_tool("calc", "divide", a=nums[0], b=nums[1])
                response = f"{nums[0]} ÷ {nums[1]} = {r}"
            else:
                response = "I only support add, subtract, multiply, divide."
        else:
            response = "Please give me two numbers and an operation."

        return Message(
            content=TextContent(text=response),
            role=MessageRole.AGENT,
            parent_message_id=message.message_id,
            conversation_id=message.conversation_id
        )

if __name__ == "__main__":
    agent = CalculatorAgent()
    run_server(agent, host="0.0.0.0", port=5000)
  
----------------------------------------------------------------------------------------------------------------------
### 4. Stock Information System
## 4.1 DuckDuckGo MCP Server (Ticker Lookup)
# duckduckgo_mcp_server.py
from python_a2a.mcp import FastMCP
import requests, re

duckduckgo_mcp = FastMCP(
    name="DuckDuckGo MCP",
    version="1.0.0",
    description="Lookup stock ticker symbols via DuckDuckGo"
)

@duckduckgo_mcp.tool()
def search_ticker(company_name: str) -> str:
    query = f"{company_name} stock ticker symbol"
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    data = requests.get(url).json().get("Abstract", "")
    m = re.search(r'(?:NYSE|NASDAQ):\s*([A-Z]+)', data)
    if m: return m.group(1)
    # Fallbacks
    falls = {"apple":"AAPL","microsoft":"MSFT","amazon":"AMZN","tesla":"TSLA","google":"GOOGL"}
    return falls.get(company_name.lower(), f"Unknown ticker for {company_name}")

duckduckgo_mcp.run(host="0.0.0.0", port=5001)

## 4.2 YFinance MCP Server (Stock Price)
# yfinance_mcp_server.py
from python_a2a.mcp import FastMCP
import yfinance as yf

yfinance_mcp = FastMCP(
    name="YFinance MCP",
    version="1.0.0",
    description="Fetch current stock price"
)

@yfinance_mcp.tool()
def get_stock_price(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    hist = t.history(period="1d")
    if hist.empty:
        return {"error": f"No data for {ticker}"}
    price = hist['Close'].iloc[-1]
    ts = hist.index[-1].strftime('%Y-%m-%d %H:%M:%S')
    return {"ticker": ticker, "price": price, "currency": "USD", "timestamp": ts}

yfinance_mcp.run(host="0.0.0.0", port=5002)

## 4.3 DuckDuckGo A2A Agent
# duckduckgo_agent.py
from python_a2a import A2AServer, Message, TextContent, MessageRole, run_server
from python_a2a.mcp import FastMCPAgent
import re

class DuckDuckGoAgent(A2AServer, FastMCPAgent):
    def __init__(self):
        A2AServer.__init__(self)
        FastMCPAgent.__init__(self, mcp_servers={"search":"http://localhost:5001"})

    async def handle_message_async(self, msg):
        text = msg.content.text
        m = re.search(r"ticker\s+(?:for|of)\s+(.+)", text, re.I)
        company = m.group(1).strip() if m else text.strip()
        ticker = await self.call_mcp_tool("search", "search_ticker", company_name=company)
        return Message(
            content=TextContent(text=f"The ticker for {company} is {ticker}."),
            role=MessageRole.AGENT,
            parent_message_id=msg.message_id,
            conversation_id=msg.conversation_id
        )

if __name__ == "__main__":
    run_server(DuckDuckGoAgent(), port=5003)
  
## 4.4 YFinance A2A Agent
# yfinance_agent.py
from python_a2a import A2AServer, Message, TextContent, MessageRole, run_server
from python_a2a.mcp import FastMCPAgent
import re

class YFinanceAgent(A2AServer, FastMCPAgent):
    def __init__(self):
        A2AServer.__init__(self)
        FastMCPAgent.__init__(self, mcp_servers={"finance":"http://localhost:5002"})

    async def handle_message_async(self, msg):
        m = re.search(r"\b([A-Z]{1,5})\b", msg.content.text)
        if not m:
            return Message(
                content=TextContent("Please provide a valid ticker symbol."),
                role=MessageRole.AGENT,
                parent_message_id=msg.message_id,
                conversation_id=msg.conversation_id
            )
        ticker = m.group(1)
        info = await self.call_mcp_tool("finance", "get_stock_price", ticker=ticker)
        if "error" in info:
            text = f"Error: {info['error']}"
        else:
            text = f"{ticker} is at {info['price']:.2f} {info['currency']} (as of {info['timestamp']})"
        return Message(
            content=TextContent(text),
            role=MessageRole.AGENT,
            parent_message_id=msg.message_id,
            conversation_id=msg.conversation_id
        )

if __name__ == "__main__":
    run_server(YFinanceAgent(), port=5004)
  
----------------------------------------------------------------------------------------------------------------------
### 5. Orchestrator: StockAssistant
# stock_assistant.py
import os, re
from python_a2a import OpenAIA2AServer, A2AClient, Message, TextContent, MessageRole, run_server

class StockAssistant(OpenAIA2AServer):
    def __init__(self, api_key, search_ep, finance_ep):
        super().__init__(
            api_key=api_key,
            model="gpt-3.5-turbo",
            system_prompt="You are a financial assistant. Extract company names, find tickers and prices."
        )
        self.search_client = A2AClient(search_ep)
        self.finance_client = A2AClient(finance_ep)

    def handle_message(self, msg):
        text = msg.content.text
        # If user asks about stock price…
        if "price" in text.lower():
            return self._get_stock(msg)
        return super().handle_message(msg)

    def _get_stock(self, msg):
        # 1) Extract company name via LLM
        resp = super().handle_message(Message(
            content=TextContent(text=f"Extract only the company name: '{msg.content.text}'"),
            role=MessageRole.USER
        ))
        company = resp.content.text.strip().strip('"\'')
        # 2) Get ticker
        tick_r = self.search_client.send_message(
            Message(content=TextContent(text=f"ticker for {company}?"), role=MessageRole.USER)
        )
        m = re.search(r"is\s+([A-Z]{1,5})", tick_r.content.text)
        if not m: return Message(TextContent(f"Couldn’t find ticker for {company}."), role=MessageRole.AGENT, parent_message_id=msg.message_id, conversation_id=msg.conversation_id)
        ticker = m.group(1)
        # 3) Get price
        price_r = self.finance_client.send_message(
            Message(content=TextContent(text=f"price of {ticker}?"), role=MessageRole.USER)
        )
        return Message(
            content=TextContent(text=f"{company} ({ticker}): {price_r.content.text}"),
            role=MessageRole.AGENT,
            parent_message_id=msg.message_id,
            conversation_id=msg.conversation_id
        )

if __name__ == "__main__":
    key = os.getenv("OPENAI_API_KEY")
    assistant = StockAssistant(key, "http://localhost:5003/a2a", "http://localhost:5004/a2a")
    run_server(assistant, port=5000)
  
----------------------------------------------------------------------------------------------------------------------
### 6. Interactive CLI Client
# stock_client.py
import argparse
from python_a2a import A2AClient, Message, TextContent, MessageRole

def interactive(client):
    print("=== Stock Price Assistant ===\nType 'exit' to quit.")
    while True:
        q = input("> ")
        if q.lower() in ("exit","quit"): break
        resp = client.send_message(Message(content=TextContent(text=q), role=MessageRole.USER))
        print("Assistant:", resp.content.text)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--endpoint", default="http://localhost:5000/a2a")
    args=p.parse_args()
    client = A2AClient(args.endpoint)
    interactive(client)
"""
7. Run the System
   -a. Install:
       pip install "python‑a2a[all]" yfinance requests
   -b. Launch MCP servers:
       python calculator_mcp_server.py
       python duckduckgo_mcp_server.py
       python yfinance_mcp_server.py
   -c. Start A2A agents:
       python calculator_agent.py
       python duckduckgo_agent.py
       python yfinance_agent.py
   -d. Start the orchestrator:
       OPENAI_API_KEY=… python stock_assistant.py
   -e. Run the client:
       python stock_client.py
  """
