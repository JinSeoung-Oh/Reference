### From https://medium.com/ai-cloud-lab/building-multi-agent-ai-app-with-googles-a2a-agent2agent-protocol-adk-and-mcp-a-deep-a94de2237200

#Setup virtual env
python -n venv .venv 

#Activate venv
source .venv/bin/activate

#Install dependancies
pip install fastapi uvicorn streamlit httpx python-dotenv pydantic
pip install google-generativeai google-adk langchain langchain-openai

#Install mcp search hotel
pip install mcp-hotel-search

#Install mcp flight search 
pip install mcp-flight-search

 GOOGLE_API_KEY=your_google_api_key
 OPENAI_API_KEY=your_openai_api_key
 SERP_API_KEY=your_serp_api_key

--------------------------------------------------------------------------------------------
## ADK agent implementation as MCP Client to fetch tools from MCP Server from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

..
..
# Fetch tools from MCP Server 
server_params = StdioServerParameters(
            command="mcp-flight-search",
            args=["--connection_type", "stdio"],
            env={"SERP_API_KEY": serp_api_key},)

        
tools, exit_stack = await MCPToolset.from_server(
            connection_params=server_params)
..
..
--------------------------------------------------------------------------------------------
## ADK Server Entry point definition using common A2A server components and types and google ADK runners , sessions and Agent
from google.adk.runners import Runner 
from google.adk.sessions import InMemorySessionService 
from google.adk.agents import Agent 
from .agent import get_agent_async

# Import common A2A server components and types
from common.server.server import A2AServer
from common.server.task_manager import InMemoryTaskManager
from common.types import (
    AgentCard, 
    SendTaskRequest, 
    SendTaskResponse, 
    Task, 
    TaskStatus, 
    Message, 
    TextPart, 
    TaskState, 
)


# --- Custom Task Manager for Flight Search --- 
class FlightAgentTaskManager(InMemoryTaskManager):
    """Task manager specific to the ADK Flight Search agent."""
    def __init__(self, agent: Agent, runner: Runner, session_service: InMemorySessionService):
        super().__init__()
        self.agent = agent 
        self.runner = runner 
        self.session_service = session_service 
        logger.info("FlightAgentTaskManager initialized.")

...
...
---------------------------------------------------------------------------------------------
## Create A2A Server instance using Agent Card
# --- Main Execution Block --- 
async def run_server():
    """Initializes services and starts the A2AServer."""
    logger.info("Starting Flight Search A2A Server initialization...")
    
    session_service = None
    exit_stack = None
    try:
        session_service = InMemorySessionService()
        agent, exit_stack = await get_agent_async()
        runner = Runner(
            app_name='flight_search_a2a_app',
            agent=agent,
            session_service=session_service,
        )
        
        # Create the specific task manager
        task_manager = FlightAgentTaskManager(
            agent=agent, 
            runner=runner, 
            session_service=session_service
        )
        
        # Define Agent Card
        port = int(os.getenv("PORT", "8000"))
        host = os.getenv("HOST", "localhost")
        listen_host = "0.0.0.0"

        agent_card = AgentCard(
            name="Flight Search Agent (A2A)",
            description="Provides flight information based on user queries.",
            url=f"http://{host}:{port}/", 
            version="1.0.0",
            defaultInputModes=["text"],
            defaultOutputModes=["text"],
            capabilities={"streaming": False}, 
            skills=[
                {
                    "id": "search_flights",
                    "name": "Search Flights",
                    "description": "Searches for flights based on origin, destination, and date.",
                    "tags": ["flights", "travel"],
                    "examples": ["Find flights from JFK to LAX tomorrow"]
                }
            ]
        )

        # Create the A2AServer instance
        a2a_server = A2AServer(
            agent_card=agent_card,
            task_manager=task_manager,
            host=listen_host, 
            port=port
        )
        # Configure Uvicorn programmatically
        config = uvicorn.Config(
            app=a2a_server.app, # Pass the Starlette app from A2AServer
            host=listen_host,
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
...
...
--------------------------------------------------------------------------------------
## LangChain Agent implementation as MCP Client with OpenAI LLM
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.client import MultiServerMCPClient

# MCP client configuration
MCP_CONFIG = {
    "hotel_search": {
        "command": "mcp-hotel-search",
        "args": ["--connection_type", "stdio"],
        "transport": "stdio",
        "env": {"SERP_API_KEY": os.getenv("SERP_API_KEY")},
    }
}

class HotelSearchAgent:
    """Hotel search agent using LangChain MCP adapters."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
    def _create_prompt(self):
        """Create a prompt template with our custom system message."""
        system_message = """You are a helpful hotel search assistant. 
        """
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
..
..
async def process_query(self, query):
...

            # Create MCP client for this query
            async with MultiServerMCPClient(MCP_CONFIG) as client:
                # Get tools from this client instance
                tools = client.get_tools()
                
                # Create a prompt
                prompt = self._create_prompt()
                
                # Create an agent with these tools
                agent = create_openai_functions_agent(
                    llm=self.llm,
                    tools=tools,
                    prompt=prompt
                )
                
                # Create an executor with these tools
                executor = AgentExecutor(
                    agent=agent,
                    tools=tools,
                    verbose=True,
                    handle_parsing_errors=True,
                )


---------------------------------------------------------------------------------------
## Create A2AServer Instance using common A2A server components and types
# Use the underlying agent directly
from hotel_search_app.langchain_agent import get_agent, HotelSearchAgent 

# Import common A2A server components and types
from common.server.server import A2AServer
from common.server.task_manager import InMemoryTaskManager
from common.types import (
    AgentCard,
    SendTaskRequest,
    SendTaskResponse,
    Task,
    TaskStatus,
    Message,
    TextPart,
    TaskState
)
..
..

class HotelAgentTaskManager(InMemoryTaskManager):
    """Task manager specific to the Hotel Search agent."""
    def __init__(self, agent: HotelSearchAgent):
        super().__init__()
        self.agent = agent # The HotelSearchAgent instance
        logger.info("HotelAgentTaskManager initialized.")

    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """Handles the tasks/send request by calling the agent's process_query."""
        task_params = request.params
        task_id = task_params.id
        user_message_text = None

        logger.info(f"HotelAgentTaskManager handling task {task_id}")


# --- Main Execution Block --- 
async def run_server():
    """Initializes services and starts the A2AServer for hotels."""
    logger.info("Starting Hotel Search A2A Server initialization...")
    
    agent_instance: Optional[HotelSearchAgent] = None
    try:
        agent_instance = await get_agent()
        if not agent_instance:
             raise RuntimeError("Failed to initialize HotelSearchAgent")

        # Create the specific task manager
        task_manager = HotelAgentTaskManager(agent=agent_instance)
        
        # Define Agent Card
        port = int(os.getenv("PORT", "8003")) # Default port 8003
        host = os.getenv("HOST", "localhost")
        listen_host = "0.0.0.0"

        agent_card = AgentCard(
            name="Hotel Search Agent (A2A)",
            description="Provides hotel information based on location, dates, and guests.",
            url=f"http://{host}:{port}/", 
            version="1.0.0",
            defaultInputModes=["text"],
            defaultOutputModes=["text"],
            capabilities={"streaming": False},
            skills=[
                {
                    "id": "search_hotels",
                    "name": "Search Hotels",
                    "description": "Searches for hotels based on location, check-in/out dates, and number of guests.",
                    "tags": ["hotels", "travel", "accommodation"],
                    "examples": ["Find hotels in London from July 1st to July 5th for 2 adults"]
                }
            ]
        )

        # Create the A2AServer instance WITHOUT endpoint parameter
        a2a_server = A2AServer(
            agent_card=agent_card,
            task_manager=task_manager,
            host=listen_host, 
            port=port
        )

        config = uvicorn.Config(
            app=a2a_server.app, # Pass the Starlette app from A2AServer
            host=listen_host,
            port=port,
            log_level="info"
        )

----------------------------------------------------------------------------------------------
## A2A Protocol implementation using Flight and Hotel API URL

# Base URLs for the A2A compliant agent APIs
FLIGHT_SEARCH_API_URL = os.getenv("FLIGHT_SEARCH_API_URL", "http://localhost:8000")
HOTEL_SEARCH_API_URL = os.getenv("HOTEL_SEARCH_API_URL", "http://localhost:8003")

class A2AClientBase:
    """Base client for communicating with A2A-compliant agents via the root endpoint."""


    async def send_a2a_task(self, user_message: str, task_id: Optional[str] = None, agent_type: str = "generic") -> Dict[str, Any]:
    ...
    ....
        # Construct the JSON-RPC payload with the A2A method and corrected params structure
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send", 
            "params": { 
                "id": task_id, 
                "taskId": task_id, 
                "message": {
                    "role": "user",
                    "parts": [
                        {"type": "text", "text": user_message}
                    ]
                }
            },
            "id": task_id 
        }

--------------------------------------------------------------------------------------------------
## Itinerary Planner Agent Card
{
    "name": "Travel Itinerary Planner",
    "displayName": "Travel Itinerary Planner",
    "description": "An agent that coordinates flight and hotel information to create comprehensive travel itineraries",
    "version": "1.0.0",
    "contact": "code.aicloudlab@gmail.com",
    "endpointUrl": "http://localhost:8005",
    "authentication": {
        "type": "none"
    },
    "capabilities": [
        "streaming"
    ],
    "skills": [
        {
            "name": "createItinerary",
            "description": "Create a comprehensive travel itinerary including flights and accommodations",
            "inputs": [
                {
                    "name": "origin",
                    "type": "string",
                    "description": "Origin city or airport code"
                },
                {
                    "name": "destination",
                    "type": "string",
                    "description": "Destination city or area"
                },
                {
                    "name": "departureDate",
                    "type": "string",
                    "description": "Departure date in YYYY-MM-DD format"
                },
                {
                    "name": "returnDate",
                    "type": "string",
                    "description": "Return date in YYYY-MM-DD format (optional)"
                },
                {
                    "name": "travelers",
                    "type": "integer",
                    "description": "Number of travelers"
                },
                {
                    "name": "preferences",
                    "type": "object",
                    "description": "Additional preferences like budget, hotel amenities, etc."
                }
            ],
            "outputs": [
                {
                    "name": "itinerary",
                    "type": "object",
                    "description": "Complete travel itinerary with flights, hotels, and schedule"
                }
            ]
        }
    ]
}
------------------------------------------------------------------------------------------------
## Itinerary agent using Google-GenAI SDK
import google.generativeai as genai # Use direct SDK
..
..
from itinerary_planner.a2a.a2a_client import FlightSearchClient, HotelSearchClient

# Configure the Google Generative AI SDK
genai.configure(api_key=api_key)

class ItineraryPlanner:
    """A planner that coordinates between flight and hotel search agents to create itineraries using the google.generativeai SDK."""
    
    def __init__(self):
        """Initialize the itinerary planner."""
        logger.info("Initializing Itinerary Planner with google.generativeai SDK")
        self.flight_client = FlightSearchClient()
        self.hotel_client = HotelSearchClient()
        
        # Create the Gemini model instance using the SDK
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash", 
        )
..
..
----------------------------------------------------------------------------------------------
## Building Multi-Agent with Google's A2A (Agent2Agent) Protocol, Agent Development Kit(ADK), and MCP (Model Context Protocol) - A Deep Dive(Full Code) | AI Cloud Lab
from fastapi import FastAPI, HTTPException, Request

from itinerary_planner.itinerary_agent import ItineraryPlanner

@app.post("/v1/tasks/send")
async def send_task(request: TaskRequest):
    """Handle A2A tasks/send requests."""
    global planner
    
    if not planner:
        raise HTTPException(status_code=503, detail="Planner not initialized")
    
    try:
        task_id = request.taskId
        
        # Extract the message from the user
        user_message = None
        for part in request.message.get("parts", []):
            if "text" in part:
                user_message = part["text"]
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="No text message found in request")
        
        # Generate an itinerary based on the query
        itinerary = await planner.create_itinerary(user_message)
        
        # Create the A2A response
        response = {
            "task": {
                "taskId": task_id,
                "state": "completed",
                "messages": [
                    {
                        "role": "user",
                        "parts": [{"text": user_message}]
                    },
                    {
                        "role": "agent",
                        "parts": [{"text": itinerary}]
                    }
                ],
                "artifacts": []
            }
        }
        
        return response

---------------------------------------------------------------------------------------------------
## Streamlit_ui — User interface built with Streamlit and provides forms for travel planning and displays results in a user-friendly format

...
...
# API endpoint
API_URL = "http://localhost:8005/v1/tasks/send"

def generate_itinerary(query: str):
    """Send a query to the itinerary planner API."""
    try:
        task_id = "task-" + datetime.now().strftime("%Y%m%d%H%M%S")
        
        payload = {
            "taskId": task_id,
            "message": {
                "role": "user",
                "parts": [
                    {
                        "text": query
                    }
                ]
            }
        }
        
        # Log the user query and the request to the event log
        log_user_query(query)
        log_itinerary_request(payload)
        
        response = requests.post(
            API_URL,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Extract the agent's response message
        agent_message = None
        for message in result.get("task", {}).get("messages", []):
            if message.get("role") == "agent":
                for part in message.get("parts", []):
                    if "text" in part:
                        agent_message = part["text"]
                        break
                if agent_message:
                    break
..
..
...
