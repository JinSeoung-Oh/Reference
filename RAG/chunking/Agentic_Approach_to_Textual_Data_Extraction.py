### From https://towardsdatascience.com/an-agentic-approach-to-textual-data-extraction-using-llms-and-langgraph-8abb12af16f2
### From https://github.com/CVxTz/travel_dataset

### Define LangGraph freamwork
from typing import Annotated, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

class OverallState(BaseModel):
    result: Optional[int] = None
    messages: Annotated[list[AnyMessage], add_messages]

def step_1(state: OverallState):
    return {"messages": ["step_1 message"]}
def step_2(state: OverallState):
    return {"messages": ["step_2 message"]}
def step_3(state: OverallState):
    return {"messages": ["step_3 message"], "result": len(state.messages)}

# Define the function that determines whether to continue or not
def which_step_next(state: OverallState):
    if len(state.messages) < 20:
        return "step_1"
    return END

# Define a new graph
workflow = StateGraph(OverallState)
# Define the two nodes we will cycle between
workflow.add_node("step_1", step_1)
workflow.add_node("step_2", step_2)
workflow.add_node("step_3", step_3)
# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "step_1")
workflow.add_edge("step_1", "step_2")
workflow.add_edge("step_2", "step_3")
# We now add a conditional edge
workflow.add_conditional_edges(
    "step_3",
    which_step_next,
)
app = workflow.compile()
# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="Init input")]},
    config={"configurable": {"thread_id": 42}}
)
print(final_state["result"])

-----------------------------------------------------------------------------------------------
### Define data structure with pydantic
from pydantic import BaseModel, Field
from typing import List, Optional

class Attraction(BaseModel):
    """Model for an attraction"""
    name: str = Field(..., description="Name of the attraction")
    description: str = Field(..., description="Description of the attraction")
    city: str = Field(..., description="City where the attraction is located or is closest to")
    countryw: str = Field(..., description="Country where the attraction is or is closest to")
    activity_types: List[ActivityType] = Field(..., description="List of activity types and attractions available at the attraction")
    tags: List[str] = Field(..., description="List of tags describing the attraction (e.g., accessible, sustainable, sunny, cheap, pricey)", min_length=1)

class City(BaseModel):
    """Model for a tourist city"""
    name: str = Field(..., description="Name of the city")
    description: str = Field(..., description="Few sentences description of the city, can be long if there is enough relevant information. Includes what the city is famous for and why people might visit it.")
    country: str = Field(..., description="Country")
    continent: Optional[str] = Field(None, description="Continent if applicable, otherwise leave empty")
    location_lat_long: Optional[List[float]] = Field([], description="Geographic coordinates [latitude, longitude] of the location, [] if unknown")
    climate_type: ClimateType = Field(..., description="Type of climate at the city")
    class Config:
        use_enum_values = True
        extra = "forbid"

-----------------------------------------------------------------------------------------------
### Define Node state with Pydantic
class OverallState(BaseModel):
    page_title: str
    page_content: str
    page_type: PageType = PageType.unknown
    cities: Optional[Cities] = None
    attractions: Optional[Attractions] = None
    url: Optional[str] = None

-----------------------------------------------------------------------------------------------
### Define node for langgraph freamwork
def predict_page_type(state: OverallState) -> dict:
    logger.info(f"Entering predict_page_type function. State: {state}")
    page_summary = f"{state.page_title} {state.page_content[:400]}"
    messages = [
        SystemMessage(
            content=f"You are a helpful assistant that outputs in JSON. Follow this schema {Page.model_json_schema()}"
        ),
        HumanMessage(content="Give me an example Json of such output"),
        AIMessage(content=page_example.model_dump_json()),
        HumanMessage(content=f"What is the type of this page?\n {page_summary}"),
    ]
    local_client = client_medium.with_structured_output(Page)
    try:
        page = local_client.invoke(messages)
        logger.info(f"Page type prediction successful: {page}")
        return {"page_type": page.page_type}
    except Exception as e:
        logger.error(f"Error in predict_page_type: {e}")
        return {"page_type": None}

def parse_city(state: OverallState) -> dict:
    logger.info(f"Entering parse_city function. State: {state}")
    messages = [
        SystemMessage(
            content=f"You are a helpful assistant that outputs in JSON. Follow this schema {Cities.model_json_schema()}. Only answer with information from the context. Keep missing information empty."
        ),
        HumanMessage(content="Give me an example Json of such output"),
        AIMessage(content=Cities(cities=[city_example]).model_dump_json()),
        HumanMessage(content=f"What are the cities mentioned in this page?\n {state.page_content}"),
    ]
    local_client = client_medium.with_structured_output(Cities)
    try:
        cities = local_client.invoke(messages)
        logger.info(f"City parsing successful: {cities}")
        return {"cities": cities}
    except Exception as e:
        logger.error(f"Error in parse_city: {e}")
        return {"cities": None}

--------------------------------------------------------------------------------------------------
### Define workflow
workflow = StateGraph(OverallState)
# Add nodes
workflow.add_node(NodeNames.predict_page_type.value, predict_page_type)
workflow.add_node(NodeNames.parse_city.value, parse_city)
workflow.add_node(NodeNames.parse_attraction.value, parse_attractions)
# Add edges
workflow.add_edge(START, NodeNames.predict_page_type.value)
workflow.add_conditional_edges(NodeNames.predict_page_type.value, parse_city_edge)
workflow.add_conditional_edges(NodeNames.predict_page_type.value, parse_attraction_edge)
workflow.add_edge(NodeNames.parse_attraction.value, END)
workflow.add_edge(NodeNames.parse_city.value, END)
app = workflow.compile()


