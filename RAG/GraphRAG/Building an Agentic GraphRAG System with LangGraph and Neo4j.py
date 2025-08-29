### From https://ai.plainenglish.io/building-a-graphrag-multi-agent-system-with-langgraph-and-neo4j-08fc2e2cb64c

import os
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document


os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

llm_transformer_filtered = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Recipe", "Foodproduct"],
    allowed_relationships=["CONTAINS"],
)

text = """
My favorite culinary creation is the irresistible Vegan Chocolate Cake Recipe. This delightful dessert is celebrated for its intense cocoa flavor and its incredibly soft and moist texture. It's completely vegan, dairy-free, and, thanks to the use of a special gluten-free flour blend, also gluten-free.
To make this cake, the recipe contains the following food products with their respective quantities: 250 grams of gluten-free flour blend, 80 grams of high-quality cocoa powder, 200 grams of granulated sugar, and 10 grams of baking powder. To enrich the taste and ensure a perfect rise, the recipe also contains 5 grams of vanilla extract. Among the liquid ingredients, 240 ml of almond milk and 60 ml of vegetable oil are needed.
This recipe produces a single chocolate cake, considered a FoodProduct of type dessert.
"""
documents = [Document(page_content=text)]
graph_documents_filtered = await llm_transformer_filtered.aconvert_to_graph_documents(
    documents
)

---------------------------------------------------------------------------

import os
from langchain_neo4j import Neo4jGraph

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"

graph = Neo4jGraph(refresh_schema=False)
graph.add_graph_documents(graph_documents_filtered)
---------------------------------------------------------------------------

import openai
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

recipe_id = "Vegan Chocolate Cake Recipe"
recipe_embedding = openai.embeddings.create(model="text-embedding-3-small", input=recipe_id).data[0].embedding

with driver.session() as session:
  # Create the embedding field
  session.run(
      "MATCH (r:Recipe {id: $recipe_id}) SET r.embedding = $embedding",
      recipe_id=recipe_id,
      embedding=recipe_embedding
  )
  # Create the vector index
  session.run(
      "CREATE VECTOR INDEX recipe_index IF NOT EXISTS FOR (r:Recipe) ON (r.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}"
  )

query = "a chocolate cake recipe that is vegan"
query_embedding = openai.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

with driver.session() as session:
    result = session.run(
        """
        CALL db.index.vector.queryNodes('recipe_index', 1, $embedding)
        YIELD node, score
        RETURN node.id AS name, score
        ORDER BY score DESC
        """,
        embedding=query_embedding
    )
    for record in result:
        print(record["name"], "=>", record["score"])


------------------------------------------------------------------------
### LangGrah
from typing import Literal
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from dataclasses import dataclass, field
from typing import Annotated
from utils.utils import update_knowledge
from core.state_graph.states.main_graph.input_state import InputState
from core.state_graph.states.main_graph.router import Router
from core.state_graph.states.step import Step

class Router(BaseModel):
    """Classify user query."""
    logic: str
    type: Literal["more-info", "valid", "general"]

@dataclass(kw_only=True)
class InputState:
    """
    Represents the input state containing a list of messages.

    Attributes:
        messages (list[AnyMessage]): The list of messages associated with the state, 
            processed using the add_messages function.
    """
    messages: Annotated[list[AnyMessage], add_messages]

@dataclass(kw_only=True)
class AgentState(InputState):
    """
    Represents the state of an agent within the main state graph.

    Attributes:
        router (Router): The routing logic for the agent.
        steps (list[Step]): The sequence of steps taken by the agent.
        knowledge (list[dict]): The agent's accumulated knowledge, updated via the update_knowledge function.
    """
    router: Router = field(default_factory=lambda: Router(type="general", logic=""))
    steps: list[Step] = field(default_factory=list)
    knowledge: Annotated[list[dict], update_knowledge] = field(default_factory=list)

#### Step 1: Analyze and route query
async def analyze_and_route_query(state: AgentState, *, config: RunnableConfig) -> dict[str, Router]:
    """
    Analyzes the current agent state and determines the routing logic for the next step.

    Args:
        state (AgentState): The current state of the agent, including messages and context.
        config (RunnableConfig): Configuration for the runnable execution.

    Returns:
        dict[str, Router]: A dictionary containing the updated router object.
    """
    model = init_chat_model(
        name="analyze_and_route_query", **app_config["inference_model_params"]
    )
    messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}] + state.messages
    print("---ANALYZE AND ROUTE QUERY---")
    print(f"MESSAGES: {state.messages}")
    response = cast(
        Router, await model.with_structured_output(Router).ainvoke(messages)
    )
    return {"router": response}

def route_query(state: AgentState) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """
    Determines the next action for the agent based on the router type in the current state.

    Args:
        state (AgentState): The current state of the agent, including the router type.

    Returns:
        Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
            The next node/action to execute in the state graph.

    Raises:
        ValueError: If the router type is unknown.
    """
    _type = state.router.type
    if _type == "valid":
        return "create_research_plan"
    elif _type == "more-info":
        return "ask_for_more_info"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type {_type}")

##### Step 1.1 Out of scope / More informations needed
async def ask_for_more_info(state: AgentState, *, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
    """
    Asks the user for more information based on the current routing logic.

    Args:
        state (AgentState): The current state of the agent, including routing logic and messages.
        config (RunnableConfig): Configuration for the runnable execution.

    Returns:
        dict[str, list[BaseMessage]]: A dictionary containing the new message(s) requesting more information.
    """
    model = init_chat_model(
        name="ask_for_more_info", **app_config["inference_model_params"]
    )
    system_prompt = MORE_INFO_SYSTEM_PROMPT.format(logic=state.router.logic)
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def respond_to_general_query(state: AgentState, *, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
    """
    Generates a response to a general user query based on the agent's current state and routing logic.

    Args:
        state (AgentState): The current state of the agent, including routing logic and messages.
        config (RunnableConfig): Configuration for the runnable execution.

    Returns:
        dict[str, list[BaseMessage]]: A dictionary containing the generated response message(s).
    """
    model = init_chat_model(
        name="respond_to_general_query", **app_config["inference_model_params"]
    )
    system_prompt = GENERAL_SYSTEM_PROMPT.format(logic=state.router.logic)
    print("---RESPONSE GENERATION---")
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}

##### Step 2: Create a research plan
async def review_research_plan(plan: Plan) -> Plan:
    """
    Reviews a research plan to ensure its quality and relevance.

    Args:
        plan (Plan): The research plan to be reviewed.

    Returns:
        Plan: The reviewed and potentially modified research plan.
    """
    formatted_plan = ""
    for i, step in enumerate(plan["steps"]):
        formatted_plan += f"{i+1}. ({step['type']}): {step['question']}\n"

    model = init_chat_model(
        name="create_research_plan", **app_config["inference_model_params"]
    )
    system_prompt = REVIEW_RESEARCH_PLAN_SYSTEM_PROMPT.format(
        schema=neo4j_graph.get_structured_schema, plan=formatted_plan
    )

    reviewed_plan = cast(
        Plan, await model.with_structured_output(Plan).ainvoke(system_prompt)
    )
    return reviewed_plan


async def reduce_research_plan(plan: Plan) -> Plan:
    """
    Reduces a research plan by simplifying or condensing its steps.

    Args:
        plan (Plan): The research plan to be reduced.

    Returns:
        Plan: The reduced research plan.
    """
    formatted_plan = ""
    for i, step in enumerate(plan["steps"]):
        formatted_plan += f"{i+1}. ({step['type']}): {step['question']}\n"

    model = init_chat_model(
        name="reduce_research_plan", **app_config["inference_model_params"]
    )
    system_prompt = REDUCE_RESEARCH_PLAN_SYSTEM_PROMPT.format(
        schema=neo4j_graph.get_structured_schema, plan=formatted_plan
    )

    reduced_plan = cast(
        Plan, await model.with_structured_output(Plan).ainvoke(system_prompt)
    )
    return reduced_plan


async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """
    Creates, reduces, and reviews a research plan based on the agent's current knowledge and messages.

    Args:
        state (AgentState): The current state of the agent, including knowledge and messages.
        config (RunnableConfig): Configuration for the runnable execution.

    Returns:
        dict[str, list[str] | str]: A dictionary containing the final steps of the reviewed plan and an empty knowledge list.
    """
    formatted_knowledge = "\n".join([item["content"] for item in state.knowledge])
    model = init_chat_model(
        name="create_research_plan", **app_config["inference_model_params"]
    )
    system_prompt = RESEARCH_PLAN_SYSTEM_PROMPT.format(
        schema=neo4j_graph.get_structured_schema, context=formatted_knowledge
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    print("---PLAN GENERATION---")

    # Generate plan
    plan = cast(Plan, await model.with_structured_output(Plan).ainvoke(messages))
    print("Plan")
    for i, step in enumerate(plan["steps"]):
        print(f"{i+1}. ({step['type']}): {step['question']}")

    # Reduce plan
    reduced_plan = cast(Plan, await reduce_research_plan(plan=plan))
    print("Reduced Plan")
    for i, step in enumerate(reduced_plan["steps"]):
        print(f"{i+1}. ({step['type']}): {step['question']}")

    # Review plan
    reviewed_plan = cast(Plan, await review_research_plan(plan=reduced_plan))

    print("Reviewed Plan")
    for i, step in enumerate(reviewed_plan["steps"]):
        print(f"{i+1}. ({step['type']}): {step['question']}")

    return {"steps": reviewed_plan["steps"], "knowledge": []}

##### Step 3: Conduct research
async def conduct_research(state: AgentState) -> dict[str, Any]:
    """
    Executes a research step using the research graph and updates the agent's knowledge.

    Args:
        state (AgentState): The current state of the agent, including steps and knowledge.

    Returns:
        dict[str, Any]: A dictionary containing the updated knowledge and remaining steps.
    """
    response = await research_graph.ainvoke(
        {"step": state.steps[0], "knowledge": state.knowledge}
    )
    knowledge = response["knowledge"]
    step = state.steps[0]
    print(
        f"\n{len(knowledge)} pieces of knowledge retrieved in total for the step: {step}."
    )
    return {"knowledge": knowledge, "steps": state.steps[1:]}

##### Step 4: Research subgraph building
@dataclass(kw_only=True)
class QueryState:
    """State class for managing research queries in the research graph."""
    query: str

class Step(TypedDict):
    """Single research step"""
    question: str
    type: Literal["semantic_search", "query_search"]

@dataclass(kw_only=True)
class ResearcherState:
    """State of the researcher graph."""
    step: Step
    queries: list[str] = field(default_factory=list)
    knowledge: Annotated[list[dict], update_knowledge] = field(default_factory=list)

###### Step 4.1 Semantic search
def execute_semantic_search(node_label: str, attribute_name: str, query: str):
    """Execute a semantic search on Neo4j vector indexes.
    
    This function performs vector-based similarity search using OpenAI embeddings
    to find nodes in the Neo4j graph database that are semantically similar to
    the provided query. It converts the query to an embedding vector and searches
    the corresponding vector index for the most similar nodes.
    
    Args:
        node_label (str): The label of the node type to search (e.g., 'Recipe', 'FoodProduct').
        attribute_name (str): The attribute/property of the node to search within (e.g., 'name', 'description').
        query (str): The search query to find semantically similar content.
        
    Returns:
        list: A list of dictionaries containing the matching nodes with their attributes,
              ordered by similarity score (highest first).
    """
    index_name = f"{node_label.lower()}_{attribute_name}_index"
    top_k = 1
    query_embedding = (
        openai.embeddings.create(model=app_config["embedding_model"], input=query)
        .data[0]
        .embedding
    )

    nodes = (
        f"node.name as name, node.{attribute_name} as {attribute_name}"
        if attribute_name != "name"
        else f"node.{attribute_name} as name"
    )
    response = neo4j_graph.query(
        f"""
        CALL db.index.vector.queryNodes('{index_name}', {top_k}, {query_embedding})
        YIELD node, score
        RETURN {nodes}
        ORDER BY score DESC"""
    )
    print(
        f"Semantic Search Tool invoked with parameters: node_label: '{node_label}', attribute_name: '{attribute_name}', query: '{query}'"
    )
    print(f"Semantic Search response: {response}")
    return response


async def semantic_search(state: ResearcherState, *, config: RunnableConfig):
    """Perform semantic search to find relevant nodes in the research graph.
    
    This function analyzes a research question to determine optimal search parameters
    and executes a semantic search on the Neo4j graph database. It uses an LLM to
    identify which node type and attribute should be searched, then performs vector-based
    similarity search to find semantically related content that can help answer the question.
    
    Args:
        state (ResearcherState): The current researcher state containing the
            research step question and accumulated knowledge.
        config (RunnableConfig): Configuration for the runnable execution.
        
    Returns:
        dict[str, list]: A dictionary with a "knowledge" key containing
            a list with the semantic search results formatted as knowledge items.
    """
    class Response(TypedDict):
        node_label: str
        attribute_name: str
        query: str

    model = init_chat_model(
        name="semantic_search", **app_config["inference_model_params"]
    )

    vector_indexes = neo4j_graph.query("SHOW VECTOR INDEXES YIELD name RETURN name;")
    print(f"vector_indexes: {vector_indexes}")

    system_prompt = SEMANTIC_SEARCH_SYSTEM_PROMPT.format(
        schema=neo4j_graph.get_structured_schema,
        vector_indexes=str(vector_indexes)
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "human", "content": state.step["question"]},
    ]
    response = cast(
        Response, await model.with_structured_output(Response).ainvoke(messages)
    )
    sem_search_response = execute_semantic_search(
        node_label=response["node_label"],
        attribute_name=response["attribute_name"],
        query=response["query"],
    )

    search_names = [f"'{record['name']}'" for record in sem_search_response]
    joined_search_names = ", ".join(search_names)
    knowledge = {
        "id": new_uuid(),
        "content": f"Executed Semantic Search on {response['node_label']}.{response['attribute_name']} for values similar to: '{response['query']}'\nResults: {joined_search_names}",
    }

    return {"knowledge": [knowledge]}

##### Step 4.2 Generate queries
async def correct_query_by_llm(query: str) -> str:
    """Correct a Cypher query using a language model.
    
    This function uses an LLM to review and correct a Cypher query based on
    the Neo4j graph schema. It provides schema-aware correction to ensure
    the query is properly formatted and uses valid relationships and nodes.
    
    Args:
        query (str): The Cypher query to be corrected.
        
    Returns:
        str: The corrected Cypher query.
    """
    model = init_chat_model(
        name="correct_query_by_llm", **app_config["inference_model_params"]
    )
    system_prompt = FIX_QUERY_SYSTEM_PROMPT.format(
        schema=neo4j_graph.get_structured_schema
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "human", "content": query},
    ]
    response = await model.ainvoke(messages)
    return response.content


def correct_query_by_parser(query: str) -> str:
    """Correct a Cypher query using a parser-based corrector.
    
    This function uses the CypherQueryCorrector to parse and correct
    Cypher queries based on the graph schema. It extracts the Cypher
    query from the text and applies structural corrections.
    
    Args:
        query (str): The text containing the Cypher query to be corrected.
        
    Returns:
        str: The corrected Cypher query.
    """
    corrector_schema = [
        Schema(el["start"], el["type"], el["end"])
        for el in neo4j_graph.get_structured_schema.get("relationships", [])
    ]
    cypher_query_corrector = CypherQueryCorrector(corrector_schema)

    extracted_query = extract_cypher(text=query)
    corrected_query = cypher_query_corrector(extracted_query)
    return corrected_query


async def generate_queries(
    state: ResearcherState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate and correct Cypher queries for a research step.
    
    This function generates multiple Cypher queries based on a research question
    and existing knowledge context. It uses an LLM to generate initial queries,
    then applies both LLM-based and parser-based corrections to ensure the
    queries are valid and properly formatted for the Neo4j graph database.
    
    Args:
        state (ResearcherState): The current researcher state containing the
            research step question and accumulated knowledge.
        config (RunnableConfig): Configuration for the runnable execution.
        
    Returns:
        dict[str, list[str]]: A dictionary with a "queries" key containing
            a list of corrected Cypher queries.
    """
    
    class Response(TypedDict):
        queries: list[str]

    print("---GENERATE QUERIES---")
    formatted_knowledge = "\n\n".join(
        [f"{i+1}. {item['content']}" for i, item in enumerate(state.knowledge)]
    )
    model = init_chat_model(
        name="generate_queries", **app_config["inference_model_params"]
    )
    system_prompt = GENERATE_QUERIES_SYSTEM_PROMPT.format(
        schema=neo4j_graph.get_schema, context=formatted_knowledge
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "human", "content": state.step["question"]},
    ]
    response = cast(
        Response, await model.with_structured_output(Response).ainvoke(messages)
    )
    response["queries"] = [
        await correct_query_by_llm(query=q) for q in response["queries"]
    ]
    response["queries"] = [
        correct_query_by_parser(query=q) for q in response["queries"]
    ]

    print(f"Queries: {response['queries']}")
    return {"queries": response["queries"]}


##### Step 4.3 Building subgraph
def build_research_graph():
    builder = StateGraph(ResearcherState)
    builder.add_node(generate_queries)
    builder.add_node(execute_query)
    builder.add_node(semantic_search)

    builder.add_conditional_edges(
        START,
        route_step,
        {"generate_queries": "generate_queries", "semantic_search": "semantic_search"},
    )
    builder.add_conditional_edges(
        "generate_queries",
        query_in_parallel,  # type: ignore
        path_map=["execute_query"],
    )
    builder.add_edge("execute_query", END)
    builder.add_edge("semantic_search", END)

    return builder.compile()


research_graph = build_research_graph()

##### Step 5: Check finished
def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """
    Determines whether the agent should respond or conduct further research based on the steps taken.

    Args:
        state (AgentState): The current state of the agent, including the steps performed.

    Returns:
        Literal["respond", "conduct_research"]: 
            "conduct_research" if there are steps present, otherwise "respond".
    """
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "respond"

###### Step 6: Respond
async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """
    Generates a final response to the user based on the agent's accumulated knowledge and messages.

    Args:
        state (AgentState): The current state of the agent, including knowledge and messages.
        config (RunnableConfig): Configuration for the runnable execution.

    Returns:
        dict[str, list[BaseMessage]]: A dictionary containing the generated response message(s).
    """
    print("--- RESPONSE GENERATION STEP ---")
    model = init_chat_model(name="respond", **app_config["inference_model_params"])
    formatted_knowledge = "\n\n".join([item["content"] for item in state.knowledge])
    prompt = RESPONSE_SYSTEM_PROMPT.format(context=formatted_knowledge)
    messages = [{"role": "system", "content": prompt}] + state.messages
    response = await model.ainvoke(messages)

    return {"messages": [response]}

##### Step 7: Building main graph
def build_main_graph():
    builder = StateGraph(AgentState, input=InputState)
    builder.add_node(analyze_and_route_query)
    builder.add_node(ask_for_more_info)
    builder.add_node(respond_to_general_query)
    builder.add_node(create_research_plan)
    builder.add_node(conduct_research)
    builder.add_node("respond", respond)

    builder.add_edge("create_research_plan", "conduct_research")
    builder.add_edge(START, "analyze_and_route_query")
    builder.add_conditional_edges("analyze_and_route_query", route_query)
    builder.add_conditional_edges("conduct_research", check_finished)
    builder.add_edge("respond", END)

    return builder.compile()
