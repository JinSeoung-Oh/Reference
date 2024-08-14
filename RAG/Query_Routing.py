## From https://ai.gopubby.com/advanced-rag-retrieval-strategies-query-routing-c7a56c6f68fa
## Very important concept for building every LLM base application

"""
1. Query Routing
   Query routing is an intelligent query distribution feature in RAG that selects the most suitable processing method or data source 
   from multiple options based on the semantic content of the user’s input. Query routing can significantly enhance the relevance and efficiency of RAG retrieval,
   making it suitable for complex information retrieval scenarios, such as distributing user queries to different knowledge bases. 
   The flexibility and intelligence of query routing make it a critical component in building efficient RAG systems.

2. The type of Query Routing
   - LLM Router
     By constructing effective prompts, LLM determines the intent of user queries. Existing implementations include LlamaIndex Router, among others.
   - Embedding Router
     By using an Embedding model, user queries are transformed into vectors, and intent is determined through similarity retrieval. 
     Existing implementations include Semantic Router, among others.
"""

########### LLM Router #############
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

# initialize router query engine (single selection, llm)
class SingleSelection(BaseModel):
    """A single selection of a choice."""

    index: int
    reason: str

DEFAULT_SINGLE_SELECT_PROMPT_TMPL = (
    "Some choices are given below. It is provided in a numbered list "
    "(1 to {num_choices}), "
    "where each item in the list corresponds to a summary.\n"
    "---------------------\n"
    "{context_list}"
    "\n---------------------\n"
    "Using only the choices above and not prior knowledge, return "
    "the choice that is most relevant to the question: '{query_str}'\n"
)

# initialize tools
list_tool = QueryEngineTool.from_defaults(
    query_engine=list_query_engine,
    description="Useful for summarization questions related to the data source",
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context related to the data source",
)

class RouterQueryEngine(BaseQueryEngine):
    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        ......
        result = self._selector.select(self._metadatas, query_bundle)
        selected_query_engine = self._query_engines[result.ind]
        final_response = selected_query_engine.query(query_bundle)
        ......
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        list_tool,
        vector_tool,
    ],
)

query_engine.query("<query>")

"""
Advantages and Disadvantages
Advantages: Simple method, easy to implement.
Disadvantages: Requires a relatively powerful LLM to correctly interpret user intent. 
               If the selection result needs to be parsed into objects, the LLM must also support Function Calling capabilities.
"""


########### Embedding Router ##############
import os
from semantic_router import Route
from semantic_router.encoders import CohereEncoder, OpenAIEncoder
from semantic_router.layer import RouteLayer

# we could use this as a guide for our chatbot to avoid political conversations
politics = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president",
        "they're going to destroy this country!",
        "they will save the country!",
    ],
)
# this could be used as an indicator to our chatbot to switch to a more
# conversational prompt
chitchat = Route(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
        "the weather is horrendous",
        "let's go to the chippy",
    ],
)
# we place both of our decisions together into single list
routes = [politics, chitchat]

# OpenAI Encoder
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
encoder = OpenAIEncoder()

rl = RouteLayer(encoder=encoder, routes=routes)

rl("don't you love politics?").name
# politics
rl("how's the weather today?").name
# chitchat

"""
Advantages and Disadvantages
Advantages: Requires only an Embedding model, which is more efficient and resource-conserving than LLM Router.
Disadvantages: Requires the pre-loading of example utterances. If the examples are insufficient or not comprehensive enough, 
               the classification performance may be suboptimal.
"""

##### Example #####

from llama_index.llms.openai import OpenAI
from llama_index.core.query_pipeline import QueryPipeline, InputComponent
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.response_synthesizers.tree_summarize import TreeSummarize
import os
from typing import Dict, Any
from llama_index.core.query_pipeline import CustomQueryComponent
from llama_index.tools.bing_search import BingSearchToolSpec
from llama_index.agent.openai import OpenAIAgent

### LLM with Query Router
llm = OpenAI(model="gpt-3.5-turbo", system_prompt="You are a helpful assistant.")
chitchat_p = QueryPipeline(verbose=True)
chitchat_p.add_modules(
    {
        "input": InputComponent(),
        "llm": llm,
    }
)
chitchat_p.add_link("input", "llm")

### RAG with Query Router
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=2)
rag_p = QueryPipeline(verbose=True)
rag_p.add_modules(
    {
        "input": InputComponent(),
        "retriever": retriever,
        "output": TreeSummarize(),
    }
)
rag_p.add_link("input", "retriever")
rag_p.add_link("input", "output", dest_key="query_str")
rag_p.add_link("retriever", "output", dest_key="nodes")

### Web serch with Query Router

class WebSearchComponent(CustomQueryComponent):
    """Web search component."""
    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        assert "input" in input, "input is required"
        return input

    @property
    def _input_keys(self) -> set:
        """Input keys dict."""
        return {"input"}

    @property
    def _output_keys(self) -> set:
        return {"output"}

    def _run_component(self, **kwargs) -> Dict[str, Any]:
        """Run the component."""
        tool_spec = BingSearchToolSpec(api_key=os.getenv("BING_SEARCH_API_KEY"))
        agent = OpenAIAgent.from_tools(tool_spec.to_tool_list())
        question = kwargs["input"]
        result = agent.chat(question)
        return {"output": result}
      
web_p = QueryPipeline(verbose=True)
web_p.add_modules(
    {
        "input": InputComponent(),
        "web_search": WebSearchComponent(),
    }
)
web_p.add_link("input", "web_search")

#######
chitchat = Route(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
        "the weather is horrendous",
        "let's go to the chippy",
    ],
)

rag = Route(
    name="rag",
    utterances=[
        "What mysterious object did Loki use in his attempt to conquer Earth?",
        "Which two members of the Avengers created Ultron?",
        "How did Thanos achieve his plan of exterminating half of all life in the universe?",
        "What method did the Avengers use to reverse Thanos' actions?",
        "Which member of the Avengers sacrificed themselves to defeat Thanos?",
    ],
)

web = Route(
    name="web",
    utterances=[
        "Search online for the top three countries in the 2024 Paris Olympics medal table.",
        "Find the latest news about the U.S. presidential election.",
        "Look up the current updates on NVIDIA's
 stock performance today.",
        "Search for what Musk said on X last month.",
        "Find the latest AI news.",
    ],
)


### Example of CustomQueryComponent

from llama_index.core.base.query_pipeline.query import (
    QueryComponent,
    QUERY_COMPONENT_TYPE,
)
from llama_index.core.bridge.pydantic import Field

class SemanticRouterComponent(CustomQueryComponent):
    """Semantic router component."""
    components: Dict[str, QueryComponent] = Field(
        ..., description="Components (must correspond to choices)"
    )

    def __init__(self, components: Dict[str, QUERY_COMPONENT_TYPE]) -> None:
        """Init."""
        super().__init__(components=components)

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        return input

    @property
    def _input_keys(self) -> set:
        """Input keys dict."""
        return {"input"}

    @property
    def _output_keys(self) -> set:
        return {"output", "selection"}

    def _run_component(self, **kwargs) -> Dict[str, Any]:
        """Run the component."""
        if len(self.components) < 1:
            raise ValueError("No components")
        if chitchat.name not in self.components.keys():
            raise ValueError("No chitchat component")
        routes = [chitchat, rag, web]
        encoder = OpenAIEncoder()
        rl = RouteLayer(encoder=encoder, routes=routes)
        question = kwargs["input"]
        selection = rl(question).name
        if selection is not None:
            output = self.components[selection].run_component(input=question)
        else:
            output = self.components["chitchat"].run_component(input=question)
        return {"output": output, "selection": selection}

p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "router": SemanticRouterComponent(
            components={
                "chitchat": chitchat_p,
                "rag": rag_p,
                "web": web_p,
            }
        ),
    }
)

output = p.run(input="hello")
# Selection: chitchat
# Output: assistant: Hello! How can I assist you today?

output = p.run(input="Which two members of the Avengers created Ultron?")
# Selection: rag
# Output: Tony Stark and Bruce Banner.

output = p.run(input="Search online for the top three countries in the 2024 Paris Olympics medal table.")
# Selection: web
# Output: The top three countries in the latest medal table for the 2024 Paris Olympics are as follows:
# 1. United States
# 2. China
# 3. Great Britain





