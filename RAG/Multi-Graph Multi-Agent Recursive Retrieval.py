### https://medium.com/enterprise-rag/legal-document-rag-multi-graph-multi-agent-recursive-retrieval-through-legal-clauses-c90e073e0052

"""
The article demonstrates an intelligent system for navigating legal document clauses using a multi-agent system based on a multi-graph approach. 
The stack includes Reducto.AI, WhyHow.AI, Langgraph, and LlamaIndex to create a document hierarchy graph and a definition graph, 
which helps in retrieving relevant clauses, recursively linking clauses, and footnotes to understand the full context of legal documents.

The specific problem addressed is building a document hierarchy in legal documents where clauses refer to other clauses,
requiring recursive retrieval to understand the full context. This recursive traversal can be applied not only to legal clauses but also to other document elements, 
including multimodal data, footnotes, and hyperlinks.

In the example, the system answers a compliance-related question from a Malaysian Central Bank regulatory document by:

Retrieving the definition of CCO from the definition page.
Navigating clauses 6.3 and 7.2.
Following links to paragraphs 7.3, 7.4, and further to 9.1, which are referenced within clauses.
The system smartly navigates through a multi-agent workflow, leveraging a Lexical Graph for document structure and a Definition Graph for legal terms.
Key components include:

Router Agent: Detects linked sections and footers, triggering traversal when required.
Recursive Retrieval Agent: Recursively retrieves clauses when links are detected.
Answering Agent: Summarizes the retrieved information into a structured response.
The workflow retrieves and intelligently connects the relevant clauses, paragraphs, and footnotes, 
building a multi-graph system that ensures precise information retrieval. 
This system shows potential for broader applications in Regulatory-Aware Generation (RAG), knowledge graph construction, and smart document traversal.

This innovative solution leverages WhyHow.AI’s Knowledge Graph Studio for modular, agentic knowledge graphs that integrate seamlessly with LLMs and developer workflows,
showing a step toward more deterministic, structured knowledge retrieval systems in regulatory and legal contexts"""

# Retrieves the definitions of terms mentioned in the query.
def definitions_search(query_prompt: str, client: Optional[WhyHow]=None) -> Dict[str, str]:
    """
    Search for definitions of terms in a question prompt and return them as a dictionary.
    """
    if client is None:
        client = WhyHow(api_key=WHYHOW_API_KEY, base_url=WHYHOW_API_URL)

    definitions_response = client.graphs.query_unstructured(
        graph_id=definitions_graph.graph_id,
        query=query_prompt,
    )
    
    response_text = definitions_response.answer
    term_def_pairs = response_text.split('\n')
    definitions_dict = {}
    
    for pair in term_def_pairs:
        if ':' in pair:
            term, definition = pair.split(':', 1)
            definitions_dict[term.strip()] = definition.strip()
    
    return definitions_dict

query_prompt = """Return me definitions for the terms in this query: "How can the Board and the CCO manage control functions?" Ensure the term-definition pairs are separated by newlines, properly capitalised"""

definitions_dict = definitions_search(query_prompt)


def print_prompt_definitions_dict(definitions_dict):
    prompt = "Relevant Definitions:\n"
    for term, definition in definitions_dict.items():
        prompt += f"{term}: {definition}\n"
    return prompt

print(print_prompt_definitions_dict(definitions_dict))

# Router Agent
def router_agent(state: AgentState) -> AgentState:
    # decide if process should should stop or continue

    starter_prompt_footer = f"""
        You are an intelligent agent overseeing a multi-agent retrieval process of graph nodes from a document. These nodes are to answer the query: 
        ```{state['query']}```
        
        Below this request is a list of nodes that were automatically retrieved. 
        
        You must determine if the list of nodes is enough to answer the query. If there isn't enough information, you must identify any relevant footer information in the nodes.
        
        A node can footer information asking to look in another section/part of the document, which will require a separate natural language search. 
        Example: If the footer says "see paragraph x", a search query e.g. "Return paragraph x to answer the query '{state['query']}'" should be made. 
    
        If there are no further nodes worth analyzing, return an empty response. ONLY RETURN QUERIES FOR FOOTERS THAT ARE RELEVANT TO ANSWERING THE QUERY
        
        Else, if any relevant nodes require a footer search, specify the node_id and the search query.
        Nodes are identified by node_id and must be quoted in backticks.     
    """
    
    starter_prompt_link = f"""
        You are an intelligent agent overseeing a multi-agent retrieval process of graph nodes from a document. These nodes are to answer the query: 
        ```{state['query']}```
        
        Below this request is a list of nodes that were automatically retrieved. 
        
        You must determine if the list of nodes is enough to answer the query. If there isn't enough information, you must identify any linked nodes that could be worth exploring.
        
        If there are no further nodes worth analyzing, return an empty response.
        
        Return a list of node_ids. ONLY RETURN NODE_IDS for NODES THAT ARE RELEVANT TO ANSWERING THE QUERY. Nodes are identified by node_id and must be quoted in backticks.
    """
    
    # collect latest nodes, and all nodes
    last_fetched_nodes_flattened: Dict[str, MultiAgentSearchLocalNode] = {}
    all_nodes_flattened: Dict[str, MultiAgentSearchLocalNode] = {}

# Supervisor Agent
def supervisor_agent(state:AgentState) -> AgentState:
    
    # Look for search failures. This might be an instance where multiple searches were made for certain parts of the document, but no relevant information was found.
    # This means that the search has to be ended prematurely to prevent infinite loops.
    printout = ""
    for node in state["previous_nodes"]:
        printout += node.print_node_prompt()
    for node in state["last_fetched_context_nodes"]:
        printout += node.print_node_prompt()
        
    prompt = f"""
You are a supervisor agent overseeing the multi-agent retrieval process of graph nodes from a document. The nodes are to answer the query:
```{state['query']}```


Below is a list of nodes that were automatically retrieved, followed by a list of errors. If there are many similar, repeated errors in the retrieval process , where no further linked or relevant nodes could be retrieved, return END to end the process. Else return CONTINUE. 
Return only a single word, either END or CONTINUE.
"""
    
    completion = openai_client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": printout},
            {"role": "user", "content": state['search_failures']},
        ],
    )

# Recursive Agent
def recursive_retrieval(state: AgentState) -> AgentState:

    current_nodes = state["last_fetched_context_nodes"]
    
    for current_node in current_nodes:
        state["previous_nodes"].append(current_node)

    new_current_nodes = []

    # look up the nodes to fetch by id    
    
    for node_id in state["node_links_to_fetch"]:
    # sometimes GPT returns node ids with or without backticks
        if node_id[0] == "`":
            node_id = node_id[1:-1]
        if node_id in local_nodes_map:
            new_current_nodes.append(local_nodes_map[node_id])
        else:
            state["search_failures"].append(f"Failed to fetch node with id: {node_id}")


    for node_id, search_query in state["node_footers_to_fetch"].items():
        # fetch nodes by keyword and bm25 search
        footer_retrieved_nodes = retrieve_with_keywords_bm25(search_query)
        # LLM prunes nodes that are not relevant
        footer_retrieved_nodes, _ = prune_nodes(search_query, footer_retrieved_nodes)

        for node in footer_retrieved_nodes:
            new_current_nodes.append(node)

        # if no nodes fetched, log failure
        if len(footer_retrieved_nodes) == 0:
            state["search_failures"].append(
                f"Failed to fetch nodes for query: {search_query}"
            )

    state["last_fetched_context_nodes"] = new_current_nodes
    state["pass_count"] += 1
    state["node_footers_to_fetch"] = {}
    state["node_links_to_fetch"] = []

    return state

# Answering Agent
def answering_agent(state: AgentState) -> AgentState:
    # answer the query
    prompt = f"""
You are an answering agent. You will be given a list of document nodes that were automatically retrieved by the system. These nodes are to answer the query:
```{state['query']}```

Give references to sections/paragraphs if possible, but do not output full node ids with backticks and the hash. 
"""


