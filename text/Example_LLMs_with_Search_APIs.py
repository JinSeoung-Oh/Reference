## From https://ai.gopubby.com/enhancing-llms-with-search-apis-09182d9dab97
## Have to check : https://python.langchain.com/docs/modules/agents/tools/custom_tools

# YOU.COM API
@tool
def you_com_api(query: str) -> str:
    """Use this tool to search for latest information on the internet."""
    headers = {"X-API-Key": "XXX"}
    params = {"query": query, "num_web_results": 3}
    results = requests.get(
        f"https://api.ydc-index.io/search",
        params=params,
        headers=headers,
    ).json()
    return json.dumps(results)


# Exa Search API
@tool
def exa_search_api(query: str) -> str:
    """Use this tool to search for latest information on the internet."""
    url = "https://api.exa.ai/search"

    payload = {
        "query": query,
        "contents": {"text": {"includeHtmlTags": False}},
        "numResults": 3,
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": "XXX",
    }
    response = requests.post(url, json=payload, headers=headers)

    return response.text


# Tavily Search
@tool
def tavily_search_api(query: str) -> str:
    """Use this tool to search for latest information on the internet."""
    tavily = TavilyClient(api_key="XXX")
    result = tavily.search(
        query=query,
        search_depth="advanced",
        max_results=3,
        include_answer=True,
        include_raw_content=False,
    )
    return json.dumps(result)


# Perplexity AI
@tool
def perplexity_ai_api(query: str) -> str:
    """Use this tool to search for latest information on the internet."""
    url = "https://api.perplexity.ai/chat/completions"

    payload = {
        "model": "sonar-medium-online",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": query},
        ],
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Bearer pplx-XXX",
    }

    response = requests.post(url, json=payload, headers=headers)
    return response.text

# Prepare agent
# Following https://python.langchain.com/docs/modules/agents/agent_types/openai_tools
prompt = hub.pull("hwchase17/openai-tools-agent")


def ask_agent(query, tool):

    # Bind tools to agent
    llm = ChatOpenAI(openai_api_key="XXX", model="gpt-4-0125-preview", temperature=0)
    llm_agent = llm.bind_tools([tool])

    # Define proper promt that includes tools
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are very powerful assistant, but don't know current events.
              Always try to list the sources your answer is based on.
              """,
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Assembling llm and parts to get an agent
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_agent
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=True)
    result = agent_executor.invoke({"input": query})
    return result["output"]
