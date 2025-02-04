### From https://medium.com/towards-data-science/using-llamaindex-workflow-to-implement-an-agent-handoff-feature-like-openai-swarm-9a63420c8540
### https://github.com/qtalen/multi-agent-customer-service/tree/main/src

llm = OpenAILike(
    model="qwen-max-latest",
    is_chat_model=True,
    is_function_calling_model=True,
    temperature=0.35
)
Settings.llm = llm

GREETINGS = "Hello, what can I do for you?"


def ready_my_workflow() -> CustomerService:
    memory = ChatMemoryBuffer(
        llm=llm,
        token_limit=5000
    )

    agent = CustomerService(
        memory=memory,
        timeout=None,
        user_state=initialize_user_state()
    )
    return agent


def initialize_user_state() -> dict[str, str | None]:
    return {
        "name": None
    }


@cl.on_chat_start
async def start():
    workflow = ready_my_workflow()
    cl.user_session.set("workflow", workflow)

    await cl.Message(
        author="assistant", content=GREETINGS
    ).send()

@cl.step(type="run", show_input=False)
async def on_progress(message: str):
    return message

@cl.on_message
async def main(message: cl.Message):
    workflow: CustomerService = cl.user_session.get("workflow")
    context = cl.user_session.get("context")
    msg = cl.Message(content="", author="assistant")
    user_msg = message.content
    handler = workflow.run(
        msg=user_msg,
        ctx=context
    )
    async for event in handler.stream_events():
        if isinstance(event, ProgressEvent):
            await on_progress(event.msg)

    await msg.send()
    result = await handler
    msg.content = result
    await msg.update()
    cl.user_session.set("context", handler.ctx)

class CustomerService(Workflow):
    def __init__(
            self,
            llm: OpenAILike | None = None,
            memory: ChatMemoryBuffer = None,
            user_state: dict[str, str | None] = None,
            *args,
            **kwargs
    ):
        self.llm = llm or Settings.llm
        self.memory = memory or ChatMemoryBuffer()
        self.user_state = user_state
        super().__init__(*args, **kwargs)

    @step
    async def start(self, ctx: Context, ev: StartEvent) -> StopEvent:
        ctx.write_event_to_stream(ProgressEvent(msg="We're making some progress."))
        return StopEvent(result="Hello World")

SKUS_TEMPLATE_EN = """
    You are the owner of an online drone store, please generate a description in English of all the drones for sale.
    Include the drone model number, selling price, detailed specifications, and a detailed description in more than 400 words.
    Do not include brand names.
    No less than 20 types of drones, ranging from consumer to industrial use.
"""

TERMS_TEMPLATE_EN = """
    You are the head of a brand's back office department, and you are asked to generate a standardized response to after-sales FAQs in English that is greater than 25,000 words.
    The text should include common usage questions, as well as questions related to returns and repairs after the sale.
    This text will be used as a reference for the customer service team when answering customer questions about after-sales issues.
    Only the body text is generated, no preamble or explanation is added.
"""

def get_index(collection_name: str,
              files: list[str]) -> VectorStoreIndex:
    chroma_client = chromadb.PersistentClient(path="temp/.chroma")

    collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    ready = collection.count()
    if ready > 0:
        print("File already loaded")
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    else:
        print("File not loaded.")
        docs = SimpleDirectoryReader(input_files=files).load_data()
        index = VectorStoreIndex.from_documents(
            docs, storage_context=storage_context, embed_model=embed_model,
            transformer=[SentenceSplitter(chunk_size=512, chunk_overlap=20)]
        )

    return index


INDEXES = {
    "SKUS": get_index("skus_docs", ["data/skus_en.txt"]),
    "TERMS": get_index("terms_docs", ["data/terms_en.txt"])
}

async def query_docs(
        index: VectorStoreIndex, query: str,
        similarity_top_k: int = 1
) -> str:
    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    nodes = await retriever.aretrieve(query)
    result = ""
    for node in nodes:
        result += node.get_content() + "\n\n"
    return result

class AgentConfig(BaseModel):
    """
    Detailed configuration for an agent
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(description="agent name")
    description: str = Field(
        description="agent description, which describes what the agent does"
    )
    system_prompt: str | None = None
    tools: list[BaseTool] | None = Field(
        description="function tools available for this agent"
    )

class TransferToAgent(BaseModel):
    """Used to explain which agent to transfer to next."""
    agent_name: str = Field(description="The name of the agent to transfer to.")

class RequestTransfer(BaseModel):
    """
    Used to indicate that you don't have the necessary permission to complete the user's request,
    or that you've already completed the user's request and want to transfer to another agent.
    """
    pass

def get_authentication_tools() -> list[BaseTool]:
    async def login(ctx: Context, username: str) -> bool:
        """When the user provides their name, you can use this method to update their status.。
        :param username The user's title or name.
        """
        if not username:
            return False

        user_state = await ctx.get("user_state", None)
        user_state["name"] = username.strip()
        await ctx.set("user_state", user_state)
        return True

    return [FunctionToolWithContext.from_defaults(async_fn=login)]

def get_pre_sales_tools() -> list[BaseTool]:
    async def skus_info_retrieve(ctx: Context, query: str) -> str:
        """
        When the user asks about a product, you can use this tool to look it up.
        :param query: The user's request.
        :return: The information found.
        """
        sku_info = await query_docs(INDEXES["SKUS"], query)
        return sku_info

    return [FunctionToolWithContext.from_defaults(async_fn=skus_info_retrieve)]


def get_after_sales_tools() -> list[BaseTool]:
    async def terms_info_retrieve(ctx: Context, query: str) -> str:
        """
        When the user asks about how to use a product, or about after-sales and repair options, you can use this tool to look it up.
        :param query: The user's request.
        :return: The information found.
        """
        terms_info = await query_docs(INDEXES["TERMS"], query)
        return terms_info
    return [FunctionToolWithContext.from_defaults(async_fn=terms_info_retrieve)]

def _get_agent_configs() -> list[AgentConfig]:
    return [
        AgentConfig(
            name="Authentication Agent",
            description="Record the user's name. If there's no name, you need to ask this from the customer.",
            system_prompt="""
            You are a front desk customer service agent for registration.
            If the user hasn't provided their name, you need to ask them.
            When the user has other requests, transfer the user's request.
            """,
            tools=get_authentication_tools()
        ),
        AgentConfig(
            name="Pre Sales Agent",
            description="When the user asks about product information, you need to consult this customer service agent.",
            system_prompt="""
            You are a customer service agent answering pre-sales questions for customers.
            You will respond to users' inquiries based on the context of the conversation.
            
            When the context is not enough, you will use tools to supplement the information.
            You can only handle user inquiries related to product pre-sales. 
            Please use the RequestTransfer tool to transfer other user requests.
            """,
            tools=get_pre_sales_tools()
        ),
        AgentConfig(
            name="After Sales Agent",
            description="When the user asks about after-sales information, you need to consult this customer service agent.",
            system_prompt="""
            You are a customer service agent answering after-sales questions for customers, including how to use the product, return and exchange policies, and repair solutions.
            You respond to users' inquiries based on the context of the conversation.
            When the context is not enough, you will use tools to supplement the information.
            You can only handle user inquiries related to product after-sales. 
            Please use the RequestTransfer tool to transfer other user requests.
            """,
            tools=get_after_sales_tools()
        )
    ]

def get_agent_config_pair() -> dict[str, AgentConfig]:
    agent_configs = _get_agent_configs()
    return {agent.name: agent for agent in agent_configs}


def get_agent_configs_str() -> str:
    agent_configs = _get_agent_configs()
    pair_list = [f"{agent.name}: {agent.description}" for agent in agent_configs]
    return "\n".join(pair_list)

ORCHESTRATION_PROMPT = """  
    You are a customer service manager for a drone store.
    Based on the user's current status, latest request, and the available customer service agents, you help the user decide which agent to consult next.

    You don't focus on the dependencies between agents; the agents will handle those themselves.
    If the user asks about something unrelated to drones, you should politely and briefly decline to answer.

    Here is the list of available customer service agents:
    {agent_configs_str}

    Here is the user's current status:
    {user_state_str}
"""

class OrchestrationEvent(Event):
    query: str


class ActiveSpeakerEvent(Event):
    query: str


class ToolCallEvent(Event):
    tool_call: ToolSelection
    tools: list[BaseTool]


class ToolCallResultEvent(Event):
    chat_message: ChatMessage


class ProgressEvent(Event):
    msg: str

class CustomerService(Workflow):
    ...
    
    @step
    async def start(
            self, ctx: Context, ev: StartEvent
    ) -> ActiveSpeakerEvent | OrchestrationEvent:
        self.memory.put(ChatMessage(
            role="user",
            content=ev.msg
        ))
        user_state = await ctx.get("user_state", None)
        if not user_state:
            await ctx.set("user_state", self.user_state)

        user_msg = ev.msg
        active_speaker = await ctx.get("active_speaker", default=None)

        if active_speaker:
            return ActiveSpeakerEvent(query=user_msg)
        else:
            return OrchestrationEvent(query=user_msg)

class CustomerService(Workflow):
    ...

    @step
    async def orchestrate(
            self, ctx: Context, ev: OrchestrationEvent
    ) -> ActiveSpeakerEvent | StopEvent:
        chat_history = self.memory.get()
        user_state_str = await self._get_user_state_str(ctx)
        system_prompt = ORCHESTRATION_PROMPT.format(
            agent_configs_str=get_agent_configs_str(),
            user_state_str=user_state_str
        )
        messages = [ChatMessage(role="system", content=system_prompt)] + chat_history
        tools = [get_function_tool(TransferToAgent)]
        event, tool_calls, _ = await self.achat_to_tool_calls(ctx, tools, messages)
        if event is not None:
            return event
        tool_call = tool_calls[0]
        selected_agent = tool_call.tool_kwargs["agent_name"]
        await ctx.set("active_speaker", selected_agent)
        ctx.write_event_to_stream(
            ProgressEvent(msg=f"In step orchestrate:\nTransfer to agent: {selected_agent}")
        )
        return ActiveSpeakerEvent(query=ev.query)

class CustomerService(Workflow):
    ...

    async def achat_to_tool_calls(self,
                              ctx: Context,
                              tools: list[FunctionTool],
                              chat_history: list[ChatMessage]
    ) -> tuple[StopEvent | None, list[ToolSelection], ChatResponse]:
    response = await self.llm.achat_with_tools(tools, chat_history=chat_history)
    tool_calls: list[ToolSelection] = self.llm.get_tool_calls_from_response(
        response=response, error_on_no_tool_call=False
    )
    stop_event = None
    if len(tool_calls) == 0:
        await self.memory.aput(response.message)
        stop_event = StopEvent(
            result=response.message.content
        )
    return stop_event, tool_calls, response

    @staticmethod
    async def _get_user_state_str(ctx: Context) -> str:
        user_state = await ctx.get("user_state", None)
        user_state_list = [f"{k}: {v}" for k, v in user_state.items()]
        return "\n".join(user_state_list)

class CustomerService(Workflow):
    ...

    @step
    async def speak_with_sub_agent(
            self, ctx: Context, ev: ActiveSpeakerEvent
    ) -> OrchestrationEvent | ToolCallEvent | StopEvent:
        active_speaker = await ctx.get("active_speaker", default="")
        agent_config: AgentConfig = get_agent_config_pair()[active_speaker]
        chat_history = self.memory.get()
        user_state_str = await self._get_user_state_str(ctx)

        system_prompt = (
                agent_config.system_prompt.strip()
                + f"\n\n<user state>:\n{user_state_str}"
        )
        llm_input = [ChatMessage(role="system", content=system_prompt)] + chat_history
        tools = [get_function_tool(RequestTransfer)] + agent_config.tools
        event, tool_calls, response = await self.achat_to_tool_calls(ctx, tools, llm_input)

        if event is not None:
            return event
        await ctx.set("num_tool_calls", len(tool_calls))
        for tool_call in tool_calls:
            if tool_call.tool_name == "RequestTransfer":
                await ctx.set("active_speaker", None)
                ctx.write_event_to_stream(
                    ProgressEvent(msg="The agent is requesting a transfer, please hold on...")
                )
                return OrchestrationEvent(query=ev.query)
            else:
                ctx.send_event(
                    ToolCallEvent(tool_call=tool_call, tools=agent_config.tools)
                )
        await self.memory.aput(response.message)

class CustomerService(Workflow):
    ...

    @step(num_workers=4)
    async def handle_tool_calls(
            self, ctx: Context, ev: ToolCallEvent
    ) -> ToolCallResultEvent:
        tool_call = ev.tool_call
        tools_by_name = {tool.metadata.get_name(): tool for tool in ev.tools}
        tool_msg = None
        tool = tools_by_name[tool_call.tool_name]
        additional_kwargs = {
            "tool_call_id": tool_call.tool_id,
            "name": tool.metadata.get_name()
        }
        if not tool:
            tool_msg = ChatMessage(
                role="tool",
                content=f"Tool {tool_call.tool_name} does not exists.",
                additional_kwargs=additional_kwargs
            )
            return ToolCallResultEvent(chat_message=tool_msg)

        try:
            if isinstance(tool, FunctionToolWithContext):
                tool_output = await tool.acall(ctx, **tool_call.tool_kwargs)
            else:
                tool_output = await tool.acall(**tool_call.tool_kwargs)

            tool_msg = ChatMessage(
                role="tool",
                content=tool_output.content,
                additional_kwargs=additional_kwargs
            )
        except Exception as e:
            tool_msg = ChatMessage(
                role="tool",
                content=f"Encountered error in tool call: {e}",
                additional_kwargs=additional_kwargs
            )
        return ToolCallResultEvent(chat_message=tool_msg)

class CustomerService(Workflow):
    ...

    @step
    async def aggregate_tool_results(
            self, ctx: Context, ev: ToolCallResultEvent
    ) -> ActiveSpeakerEvent | None:
        num_tool_calls = await ctx.get("num_tool_calls")
        results = ctx.collect_events(ev, [ToolCallResultEvent] * num_tool_calls)
        if not results:
            return None

        for result in results:
            await self.memory.aput(result.chat_message)
        return ActiveSpeakerEvent(query="")


