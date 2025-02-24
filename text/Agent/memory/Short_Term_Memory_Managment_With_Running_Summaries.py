### From https://ai.gopubby.com/short-term-memory-managment-with-running-summaries-e0dee5c631ea

import dotenv

%load_ext dotenv
%dotenv

from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

class State(MessagesState):
    summary: str

llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
)

def summarize_conversation(state: State) -> State:
    # Get summary
    summary = state.get("summary")
    
    if summary:
        sys_msg = f"""Given this conversation summary: {summary}

        Your task:
        1. Analyze new messages provided above
        2. Identify key updates in topic, context, or user intent
        3. Integrate these updates with existing summary
        4. Maintain chronological flow and contextual relevance
        5. Focus on information essential for conversation continuity

        Generate an updated summary that maintains clarity and coherence."""
                
    else:
        sys_msg = """Analyze the conversation above and create a concise summary that:
        1. Captures main topics and key points discussed
        2. Preserves essential context and decisions made
        3. Notes any unresolved questions or action items
        4. Maintains chronological order of major developments

        Focus on information needed for conversation continuity."""
    
    message = state["messages"] + [HumanMessage(content=sys_msg)]
    response = llm.invoke(message)
    
    # Delete unwanted messages, only keep the last two
    deleted_messages = [RemoveMessage(m.id) for m in state["messages"][:-2]]
    
    return {"messages": deleted_messages, "summary": response.content}

def model_invocation(state: State) -> State:
    # get summary
    summary = state.get("summary")
    
    if summary:
        sys_msg = f"Summary of previous conversation with the user: {summary}"
        
        messages = [SystemMessage(sys_msg)] + state["messages"]
        
    else:
        messages = state["messages"]
        
    response = llm.invoke(messages)
    
    return {"messages": response}

def should_continue(state: State) -> str:
    """Return the next node to execute, decides wherether we should generate a running summary or not"""
    
    messages = state.get("messages")
    
    # Summarize the conversatoin if there are more than 6 messages
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise, end
    return END

builder = StateGraph(State)

builder.add_node("conversation", model_invocation)
builder.add_node("summarize_conversation", summarize_conversation)

builder.add_edge(START, "conversation")
builder.add_conditional_edges("conversation", should_continue)
builder.add_edge("summarize_conversation", END)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)
