### From https://levelup.gitconnected.com/building-long-term-memory-in-agentic-ai-2941b0cca3bf

# Import the InMemoryStore class for storing memories in memory (no persistence)
from langgraph.store.memory import InMemoryStore

# Initialize an in-memory store instance for use in this notebook
in_memory_store = InMemoryStore()

# Define a user ID for memory storage
user_id = "1"

# Set the namespace for storing and retrieving memories
namespace_for_memory = (user_id, "memories")
---------------------------------------------------------------------------------
import uuid

# Generate a unique ID for the memory
memory_id = str(uuid.uuid4())

# Create a memory dictionary
memory = {"food_preference": "I like pizza"}

# Save the memory in the defined namespace
in_memory_store.put(namespace_for_memory, memory_id, memory)
# Retrieve all stored memories for the given namespace
memories = in_memory_store.search(namespace_for_memory)

# View the latest memory
memories[-1].dict()
-----------------------------------------------------------------------------------
# To enable threads (conversations)
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# To enable across-thread memory
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()

# Compile the graph with the checkpointer and store
# graph = graph.compile(checkpointer=checkpointer, store=in_memory_store)
------------------------------------------------------------------------------------
# Import the necessary libraries from Pydantic and Python's typing module
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal

# Define a Pydantic model for our router's structured output.
class RouterSchema(BaseModel):
    """Analyze the unread email and route it according to its content."""
    
    # Add a field for the LLM to explain its step-by-step reasoning.
    reasoning: str = Field(description="Step-by-step reasoning behind the classification.")
    
    # Add a field to hold the final classification.
    # The `Literal` type restricts the output to one of these three specific strings.
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email."
    )

# Import the base state class from LangGraph
from langgraph.graph import MessagesState

# Define the central state object for our graph.
class State(MessagesState):
    # This field will hold the initial raw email data.
    email_input: dict
    
    # This field will store the decision made by our triage router.
    classification_decision: Literal["ignore", "respond", "notify"]

# Define a TypedDict for the initial input to our entire workflow.
class StateInput(TypedDict):
    # The workflow must be started with a dictionary containing an 'email_input' key.
    email_input: dict

# Define a default persona for the agent.
default_background = """ 
I'm Lance, a software engineer at LangChain.
"""

# Define the initial rules for the triage LLM.
default_triage_instructions = """
Emails that are not worth responding to:
- Marketing newsletters and promotional emails
- Spam or suspicious emails
- CC'd on FYI threads with no direct questions

Emails that require notification but no response:
- Team member out sick or on vacation
- Build system notifications or deployments
Emails that require a response:
- Direct questions from team members
- Meeting requests requiring confirmation
"""

# Define the default preferences for how the agent should compose emails.
default_response_preferences = """
Use professional and concise language.
If the e-mail mentions a deadline, make sure to explicitly acknowledge
and reference the deadline in your response.

When responding to meeting scheduling requests:
- If times are proposed, verify calendar availability and commit to one.
- If no times are proposed, check your calendar and propose multiple options.
"""

# Define the default preferences for scheduling meetings.
default_cal_preferences = """
30 minute meetings are preferred, but 15 minute meetings are also acceptable.
"""

# Define the system prompt for the initial triage step.
triage_system_prompt = """

< Role >
Your role is to triage incoming emails based on background and instructions.
</ Role >

< Background >
{background}
</ Background >

< Instructions >
Categorize each email into IGNORE, NOTIFY, or RESPOND.
</ Instructions >

< Rules >
{triage_instructions}
</ Rules >
"""

# Define the user prompt for triage, which will format the raw email.
triage_user_prompt = """
Please determine how to handle the following email:
From: {author}
To: {to}
Subject: {subject}
{email_thread}"""


# Import the datetime library to include the current date in the prompt.
from datetime import datetime

# Define the main system prompt for the response agent.
agent_system_prompt_hitl_memory = """
< Role >
You are a top-notch executive assistant. 
</ Role >

< Tools >
You have access to the following tools: {tools_prompt}
</ Tools >

< Instructions >
1. Analyze the email content carefully.
2. Always call one tool at a time until the task is complete.
3. Use Question to ask the user for clarification.
4. Draft emails using write_email.
5. For meetings, check availability and schedule accordingly.
   - Today's date is """ + datetime.now().strftime("%Y-%m-%d") + """
6. After sending emails, use the Done tool.
</ Instructions >

< Background >
{background}
</ Background >

< Response Preferences >
{response_preferences}
</ Response Preferences >

< Calendar Preferences >
{cal_preferences}
</ Calendar Preferences >
"""
# Define the system prompt for our specialized memory update manager LLM.
MEMORY_UPDATE_INSTRUCTIONS = """
# Role
You are a memory profile manager for an email assistant.

# Rules
- NEVER overwrite the entire profile
- ONLY add new information
- ONLY update facts contradicted by feedback
- PRESERVE all other information

# Reasoning Steps
1. Analyze the current memory profile.
2. Review feedback messages.
3. Extract relevant preferences.
4. Compare to existing profile.
5. Identify facts to update.
6. Preserve everything else.
7. Output updated profile.

# Process current profile for {namespace}
<memory_profile>
{current_profile}
</memory_profile>
"""

# Define a reinforcement prompt to remind the LLM of the most critical rules.
MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT = """
Remember:
- NEVER overwrite the entire profile
- ONLY make targeted additions
- ONLY update specific facts contradicted by feedback
- PRESERVE all other information
"""

# A simple string describing the available tools for the LLM.
HITL_MEMORY_TOOLS_PROMPT = """
1. write_email(to, subject, content) - Send emails to specified recipients
2. schedule_meeting(attendees, subject, duration_minutes, preferred_day, start_time) - Schedule calendar meetings
3. check_calendar_availability(day) - Check available time slots
4. Question(content) - Ask follow-up questions
5. Done - Mark the email as sent
"""

# This utility unpacks the email input dictionary for easier access.
def parse_email(email_input: dict) -> tuple[str, str, str, str]:
    """Parse an email input dictionary into its constituent parts."""
    
    # Return a tuple containing the author, recipient, subject, and body of the email.
    return (
        email_input["author"],
        email_input["to"],
        email_input["subject"],
        email_input["email_thread"],
    )

# This function formats the raw email data into clean markdown for the LLM.
def format_email_markdown(subject, author, to, email_thread):
    """Format email details into a nicely formatted markdown string."""
    
    # Use f-string formatting to create a structured string with clear labels.
    return f"""
             **Subject**: {subject}
             **From**: {author}
             **To**: {to}
             {email_thread}
             ---
             """

# This function creates a human-friendly view of a tool call for the HITL interface.
def format_for_display(tool_call: dict) -> str:
    """Format a tool call into a readable string for the user."""
    
    # Initialize an empty string to build our display.
    display = ""
    
    # Use conditional logic to create custom, readable formats for our main tools.
    if tool_call["name"] == "write_email":
        display += f'# Email Draft\n\n**To**: {tool_call["args"].get("to")}\n**Subject**: {tool_call["args"].get("subject")}\n\n{tool_call["args"].get("content")}'
    elif tool_call["name"] == "schedule_meeting":
        display += f'# Calendar Invite\n\n**Meeting**: {tool_call["args"].get("subject")}\n**Attendees**: {", ".join(tool_call["args"].get("attendees"))}'
    elif tool_call["name"] == "Question":
        display += f'# Question for User\n\n{tool_call["args"].get("content")}'
    # Provide a generic fallback for any other tools.
    else:
        display += f'# Tool Call: {tool_call["name"]}\n\nArguments:\n{tool_call["args"]}'
        
    # Return the final formatted string.
    return display

# A function to retrieve memory from the store or initialize it with defaults.
def get_memory(store, namespace, default_content=None):
    """Get memory from the store or initialize with default if it doesn't exist."""
    
    # Use the store's .get() method to search for an item with a specific key.
    user_preferences = store.get(namespace, "user_preferences")
    
    # If the item exists, return its value (the stored string).
    if user_preferences:
        return user_preferences.value
    
    # If the item does not exist, this is the first time we're accessing this memory.
    else:
        # Use the store's .put() method to create the memory item with default content.
        store.put(namespace, "user_preferences", default_content)
        # Return the default content to be used in this run.
        return default_content

# A Pydantic model to structure the output of our memory update LLM call.
class UserPreferences(BaseModel):
    """Updated user preferences based on user's feedback."""
    
    # A field for the LLM to explain its reasoning, useful for debugging.
    chain_of_thought: str = Field(description="Reasoning about which user preferences need to add / update if required")
    
    # The final, updated string of user preferences.
    user_preferences: str = Field(description="Updated user preferences")

# Import AIMessage to help filter messages before sending them to the memory updater.
from langchain_core.messages import AIMessage

# This function intelligently updates the memory store based on user feedback.
def update_memory(store, namespace, messages):
    """Update memory profile in the store."""
    # First, get the current memory from the store so we can provide it as context.
    user_preferences = store.get(namespace, "user_preferences")
    # Initialize a new LLM instance specifically for this task, configured for structured output.
    memory_updater_llm = llm.with_structured_output(UserPreferences)
    
    # This is a small but important fix: filter out any previous AI messages with tool calls.
    # Passing these complex objects can sometimes cause errors in the downstream LLM call.
    messages_to_send = [
        msg for msg in messages
        if not (isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls)
    ]
    
    # Invoke the LLM with the memory prompt, current preferences, and the user's feedback.
    result = memory_updater_llm.invoke(
        [
            # The system prompt that instructs the LLM on how to update memory.
            {"role": "system", "content": MEMORY_UPDATE_INSTRUCTIONS.format(current_profile=user_preferences.value, namespace=namespace)},
        ] 
        # Append the filtered conversation messages containing the feedback.
        + messages_to_send
    )
    
    # Save the newly generated preference string back into the store, overwriting the old one.
    store.put(namespace, "user_preferences", result.user_preferences)

# Import the Command class for routing and BaseStore for type hinting
from langgraph.types import Command
from langgraph.store.base import BaseStore

# Define the first node in our graph, the triage router.
def triage_router(state: State, store: BaseStore) -> Command:
    """Analyze email content to decide the next step."""
    # Unpack the raw email data using our utility function.
    author, to, subject, email_thread = parse_email(state["email_input"])
    
    # Format the email content into a clean string for the LLM.
    email_markdown = format_email_markdown(subject, author, to, email_thread)
    
    # Here is the memory integration: fetch the latest triage instructions.
    # If they don't exist, it will use the `default_triage_instructions`.
    triage_instructions = get_memory(store, ("email_assistant", "triage_preferences"), default_triage_instructions)
    
    # Format the system prompt, injecting the retrieved triage instructions.
    system_prompt = triage_system_prompt.format(
        background=default_background,
        triage_instructions=triage_instructions,
    )
    
    # Format the user prompt with the specific details of the current email.
    user_prompt = triage_user_prompt.format(
        author=author, to=to, subject=subject, email_thread=email_thread
    )
    # Invoke the LLM router, which is configured to return our `RouterSchema`.
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    # Based on the LLM's classification, decide which node to go to next.
    if result.classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        # Set the next node to be the 'response_agent'.
        goto = "response_agent"
        # Update the state with the decision and the formatted email for the agent.
        update = {
            "classification_decision": result.classification,
            "messages": [{"role": "user", "content": f"Respond to the email: {email_markdown}"}],
        }
    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        # End the workflow immediately.
        goto = END
        # Update the state with the classification decision.
        update = {"classification_decision": result.classification}
    elif result.classification == "notify":
        print("🔔 Classification: NOTIFY - This email contains important information")
        # Go to the human-in-the-loop handler for notification.
        goto = "triage_interrupt_handler"
        # Update the state with the classification decision.
        update = {"classification_decision": result.classification}
    else:
        # Raise an error if the classification is invalid.
        raise ValueError(f"Invalid classification: {result.classification}")
    
    # Return a Command object to tell LangGraph where to go next and what to update.
    return Command(goto=goto, update=update)

# This is the primary reasoning node for the response agent.
def llm_call(state: State, store: BaseStore):
    """LLM decides whether to call a tool or not, using stored preferences."""

    # Fetch the user's latest calendar preferences from the memory store.
    cal_preferences = get_memory(store, ("email_assistant", "cal_preferences"), default_cal_preferences)
    
    # Fetch the user's latest response (writing style) preferences.
    response_preferences = get_memory(store, ("email_assistant", "response_preferences"), default_response_preferences)
    # Filter out previous AI messages with tool calls to prevent API errors.
    messages_to_send = [
        msg for msg in state["messages"]
        if not (isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls)
    ]

    # Invoke the main LLM, which is bound to our set of tools.
    # The prompt is formatted with the preferences retrieved from memory.
    response = llm_with_tools.invoke(
        [
            {"role": "system", "content": agent_system_prompt_hitl_memory.format(
                tools_prompt=HITL_MEMORY_TOOLS_PROMPT,
                background=default_background,
                response_preferences=response_preferences, 
                cal_preferences=cal_preferences
            )}
        ]
        + messages_to_send
    )
    
    # Return the LLM's response to be added to the state.
    return {"messages": [response]}

# Import the `interrupt` function from LangGraph.
from langgraph.types import interrupt

# Define the interrupt handler for the triage step.
def triage_interrupt_handler(state: State, store: BaseStore) -> Command:
    """Handles interrupts from the triage step, pausing for user input."""
    
    # Parse the email input to format it for display.
    author, to, subject, email_thread = parse_email(state["email_input"])
    email_markdown = format_email_markdown(subject, author, to, email_thread)
    # This is the data structure that defines the interrupt.
    # It specifies the action, the allowed user responses, and the content to display.
    request = {
        "action_request": {
            "action": f"Email Assistant: {state['classification_decision']}",
            "args": {}
        },
        "config": { "allow_ignore": True, "allow_respond": True },
        "description": email_markdown,
    }
    # The `interrupt()` function pauses the graph and sends the request to the user.
    # It waits here until it receives a response.
    response = interrupt([request])[0]
    # Now, we process the user's response.
    if response["type"] == "response":
        # The user decided to respond, overriding the 'notify' classification.
        user_input = response["args"]
        # We create a message to pass to the memory updater.
        messages = [{"role": "user", "content": f"The user decided to respond to the email, so update the triage preferences to capture this."}]
        
        # This is a key step: we call `update_memory` to teach the agent.
        update_memory(store, ("email_assistant", "triage_preferences"), messages)
        
        # Prepare to route to the main response agent.
        goto = "response_agent"
        # Update the state with the user's feedback.
        update = {"messages": [{"role": "user", "content": f"User wants to reply. Use this feedback: {user_input}"}]}
    elif response["type"] == "ignore":
        # The user confirmed the email should be ignored.
        messages = [{"role": "user", "content": f"The user decided to ignore the email even though it was classified as notify. Update triage preferences to capture this."}]
        
        # We still update memory to reinforce this preference.
        update_memory(store, ("email_assistant", "triage_preferences"), messages)
        
        # End the workflow.
        goto = END
        update = {} # No message update needed.
    else:
        raise ValueError(f"Invalid response: {response}")
    # Return a Command to direct the graph's next step.
    return Command(goto=goto, update=update)

# The main interrupt handler for reviewing tool calls.
def interrupt_handler(state: State, store: BaseStore) -> Command:
    """Creates an interrupt for human review of tool calls and updates memory."""
    
    # We'll build up a list of new messages to add to the state.
    result = []
    # By default, we'll loop back to the LLM after this.
    goto = "llm_call"


    # The agent can propose multiple tool calls, so we loop through them.
    for tool_call in state["messages"][-1].tool_calls:

        # We only want to interrupt for certain "high-stakes" tools.
        hitl_tools = ["write_email", "schedule_meeting", "Question"]
        if tool_call["name"] not in hitl_tools:
            # For other tools (like check_calendar), execute them without interruption.
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
            continue
            
        # Format the proposed action for display to the human reviewer.
        tool_display = format_for_display(tool_call)
        
        # Define the interrupt request payload.
        request = {
            "action_request": {"action": tool_call["name"], "args": tool_call["args"]},
            "config": { "allow_ignore": True, "allow_respond": True, "allow_edit": True, "allow_accept": True },
            "description": tool_display,
        }

        # Pause the graph and wait for the user's response.
        response = interrupt([request])[0]
        # --- MEMORY UPDATE LOGIC BASED ON USER RESPONSE ---
        
        if response["type"] == "edit":

            # The user directly edited the agent's proposed action.
            initial_tool_call = tool_call["args"]
            edited_args = response["args"]["args"]
            
            # This is the most direct form of feedback. We call `update_memory`.
            if tool_call["name"] == "write_email":
                update_memory(store, ("email_assistant", "response_preferences"), [{"role": "user", "content": f"User edited the email. Initial draft: {initial_tool_call}. Edited draft: {edited_args}."}])
            elif tool_call["name"] == "schedule_meeting":
                update_memory(store, ("email_assistant", "cal_preferences"), [{"role": "user", "content": f"User edited the meeting. Initial invite: {initial_tool_call}. Edited invite: {edited_args}."}])
            
            # Execute the tool with the user's edited arguments.
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(edited_args)
            result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})

        elif response["type"] == "response":

            # The user gave natural language feedback.
            user_feedback = response["args"]
            
            # We capture this feedback and use it to update memory.
            if tool_call["name"] == "write_email":
                update_memory(store, ("email_assistant", "response_preferences"), [{"role": "user", "content": f"User gave feedback on the email draft: {user_feedback}"}])
            elif tool_call["name"] == "schedule_meeting":
                update_memory(store, ("email_assistant", "cal_preferences"), [{"role": "user", "content": f"User gave feedback on the meeting invite: {user_feedback}"}])

            # We don't execute the tool. Instead, we pass the feedback back to the agent.
            result.append({"role": "tool", "content": f"User gave feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
            
        elif response["type"] == "ignore":
            # The user decided this action should not be taken. This is triage feedback.
            update_memory(store, ("email_assistant", "triage_preferences"), [{"role": "user", "content": f"User ignored the proposal to {tool_call['name']}. This email should not have been classified as 'respond'."}])
            result.append({"role": "tool", "content": "User ignored this. End the workflow.", "tool_call_id": tool_call["id"]})
            goto = END
        elif response["type"] == "accept":
            # The user approved the action. No memory update is needed.
            tool = tools_by_name[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})

    # Return a command with the next node and the messages to add to the state.
    return Command(goto=goto, update={"messages": result})

# This function determines the next step after the LLM has made its decision.
def should_continue(state: State) -> Literal["interrupt_handler", END]:
    """Route to the interrupt handler or end the workflow if the 'Done' tool is called."""
    
    # Get the list of messages from the current state.
    messages = state["messages"]
    # Get the most recent message, which contains the agent's proposed action.
    last_message = messages[-1]
    
    # Check if the last message contains any tool calls.
    if last_message.tool_calls:
        # Loop through each proposed tool call.
        for tool_call in last_message.tool_calls: 
            # If the agent has decided it's finished, we end the workflow.
            if tool_call["name"] == "Done":
                return END
            # For any other tool, we proceed to the human review step.
            else:
                return "interrupt_handler"

# Import the main graph-building class from LangGraph.
from langgraph.graph import StateGraph, START, END

# --- Part 1: Build the Response Agent Subgraph ---
# Initialize a new state graph with our defined `State` schema.
agent_builder = StateGraph(State)

# Add the 'llm_call' node to the graph.
agent_builder.add_node("llm_call", llm_call)

# Add the 'interrupt_handler' node to the graph.
agent_builder.add_node("interrupt_handler", interrupt_handler)

# Set the entry point of this subgraph to be the 'llm_call' node.
agent_builder.add_edge(START, "llm_call")

# Add the conditional edge that routes from 'llm_call' to either 'interrupt_handler' or END.
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "interrupt_handler": "interrupt_handler",
        END: END,
    },
)

# After the interrupt handler, the graph always loops back to the LLM to continue the task.
agent_builder.add_edge("interrupt_handler", "llm_call")

# Compile the subgraph into a runnable object.
response_agent = agent_builder.compile()

# --- Part 2: Build the Overall Workflow ---
# Initialize the main graph, defining its input schema as `StateInput`.
overall_workflow = (
    StateGraph(State, input=StateInput)
    # Add the triage router as the first node.
    .add_node("triage_router", triage_router)
    # Add the triage interrupt handler node.
    .add_node("triage_interrupt_handler", triage_interrupt_handler)
    # Add our entire compiled `response_agent` subgraph as a single node.
    .add_node("response_agent", response_agent)
    # Set the entry point for the entire workflow.
    .add_edge(START, "triage_router")
    # Define the edges from the triage router to the appropriate next steps.
    .add_edge("triage_router", "response_agent")
    .add_edge("triage_router", "triage_interrupt_handler")
    .add_edge("triage_interrupt_handler", "response_agent")
)

# Compile the final, complete graph.
email_assistant = overall_workflow.compile()

################# Test
# Import necessary libraries for testing.
import uuid 
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langgraph.store.memory import InMemoryStore


# Define a helper function to display the content of our memory store.
def display_memory_content(store, namespace=None):
    """A utility to print the current state of the memory store."""
    
    # Print a header for clarity.
    print("\n======= CURRENT MEMORY CONTENT =======")
    
    # If a specific namespace is requested, show only that one.
    if namespace:
        # Retrieve the memory item for the specified namespace.
        memory = store.get(namespace, "user_preferences")
        print(f"\n--- {namespace[1]} ---")
        if memory:
            print(memory.value)
        else:
            print("No memory found")
            
    # If no specific namespace is given, show all of them.
    else:
        # Define the list of all possible namespaces we are using.
        for ns in [
            ("email_assistant", "triage_preferences"),
            ("email_assistant", "response_preferences"),
            ("email_assistant", "cal_preferences"),
            ("email_assistant", "background")
        ]:
            # Retrieve and print the memory content for each namespace.
            memory = store.get(ns, "user_preferences")
            print(f"\n--- {ns[1]} ---")
            if memory:
                print(memory.value)
            else:
                print("No memory found")
            print("=======================================\n")


