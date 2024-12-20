### From https://medium.com/@sabber/context-aware-ai-agent-memory-management-and-state-tracking-3c904622edd7

"""
1. Context Awareness in AI Agents
   The article discusses how context awareness sets advanced systems apart from basic chatbots.
   A context-aware agent can maintain conversation history, remember user preferences, understand situational events, 
   track and update states, and handle context switches gracefully. 
   A diagram (not shown in code) conceptually illustrates how user input flows into an NLU agent, interacts with a context manager and memory system,
   and then guides the core agent’s response generation.

2. Designing a Context-Aware System Using LLMs
The text proposes building a context-aware AI agent leveraging modern LLM technologies. The system consists of:
"""
## a. Context Manager:
# Maintains conversation history, user preferences, and current context (e.g., timestamp, topic, state).
class ContextManager:
    def __init__(self):
        self.conversation_history: List[Dict] = []
        self.user_preferences: Dict = {}
        self.current_context: Dict = {
            'timestamp': None,
            'topic': None,
            'state': None
        }
      
## b. Conversation Manager:
# Handles conversation history and ensures the agent can retrieve recent dialogue for contextual prompts. 
# It stores up to max_history interactions, each with timestamps, user input, and agent responses.
class ConversationManager:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history = []

    def add_interaction(self, user_input: str, agent_response: str):
        interaction = {
            'timestamp': datetime.now(),
            'user_input': user_input,
            'agent_response': agent_response
        }
        self.history.append(interaction)

        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_context_window(self, window_size: int = 3) -> List[Dict]:
      
        return self.history[-window_size:]
## c. Contextual Prompt Generation:
# Combines recent conversation history, user preferences, and the current user’s input into a contextual prompt for the LLM:
def generate_contextual_prompt(
    user_input: str,
    conversation_history: List[Dict],
    user_preferences: Dict
) -> str:
    history_text = '\n'.join([
        f"User: {interaction['user_input']}\nagent: {interaction['agent_response']}"
        for interaction in conversation_history
    ])

    prompt = f"""
    Previous conversation:
    {history_text}

    User preferences:
    {user_preferences}

    Current user input:
    {user_input}

    Provide a contextually relevant response that takes into account
    the conversation history and user preferences.
    """

    return prompt

## d. Context-Aware Agent Implementation: The pseudocode integrates all components into a ContextAwareAgent class:
class ContextAwareagent:
    def __init__(self, api_key: str):
        self.conversation_manager = ConversationManager()
        self.context_manager = ContextManager()
        openai.api_key = api_key

    async def process_input(self, user_input: str) -> str:
        recent_history = self.conversation_manager.get_context_window()
        user_prefs = self.context_manager.user_preferences
        prompt = generate_contextual_prompt(user_input, recent_history, user_prefs)
        response = await self.get_llm_response(prompt)
        self.conversation_manager.add_interaction(user_input, response)
        return response

    async def get_llm_response(self, prompt: str) -> str:
        try:
            response = await openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful agent."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

# This agent maintains context by recalling user preferences, conversation history, and adjusting its responses accordingly.

"""
3. Practical Tools
   The text mentions a platform called getassisted.ai that allows building multi-agent systems without writing code. 
   The system aims to help users build assistants for learning niche topics.

4. Conclusion
   Creating a context-aware AI agent involves careful attention to:

   -a. Context Management: Maintaining conversation history and user preferences.
   -b. Memory Systems: Handling short-term and long-term memory, episodic, and semantic memories.
   -c. Contextual Prompt Generation: Dynamically building prompts that incorporate historical data and user preferences.
   -d. Resilient Design: Ensuring the system can handle context shifts and maintain coherence over time.

By following these best practices and leveraging LLM capabilities, developers can build more natural, coherent, and context-sensitive AI agents.
"""
