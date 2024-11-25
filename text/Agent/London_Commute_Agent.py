### From https://pub.towardsai.net/london-commute-agent-from-concepts-to-pretty-maps-6b2a0a28dcc8
### See this : https://gist.github.com/anderzzz

# build_agents.py

PROMPT_FOLDER = os.path.join(os.path.dirname(__file__), 'prompt_templates')

def build_agent(
        api_key_env_var: str,
        system_prompt_template: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        name: str,
        tools: Optional[ToolSet] = None,
        system_prompt_kwargs: Optional[Dict] = None,
) -> Engine:
    """Build an agent with a system prompt and optional tools
    Args:
        api_key_env_var: The name of the environment variable that holds the API key to the LLM API
        system_prompt_template: The name of the Jinja2 template file with the system prompt
        model_name: The name of the LLM model to use for the agent
        max_tokens: The maximum number of tokens to generate
        temperature: The temperature to use for generation
        name: The name of the agent, used for logging
        tools: The tool set to include with the agent
        system_prompt_kwargs: The keyword arguments to pass to the system prompt template in case the
            jinja template includes variables
    """
    system_prompt_template = Environment(
        loader=FileSystemLoader(PROMPT_FOLDER),
    ).get_template(system_prompt_template)
    if system_prompt_kwargs is None:
        system_prompt_kwargs = {}

    return Engine(
        name=name,
        api_key_env_var=api_key_env_var,
        system_prompt=system_prompt_template.render(**system_prompt_kwargs),
        message_params=AnthropicMessageParams(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        ),
        tools=tools,
    )

  
#
# Instantiate the agent that will route requests to the appropriate sub-task agent as well as
# speak with the principal.
date_now_in_london = datetime.now(pytz.timezone('Europe/London'))
date_str = date_now_in_london.strftime('%Y-%m-%d')
agent_router = build_agent(
    name='agent to route requests and speak with the principal',
    api_key_env_var='ANTHROPIC_API_KEY',
    system_prompt_template='router.j2',
    system_prompt_kwargs={
        'year_today': '2024',
        'date_today': date_str,
        'user_name': user_0.get('name', None),
        'user_shorthands': user_0.get('user location short-hands', None),
    },
    model_name='claude-3-5-sonnet-20241022',
    max_tokens=1000,
    temperature=0.5,
    tools=SubTaskAgentToolSet(
        subtask_agents={
            'preferences_and_settings': agent_handle_preferences_and_settings,
            'journey_planner': agent_handle_journey_plans,
            'output_artefacts': agent_handle_output_artefacts,
        },
        tools_to_include=('preferences_and_settings',
                          'journey_planner',
                          'output_artefacts'),
    ),
)

----------------------------------------------------------------------------------------------
# 
    system_prompt_template = Environment(
        loader=FileSystemLoader(PROMPT_FOLDER),
    ).get_template(system_prompt_template)

[
  {
    "name": "preferences_and_settings",
    "description": "Specific preferences or conditions of the journey by the user are handled and set by this tool. That includes such things as whether to prefer journeys with few interchanges, or fast journeys or what amount of walking or biking the user prefers. These changes are persistent, such that a change can be done once and used in multiple subsequent journey planning task",
    "input_schema": {
      "type": "object",
      "properties": {
        "input_prompt": {
          "type": "string",
          "description": "Free text prompt that contains the text relevant to the user's preferences and settings."
        }
      },
      "required": ["input_prompt"]
    }
  },
  {
    "name": "journey_planner",
    "description": "Compute and get the journey plans based on specific starting and destination points as well as the time of day and date.",
    "input_schema": {
      "type": "object",
      "properties": {
        "input_prompt": {
          "type": "string",
          "description": "Free text prompt that contains the text relevant to the journey the user needs to be planned."
        }
      },
      "required": ["input_prompt"]
    }
  },
  {
    "name": "output_artefacts",
    "description": "Generate the output artefacts for the journey planner, which communicate the journey plan to the user. Artefacts this tool can create are (1) map of London with the travel path drawn on it; (2) ICS file (a calendar file) that can be imported into the user's calendar; (3) detailed free text of the steps of the journey plan.",
    "input_schema": {
      "type": "object",
      "properties": {
        "input_prompt": {
        "type": "string",
        "description": "Free text prompt that contains the text relevant to the output artefact to create."
        },
        "input_structured": {
            "type": "object",
            "properties": {
                "journey_index": {
                "type": "integer",
                "description": "The index of the journey which plans to turn into one more more output artefacts"
                },
                "plan_index": {
                "type": "integer",
                "description": "The index of the plan to turn into one more more output artefacts"
                }
            },
            "required": ["journey_index", "plan_index"]
        }
      },
      "required": ["input_prompt","input_structured"]
    }
  }
]

-----------------------------------------------------------------------------------------------------------
# ToolSet
import json

class ToolSet:
    def __init__(self,
                 tool_spec_file: Optional[str] = None,
                 tools_to_include: Optional[Sequence[str]] = None,
                 ):
        if tool_spec_file is None:
            self._tools = []
        else:
            with open(tool_spec_file, 'r') as f:
                self._tools = json.load(f)
        if tools_to_include is not None:
            self._tools = [tool for tool in self._tools if tool['name'] in tools_to_include]

    def __call__(self, tool_name: str, **kwargs) -> str:
        try:
            _tool_exec = getattr(self, tool_name)
        except AttributeError:
            raise ValueError(f'Unknown tool name {tool_name}')
        return _tool_exec(**kwargs)

    @property
    def tools_spec(self) -> List[Dict[str, Any]]:
        return self._tools
      
      
TOOL_SPEC_FILE = os.path.join(os.path.dirname(__file__), 'tools.json')

class SubTaskAgentToolSet(ToolSet):
    """Sub-task agent tool set
    Args:
        subtask_agents: The sub-task agents to use as tools
        tools_to_include: The tools to include in the tool set
        tool_spec_file: The file with the tool specifications
    """
    def __init__(self,
                 subtask_agents: Dict[str, Engine],
                 tools_to_include=Optional[Sequence[str]],
                 tool_spec_file: str = TOOL_SPEC_FILE,
                 ):
        super().__init__(tool_spec_file, tools_to_include)
        self.subtask_agents = subtask_agents

        # The sub-task agents are only going to create (with LLM) and execute a call to the underlying tool
        # function, not create any conversational output or recall previous outputs.
        self.kwargs_only_use_tool_no_memory = {
            'tool_choice_type': 'any',
            'interpret_tool_use_output': False,
            'with_memory': False,
        }

    def _invoke_engine(self, agent_key: str, **kwargs) -> str:
        return self.subtask_agents[agent_key].process(**kwargs)

    def preferences_and_settings(self,
                                 input_prompt: str,
                                 ) -> str:
        return self._invoke_engine('preferences_and_settings',
                                   input_prompt=input_prompt,
                                   **self.kwargs_only_use_tool_no_memory
                                   )

    def journey_planner(self, input_prompt: str) -> str:
        return self._invoke_engine('journey_planner',
                                   input_prompt=input_prompt,
                                   **self.kwargs_only_use_tool_no_memory
                                   )

    def output_artefacts(self, input_prompt: str, input_structured: Dict[str, Any]) -> str:
        return self._invoke_engine('output_artefacts',
                                   input_prompt=input_prompt,
                                   input_structured=input_structured,
                                   **self.kwargs_only_use_tool_no_memory
                                   )

-----------------------------------------------------------------------------------------------------
# Engine
class Engine:
    """The main object to interact with the Anthropic API.
    The `process` method handles the interactions with input prompts.
    """
    def __init__(self,
                 api_key_env_var: str,
                 system_prompt: str,
                 message_params: AnthropicMessageParams,
                 name: Optional[str] = None,
                 tools: Optional[ToolSet] = None,
                 ):
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise ValueError(f'Did not find an API key in environment variable {api_key_env_var}')
        self.client = Anthropic(api_key=os.getenv(api_key_env_var))
        self.system_prompt = system_prompt
        self.name = name
        self.message_params = message_params
        if tools is None:
            self.tool_set = ToolSet()
        self.tool_set = tools
        self.tool_spec = self.tool_set.tools_spec
        self.tool_choice = None
        self.interpret_tool_use_output = True

        self._message_stack = MessageStack()

    def process(self,
                input_prompt: str,
                input_structured: Optional[Dict[str, Any]] = None,
                tool_choice_type: str = 'auto',
                tools_choice_name: Optional[str] = None,
                interpret_tool_use_output: bool = True,
                with_memory: bool = True,
                ):
        """Process the input prompt and return the output.
        Args:
            input_prompt: The input prompt to the agent
            input_structured: The structured input data, which enable the caller to the agent to pass structured data
                to be included in the text sent to the LLM model
            tool_choice_type: The type of tool choice to use
            tools_choice_name: The name of the tool to use
            interpret_tool_use_output: Whether to interpret the tool output; note that this leads to recursive calls to
                the LLM model until it returns a message without tool use
            with_memory: Whether to keep the memory of the conversation between process calls
        """
        if not with_memory:
            self._message_stack = MessageStack()

        self.tool_choice = {'type': tool_choice_type}
        if tools_choice_name:
            self.tool_choice['name'] = tools_choice_name
        self.interpret_tool_use_output = interpret_tool_use_output

        if input_structured:
            structured_content_str = json.dumps(input_structured)
            input_prompt += f'\n\n=== Structured input data ===\n{structured_content_str}\n=== End of structured input data ==='

        self._message_stack.append(MessageParam(
            role='user',
            content=[TextBlock(
                text=input_prompt,
                type='text',
            )]
        ))
        length_message_stack = len(self._message_stack)

        if self.what_does_ai_say():
            added_messages = self._message_stack[length_message_stack:]
            if self.interpret_tool_use_output:
                text_out = added_messages.pull_text_by_role('assistant')
            else:
                text_out = added_messages.pull_tool_result()
            return '\n\n'.join([text for text in text_out])

------------------------------------------------------------------------------------------------------
