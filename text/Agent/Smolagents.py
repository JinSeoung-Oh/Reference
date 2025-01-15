### From https://nayakpplaban.medium.com/build-an-agent-to-identify-the-most-predictive-set-of-features-in-a-linear-model-using-smolagents-d0d44ac7e721

"""
1. What Are Agents? 
   -a. Definition:
       Agents are systems enabling AI, particularly large language models (LLMs), to interact with the real world.  
       They can access external information (e.g., via search) or execute tasks (e.g., through specific programs). 
       Essentially, agents give LLMs the ability to act independently—known as having agency—where the model's outputs directly dictate 
       task execution in a workflow.
    -b. When to Use Agents:
        Use agents when you need workflow flexibility or need to handle complex, unpredictable requests.
    -c. When to Avoid Agents:
        Avoid agents when workflows are simple, predetermined, or can be categorized into fixed options, 
        as these do not require the dynamic reasoning capabilities of agents.

    Example Scenario: Surfing Trip App
    -1. For simple, predefined queries (e.g., trip information or sales inquiries), traditional fixed workflows suffice.
    -2. For complex queries that require multi-step reasoning and integration of various data sources 
        (e.g., "Can I surf on Tuesday morning if I arrive Monday but forgot my passport?"), an agentic system is more suitable.

3. When to Use Smolagents
   Smolagents is a lightweight framework by Hugging Face designed for creating robust AI agents with minimal code.

   -a. Key Reasons to Choose Smolagents:
       -1. Simplicity:
           Minimalistic codebase simplifies development.
       -2. Code Agents for Enhanced Performance:
           Agents can write and execute Python code snippets directly, improving efficiency and accuracy.
       -3. Seamless Integration:
           Easy integration with various LLMs and access to tools and shared resources via the Hugging Face Hub.
       -4. Agent Types Supported:
           -1) CodeAgent:
               Capable of writing and executing Python code for dynamic task execution.
           -2) ToolCallingAgent:
               Generates actions as JSON/text without executing code, suitable for simpler tasks.
   -b. Additional Features:
       -1. Tools:
           -1) Access to various built-in tools (e.g., DuckDuckGo Search Tool, Python Code Interpreter).
           -2) Support for creating and integrating custom tools.
       -2. Models:
           -1) Smolagents supports multiple LLMs from providers like Hugging Face, OpenAI, Anthropic.
       -3. Sandboxed Execution:
           -1) Executes code in a secure environment to mitigate vulnerabilities.
       -4. Memory Management:
           -1) Maintains memory across interactions, crucial for multi-step tasks requiring context.
       -5. Hub Integration:
           -1) Enables sharing and loading tools from the Hugging Face Hub, promoting community collaboration.
       -6. Multi-Agent Systems:
           -1) Supports coordination of multiple agents working together on complex tasks.

4. Building an Agent with Smolagents
   Requirements to Build an Agent:
   -a. Tools:
       -1) A list of tools the agent can access (e.g., to query data, control smart devices, etc.).
   -b. Model:
       -1) An LLM that serves as the engine of the agent. Options include:
           - HfApiModel: Uses Hugging Face’s free inference API.
           - LiteLLMModel: Leverages litellm to select from a variety of cloud LLMs.
   Example Use Case Mentioned:
   -1) The text suggests implementing a code agent that selects the most predictive set of features in a linear model.
   -2) It also mentions comparing results using different LLMs such as OpenAI, Ollama/llama3.2, and Ollama/deepseek-v2.

5. Summary of Key Points
   -a. Agents provide LLMs with agency, enabling them to perform complex, real-world tasks by interacting with external tools and executing actions.
   -b. Smolagents is highlighted as a framework for creating such agents with minimal code, offering simplicity, security, flexibility, 
       and integration with multiple LLMs and tools.
   -c. Agent Types:
       -1) CodeAgent: Executes Python code for dynamic task execution.
       -2) ToolCallingAgent: Generates actions in JSON/text form for simpler tasks.
   -d. Framework Components:
       -1) Tools, models, sandboxed execution, memory management, Hub integration, and support for multi-agent systems.
   -e. Building an Agent:
       -1) Requires defining a set of tools and selecting an appropriate LLM model.
       -2) The article hints at an upcoming implementation example focusing on selecting predictive features using different LLMs.
"""
from google.colab import userdata
import os
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')

from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
import os
deepseek_model = LiteLLMModel(model_id="ollama/deepseek-v2:16b")
codellama_model = LiteLLMModel(model_id="ollama/codellama")
llama_model = LiteLLMModel(model_id="ollama/llama3.2")
openai_model = LiteLLMModel(model_id="openai/gpt-4o-mini")

#

# Task for the agent
task = """
1. Load the Diabetes dataset from the 'sklearn' library using the following code:
        from sklearn.datasets import load_diabetes
        import pandas as pd

        # Load the dataset
        data, target = load_diabetes(return_X_y=True, as_frame=False)

        # Create a DataFrame
        df = pd.DataFrame(data, columns=load_diabetes().feature_names)
        df['target'] = target
2. Split data with a train/test split of 75%/25%
3. Create a linear regression model on the training data predicting the target variable using the "sklearn" or "statsmodels" library.
4. Execute on a strategy of combination of up to 3 predictors that attains the lowest root mean square error (RMSE) on the testing data. 
   (You can't use the target variable).
5. Use feature engineering as needed to improve model performance.
6. Based on the lowest RMSE of each model for the testing data, provide a final list of predictors for the top 5 models
7. Only Output The predictors as a table in Markdown format.Do not provide any other Reasoning or explanation.
"""

# Define the Feature Selection Agent with deepseek_model
feature_selection_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool], # search internet if necessary
    additional_authorized_imports=['pandas','statsmodels','sklearn','numpy','json'], # packages for code interpreter
    model=deepseek_model # model set above
)
result = feature_selection_agent.run(task)

# Define the Feature Selection Agent with codellama_model
feature_selection_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool], # search internet if necessary
    additional_authorized_imports=['pandas','statsmodels','sklearn','numpy','json'], # packages for code interpreter
    model=codellama_model # model set above
)
result = feature_selection_agent.run(task)

# Define the Feature Selection Agent with llama_model
feature_selection_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool], # search internet if necessary
    additional_authorized_imports=['pandas','statsmodels','sklearn','numpy','json'], # packages for code interpreter
    model=llama_model # model set above
)
result = feature_selection_agent.run(task)

# print result
print(result.content)

# print System Prompt
print(feature_selection_agent.system_prompt_template)

