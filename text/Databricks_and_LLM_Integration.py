### From https://generativeai.pub/building-ai-powered-tools-using-large-language-models-llms-on-databricks-a566e5c8717a

!pip install -U --quiet databricks-sdk==0.28.0 databricks-agents mlflow langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 Faker==27.0.0

#Create the few shot learning examples
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
examples = [
  {
    "input":"""I want to generate a custom dataset to mimic the real online business, the sample dataset looks like below:
    UserID|Username|Email|Country|AccountStatus
    123456789|John Doe|johndoe@gmail.com|USA|Active
    23456789|Jane Smith|janes@hotmail.com|Canada|Active
    3456789|Alice Johnson|alicej@gmail.com|UK|Active
    456789|Bob Brown|bobbrown@outlook.com|UK|Active
    """,
    "output":"""
    from faker import Faker
    import pandas as pd
    import random
    from datetime import datetime, timedelta
    # Initialize Faker
    fake = Faker()
    # Function to generate user data
    def generate_user_data(num_rows=10000):
        user_data = []
        for _ in range(num_rows):
            user = {
                "UserID": fake.uuid4(),
                "Username": fake.user_name(),
                "Email": fake.email(),
                "Country": fake.country(),
                "AccountStatus": random.choice(["Active", "Suspended", "Inactive"])
            }
            user_data.append(user)
        return pd.DataFrame(user_data)
    # Generate the user data
    df = generate_user_data(10000)
    """,
  }

]

# This is a prompt template used to format each individual example.
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
         You are an AI dataset generation tool. Use the following pieces of retrieved context to generate the dataset. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.
        Please output **only** valid, bug-free Python code to generate the dataset based on the input. Ensure that:
        1. The code is properly indented.
        2. Do not include any markdown or formatting characters like backticks (`), triple quotes (```), or explanations.
        3. The code should be plain Python without any comments, explanations, or descriptions.
        4. Ensure the code is formatted to be executable without further modifications, with the correct indentation maintained throughout.
        5. Do not include the save_dataframe function unless the user specifies the locations for catalog, schema and table name.
         """),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

###########################
from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatDatabricks
# Initialize the ChatDatabricks model with endpoint and token settings
dbrx_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens=4096)
# Initialize the chat model with the prompt template
cgt_chain = LLMChain(
    llm=dbrx_model,
    #llm = llma3_model,
    prompt=final_prompt
)

# Main automated function integrating DBRX, few-shot learning, and dataset saving
def ai_data_generation_tool_automated(catalog: str, schema: str, table_name: str, sample_data: str):
    # Step 1: Generate code using DBRX with few-shot learning
    generated_code = generate_code_with_few_shot(sample_data)
    print("Generated PySpark Code:\n", generated_code)
    
    # Step 2: Execute the generated code to create DataFrame
    try:
        df = execute_generated_code(generated_code)
    except Exception as e:
        print(f"Error executing generated code: {e}")
        return
    
    # Step 3: Save the generated DataFrame to Delta table
    save_dataframe(df, catalog, schema, table_name)

###########################
import mlflow.pyfunc
from langchain.chains.llm import LLMChain
from langchain.agents import initialize_agent, Tool
from langchain_community.chat_models import ChatDatabricks
import textwrap
import re
import pandas as pd
import random
from faker import Faker

# Initialize Faker
fake = Faker()

class LangChainAgentModel2(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context):
        # Initialize DBRX model and other components here
        self.dbrx_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens=2000)
        # Initialize the few-shot learning chain with the model and prompt
        self.cgt_chain = LLMChain(llm=self.dbrx_model, prompt=final_prompt)
        self.initialize_agent()

    def initialize_agent(self):
        # Define the tools for the agent to use
        def generate_pyspark_code_with_few_shot(sample_data):
            return self.generate_code_with_few_shot(sample_data)

        # Other tools: execute PySpark code and save DataFrame (reuse previous functions)
        def execute_pyspark_code(generated_code):
            return self.execute_generated_code(generated_code)
        
        def save_data_to_delta(df, catalog, schema, table_name):
            return self.save_dataframe(df, catalog, schema, table_name)

        # Tools to be used by the agent
        self.tools = [
            Tool(name="Generate PySpark Code", func=generate_pyspark_code_with_few_shot, description="Generate PySpark code based on input data."),
            Tool(name="Execute PySpark Code", func=execute_pyspark_code, description="Execute PySpark code to create a DataFrame."),
            Tool(name="Save Data", func=save_data_to_delta, description="Save the DataFrame to Delta Lake.")
        ]
        
        # Initialize the agent
        self.agent = initialize_agent(tools=self.tools, agent_type="zero-shot-react-description", llm=self.dbrx_model)

    # Function to generate code using the LLM model with few-shot learning
    def generate_code_with_few_shot(self, sample_data: str) -> str:
        # Generate the code using the LLM chain
        generated_code = self.cgt_chain.run({"input": sample_data})
        
        # Print the generated code for debugging purposes
        print("Generated Code (Raw):\n", generated_code)
        
        # Ensure the code is properly formatted by removing any extra spaces/tabs
        generated_code = textwrap.dedent(generated_code)
        
        # Apply custom post-processing to fix indentation
        formatted_code = self.fix_indentation(generated_code)
        
        return formatted_code

    # Function to apply custom post-processing and fix indentation issues
    def fix_indentation(self, code: str) -> str:
        # Split the code into lines
        lines = code.splitlines()
        formatted_lines = []
        
        indent_level = 0
        indent_size = 4  # Use 4 spaces per indent

        for line in lines:
            stripped_line = line.strip()
            
            if stripped_line.endswith(":"):
                formatted_lines.append(" " * indent_level + stripped_line)
                indent_level += indent_size
            elif stripped_line == "":
                formatted_lines.append("")
            else:
                if re.match(r"return|break|continue|pass", stripped_line):
                    indent_level = max(0, indent_level - indent_size)
                formatted_lines.append(" " * indent_level + stripped_line)

                if re.match(r"return|break|continue|pass", stripped_line):
                    indent_level = max(0, indent_level - indent_size)
        
        return "\n".join(formatted_lines)

    # Function to execute the generated code and create a PySpark DataFrame
    def execute_generated_code(self, generated_code: str):
        try:
            # Print the generated code for debugging
            print("Executing the following code:\n", generated_code)
            
            # Create a local scope for execution and pass required imports
            local_vars = {
                'Faker': Faker,
                'pd': pd,
                'random': random,
                'fake': fake  # Ensure fake object is passed into scope
            }
            
            # Execute the code and ensure 'df' is created
            exec(generated_code, globals(), local_vars)
            
            if 'df' in local_vars:
                return spark.createDataFrame(local_vars['df'])
            else:
                raise ValueError("The generated code did not create a DataFrame named 'df'.")
        except Exception as e:
            print(f"Error executing generated code: {e}")
            raise

    # Function to save the DataFrame to Delta
    def save_dataframe(self, df, catalog: str, schema: str, table_name: str):
        full_table_name = f"{catalog}.{schema}.{table_name}"
        df.write.mode("overwrite").format("delta").saveAsTable(full_table_name)
        print(f"Table '{full_table_name}' created successfully.")
        return f"Table '{full_table_name}' created successfully."

    def predict(self, context, model_input):
        # Extract inputs from the input dataframe
        sample_data = model_input['sample_data'].iloc[0]  # Get the first value of sample_data
        catalog = model_input['catalog'].iloc[0]
        schema = model_input['schema'].iloc[0]
        table_name = model_input['table_name'].iloc[0]

        # Step 1: Generate PySpark code using few-shot learning
        generated_code = self.generate_code_with_few_shot(sample_data)
        print("Generated PySpark Code:\n", generated_code)
        
        # Step 2: Execute the generated code to create a PySpark DataFrame
        df = self.execute_generated_code(generated_code)
        
        # Step 3: Save the DataFrame to the specified Delta table
        result = self.save_dataframe(df, catalog, schema, table_name)
        
        return result

########################
import pandas as pd
import mlflow.pyfunc
from mlflow.models.signature import infer_signature

# Sample input data for the model
sample_input = pd.DataFrame({
    "sample_data": ["UserID | Username | Email | Country | AccountStatus\n123456789 | John Doe | johndoe@gmail.com | USA | Active"],
    "catalog": ["main"],
    "schema": ["test"],
    "table_name": ["user_table"]
})

# Sample output: A success message or any other output your model returns
sample_output = pd.DataFrame({
    "result": ["Data saved successfully to default.public.generated_table"]
})

# Infer the model signature based on the sample input and output
signature = infer_signature(sample_input, sample_output)

# Log the model with the conda environment
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        artifact_path="langchain_agent_model_2",
        python_model=LangChainAgentModel2(),
        conda_env=conda_env,  # Add the environment
        signature=signature,
        registered_model_name="LangChainAgentModel_2"
    )

