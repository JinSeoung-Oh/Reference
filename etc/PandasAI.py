## From https://generativeai.pub/pandasai-revolutionizing-data-analysis-with-the-power-of-natural-language-442cfd53acc2

##** with pandas
import pandas as pd

df = pd.read_csv("employee_data.csv")
average_salary = df[df["department"] == ["Marketing"]["salary"].mean()
print(average_salary)

##** with pandasai
import os
import pandas as pd
from pandasai import SmartDataframe

df = pd.read_csv("employee_data.csv")

# By default, unless you choose a different LLM, it will use BambooLLM.
# You can get your free API key signing up at https://pandabi.ai (you can also configure it in your .env file)
os.environ["PANDASAI_API_KEY"] = "YOUR_API_KEY"

df_smart = SmartDataframe(df)
df_smart.chat('What is the average salary of employees in the marketing department?')

########################################################################################################

##** with pandas
import pandas as pd

# Load data
sales_data = pd.read_csv("sales_data.csv")

# Analyze sales by product category
sales_by_category = sales_data.groupby("product_category")["sales"].sum()

# Print results
print(sales_by_category)

##** with pandasai
import os
from pandasai import SmartDataframe

# By default, unless you choose a different LLM, it will use BambooLLM.
# You can get your free API key signing up at https://pandabi.ai (you can also configure it in your .env file)
os.environ["PANDASAI_API_KEY"] = "YOUR_API_KEY"

# You can instantiate a SmartDataframe with a path to a CSV file
sdf = SmartDataframe("data/sales_data.csv")

response = sdf.chat("What are the total sales for each product category?")

########################################################################################################
## SQL Data Analysis: A Seamless Experience
SELECT product_name, SUM(quantity_sold) AS total_sold
FROM sales
JOIN products ON sales.product_id = products.product_id
GROUP BY product_name
ORDER BY total_sold DESC;

from pandasai import SmartDataframe
from pandasai.connectors import MySQLConnector

mysql_connector = MySQLConnector(
    config={
        "host": "localhost",
        "port": 3306,
        "database": "mydb",
        "username": "root",
        "password": "root",
        "table": "loans",
        "where": [
            # this is optional and filters the data to
            # reduce the size of the dataframe
            ["loan_status", "=", "PAIDOFF"],
        ],
    }
)

df = SmartDataframe(mysql_connector)
df.chat('What is the total amount of loans in the last year?')

########################################################################################################
## Example 1: BambooLLM for Basic Analysis
from pandasai import SmartDataframe
from pandasai.llm import BambooLLM

llm = BambooLLM(api_key="my-bamboo-api-key")
df = SmartDataframe("data.csv", config={"llm": llm})

response = df.chat("Calculate the sum of the gdp of north american countries")
print(response)

## Example 2: OPEN Ai models
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

llm = OpenAI(api_token="my-openai-api-key")
pandas_ai = SmartDataframe("data.csv", config={"llm": llm})

## Example 3: Locally hosted Ollama models
from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM

ollama_llm = LocalLLM(api_base="http://localhost:11434/v1", model="codellama")
df = SmartDataframe("data.csv", config={"llm": ollama_llm})

########################################################################################################
### SmartDatalake: Working with Multiple DataFrames ###
import os
import pandas as pd
from pandasai import SmartDatalake

employees_data = {
    'EmployeeID': [1, 2, 3, 4, 5],
    'Name': ['John', 'Emma', 'Liam', 'Olivia', 'William'],
    'Department': ['HR', 'Sales', 'IT', 'Marketing', 'Finance']
}

salaries_data = {
    'EmployeeID': [1, 2, 3, 4, 5],
    'Salary': [5000, 6000, 4500, 7000, 5500]
}

employees_df = pd.DataFrame(employees_data)
salaries_df = pd.DataFrame(salaries_data)

# By default, unless you choose a different LLM, it will use BambooLLM.
# You can get your free API key signing up at https://pandabi.ai (you can also configure it in your .env file)
os.environ["PANDASAI_API_KEY"] = "YOUR_API_KEY"

lake = SmartDatalake([employees_df, salaries_df])
lake.chat("Who gets paid the most?")

### Skills: Tailoring PandasAI to Your Needs ###
import os
import pandas as pd
from pandasai import Agent
from pandasai.skills import skill

employees_data = {
    "EmployeeID": [1, 2, 3, 4, 5],
    "Name": ["John", "Emma", "Liam", "Olivia", "William"],
    "Department": ["HR", "Sales", "IT", "Marketing", "Finance"],
}

salaries_data = {
    "EmployeeID": [1, 2, 3, 4, 5],
    "Salary": [5000, 6000, 4500, 7000, 5500],
}

employees_df = pd.DataFrame(employees_data)
salaries_df = pd.DataFrame(salaries_data)

# Function doc string to give more context to the model for use this skill
@skill
def plot_salaries(names: list[str], salaries: list[int]):
    """
    Displays the bar chart  having name on x-axis and salaries on y-axis
    Args:
        names (list[str]): Employees' names
        salaries (list[int]): Salaries
    """
    # plot bars
    import matplotlib.pyplot as plt

    plt.bar(names, salaries)
    plt.xlabel("Employee Name")
    plt.ylabel("Salary")
    plt.title("Employee Salaries")
    plt.xticks(rotation=45)

# By default, unless you choose a different LLM, it will use BambooLLM.
# You can get your free API key signing up at https://pandabi.ai (you can also configure it in your .env file)
os.environ["PANDASAI_API_KEY"] = "YOUR_API_KEY"

agent = Agent([employees_df, salaries_df], memory_size=10)
agent.add_skills(plot_salaries)

# Chat with the agent
response = agent.chat("Plot the employee salaries against names")

### Agents driven Conversations: Interactive Analysis ### 
import os
from pandasai import Agent
import pandas as pd

# Sample DataFrames
sales_by_country = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "sales": [5000, 3200, 2900, 4100, 2300, 2100, 2500, 2600, 4500, 7000],
    "deals_opened": [142, 80, 70, 90, 60, 50, 40, 30, 110, 120],
    "deals_closed": [120, 70, 60, 80, 50, 40, 30, 20, 100, 110]
})


# By default, unless you choose a different LLM, it will use BambooLLM.
# You can get your free API key signing up at https://pandabi.ai (you can also configure it in your .env file)
os.environ["PANDASAI_API_KEY"] = "YOUR_API_KEY"

agent = Agent(sales_by_country)
agent.chat('Which are the top 5 countries by sales?')

### Data Visualization with PandasAI ### 
import os
from pandasai import SmartDataframe

# By default, unless you choose a different LLM, it will use BambooLLM.
# You can get your free API key signing up at https://pandabi.ai (you can also configure it in your .env file)
os.environ["PANDASAI_API_KEY"] = "YOUR_API_KEY"

sdf = SmartDataframe("data/Countries.csv")
response = sdf.chat(
    "Plot the histogram of countries showing for each the gpd, using different colors for each bar",
)
print(response)














