## From https://medium.com/intel-tech/tabular-data-rag-llms-improve-results-through-data-table-prompting-bcb42678914b

import os 
import sys
import pandas as pd
from typing import List

!pip install beautifultable
from beautifultable import BeautifulTable

!pip install opencv-python
!pip install camelot-py
!pip install ghostscript

import camelot

file_path="./World_Billionaires_Wikipedia.pdf"

# use camelot to parse tables   
def get_tables(path: str, pages: List[int]):    
    for page in pages:
        table_list = camelot.read_pdf(path, pages=str(page))
        if table_list.n>0:
            for tab in range(table_list.n):
                
                # Conversion of the the tables into the dataframes.
                table_df = table_list[tab].df 
                
                table_df = (
                    table_df.rename(columns=table_df.iloc[0])
                    .drop(table_df.index[0])
                    .reset_index(drop=True)
                )        
                     
                table_df = table_df.apply(lambda x: x.str.replace('\n',''))
                
                # Change column names to be valid as XML tags
                table_df.columns = [col.replace('\n', ' ').replace(' ', '') for col in table_df.columns]
                table_df.columns = [col.replace('(', '').replace(')', '') for col in table_df.columns]
    
    return table_df
# extract data table from page number
df = get_tables(file_path, pages=[3])

# prepare test set
eval_df = pd.DataFrame(columns=["Data Format", "Data raw"]) # , "Question", "Answer"

# Save the data in JSON format
data_json = df.to_json(orient='records')
eval_df.loc[len(eval_df)] = ["JSON", data_json]

# Save the data as a list of dictionaries
data_list_dict = df.to_dict(orient='records')
eval_df.loc[len(eval_df)] = ["DICT", data_list_dict]

# Save the data in CSV format
csv_data = df.to_csv(index=False)
eval_df.loc[len(eval_df)] = ["CSV", csv_data]

# Save the data in tab-separated format
tsv_data = df.to_csv(index=False, sep='\t')
eval_df.loc[len(eval_df)] = ["TSV (tab-separated)", tsv_data]

# Save the data in HTML format
html_data = df.to_html(index=False)
eval_df.loc[len(eval_df)] = ["HTML", html_data]

# Save the data in LaTeX format
latex_data = df.to_latex(index=False)
eval_df.loc[len(eval_df)] = ["LaTeX", latex_data]

# Save the data in Markdown format
markdown_data = df.to_markdown(index=False)
eval_df.loc[len(eval_df)] = ["Markdown", markdown_data]

# Save the data as a string
string_data = df.to_string(index=False)
eval_df.loc[len(eval_df)] = ["STRING", string_data]

# Save the data as a NumPy array
numpy_data = df.to_numpy()
eval_df.loc[len(eval_df)] = ["NumPy", numpy_data]

# Save the data in XML format
xml_data = df.to_xml(index=False)
eval_df.loc[len(eval_df)] = ["XML", xml_data]

from pandas import option_context
with option_context('display.max_colwidth', 150):
    display(eval_df.head(10))

###### Set our model for validation
MESSAGE_SYSTEM_CONTENT = """You are a customer service agent that helps a customer with answering questions. 
Please answer the question based on the provided context below. 
Make sure not to make any changes to the context, if possible, when preparing answers to provide accurate responses. 
If the answer cannot be found in context, just politely say that you do not know, do not try to make up an answer."""

from openai import AzureOpenAI

client = AzureOpenAI(
    api_key=OAI_API_Key, 
    api_version=OAI_API_Version, 
    azure_endpoint=OAI_API_Base)

def response_test(question:str, context:str, model:str = "gpt-4"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": MESSAGE_SYSTEM_CONTENT,
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": context},
        ],
    )
    
    return response.choices[0].message.content

def run_question_test(query: str, eval_df:str):

    questions = []
    answers = []

    for index, row in eval_df.iterrows():
        questions.append(query)
        response = response_test(query, str(row['Data raw']))
        answers.append(response)
        
    eval_df['Question'] = questions
    eval_df['Answer'] = answers
    
    return eval_df

def BeautifulTableformat(query:str, results:pd.DataFrame, MaxWidth:int = 250):
    table = BeautifulTable(maxwidth=MaxWidth, default_alignment=BeautifulTable.ALIGN_LEFT)
    table.columns.header = ["Data Format", "Query", "Answer"]
    for index, row in results.iterrows():
        table.rows.append([row['Data Format'], query, row['Answer']])
    
    return table


