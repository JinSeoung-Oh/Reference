### From https://medium.com/the-ai-forum/building-a-self-healing-sql-query-generator-agent-with-pydantic-ai-and-groq-7045910265c0

%pip install 'pydantic-ai-slim[openai,groq,logfire]'
%pip install aiosqlite

from google.colab import userdata
import os
os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.groq import GroqModel
import aiosqlite
import asyncio
from typing import Union, TypeAlias, Annotated,Optional,Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field
from annotated_types import MinLen
from pydantic_ai import Agent, ModelRetry, RunContext
from typing import Tuple, Optional

import nest_asyncio
nest_asyncio.apply()

openai_model = OpenAIModel('gpt-4o-mini')
groq_model = GroqModel("llama3-groq-70b-8192-tool-use-preview")

# Models for type safety
class Success(BaseModel):
    type: str = Field("Success", pattern="^Success$")
    sql_query: Annotated[str, MinLen(1)]
    explanation: str

class InvalidRequest(BaseModel):
    type: str = Field("InvalidRequest", pattern="^InvalidRequest$")
    error_message: str

@dataclass
class Deps:
    conn: aiosqlite.Connection
    db_schema: str = DB_SCHEMA

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    title TEXT NOT NULL,
    content TEXT,
    published BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""

sqlagent = Agent(
    openai_model,
    deps_type=Deps,
    retries=3,
    result_type=Response,
    system_prompt=("""You are a proficient Database Administrator  having expertise in generating SQL queries. Your task is to convert natural language requests into SQL queries for a SQLite database.
You must respond with a Success object containing a sql_query and an explanation.

Database schema:
{DB_SCHEMA}

Format your response exactly like this, with no additional text or formatting:
{{
    "type": "Success",
    "sql_query": "<your SQL query here>",
    "explanation": "<your explanation here>"
}}

Examples:
    User: show me all users who have published posts
    {{
        "type": "Success",
        "sql_query": "SELECT DISTINCT users.* FROM users JOIN posts ON users.id = posts.user_id WHERE posts.published = TRUE",
        "explanation": "This query finds all users who have at least one published post by joining the users and posts tables."
    }}

    User: count posts by user
    {{
        "type": "Success",
        "sql_query": "SELECT users.name, COUNT(posts.id) as post_count FROM users LEFT JOIN posts ON users.id = posts.user_id GROUP BY users.id, users.name",
        "explanation": "This query counts the number of posts for each user, including users with no posts using LEFT JOIN."
    }}

    If you receive an error message about a previous query, analyze the error and fix the issues in your new query.
    Common fixes include:
    - Correcting column names
    - Fixing JOIN conditions
    - Adjusting GROUP BY clauses
    - Handling NULL values properly

If you cannot generate a valid query, respond with:
{{
    "type": "InvalidRequest",
    "error_message": "<explanation of why the request cannot be processed>"
}}

Important:
1. Respond with ONLY the JSON object, no additional text
2. Always include the "type" field as either "Success" or "InvalidRequest"
3. All queries must be SELECT statements
4. Provide clear explanations
5. Use proper JOIN conditions and WHERE clauses as needed
""")
)

async def init_database(db_path: str = "test.db") -> aiosqlite.Connection:
    """Initialize the database with schema"""
    conn = await aiosqlite.connect(db_path)
    
    # Enable foreign keys
    await conn.execute("PRAGMA foreign_keys = ON")
    
    # Create schema
    await conn.executescript(DB_SCHEMA)
    
    # Add some sample data if the tables are empty
    async with conn.execute("SELECT COUNT(*) FROM users") as cursor:
        count = await cursor.fetchone()
        if count[0] == 0:
            sample_data = """
            INSERT INTO users (name, email) VALUES 
                ('John Doe', 'john@example.com'),
                ('Jane Smith', 'jane@example.com');
                
            INSERT INTO posts (user_id, title, content, published) VALUES 
                (1, 'First Post', 'Hello World', TRUE),
                (1, 'Draft Post', 'Work in Progress', FALSE),
                (2, 'Jane''s Post', 'Hello from Jane', TRUE);
            """
            await conn.executescript(sample_data)
            await conn.commit()
    
    return conn
  
async def execute_query(conn: aiosqlite.Connection, query: str) -> Tuple[bool, Optional[str]]:
    """
    Execute a SQL query and return success status and error message if any.
    Returns: (success: bool, error_message: Optional[str])
    """
    try:
        async with conn.execute(query) as cursor:
            await cursor.fetchone()
        return True, None
    except Exception as e:
        return False, str(e)

async def query_database(prompt: str, conn: aiosqlite.Connection) -> Response:
    max_retries = 3
    last_error: Optional[str] = None
    
    for attempt in range(max_retries):
        try:
            result = await agent.run(prompt, deps=deps)
            success, error = await execute_query(conn, result.sql_query)
            if success:
                return result
            
            last_error = error
            prompt = f"""
Previous query failed with error: {error}
Please generate a corrected SQL query for the original request: {prompt}
"""
        except Exception as e:
            last_error = str(e)
            continue


async def main():
    # Ensure GROQ API key is set
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Please set GROQ_API_KEY environment variable")

    # Initialize database
    conn = await init_database("test.db")
    
    try:
        # Example queries to test
        test_queries = [
            "show me all users and the number of posts posted",
            "find users who have published posts",
            "show me all draft posts with their authors",
            "what is the count of users table",
            "show me the title of the posts published",
            "show me the structure of the posts",
            "show me the names of all the users"
        ]

        for query in test_queries:
            print(f"\nProcessing query: {query}")
            result = await query_database(query, conn)
            print(f"\nProcessing query result: {result}")
            if isinstance(result, InvalidRequest):
                print(f"Error: {result.error_message}")
            else:

                print("\n✅ Generated SQL:")
                print(result.data.sql_query)
                print("\n✅ Explanation:")
                print(result.data.explanation)
                print("\n✅ Cost:")
                print(result._usage)
                
                # Execute the query to show results
                try:
                    async with conn.execute(result.data.sql_query) as cursor:
                        rows = await cursor.fetchall()
                        print("\n📊 Results:")
                        for row in rows:
                            print(row)
                except Exception as e:
                    print(f"Error executing query: {query}")
                    continue
                          
            
            print("\n" + "="*50)

    finally:
        await conn.close()
