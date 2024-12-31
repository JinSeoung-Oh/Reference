## From https://generativeai.pub/enhancing-ai-with-graph-rag-models-a-practical-guide-906eb3e8721a

from neo4j import GraphDatabase

# Connect to Neo4j database
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

# Define a function to create nodes and relationships
def create_graph(tx):
    tx.run("CREATE (a:Disease {name: 'Flu'})")
    tx.run("CREATE (b:Symptom {name: 'Fever'})")
    tx.run("CREATE (c:Symptom {name: 'Cough'})")
    tx.run("CREATE (a)-[:HAS_SYMPTOM]->(b)")
    tx.run("CREATE (a)-[:HAS_SYMPTOM]->(c)")

# Add data to the graph
with driver.session() as session:
    session.write_transaction(create_graph)

driver.close()

def fetch_symptoms(tx, disease_name):
    query = """
    MATCH (d:Disease {name: $disease})-[:HAS_SYMPTOM]->(s:Symptom)
    RETURN s.name AS symptom
    """
    result = tx.run(query, disease=disease_name)
    return [record["symptom"] for record in result]

with driver.session() as session:
    symptoms = session.read_transaction(fetch_symptoms, "Flu")
    print("Symptoms of Flu:", symptoms)

import openai

# Combine graph-retrieved data with a prompt for the LLM
retrieved_data = "Symptoms of Flu: Fever, Cough"
prompt = f"Based on the following data, explain the symptoms and treatment options for Flu: {retrieved_data}"

# Generate response
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=100
)

print(response.choices[0].text.strip())


