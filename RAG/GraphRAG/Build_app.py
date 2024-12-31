### From https://towardsdatascience.com/how-to-build-a-graph-rag-app-b323fc33ba06

import weaviate
from weaviate.util import generate_uuid5
from weaviate.classes.init import Auth
import os
import json
import pandas as pd
import numpy as np
import urllib.parse
from rdflib import Graph, RDF, RDFS, Namespace, URIRef, Literal
from weaviate.classes.config import Configure

client = weaviate.connect_to_weaviate_cloud(
    cluster_url="XXX",  # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key("XXX"),  # Replace with your Weaviate Cloud key
    headers={'X-OpenAI-Api-key': "XXX"}  # Replace with your OpenAI API key
)

df = spark.sql("SELECT * FROM workspace.default.pub_med_multi_label_text_classification_dataset_processed").toPandas()
df = pd.read_csv("PubMed Multi Label Text Classification Dataset Processed.csv")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna('', inplace=True)

# Convert columns to string type
df['Title'] = df['Title'].astype(str)
df['abstractText'] = df['abstractText'].astype(str)
df['meshMajor'] = df['meshMajor'].astype(str)

# Function to create a valid URI
def create_valid_uri(base_uri, text):
    if pd.isna(text):
        return None
    # Encode text to be used in URI
    sanitized_text = urllib.parse.quote(text.strip().replace(' ', '_').replace('"', '').replace('<', '').replace('>', '').replace("'", "_"))
    return URIRef(f"{base_uri}/{sanitized_text}")


# Function to create a valid URI for Articles
def create_article_uri(title, base_namespace="http://example.org/article/"):
    """
    Creates a URI for an article by replacing non-word characters with underscores and URL-encoding.

    Args:
        title (str): The title of the article.
        base_namespace (str): The base namespace for the article URI.

    Returns:
        URIRef: The formatted article URI.
    """
    if pd.isna(title):
        return None
    # Replace non-word characters with underscores
    sanitized_title = re.sub(r'\W+', '_', title.strip())
    # Condense multiple underscores into a single underscore
    sanitized_title = re.sub(r'_+', '_', sanitized_title)
    # URL-encode the term
    encoded_title = quote(sanitized_title)
    # Concatenate with base_namespace without adding underscores
    uri = f"{base_namespace}{encoded_title}"
    return URIRef(uri)

# Add a new column to the DataFrame for the article URIs
df['Article_URI'] = df['Title'].apply(lambda title: create_valid_uri("http://example.org/article", title))

# Function to clean and parse MeSH terms
def parse_mesh_terms(mesh_list):
    if pd.isna(mesh_list):
        return []
    return [
        term.strip().replace(' ', '_')
        for term in mesh_list.strip("[]'").split(',')
    ]

# Function to create a valid URI for MeSH terms
def create_valid_uri(base_uri, text):
    if pd.isna(text):
        return None
    sanitized_text = urllib.parse.quote(
        text.strip()
        .replace(' ', '_')
        .replace('"', '')
        .replace('<', '')
        .replace('>', '')
        .replace("'", "_")
    )
    return f"{base_uri}/{sanitized_text}"

# Extract and process all MeSH terms
all_mesh_terms = []
for mesh_list in df["meshMajor"]:
    all_mesh_terms.extend(parse_mesh_terms(mesh_list))

# Deduplicate terms
unique_mesh_terms = list(set(all_mesh_terms))

# Create a DataFrame of MeSH terms and their URIs
mesh_df = pd.DataFrame({
    "meshTerm": unique_mesh_terms,
    "URI": [create_valid_uri("http://example.org/mesh", term) for term in unique_mesh_terms]
})

# Display the DataFrame
print(mesh_df)


#define the collection
articles = client.collections.create(
    name = "Article",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(),  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
    generative_config=Configure.Generative.openai(),  # Ensure the `generative-openai` module is used for generative queries
)

#add ojects
articles = client.collections.get("Article")

with articles.batch.dynamic() as batch:
    for index, row in df.iterrows():
        batch.add_object({
            "title": row["Title"],
            "abstractText": row["abstractText"],
            "Article_URI": row["Article_URI"],
            "meshMajor": row["meshMajor"],
        })

#define the collection
terms = client.collections.create(
    name = "term",
    vectorizer_config=Configure.Vectorizer.text2vec_openai(),  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
    generative_config=Configure.Generative.openai(),  # Ensure the `generative-openai` module is used for generative queries
)

#add ojects
terms = client.collections.get("term")

with terms.batch.dynamic() as batch:
    for index, row in mesh_df.iterrows():
        batch.add_object({
            "meshTerm": row["meshTerm"],
            "URI": row["URI"],
        })

---------------------------------------------------------------------------------
from rdflib import Graph, RDF, RDFS, Namespace, URIRef, Literal
from rdflib.namespace import SKOS, XSD
import pandas as pd
import urllib.parse
import random
from datetime import datetime, timedelta
import re
from urllib.parse import quote

# --- Initialization ---
g = Graph()

# Define namespaces
schema = Namespace('http://schema.org/')
ex = Namespace('http://example.org/')
prefixes = {
    'schema': schema,
    'ex': ex,
    'skos': SKOS,
    'xsd': XSD
}
for p, ns in prefixes.items():
    g.bind(p, ns)

# Define classes and properties
Article = URIRef(ex.Article)
MeSHTerm = URIRef(ex.MeSHTerm)
g.add((Article, RDF.type, RDFS.Class))
g.add((MeSHTerm, RDF.type, RDFS.Class))

title = URIRef(schema.name)
abstract = URIRef(schema.description)
date_published = URIRef(schema.datePublished)
access = URIRef(ex.access)

g.add((title, RDF.type, RDF.Property))
g.add((abstract, RDF.type, RDF.Property))
g.add((date_published, RDF.type, RDF.Property))
g.add((access, RDF.type, RDF.Property))

# Function to clean and parse MeSH terms
def parse_mesh_terms(mesh_list):
    if pd.isna(mesh_list):
        return []
    return [term.strip() for term in mesh_list.strip("[]'").split(',')]

# Enhanced convert_to_uri function
def convert_to_uri(term, base_namespace="http://example.org/mesh/"):
    """
    Converts a MeSH term into a standardized URI by replacing spaces and special characters with underscores,
    ensuring it starts and ends with a single underscore, and URL-encoding the term.

    Args:
        term (str): The MeSH term to convert.
        base_namespace (str): The base namespace for the URI.

    Returns:
        URIRef: The formatted URI.
    """
    if pd.isna(term):
        return None  # Handle NaN or None terms gracefully
    
    # Step 1: Strip existing leading and trailing non-word characters (including underscores)
    stripped_term = re.sub(r'^\W+|\W+$', '', term)
    
    # Step 2: Replace non-word characters with underscores (one or more)
    formatted_term = re.sub(r'\W+', '_', stripped_term)
    
    # Step 3: Replace multiple consecutive underscores with a single underscore
    formatted_term = re.sub(r'_+', '_', formatted_term)
    
    # Step 4: URL-encode the term to handle any remaining special characters
    encoded_term = quote(formatted_term)
    
    # Step 5: Add single leading and trailing underscores
    term_with_underscores = f"_{encoded_term}_"
    
    # Step 6: Concatenate with base_namespace without adding an extra underscore
    uri = f"{base_namespace}{term_with_underscores}"

    return URIRef(uri)

# Function to generate a random date within the last 5 years
def generate_random_date():
    start_date = datetime.now() - timedelta(days=5*365)
    random_days = random.randint(0, 5*365)
    return start_date + timedelta(days=random_days)

# Function to generate a random access value between 1 and 10
def generate_random_access():
    return random.randint(1, 10)

# Function to create a valid URI for Articles
def create_article_uri(title, base_namespace="http://example.org/article"):
    """
    Creates a URI for an article by replacing non-word characters with underscores and URL-encoding.

    Args:
        title (str): The title of the article.
        base_namespace (str): The base namespace for the article URI.

    Returns:
        URIRef: The formatted article URI.
    """
    if pd.isna(title):
        return None
    # Encode text to be used in URI
    sanitized_text = urllib.parse.quote(title.strip().replace(' ', '_').replace('"', '').replace('<', '').replace('>', '').replace("'", "_"))
    return URIRef(f"{base_namespace}/{sanitized_text}")

# Loop through each row in the DataFrame and create RDF triples
for index, row in df.iterrows():
    article_uri = create_article_uri(row['Title'])
    if article_uri is None:
        continue
    
    # Add Article instance
    g.add((article_uri, RDF.type, Article))
    g.add((article_uri, title, Literal(row['Title'], datatype=XSD.string)))
    g.add((article_uri, abstract, Literal(row['abstractText'], datatype=XSD.string)))
    
    # Add random datePublished and access
    random_date = generate_random_date()
    random_access = generate_random_access()
    g.add((article_uri, date_published, Literal(random_date.date(), datatype=XSD.date)))
    g.add((article_uri, access, Literal(random_access, datatype=XSD.integer)))
    
    # Add MeSH Terms
    mesh_terms = parse_mesh_terms(row['meshMajor'])
    for term in mesh_terms:
        term_uri = convert_to_uri(term, base_namespace="http://example.org/mesh/")
        if term_uri is None:
            continue
        
        # Add MeSH Term instance
        g.add((term_uri, RDF.type, MeSHTerm))
        g.add((term_uri, RDFS.label, Literal(term.replace('_', ' '), datatype=XSD.string)))
        
        # Link Article to MeSH Term
        g.add((article_uri, schema.about, term_uri))

# Path to save the file
file_path = "/Workspace/PubMedGraph.ttl"

# Save the file
g.serialize(destination=file_path, format='turtle')

print(f"File saved at {file_path}")

-----------------------------------------------------------------------------------------
# --- TAB 1: Search Articles ---
with tab_search:
    st.header("Search Articles (Vector Query)")
    query_text = st.text_input("Enter your vector search term (e.g., Mouth Neoplasms):", key="vector_search")

    if st.button("Search Articles", key="search_articles_btn"):
        try:
            client = initialize_weaviate_client()
            article_results = query_weaviate_articles(client, query_text)

            # Extract URIs here
            article_uris = [
                result["properties"].get("article_URI")
                for result in article_results
                if result["properties"].get("article_URI")
            ]

            # Store article_uris in the session state
            st.session_state.article_uris = article_uris

            st.session_state.article_results = [
                {
                    "Title": result["properties"].get("title", "N/A"),
                    "Abstract": (result["properties"].get("abstractText", "N/A")[:100] + "..."),
                    "Distance": result["distance"],
                    "MeSH Terms": ", ".join(
                        ast.literal_eval(result["properties"].get("meshMajor", "[]"))
                        if result["properties"].get("meshMajor") else []
                    ),

                }
                for result in article_results
            ]
            client.close()
        except Exception as e:
            st.error(f"Error during article search: {e}")

    if st.session_state.article_results:
        st.write("**Search Results for Articles:**")
        st.table(st.session_state.article_results)
    else:
        st.write("No articles found yet.")

# Function to query Weaviate for Articles
def query_weaviate_articles(client, query_text, limit=10):
    # Perform vector search on Article collection
    response = client.collections.get("Article").query.near_text(
        query=query_text,
        limit=limit,
        return_metadata=MetadataQuery(distance=True)
    )

    # Parse response
    results = []
    for obj in response.objects:
        results.append({
            "uuid": obj.uuid,
            "properties": obj.properties,
            "distance": obj.metadata.distance,
        })
    return results

# Fetch alternative names and triples for a MeSH term
def get_concept_triples_for_term(term):
    term = sanitize_term(term)  # Sanitize input term
    sparql = SPARQLWrapper("https://id.nlm.nih.gov/mesh/sparql")
    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
    PREFIX mesh: <http://id.nlm.nih.gov/mesh/>

    SELECT ?subject ?p ?pLabel ?o ?oLabel
    FROM <http://id.nlm.nih.gov/mesh>
    WHERE {{
        ?subject rdfs:label "{term}"@en .
        ?subject ?p ?o .
        FILTER(CONTAINS(STR(?p), "concept"))
        OPTIONAL {{ ?p rdfs:label ?pLabel . }}
        OPTIONAL {{ ?o rdfs:label ?oLabel . }}
    }}
    """
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        triples = set()
        for result in results["results"]["bindings"]:
            obj_label = result.get("oLabel", {}).get("value", "No label")
            triples.add(sanitize_term(obj_label))  # Sanitize term before adding

        # Add the sanitized term itself to ensure it's included
        triples.add(sanitize_term(term))
        return list(triples)

    except Exception as e:
        print(f"Error fetching concept triples for term '{term}': {e}")
        return []

