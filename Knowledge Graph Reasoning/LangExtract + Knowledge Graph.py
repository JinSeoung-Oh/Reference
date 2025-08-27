### From https://medium.com/data-science-collective/langextract-knowledge-graph-googles-new-library-for-nlp-tasks-859e94324718

import os
import textwrap
import langextract as lx
import logging
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph
from typing import List, Dict, Any, Optional
import json

def document_extractor_tool(unstructured_text: str, user_query: str) -> dict:
    """
    Extracts structured information from a given unstructured text based on a user's query.
    """
    prompt = textwrap.dedent(f"""
    You are an expert at extracting specific information from documents.
    Based on the user's query, extract the relevant information from the provided text.
    The user's query is: '{user_query}'
    Provide the output in a structured JSON format.
    """)

    # Dynamic Few-Shot Example Selection
    examples = []
    query_lower = user_query.lower()
    if any(keyword in query_lower for keyword in ["financial", "revenue", "company", "fiscal"]):
        financial_example = lx.data.ExampleData(
            text="In Q1 2023, Innovate Inc. reported a revenue of $15 million.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="company_name",
                    extraction_text="Innovate Inc.",
                    attributes={"name": "Innovate Inc."},
                ),
                lx.data.Extraction(
                    extraction_class="revenue",
                    extraction_text="$15 million",
                    attributes={"value": 15000000, "currency": "USD"},
                ),
                lx.data.Extraction(
                    extraction_class="fiscal_period",
                    extraction_text="Q1 2023",
                    attributes={"period": "Q1 2023"},
                ),
            ]
        )
        examples.append(financial_example)
    elif any(keyword in query_lower for keyword in ["legal", "agreement", "parties", "effective date"]):
        legal_example = lx.data.ExampleData(
            text="This agreement is between John Doe and Jane Smith, effective 2024-01-01.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="party",
                    extraction_text="John Doe",
                    attributes={"name": "John Doe"},
                ),
                lx.data.Extraction(
                    extraction_class="party",
                    extraction_text="Jane Smith",
                    attributes={"name": "Jane Smith"},
                ),
                lx.data.Extraction(
                    extraction_class="effective_date",
                    extraction_text="2024-01-01",
                    attributes={"date": "2024-01-01"},
                ),
            ]
        )
        examples.append(legal_example)
    elif any(keyword in query_lower for keyword in ["social", "post", "feedback", "restaurant", "菜式", "評價"]):
        social_media_example = lx.data.ExampleData(
            text="I tried the new 'Taste Lover' restaurant in TST today. The black truffle risotto was amazing, but the Tiramisu was just average.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="restaurant_name",
                    extraction_text="Taste Lover",
                    attributes={"name": "Taste Lover"},
                ),
                lx.data.Extraction(
                    extraction_class="dish",
                    extraction_text="black truffle risotto",
                    attributes={"name": "black truffle risotto", "sentiment": "positive"},
                ),
                lx.data.Extraction(
                    extraction_class="dish",
                    extraction_text="Tiramisu",
                    attributes={"name": "Tiramisu", "sentiment": "neutral"},
                ),
            ]
        )
        examples.append(social_media_example)
    else:
        # Default generic example if no specific keywords match
        generic_example = lx.data.ExampleData(
            text="Juliet looked at Romeo with a sense of longing.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="character", extraction_text="Juliet", attributes={"name": "Juliet"}
                ),
                lx.data.Extraction(
                    extraction_class="character", extraction_text="Romeo", attributes={"name": "Romeo"}
                ),
                lx.data.Extraction(
                    extraction_class="emotion", extraction_text="longing", attributes={"type": "longing"}
                ),
            ]
        )
        examples.append(generic_example)

    logging.info(f"Selected {len(examples)} few-shot example(s).")

    result = lx.extract(
        text_or_documents=unstructured_text,
        prompt_description=prompt,
        examples=examples,
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    logging.info(f"Extraction result: {result}")

    # Convert the result to a JSON-serializable format
    extractions = [
        {"text": e.extraction_text, "class": e.extraction_class, "attributes": e.attributes}
        for e in result.extractions
    ]
    
    return {
        "extracted_data": extractions
    }

# Streamlit utility functions
def load_gemini_key() -> tuple[str, bool]:
    """Load the Gemini API key from the environment variable or user input."""
    key = ""
    is_key_provided = False
    secrets_file = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(secrets_file) and "GOOGLE_API_KEY" in st.secrets.keys():
        key = st.secrets["GOOGLE_API_KEY"]
        st.sidebar.success('Using Gemini Key from secrets.toml')
        is_key_provided = True
    else:
        key = st.sidebar.text_input(
            'Add Gemini API key and press \'Enter\'', type="password")
        if len(key) > 0:
            st.sidebar.success('Using the provided Gemini Key')
            is_key_provided = True
        else:
            st.sidebar.error('No Gemini Key')
    return key, is_key_provided

def format_output_agraph(output):
    nodes = []
    edges = []
    for node in output["nodes"]:
        nodes.append(
            Node(id=node["id"], label=node["label"], size=8, shape="diamond"))
    for edge in output["edges"]:
        edges.append(Edge(source=edge["source"], label=edge["relation"],
                     target=edge["target"], color="#4CAF50", arrows="to"))
    return nodes, edges

def display_agraph(nodes, edges):
    config = Config(width=950,
                    height=950,
                    directed=True,
                    physics=True,
                    hierarchical=True,
                    nodeHighlightBehavior=False,
                    highlightColor="#F7A7A6",
                    collapsible=False,
                    node={'labelProperty': 'label'},
                    )
    return agraph(nodes=nodes, edges=edges, config=config)

# Core GraphRAG functions
def extract_entities(documents: List[str]) -> List[Dict[str, Any]]:
    """Extract entities from documents"""
    all_entities = []
    
    for doc in documents:
        result = document_extractor_tool(
            doc, 
            "Extract financial entities including company names, revenue figures, and fiscal periods from business documents"
        )
        all_entities.extend(result["extracted_data"])
    
    return all_entities

def extract_relationships(documents: List[str]) -> List[Dict[str, Any]]:
    """Extract relationships between entities"""
    all_relationships = []
    
    for doc in documents:
        result = document_extractor_tool(
            doc,
            "Extract financial relationships and revenue connections between companies and fiscal periods"
        )
        all_relationships.extend(result["extracted_data"])
    
    return all_relationships

def build_graph_data(entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build graph data for visualization"""
    nodes = []
    edges = []
    
    # Create nodes from entities
    entity_map = {}
    for i, entity in enumerate(entities):
        node_id = str(i)
        nodes.append({
            "id": node_id,
            "label": entity["text"],
            "type": entity["class"]
        })
        entity_map[entity["text"].lower()] = node_id
    
    # Create edges from relationships and simple co-occurrence
    for rel in relationships:
        rel_text = rel["text"].lower()
        found_entities = []
        
        # Find entities mentioned in this relationship
        for entity_text, entity_id in entity_map.items():
            if entity_text in rel_text:
                found_entities.append(entity_id)
        
        # Create edges between found entities
        for i in range(len(found_entities)):
            for j in range(i + 1, len(found_entities)):
                edges.append({
                    "source": found_entities[i],
                    "target": found_entities[j],
                    "relation": rel["class"]
                })
    
    # If no relationships found, create simple co-occurrence edges
    if not edges:
        st.write("No relationship edges found, creating fallback edges...")
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                if i < j:
                    # Create edges between all entities
                    edges.append({
                        "source": str(i),
                        "target": str(j),
                        "relation": "related_to"
                    })
    
    return {"nodes": nodes, "edges": edges}

def answer_query(entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """Answer query using extracted entities and relationships"""
    if not query:
        return None
    
    # Find relevant entities
    relevant_entities = [
        e for e in entities 
        if any(word.lower() in e["text"].lower() or word.lower() in str(e["attributes"]).lower() 
               for word in query.split())
    ]
    
    # Find relevant relationships
    relevant_relationships = [
        r for r in relationships
        if any(word.lower() in r["text"].lower() or word.lower() in str(r["attributes"]).lower()
               for word in query.split())
    ]
    
    return {
        "query": query,
        "relevant_entities": relevant_entities,
        "relevant_relationships": relevant_relationships,
        "entity_count": len(relevant_entities),
        "relationship_count": len(relevant_relationships)
    }

def process_documents(documents: List[str], query: str = None) -> Dict[str, Any]:
    """Process documents and optionally answer a query"""
    # Extract entities and relationships
    entities = extract_entities(documents)
    relationships = extract_relationships(documents)
    
    # Debug info
    st.write(f"Debug: Found {len(entities)} entities, {len(relationships)} relationships")
    
    # Build graph data
    graph_data = build_graph_data(entities, relationships)
    
    # Debug graph data
    st.write(f"Debug: Graph has {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")
    
    # Answer query if provided
    results = answer_query(entities, relationships, query) if query else None
    
    return {
        "entities": entities,
        "relationships": relationships,
        "graph_data": graph_data,
        "results": results
    }

# Streamlit app
def main():
    st.set_page_config(page_title="GraphRAG with LangExtract", layout="wide")
    st.title("GraphRAG with LangExtract")
    
    # Load API key
    api_key, is_key_provided = load_gemini_key()
    
    if not is_key_provided:
        st.warning("Please provide an API key to continue")
        return
    
    # Set environment variable
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Predefined documents
    documents = [
        "Apple Inc. was founded by Steve Jobs and Steve Wozniak in 1976. The company is headquartered in Cupertino, California. Steve Jobs served as CEO until his death in 2011.",
        "Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975. It's based in Redmond, Washington. Bill Gates was the CEO for many years.",
        "Both Apple and Microsoft are major technology companies that compete in various markets including operating systems and productivity software. They have a long history of rivalry.",
        "Google was founded by Larry Page and Sergey Brin in 1998. The company started as a search engine but has expanded into many areas including cloud computing and artificial intelligence."
    ]
    
    st.success(f"Using {len(documents)} predefined documents about tech companies")
        
    # Query input
    query = st.text_input("Enter your query (optional):")
    
    if st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            result = process_documents(documents, query if query else None)
            
            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Graph Visualization", "Entities", "Relationships", "Query Results"])
            
            with tab1:
                if result["graph_data"]:
                    st.subheader("Knowledge Graph")
                    
                    nodes, edges = format_output_agraph(result["graph_data"])
                    if nodes:
                        display_agraph(nodes, edges)
                    else:
                        st.info("No graph data to display")
            
            with tab2:
                st.subheader("Extracted Entities")
                if result["entities"]:
                    for i, entity in enumerate(result["entities"]):
                        with st.expander(f"{entity['text']} ({entity['class']})"):
                            st.json(entity["attributes"])
                else:
                    st.info("No entities extracted")
            
            with tab3:
                st.subheader("Extracted Relationships")
                if result["relationships"]:
                    for i, rel in enumerate(result["relationships"]):
                        with st.expander(f"{rel['text']} ({rel['class']})"):
                            st.json(rel["attributes"])
                else:
                    st.info("No relationships extracted")
            
            with tab4:
                if query and result["results"]:
                    st.subheader("Query Results")
                    st.json(result["results"])
                else:
                    st.info("No query provided or no results")

if __name__ == "__main__":
    main()

