# https://towardsdatascience.com/text-to-knowledge-graph-made-easy-with-graph-maker-f3f890c0dbe8

## Define ontology
ontology = Ontology(
# labels of the entities to be extracted. Can be a string or an object, like the following.
labels=[
{"Person": "Person name without any adjectives, Remember a person may be referenced by their name or using a pronoun"},
{"Object": "Do not add the definite article 'the' in the object name"},
{"Event": "Event event involving multiple people. Do not include qualifiers or verbs like gives, leaves, works etc."},
"Place",
"Document",
"Organisation",
"Action",
{"Miscellaneous": "Any important concept can not be categorised with any other given label"},
],
# Relationships that are important for your application.
# These are more like instructions for the LLM to nudge it to focus on specific relationships.
# There is no guarantee that only these relationships will be extracted, but some models do a good job overall at sticking to these relations.
relationships=[
"Relation between any pair of Entities",
],
)

## Make test chunk and  Convert these chunks into Documents.
# After making chunk


from graph_maker import GraphMaker, Ontology, GroqClient
from graph_maker import Neo4jGraphModel

class Document(BaseModel):
  text: str
  metadata: dict

class Node(BaseModel):
  label: str
  name: str
 
class Edge(BaseModel):
  node_1: Node
  node_2: Node
  relationship: str
  metadata: dict = {}
  order: Union[int, None] = None

model = "mixtral-8x7b-32768"
llm = GroqClient(model=model, temperature=0.1, top_p=0.5)
graph_maker = GraphMaker(ontology=ontology, llm_client=llm, verbose=False)

graph = graph_maker.from_documents(docs)

create_indices = False
neo4j_graph = Neo4jGraphModel(edges=graph, create_indices=create_indices)
neo4j_graph.save()
