### From https://towardsdatascience.com/from-retrieval-to-intelligence-exploring-rag-agent-rag-and-evaluation-with-trulens-3c518af836ce

"""
1. Overview
   The article explores enhancing Large Language Models (LLMs) with Retrieval Augmented Generation (RAG) using the LlamaIndex framework and a Neo4j database. 
   The motivation is to solve two main issues faced by LLMs:

    -1. Depth of Knowledge: While general-purpose LLMs know broad facts, they struggle with very specific or detailed information 
                           (e.g., exact sales figures or specialized technical details).
    -2. Up-to-date Information: LLMs are trained on static data and have a knowledge cutoff. They don’t inherently know about recent events or updates after their training period.

    By using RAG, we can feed the LLM with relevant, custom data at query time from a vector database. This ensures the model can provide accurate, detailed, 
    and current information. The article shows how to combine RAG with LlamaIndex, Neo4j as a vector store, and TruLens for evaluation.

2. Key Steps:
   -1. Generate a private text corpus unknown to the base model.
   -2. Store the processed corpus in a vector database (Neo4j) for retrieval.
   -3. Implement a RAG pipeline using LlamaIndex to query relevant data chunks.
   -4. Compare and test different LLM backends (OpenAI GPT-4o and Meta’s Llama 3.2 models).
   -5. Integrate the RAG logic into an agent that can call retrieval as a tool.
   -6. Evaluate the system using TruLens and RAG Triad metrics.
"""

### Data Generation
### Since most public data is likely known by large foundation models, the author creates a private corpus. 
### The corpus describes four imaginary companies, including “Ukraine Boats Inc.”, with details about products, pricing, manufacturing, and client success stories.

### A snippet from “Ukraine Boats Inc.” file is shown in the article:
-----------------------------------------------------------------------------------------------------------------
## **Ukraine Boats Inc.**
**Corporate Overview:**
Ukraine Boats Inc. is a premier manufacturer and supplier of high-quality boats and maritime solutions based in Odessa, Ukraine. The company prides itself on blending traditional craftsmanship with modern technology to serve clients worldwide. Founded in 2005, the company has grown to be a leader in the boating industry, specializing in recreational, commercial, and luxury vessels.
 - -
### **Product Lineup**
#### **Recreational Boats:**
1. **WaveRunner X200**
- **Description:** A sleek speedboat designed for water sports enthusiasts. Equipped with advanced navigation and safety features.
- **Price:** $32,000
- **Target Market:** Young adventurers and watersport lovers.
- **Features:**
- Top speed of 85 mph
- Built-in GPS with autopilot mode
- Seating capacity: 4
- Lightweight carbon-fiber hull
...
-----------------------------------------------------------------------------------------------------------------   
## The author stores multiple such documents:

# nova-drive-motors.txt
# aero-vance-aviation.txt
# ukraine-boats.txt
# city-solve.txt
# Token counts for these four files total about 12k tokens.

    
# Additionally, a Q&A dataset with 10 question-answer pairs about Ukraine Boats Inc. is generated for testing:
-----------------------------------------------------------------------------------------------------------------   
[
    {
        "question": "What is the primary focus of Ukraine Boats Inc.?",
        "answer": "Ukraine Boats Inc. specializes in manufacturing high-quality recreational, luxury, and commercial boats, blending traditional craftsmanship with modern technology."
    },
    {
        "question": "What is the price range for recreational boats offered by Ukraine Boats Inc.?",
        "answer": "Recreational boats range from $32,000 for the WaveRunner X200 to $55,000 for the SolarGlide EcoBoat."
    },
    {
        "question": "Which manufacturing facility focuses on bespoke yachts and customizations?",
        "answer": "The Lviv Custom Craft Workshop specializes in bespoke yachts and high-end customizations, including handcrafted woodwork and premium materials."
    },
    {
        "question": "What is the warranty coverage offered for boats by Ukraine Boats Inc.?",
        "answer": "All boats come with a 5-year warranty for manufacturing defects, while engines are covered under a separate 3-year engine performance guarantee."
    },
    {
        "question": "Which client used the Neptune Voyager catamaran, and what was the impact on their business?",
        "answer": "Paradise Resorts International used the Neptune Voyager catamarans, resulting in a 45% increase in resort bookings and winning the 'Best Tourism Experience' award."
    },
    {
        "question": "What award did the SolarGlide EcoBoat win at the Global Marine Design Challenge?",
        "answer": "The SolarGlide EcoBoat won the 'Best Eco-Friendly Design' award at the Global Marine Design Challenge in 2022."
    },
    {
        "question": "How has the Arctic Research Consortium benefited from the Poseidon Explorer?",
        "answer": "The Poseidon Explorer enabled five successful Arctic research missions, increased data collection efficiency by 60%, and improved safety in extreme conditions."
    },
    {
        "question": "What is the price of the Odessa Opulence 5000 luxury yacht?",
        "answer": "The Odessa Opulence 5000 luxury yacht starts at $1,500,000."
    },
    {
        "question": "Which features make the WaveRunner X200 suitable for watersports?",
        "answer": "The WaveRunner X200 features a top speed of 85 mph, a lightweight carbon-fiber hull, built-in GPS, and autopilot mode, making it ideal for watersports."
    },
    {
        "question": "What sustainability initiative is Ukraine Boats Inc. pursuing?",
        "answer": "Ukraine Boats Inc. is pursuing the Green Maritime Initiative (GMI) to reduce the carbon footprint by incorporating renewable energy solutions in 50% of their fleet by 2030."
    }
]
    
-----------------------------------------------------------------------------------------------------------------   
### Data Storage in Neo4j
### For RAG, a vector database is needed. The author uses Neo4j, which can store vector embeddings and allows semantic search.

### Configuration (pyproject.toml):
[configuration]
similarity_top_k = 10
vector_store_query_mode = "default"
similarity_cutoff = 0.75
response_mode = "compact"
distance_strategy = "cosine"
embedding_dimension = 256
chunk_size = 512
chunk_overlap = 128
separator = " "
max_function_calls = 2
hybrid_search = false

[configuration.data]
raw_data_path = "../data/companies"
dataset_path = "../data/companies/dataset.json"
source_docs = ["city-solve.txt", "aero-vance-aviation.txt", "nova-drive-motors.txt", "ukraine-boats.txt"]

[configuration.models]
llm = "gpt-4o-mini"
embedding_model = "text-embedding-3-small"
temperature = 0
llm_hf = "meta-llama/Llama-3.2-3B-Instruct"
context_window = 8192
max_new_tokens = 4096
hf_token = "hf_custom-token"
llm_evaluation = "gpt-4o-mini"

[configuration.db]
url = "neo4j+s://custom-url"
username = "neo4j"
password = "custom-password"
database = "neo4j"
index_name = "article"
text_node_property = "text"

-----------------------------------------------------------------------------------------------------------------  
### Code for reading documents and embedding:
# initialize models
embed_model = OpenAIEmbedding(
  model=CFG['configuration']['models']['embedding_model'],
  api_key=os.getenv('AZURE_OPENAI_API_KEY'),
  dimensions=CFG['configuration']['embedding_dimension']
)

# get documents paths
document_paths = [Path(CFG['configuration']['data']['raw_data_path']) / document for document in CFG['configuration']['data']['source_docs']]

# initialize a file reader
reader = SimpleDirectoryReader(input_files=document_paths)

# load documents into LlamaIndex Documents
documents = reader.load_data()

-----------------------------------------------------------------------------------------------------------------
### Splitting documents into nodes and uploading to Neo4j:
# create nodes via splitter
splitter = SentenceSplitter(
   separator=CFG['configuration']['separator'],
   chunk_size=CFG['configuration']['chunk_size'],
   chunk_overlap=CFG['configuration']['chunk_overlap']
)
nodes = splitter.split_documents(documents)

neo4j_vector = Neo4jVectorStore(
    username=CFG['configuration']['db']['username'],
    password=CFG['configuration']['db']['password'],
    url=CFG['configuration']['db']['url'],
    embedding_dimension=CFG['configuration']['embedding_dimension'],
    hybrid_search=CFG['configuration']['hybrid_search']
)

storage_context = StorageContext.from_defaults(
    vector_store=neo4j_vector
)

index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
### This process stores all document chunks into Neo4j with embeddings. Hybrid search is turned off for now.

-----------------------------------------------------------------------------------------------------------------
### Querying the RAG Pipeline
### To query the data, we connect to the existing vector store and create a query engine. 
### Two approaches are demonstrated: a standalone query engine and an agent-based approach.

### Connecting to Neo4j:
# connect to existing neo4j vector index
vector_store = Neo4jVectorStore(
  username=CFG['configuration']['db']['username'],
  password=CFG['configuration']['db']['password'],
  url=CFG['configuration']['db']['url'],
  embedding_dimension=CFG['configuration']['embedding_dimension'],
  distance_strategy=CFG['configuration']['distance_strategy'],
  index_name=CFG['configuration']['db']['index_name'],
  text_node_property=CFG['configuration']['db']['text_node_property']
)
index = VectorStoreIndex.from_vector_store(vector_store)

### Using OpenAI Models
### Initialize LLM and embedding model:
llm = OpenAI(
  api_key=os.getenv('AZURE_OPENAI_API_KEY'),
  model=CFG['configuration']['models']['llm'],
  temperature=CFG['configuration']['models']['temperature']
)
embed_model = OpenAIEmbedding(
  model=CFG['configuration']['models']['embedding_model'],
  api_key=os.getenv('AZURE_OPENAI_API_KEY'),
  dimensions=CFG['configuration']['embedding_dimension']
)

Settings.llm = llm
Settings.embed_model = embed_model

-----------------------------------------------------------------------------------------------------------------
### Create a default query engine:
query_engine = index.as_query_engine()

### Query example:
response = query_engine.query("What is the primary focus of Ukraine Boats Inc.?")

for node in response.source_nodes:
  print(f'{node.node.id_}, {node.score}')

print(response.response)

-----------------------------------------------------------------------------------------------------------------
### Sample output:
ukraine-boats-3, 0.8536546230316162
ukraine-boats-4, 0.8363556861877441
The primary focus of Ukraine Boats Inc. is designing, manufacturing, and selling luxury and eco-friendly boats...

### To customize retrieval:
retriever = VectorIndexRetriever(
  index=index,
  similarity_top_k=CFG['configuration']['similarity_top_k'],
  vector_store_query_mode=CFG['configuration']['vector_store_query_mode']
)

similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=CFG['configuration']['similarity_cutoff'])
response_synthesizer = get_response_synthesizer(response_mode=CFG['configuration']['response_mode'])

query_engine = RetrieverQueryEngine(
  retriever=retriever,
  node_postprocessors=[similarity_postprocessor],
  response_synthesizer=response_synthesizer
)

-----------------------------------------------------------------------------------------------------------------
### Using an Agent with OpenAI
The article shows how to wrap the query engine in a tool and use an agent (OpenAIAgentWorker) to query that tool:
AGENT_SYSTEM_PROMPT = "You are a helpful human assistant. You always call the retrieve_semantically_similar_data tool before answering any questions. If the answer couldn't be found, respond with `Didn't find relevant information`."

TOOL_NAME = "retrieve_semantically_similar_data"
TOOL_DESCRIPTION = "Provides additional information about the companies. Input: string"

agent_worker = OpenAIAgentWorker.from_tools(
    [
        QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name=TOOL_NAME,
            description=TOOL_DESCRIPTION,
            return_direct=False,
        )
    ],
    system_prompt=AGENT_SYSTEM_PROMPT,
    llm=llm,
    verbose=True,
    max_function_calls=CFG['configuration']['max_function_calls']
)

agent = AgentRunner(agent_worker=agent_worker)


-----------------------------------------------------------------------------------------------------------------
### Interactive chat:
while True:
  current_message = input('Insert your next message:')
  print(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}|User: {current_message}")
  response = agent.chat(current_message)
  print(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]}|Agent: {response.response}")

### Sample conversation shows the agent retrieving data from Neo4j and returning accurate answers.

-----------------------------------------------------------------------------------------------------------------
### Using Llama 3.2 (Open-Source Model)
### For open-source models (e.g., Meta’s Llama 3.2), the code uses HuggingFaceLLM:
login(token=CFG['configuration']['models']['hf_token'])

SYSTEM_PROMPT = """You are an AI assistant ... (omitted for brevity) """

query_wrapper_prompt = PromptTemplate(
    "<|start_header_id|>system<|end_header_id|>\n" + SYSTEM_PROMPT + "<|eot_id|><|start_header_id|>user<|end_header_id|>{query_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
)

llm = HuggingFaceLLM(
    context_window=CFG['configuration']['models']['context_window'],
    max_new_tokens=CFG['configuration']['models']['max_new_tokens'],
    generate_kwargs={"temperature": CFG['configuration']['models']['temperature'], "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=CFG['configuration']['models']['llm_hf'],
    model_name=CFG['configuration']['models']['llm_hf'],
    device_map="cuda:0",
    model_kwargs={"torch_dtype": torch.bfloat16}
)

Settings.llm = llm

### The querying code remains similar. For an agent with a non-OpenAI model, the ReActAgentWorker is used:

agent_worker = ReActAgentWorker.from_tools(
    [
        QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name=TOOL_NAME,
            description=TOOL_DESCRIPTION,
            return_direct=False,
        )
    ],
    llm=llm,
    verbose=True,
    chat_history=[ChatMessage(content=AGENT_SYSTEM_PROMPT, role="system")]
)

agent = AgentRunner(agent_worker=agent_worker)

### Running queries: The agent now uses the open-source Llama model. The conversation works similarly, calling retrieve_semantically_similar_data tool when needed, 
###                  then returning a final answer.

-------------------------------------------------------------------------------------------------------------------
### Evaluation with TruLens
### To assess the RAG Triad (Answer Relevance, Context Relevance, Groundedness), the article uses TruLens. TruLens uses an LLM as a judge to evaluate answers. 
### The code snippet for evaluation:
experiment_name = "llama-3.2-3B-custom-retriever"

provider = OpenAIProvider(
    model_engine=CFG['configuration']['models']['llm_evaluation']
)

context_selection = TruLlama.select_source_nodes().node.text

f_context_relevance = (
    Feedback(provider.context_relevance, name="Context Relevance")
    .on_input()
    .on(context_selection)
)

f_groundedness_cot = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(context_selection.collect())
    .on_output()
)

f_qa_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input_output()
)

tru_agent = TruLlama(
    agent,
    app_name=experiment_name,
    tags="agent testing",
    feedbacks=[f_qa_relevance, f_context_relevance, f_groundedness_cot],
)

for item in tqdm(dataset):
    try:
        agent.reset()
        with tru_agent as recording:
            agent.query(item.get('question'))
        record_agent = recording.get()
        
        # wait for feedback results
        for feedback, result in record_agent.wait_for_feedback_results().items():
            logging.info(f'{feedback.name}: {result.result}')
    except Exception as e:
        logging.error(e)
        traceback.format_exc()

### TruLens then provides a UI leaderboard, record-by-record assessment, and execution traces. This helps identify how well the system is performing on each metric.

--------------------------------------------------------------------------------------------------------
"""
Conclusion
Using RAG with LlamaIndex and Neo4j allows LLMs to return accurate, detailed answers drawn from a custom private corpus.
The pipeline supports both OpenAI and open-source models (Llama 3.2).
Agent frameworks let the system reason and choose when to call the retrieval tool.
Evaluation via TruLens and the RAG Triad metrics shows that the system achieves high groundedness and good relevance.
Future improvements could include keyword search, re-ranking, neighbor chunk selection, and more.
The provided code and explanations form a complete reference implementation for building and evaluating a RAG system with LlamaIndex and Neo4j.
"""

