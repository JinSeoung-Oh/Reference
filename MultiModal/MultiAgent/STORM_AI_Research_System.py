### From https://towardsdatascience.com/running-the-storm-ai-research-system-with-your-local-documents-e413ea2ae064

"""
The STORM (Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking) system from Stanford leverages LLM agents to simulate "perspective-guided conversations,"
aimed at complex research tasks where single-step RAG methods may not suffice. 
STORM generates rich research outlines and articles, ideal for pre-writing stages, and was initially developed for retrieving information from the web,
but it can also search local document databases.

1. Key Elements and Setup
   The STORM workflow allows organizations to run research on their data by configuring it with local vector databases. 
   We’ll implement this with US FEMA disaster preparedness PDFs. Here’s the approach:

   -1. Parsing PDFs: STORM splits PDFs into manageable document chunks, either by page or into smaller sections, for effective topic retrieval.
   -2. Metadata Enrichment: STORM requires metadata, so we generate titles and descriptions with LLM prompts for each chunk.
   -3. Vector Database Construction: Using LangChain’s Qdrant Vector Store, we save parsed content, making it searchable by embedding vectors.
"""

# 2. Step
#   -1. Parsing PDFs
def parse_pdfs():
    """
    Parses all PDF files in the specified directory and loads their content.

    This function iterates through all files in the directory specified by PDF_DIR,
    checks if they have a .pdf extension, and loads their content using PyPDFLoader.
    The loaded content from each PDF is appended to a list which is then returned.

    Returns:
        list: A list containing the content of all loaded PDF documents.
    """
    docs = []
    pdfs = os.listdir(PDF_DIR)
    print(f"We have {len(pdfs)} pdfs")
    for pdf_file in pdfs:
        if not pdf_file.endswith(".pdf"):
            continue
        print(f"Loading PDF: {pdf_file}")
        file_path = f"{PDF_DIR}/{pdf_file}"
        loader = PyPDFLoader(file_path)
        docs = docs + loader.load()
        print(f"Loaded {len(docs)} documents")

    return docs


docs = parse_pdfs()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

#  -2. Metadata enrichment
#      STORM’s example documentation requires that documents have metadata fields ‘URL’, ‘title’, and ‘description’, where ‘URL’ should be unique.
def summarize_text(text, prompt):
    """
    Generate a summary of some text based on the user's prompt

    Args:

    text (str) - the text to analyze
    prompt (str) - prompt instruction on how to summarize the text, eg 'generate a title'

    Returns:

    summary (text) - LLM-generated summary

    """
    messages = [
        (
            "system",
            "You are an assistant that gives very brief single sentence description of text.",
        ),
        ("human", f"{prompt} :: \n\n {text}"),
    ]
    ai_msg = llm.invoke(messages)
    summary = ai_msg.content
    return summary


def enrich_metadata(docs):
    """
    Uses an LLM to populate 'title' and 'description' for text chunks

    Args:

    docs (list) - list of LangChain documents

    Returns:

    docs (list) - list of LangChain documents with metadata fields populated

    """
    new_docs = []
    for doc in docs:

        # pdf name is last part of doc.metadata['source']
        pdf_name = doc.metadata["source"].split("/")[-1]

        # Find row in df where pdf_name is in URL
        row = df[df["Document"].str.contains(pdf_name)]
        page = doc.metadata["page"] + 1
        url = f"{row['Document'].values[0]}?id={str(uuid4())}#page={page}"

        # We'll use an LLM to generate a summary and title of the text, used by STORM
        # This is just for the demo, proper application would have better metadata
        summary = summarize_text(doc.page_content, prompt="Please describe this text:")
        title = summarize_text(
            doc.page_content, prompt="Please generate a 5 word title for this text:"
        )

        doc.metadata["description"] = summary
        doc.metadata["title"] = title
        doc.metadata["url"] = url
        doc.metadata["content"] = doc.page_content

        # print(json.dumps(doc.metadata, indent=2))
        new_docs.append(doc)

    print(f"There are {len(docs)} docs")

    return new_docs


docs = enrich_metadata(docs)
chunks = enrich_metadata(chunks)

#  -3. Building vector databases
#      STORM already supports the Qdrant vector store. 
def build_vector_store(doc_type, docs):
    """
    Givena  list of LangChain docs, will embed and create a file-system Qdrant vector database.
    The folder includes doc_type in its name to avoid overwriting.

    Args:

    doc_type (str) - String to indicate level of document split, eg 'pages',
                     'chunks'. Used to name the database save folder
    docs (list) - List of langchain documents to embed and store in vector database

    Returns:

    Nothing returned by function, but db saved to f"{DB_DIR}_{doc_type}".

    """

    print(f"There are {len(docs)} docs")

    save_dir = f"{DB_DIR}_{doc_type}"

    print(f"Saving vectors to directory {save_dir}")

    client = QdrantClient(path=save_dir)

    client.create_collection(
        collection_name=DB_COLLECTION_NAME,
        vectors_config=VectorParams(size=num_vectors, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=DB_COLLECTION_NAME,
        embedding=embeddings,
    )

    uuids = [str(uuid4()) for _ in range(len(docs))]

    vector_store.add_documents(documents=docs, ids=uuids)


build_vector_store("pages", docs)
build_vector_store("chunks", docs)

# -4. Running STORM
def set_instructions(runner):
    """
    Adjusts templates and personas for the STORM AI Research algorithm.

    Args:

    runner - STORM runner object

    Returns:

    runner - STORM runner object with extra prompting

    """

    # Open LMs are generally weaker in following output format.
    # One way for mitigation is to add one-shot example to the prompt to exemplify the desired output format.
    # For example, we can add the following examples to the two prompts used in StormPersonaGenerator.
    # Note that the example should be an object of dspy.Example with fields matching the InputField
    # and OutputField in the prompt (i.e., dspy.Signature).
    find_related_topic_example = Example(
        topic="Knowledge Curation",
        related_topics="https://en.wikipedia.org/wiki/Knowledge_management\n"
        "https://en.wikipedia.org/wiki/Information_science\n"
        "https://en.wikipedia.org/wiki/Library_science\n",
    )
    gen_persona_example = Example(
        topic="Knowledge Curation",
        examples="Title: Knowledge management\n"
        "Table of Contents: History\nResearch\n  Dimensions\n  Strategies\n  Motivations\nKM technologies"
        "\nKnowledge barriers\nKnowledge retention\nKnowledge audit\nKnowledge protection\n"
        "  Knowledge protection methods\n    Formal methods\n    Informal methods\n"
        "  Balancing knowledge protection and knowledge sharing\n  Knowledge protection risks",
        personas="1. Historian of Knowledge Systems: This editor will focus on the history and evolution of knowledge curation. They will provide context on how knowledge curation has changed over time and its impact on modern practices.\n"
        "2. Information Science Professional: With insights from 'Information science', this editor will explore the foundational theories, definitions, and philosophy that underpin knowledge curation\n"
        "3. Digital Librarian: This editor will delve into the specifics of how digital libraries operate, including software, metadata, digital preservation.\n"
        "4. Technical expert: This editor will focus on the technical aspects of knowledge curation, such as common features of content management systems.\n"
        "5. Museum Curator: The museum curator will contribute expertise on the curation of physical items and the transition of these practices into the digital realm.",
    )
    runner.storm_knowledge_curation_module.persona_generator.create_writer_with_persona.find_related_topic.demos = [
        find_related_topic_example
    ]
    runner.storm_knowledge_curation_module.persona_generator.create_writer_with_persona.gen_persona.demos = [
        gen_persona_example
    ]

    # A trade-off of adding one-shot example is that it will increase the input length of the prompt. Also, some
    # examples may be very long (e.g., an example for writing a section based on the given information), which may
    # confuse the model. For these cases, you can create a pseudo-example that is short and easy to understand to steer
    # the model's output format.
    # For example, we can add the following pseudo-examples to the prompt used in WritePageOutlineFromConv and
    # ConvToSection.
    write_page_outline_example = Example(
        topic="Example Topic",
        conv="Wikipedia Writer: ...\nExpert: ...\nWikipedia Writer: ...\nExpert: ...",
        old_outline="# Section 1\n## Subsection 1\n## Subsection 2\n"
        "# Section 2\n## Subsection 1\n## Subsection 2\n"
        "# Section 3",
        outline="# New Section 1\n## New Subsection 1\n## New Subsection 2\n"
        "# New Section 2\n"
        "# New Section 3\n## New Subsection 1\n## New Subsection 2\n## New Subsection 3",
    )
    runner.storm_outline_generation_module.write_outline.write_page_outline.demos = [
        write_page_outline_example
    ]
    write_section_example = Example(
        info="[1]\nInformation in document 1\n[2]\nInformation in document 2\n[3]\nInformation in document 3",
        topic="Example Topic",
        section="Example Section",
        output="# Example Topic\n## Subsection 1\n"
        "This is an example sentence [1]. This is another example sentence [2][3].\n"
        "## Subsection 2\nThis is one more example sentence [1].",
    )
    runner.storm_article_generation.section_gen.write_section.demos = [
        write_section_example
    ]

    return runner

def latest_dir(parent_folder):
    """
    Find the most recent folder (by modified date) in the specified parent folder.

    Args:
        parent_folder (str): The path to the parent folder where the search for the most recent folder will be conducted. Defaults to f"{DATA_DIR}/storm_output".

    Returns:
        str: The path to the most recently modified folder within the parent folder.
    """
    # Find most recent folder (by modified date) in DATA_DIR/storm_data
    # TODO, find out how exactly storm passes back its output directory to avoid this hack
    folders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
    folder = max(folders, key=os.path.getmtime)

    return folder


def generate_footnotes(folder):
    """
    Generates footnotes from a JSON file containing URL information.

    Args:
        folder (str): The directory path where the 'url_to_info.json' file is located.

    Returns:
        str: A formatted string containing footnotes with URLs and their corresponding titles.
    """

    file = f"{folder}/url_to_info.json"

    with open(file) as f:
        data = json.load(f)

    refs = {}
    for rec in data["url_to_unified_index"]:
        val = data["url_to_unified_index"][rec]
        title = data["url_to_info"][rec]["title"].replace('"', "")
        refs[val] = f"- {val} [{title}]({rec})"

    keys = list(refs.keys())
    keys.sort()

    footer = ""
    for key in keys:
        footer += f"{refs[key]}\n"

    return footer, refs


def generate_markdown_article(output_dir):
    """
    Generates a markdown article by reading a text file, appending footnotes, 
    and saving the result as a markdown file.

    The function performs the following steps:
    1. Retrieves the latest directory using the `latest_dir` function.
    2. Generates footnotes for the article using the `generate_footnotes` function.
    3. Reads the content of a text file named 'storm_gen_article_polished.txt' 
       located in the latest directory.
    4. Appends the generated footnotes to the end of the article content.
    5. Writes the modified content to a new markdown file named 
       STORM_OUTPUT_MARKDOWN_ARTICLE in the same directory.

    Args:

    output_dir (str) - The directory where the STORM output is stored.


    """

    folder = latest_dir(output_dir)
    footnotes, refs = generate_footnotes(folder)

    with open(f"{folder}/storm_gen_article_polished.txt") as f:
        text = f.read()

    # Update text references like [10] to link to URLs
    for ref in refs:
        print(f"Ref: {ref}, Ref_num: {refs[ref]}")
        url = refs[ref].split("(")[1].split(")")[0]
        text = text.replace(f"[{ref}]", f"\[[{ref}]({url})\]")

    text += f"\n\n## References\n\n{footnotes}"

    with open(f"{folder}/{STORM_OUTPUT_MARKDOWN_ARTICLE}", "w") as f:
        f.write(text)

def run_storm(topic, model_type, db_dir):
    """
    This function runs the STORM AI Research algorithm using data
    in a QDrant local database.

    Args:

    topic (str) - The research topic to generate the article for
    model_type (str) - One of 'openai' and 'ollama' to control LLM used
    db_dir (str) - Directory where the QDrant vector database is
    
    """
    if model_type not in ["openai", "ollama"]:
        print("Unsupported model_type")
        sys.exit()

    # Clear lock so can be read
    if os.path.exists(f"{db_dir}/.lock"):
        print(f"Removing lock file {db_dir}/.lock")
        os.remove(f"{db_dir}/.lock")

    print(f"Loading Qdrant vector store from {db_dir}")

    engine_lm_configs = STORMWikiLMConfigs()

    if model_type == "openai":

        print("Using OpenAI models")

        # Initialize the language model configurations
        openai_kwargs = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 1.0,
            "top_p": 0.9,
        }

        ModelClass = (
            OpenAIModel
            if os.getenv("OPENAI_API_TYPE") == "openai"
            else AzureOpenAIModel
        )
        # If you are using Azure service, make sure the model name matches your own deployed model name.
        # The default name here is only used for demonstration and may not match your case.
        gpt_35_model_name = (
            "gpt-4o-mini"
            if os.getenv("OPENAI_API_TYPE") == "openai"
            else "gpt-35-turbo"
        )
        gpt_4_model_name = "gpt-4o"
        if os.getenv("OPENAI_API_TYPE") == "azure":
            openai_kwargs["api_base"] = os.getenv("AZURE_API_BASE")
            openai_kwargs["api_version"] = os.getenv("AZURE_API_VERSION")

        # STORM is a LM system so different components can be powered by different models.
        # For a good balance between cost and quality, you can choose a cheaper/faster model for conv_simulator_lm
        # which is used to split queries, synthesize answers in the conversation. We recommend using stronger models
        # for outline_gen_lm which is responsible for organizing the collected information, and article_gen_lm
        # which is responsible for generating sections with citations.
        conv_simulator_lm = ModelClass(
            model=gpt_35_model_name, max_tokens=10000, **openai_kwargs
        )
        question_asker_lm = ModelClass(
            model=gpt_35_model_name, max_tokens=10000, **openai_kwargs
        )
        outline_gen_lm = ModelClass(
            model=gpt_4_model_name, max_tokens=10000, **openai_kwargs
        )
        article_gen_lm = ModelClass(
            model=gpt_4_model_name, max_tokens=10000, **openai_kwargs
        )
        article_polish_lm = ModelClass(
            model=gpt_4_model_name, max_tokens=10000, **openai_kwargs
        )

    elif model_type == "ollama":

        print("Using Ollama models")

        ollama_kwargs = {
            # "model": "llama3.2:3b",
            "model": "llama3.1:latest",
            # "model": "qwen2.5:14b",
            "port": "11434",
            "url": "http://localhost",
            "stop": (
                "\n\n---",
            ),  # dspy uses "\n\n---" to separate examples. Open models sometimes generate this.
        }

        conv_simulator_lm = OllamaClient(max_tokens=500, **ollama_kwargs)
        question_asker_lm = OllamaClient(max_tokens=500, **ollama_kwargs)
        outline_gen_lm = OllamaClient(max_tokens=400, **ollama_kwargs)
        article_gen_lm = OllamaClient(max_tokens=700, **ollama_kwargs)
        article_polish_lm = OllamaClient(max_tokens=4000, **ollama_kwargs)

    engine_lm_configs.set_conv_simulator_lm(conv_simulator_lm)
    engine_lm_configs.set_question_asker_lm(question_asker_lm)
    engine_lm_configs.set_outline_gen_lm(outline_gen_lm)
    engine_lm_configs.set_article_gen_lm(article_gen_lm)
    engine_lm_configs.set_article_polish_lm(article_polish_lm)

    max_conv_turn = 4
    max_perspective = 3
    search_top_k = 10
    max_thread_num = 1
    device = "cpu"
    vector_db_mode = "offline"

    do_research = True
    do_generate_outline = True
    do_generate_article = True
    do_polish_article = True

    # Initialize the engine arguments
    output_dir=f"{STORM_OUTPUT_DIR}/{db_dir.split('db_')[1]}"
    print(f"Output directory: {output_dir}")

    engine_args = STORMWikiRunnerArguments(
        output_dir=output_dir,
        max_conv_turn=max_conv_turn,
        max_perspective=max_perspective,
        search_top_k=search_top_k,
        max_thread_num=max_thread_num,
    )

    # Setup VectorRM to retrieve information from your own data
    rm = VectorRM(
        collection_name=DB_COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        device=device,
        k=search_top_k,
    )

    # initialize the vector store, either online (store the db on Qdrant server) or offline (store the db locally):
    if vector_db_mode == "offline":
        rm.init_offline_vector_db(vector_store_path=db_dir)

    # Initialize the STORM Wiki Runner
    runner = STORMWikiRunner(engine_args, engine_lm_configs, rm)

    # Set instructions for the STORM AI Research algorithm
    runner = set_instructions(runner)

    # run the pipeline
    runner.run(
        topic=topic,
        do_research=do_research,
        do_generate_outline=do_generate_outline,
        do_generate_article=do_generate_article,
        do_polish_article=do_polish_article,
    )
    runner.post_run()
    runner.summary()

    generate_markdown_article(output_dir)



