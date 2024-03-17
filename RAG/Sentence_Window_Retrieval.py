## From https://generativeai.pub/advanced-rag-retrieval-strategies-sentence-window-retrieval-b6964b6e56f7
# The principle of sentence window retrieval is quite simple. Initially, documents are split by sentences during 
# the slicing process and then embedded and saved in the database. During retrieval, related sentences are found,
# but not only the retrieved sentences are considered as retrieval results. The sentences before and after the retrieved sentence are also included as part of the results. 
# The number of sentences included can be adjusted through parameters, and finally, the retrieval results are submitted together to the LLM to generate an answer.

## Example

from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)
documents = SimpleDirectoryReader("./data").load_data()
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
embed_model = OpenAIEmbedding()
Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = node_parser
sentence_index = VectorStoreIndex.from_documents(
    documents=documents,
)
postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
sentence_window
_engine = sentence_index.as_query_engine(
    similarity_top_k=2, node_postprocessors=[postproc]
)


## Retrieval Effect Comparison - LLM evaluation tool Trulens
from trulens_eval import Tru, Feedback, TruLlama
from trulens_eval.feedback.provider.openai import OpenAI as Trulens_OpenAI
from trulens_eval.feedback import Groundedness

tru = Tru()
openai = Trulens_OpenAI()
def rag_evaluate(query_engine, eval_name):
    grounded = Groundedness(groundedness_provider=openai)
    groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(TruLlama.select_source_nodes().node.text)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )
    qa_relevance = Feedback(
        openai.relevance_with_cot_reasons, name="Answer Relevance"
    ).on_input_output()
    qs_relevance = (
        Feedback(openai.qs_relevance_with_cot_reasons, name="Context Relevance")
        .on_input()
        .on(TruLlama.select_source_nodes().node.text)
    )
    tru_query_engine_recorder = TruLlama(
        query_engine,
        app_id=eval_name,
        feedbacks=[
            groundedness,
            qa_relevance,
            qs_relevance,
        ],
    )
    with tru_query_engine_recorder as recording:
        query_engine.query(question)

tru.reset_database()
rag_evaluate(base_engine, "base_evaluation")
rag_evaluate(sentence_window_engine, "sentence_window_evaluation")
Tru().run_dashboard()
