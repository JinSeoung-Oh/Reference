import numpy as np
from trulens_eval import Tru, Feedback, TruLlama
from trulens_eval.feedback.provider.openai import OpenAI
from trulens_eval.feedback import Groundedness, GroundTruthAgreement
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

questions = [
    "What mysterious object did Loki use in his attempt to conquer Earth?",
    "Which two members of the Avengers created Ultron?",
    "How did Thanos achieve his plan of exterminating half of all life in the universe?",
    "What method did the Avengers use to reverse Thanos' actions?",
    "Which member of the Avengers sacrificed themselves to defeat Thanos?",
]

ground_truth = [
    "The Tesseract",
    "Tony Stark (Iron Man) and Bruce Banner (The Hulk).",
    "By using the six Infinity Stones",
    "By collecting the stones through time travel.",
    "Tony Stark (Iron Man)",
]

documents = SimpleDirectoryReader("./data").load_data()
vector_index = VectorStoreIndex.from_documents(documents)
query_engine = vector_index.as_query_engine(similarity_top_k=2)

openai = OpenAI()
golden_set = [{"query": q, "response": r} for q, r in zip(questions, ground_truth)]
ground_truth = Feedback(
    GroundTruthAgreement(golden_set).agreement_measure, name="Ground Truth"
).on_input_output()
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
    .aggregate(np.mean)
)
tru_query_engine_recorder = TruLlama(
    query_engine,
    app_id="Avengers_App",
    feedbacks=[
        ground_truth,
        groundedness,
        qa_relevance,
        qs_relevance,
    ],
)
with tru_query_engine_recorder as recording:
    for question in questions:
        query_engine.query(question)
tru = Tru()
tru.run_dashboard()
