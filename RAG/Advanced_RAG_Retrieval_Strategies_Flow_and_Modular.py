## From https://ai.gopubby.com/advanced-rag-retrieval-strategies-flow-and-modular-672493acb4a7 <-- Have to check for flow

############# Basic RAG ################
import os
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_pipeline import QueryPipeline, InputComponent
from llama_index.core.response_synthesizers.simple_summarize import SimpleSummarize

documents = SimpleDirectoryReader("./data").load_data()
node_parser = SentenceSplitter()
llm = OpenAI(model="gpt-3.5-turbo")
embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = llm
Settings.embed_model = embed_model
Settings.node_parser = node_parser
if not os.path.exists("storage"):
    index = VectorStoreIndex.from_documents(documents)
    index.set_index_id("avengers")
    index.storage_context.persist("./storage")
else:
    store_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(
        storage_context=store_context, index_id="avengers"
    )

retriever =  index.as_retriever()
p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "input": InputComponent(),
        "retriever": retriever,
        "output": SimpleSummarize(),
    }
)
p.add_link("input", "retriever")
p.add_link("input", "output", dest_key="query_str")
p.add_link("retriever", "output", dest_key="nodes")

question = "Which two members of the Avengers created Ultron?"
output = p.run(input=question)


############## Adding a Reranker Module ################
# Compare with basic RAG / + mean add that line, - mean delet that line
+from llama_index.postprocessor.cohere_rerank import CohereRerank

+reranker = CohereRerank()
p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "input": InputComponent(),
        "retriever": retriever,
+        "reranker": reranker,
        "output": SimpleSummarize(),
    }
)
p.add_link("input", "retriever")
+p.add_link("input", "reranker", dest_key="query_str")
+p.add_link("retriever", "reranker", dest_key="nodes")
p.add_link("input", "output", dest_key="query_str")
-p.add_link("retriever", "output", dest_key="nodes")
+p.add_link("reranker", "output", dest_key="nodes")

##### run_with_intermediates method, which can get the intermediate results of the pipeline ##

output, intermediates = p.run_with_intermediates(input=question)
retriever_output = intermediates["retriever"].outputs["output"]
print(f"retriever output:")
for node in retriever_output:
    print(f"node id: {node.node_id}, node score: {node.score}")
reranker_output = intermediates["reranker"].outputs["nodes"]
print(f"\nreranker output:")
for node in reranker_output:
      print(f"node id: {node.node_id}, node score: {node.score}")


############## Adding a Query Rewrite Module #################
# Compare with Adding a Reranker Module RAG / + mean add that line, - mean delet that line

#define  HydeComponent class
from llama_index.core.query_pipeline import CustomQueryComponent
from typing import Dict, Any
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

class HydeComponent(CustomQueryComponent):
    """HyDE query rewrite component."""
    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        assert "input" in input, "input is required"
        return input

    @property
    def _input_keys(self) -> set:
        """Input keys dict."""
        return {"input"}

    @property
    def _output_keys(self) -> set:
        return {"output"}

    def _run_component(self, **kwargs) -> Dict[str, Any]:
        """Run the component."""
        hyde = HyDEQueryTransform(include_original=True)
        query_bundle = hyde(kwargs["input"])
        return {"output": query_bundle.embedding_strs[0]}

+query_rewriter = HydeComponent()
p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "input": InputComponent(),
+        "query_rewriter": query_rewriter,
        "retriever": retriever,
        "reranker": reranker,
        "output": SimpleSummarize(),
    }
)

-p.add_link("input", "retriever")
+p.add_link("input", "query_rewriter")
+p.add_link("query_rewriter", "retriever")
p.add_link("input", "reranker", dest_key="query_str")
p.add_link("retriever", "reranker", dest_key="nodes")
p.add_link("input", "output", dest_key="query_str")
p.add_link("reranker", "output", dest_key="nodes")

############## Replacing the Output Module #################
# Compare with Adding a Query Rewrite Module RAG / + mean add that line, - mean delet that line

+from llama_index.core.response_synthesizers.tree_summarize import TreeSummarize

p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "input": InputComponent(),
        "query_rewriter": query_rewriter,
        "retriever": retriever,
        "reranker": reranker,
-        "output": SimpleSummarize(),
+        "output": TreeSummarize(),
    }
)

############## Vis all pipeline #################
from pyvis.network import Network

net = Network(notebook=True, cdn_resources="in_line", directed=True)
net.from_nx(p.clean_dag)
net.write_html("output/pipeline_dag.html")

############## Using Sentence Window Retrieval #################
# Compare with  Replacing the Output Module RAG / + mean add that line, - mean delet that line

+from llama_index.core.node_parser import SentenceWindowNodeParser
+from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor

-node_parser = SentenceSplitter()
+node_parser = SentenceWindowNodeParser.from_defaults(
+    window_size=3,
+    window_metadata_key="window",
+    original_text_metadata_key="original_text",
+)
+meta_replacer = MetadataReplacementPostProcessor(target_metadata_key="window")
p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "input": InputComponent(),
        "query_rewriter": query_rewriter,
        "retriever": retriever,
+        "meta_replacer": meta_replacer,
        "reranker": reranker,
        "output": TreeSummarize(),
    }
)
p.add_link("input", "query_rewriter")
p.add_link("query_rewriter", "retriever")
+p.add_link("retriever", "meta_replacer")
p.add_link("input", "reranker", dest_key="query_str")
-p.add_link("retriever", "reranker", dest_key="nodes")
+p.add_link("meta_replacer", "reranker", dest_key="nodes")
p.add_link("input", "output", dest_key="query_str")
p.add_link("reranker", "output", dest_key="nodes")

output, intermediates = p.run_with_intermediates(input=question)
retriever_output = intermediates["retriever"].outputs["output"]
print(f"retriever output:")
for node in retriever_output:
    print(f"node: {node.text}\n")
meta_replacer_output = intermediates["meta_replacer"].outputs["nodes"]
print(f"meta_replacer output:")
for node in meta_replacer_output:
    print(f"node: {node.text}\n")


############## Using Adding an Evaluation Module #################
# Compare with Using Sentence Window Retrieval RAG / + mean add that line, - mean delet that line

from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate
from datasets import Dataset
from llama_index.core.query_pipeline import CustomQueryComponent
from typing import Dict, Any

metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

class RagasComponent(CustomQueryComponent):
    """Ragas evaluation component."""
    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        return input
    
    @property
    def _input_keys(self) -> set:
        """Input keys dict."""
        return {"question", "nodes", "answer", "ground_truth", }

    @property
    def _output_keys(self) -> set:
        return {"answer", "source_nodes", "evaluation"}

    def _run_component(self, **kwargs) -> Dict[str, Any]:
        """Run the component."""
        question, ground_truth, nodes, answer = kwargs.values()
        data = {
            "question": [question],
            "contexts": [[n.get_content() for n in nodes]],
            "answer": [str(answer)],
            "ground_truth": [ground_truth],
        }
        dataset = Dataset.from_dict(data)
        evaluation = evaluate(dataset, metrics)
        return {"answer": str(answer), "source_nodes": nodes, "evaluation": evaluation}

+evaluator = RagasComponent()
p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "input": InputComponent(),
        "query_rewriter": query_rewriter,
        "retriever": retriever,
        "meta_replacer": meta_replacer,
        "reranker": reranker,
        "output": TreeSummarize(),
+        "evaluator": evaluator,
    }
)
-p.add_link("input", "query_rewriter")
+p.add_link("input", "query_rewriter", src_key="input")
p.add_link("query_rewriter", "retriever")
p.add_link("retriever", "meta_replacer")
-p.add_link("input", "reranker", dest_key="query_str")
+p.add_link("input", "reranker", src_key="input", dest_key="query_str")
p.add_link("meta_replacer", "reranker", dest_key="nodes")
-p.add_link("input", "output", dest_key="query_str")
+p.add_link("input", "output", src_key="input", dest_key="query_str")
p.add_link("reranker", "output", dest_key="nodes")
+p.add_link("input", "evaluator", src_key="input", dest_key="question")
+p.add_link("input", "evaluator", src_key="ground_truth", dest_key="ground_truth")
+p.add_link("reranker", "evaluator", dest_key="nodes")
+p.add_link("output", "evaluator", dest_key="answer")

question = "Which two members of the Avengers created Ultron?"
ground_truth = "Tony Stark (Iron Man) and Bruce Banner (The Hulk)."
output = p.run(input=question, ground_truth=ground_truth)
print(f"answer: {output['answer']}")
print(f"evaluation: {output['evaluation']}")


