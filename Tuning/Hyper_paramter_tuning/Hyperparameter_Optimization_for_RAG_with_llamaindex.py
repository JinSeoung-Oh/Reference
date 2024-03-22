## From https://docs.llamaindex.ai/en/stable/examples/param_optimizer/param_optimizer.html

%pip install llama-index-llms-openai
%pip install llama-index-embeddings-openai
%pip install llama-index-readers-file pymupdf
%pip install llama-index-experimental-param-tuner

!pip install llama-index llama-hub
!mkdir data && wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"

import nest_asyncio

nest_asyncio.apply()

from pathlib import Path
from llama_index.readers.file import PDFReader
from llama_index.readers.file import UnstructuredReader
from llama_index.readers.file import PyMuPDFReader

loader = PDFReader()
docs0 = loader.load_data(file=Path("./data/llama2.pdf"))

from llama_index.core import Document

doc_text = "\n\n".join([d.get_content() for d in docs0])
docs = [Document(text=doc_text)]

from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import IndexNode

!wget "https://www.dropbox.com/scl/fi/fh9vsmmm8vu0j50l3ss38/llama2_eval_qr_dataset.json?rlkey=kkoaez7aqeb4z25gzc06ak6kb&dl=1" -O data/llama2_eval_qr_dataset.json

from llama_index.core.evaluation import QueryResponseDataset

# optional
eval_dataset = QueryResponseDataset.from_json(
    "data/llama2_eval_qr_dataset.json"
)

eval_qs = eval_dataset.questions
ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]

### Define Objective Function + Parameters ###
from llama_index.core import (
    VectorStoreIndex,
    load_index_from_storage,
    StorageContext,
)
from llama_index.experimental.param_tuner import ParamTuner
from llama_index.core.param_tuner.base import TunedResult, RunResult
from llama_index.core.evaluation.eval_utils import (
    get_responses,
    aget_responses,
)
from llama_index.core.evaluation import (
    SemanticSimilarityEvaluator,
    BatchEvalRunner,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

import os
import numpy as np
from pathlib import Path

def _build_index(chunk_size, docs):
    index_out_path = f"./storage_{chunk_size}"
    if not os.path.exists(index_out_path):
        Path(index_out_path).mkdir(parents=True, exist_ok=True)
        # parse docs
        node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
        base_nodes = node_parser.get_nodes_from_documents(docs)

        # build index
        index = VectorStoreIndex(base_nodes)
        # save index to disk
        index.storage_context.persist(index_out_path)
    else:
        # rebuild storage context
        storage_context = StorageContext.from_defaults(
            persist_dir=index_out_path
        )
        # load index
        index = load_index_from_storage(
            storage_context,
        )
    return index

def _get_eval_batch_runner():
    evaluator_s = SemanticSimilarityEvaluator(embed_model=OpenAIEmbedding())
    eval_batch_runner = BatchEvalRunner(
        {"semantic_similarity": evaluator_s}, workers=2, show_progress=True
    )

    return eval_batch_runner

def objective_function(params_dict):
    chunk_size = params_dict["chunk_size"]
    docs = params_dict["docs"]
    top_k = params_dict["top_k"]
    eval_qs = params_dict["eval_qs"]
    ref_response_strs = params_dict["ref_response_strs"]

    # build index
    index = _build_index(chunk_size, docs)

    # query engine
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    # get predicted responses
    pred_response_objs = get_responses(
        eval_qs, query_engine, show_progress=True
    )

    # run evaluator
    # NOTE: can uncomment other evaluators
    eval_batch_runner = _get_eval_batch_runner()
    eval_results = eval_batch_runner.evaluate_responses(
        eval_qs, responses=pred_response_objs, reference=ref_response_strs
    )

    # get semantic similarity metric
    mean_score = np.array(
        [r.score for r in eval_results["semantic_similarity"]]
    ).mean()

    return RunResult(score=mean_score, params=params_dict)

async def aobjective_function(params_dict):
    chunk_size = params_dict["chunk_size"]
    docs = params_dict["docs"]
    top_k = params_dict["top_k"]
    eval_qs = params_dict["eval_qs"]
    ref_response_strs = params_dict["ref_response_strs"]

    # build index
    index = _build_index(chunk_size, docs)

    # query engine
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    # get predicted responses
    pred_response_objs = await aget_responses(
        eval_qs, query_engine, show_progress=True
    )

    # run evaluator
    # NOTE: can uncomment other evaluators
    eval_batch_runner = _get_eval_batch_runner()
    eval_results = await eval_batch_runner.aevaluate_responses(
        eval_qs, responses=pred_response_objs, reference=ref_response_strs
    )

    # get semantic similarity metric
    mean_score = np.array(
        [r.score for r in eval_results["semantic_similarity"]]
    ).mean()

    return RunResult(score=mean_score, params=params_dict)

param_dict = {"chunk_size": [256, 512, 1024], "top_k": [1, 2, 5]}
# param_dict = {
#     "chunk_size": [256],
#     "top_k": [1]
# }
fixed_param_dict = {
    "docs": docs,
    "eval_qs": eval_qs[:10],
    "ref_response_strs": ref_response_strs[:10],
}


### Run ParamTuner (default) ###
from llama_index.experimental.param_tuner import ParamTuner

param_tuner = ParamTuner(
    param_fn=objective_function,
    param_dict=param_dict,
    fixed_param_dict=fixed_param_dict,
    show_progress=True,
)
results = param_tuner.tune()

best_result = results.best_run_result
best_top_k = results.best_run_result.params["top_k"]
best_chunk_size = results.best_run_result.params["chunk_size"]
print(f"Score: {best_result.score}")
print(f"Top-k: {best_top_k}")
print(f"Chunk size: {best_chunk_size}")

test_idx = 6
p = results.run_results[test_idx].params
(results.run_results[test_idx].score, p["top_k"], p["chunk_size"])

### Run ParamTuner (Async) ###
from llama_index.experimental.param_tuner import AsyncParamTuner
aparam_tuner = AsyncParamTuner(
    aparam_fn=aobjective_function,
    param_dict=param_dict,
    fixed_param_dict=fixed_param_dict,
    num_workers=2,
    show_progress=True,
)
results = await aparam_tuner.atune()
best_result = results.best_run_result
best_top_k = results.best_run_result.params["top_k"]
best_chunk_size = results.best_run_result.params["chunk_size"]
print(f"Score: {best_result.score}")
print(f"Top-k: {best_top_k}")
print(f"Chunk size: {best_chunk_size}")

### Run ParamTuner (Ray Tune) ###
from llama_index.experimental.param_tuner import RayTuneParamTuner
param_tuner = RayTuneParamTuner(
    param_fn=objective_function,
    param_dict=param_dict,
    fixed_param_dict=fixed_param_dict,
    run_config_dict={"storage_path": "/tmp/custom/ray_tune", "name": "my_exp"},
)
results = param_tuner.tune()
results.best_run_result.params.keys()
results.best_idx
best_result = results.best_run_result

best_top_k = results.best_run_result.params["top_k"]
best_chunk_size = results.best_run_result.params["chunk_size"]
print(f"Score: {best_result.score}")
print(f"Top-k: {best_top_k}")
print(f"Chunk size: {best_chunk_size}")







