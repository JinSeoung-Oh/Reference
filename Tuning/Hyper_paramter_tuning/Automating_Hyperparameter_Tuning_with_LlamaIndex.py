## https://medium.com/gitconnected/automating-hyperparameter-tuning-with-llamaindex-72fdd68e3b90

### Step 1 : Load document ###
### Step 2 : Generate evaluation question/answer pairs ###

from llama_index.evaluation import (
    DatasetGenerator,
    QueryResponseDataset,
)

eval_service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-1106-preview"))

# load eval question/answer dataset from JSON file if exists
if os.path.exists("data/eval_qr_dataset.json"):
    eval_dataset = QueryResponseDataset.from_json("data/eval_qr_dataset.json")
else:
    # construct dataset_generator
    dataset_generator = DatasetGenerator(
        nodes[:8],
        service_context=eval_service_context,
        show_progress=True,
        num_questions_per_chunk=2,
    )

    # generate queries and responses
    eval_dataset = dataset_generator.generate_dataset_from_nodes()

    # save the dataset into a file
    eval_dataset.save_json("data/eval_qr_dataset.json")

import json

# Load dataset from JSON file
with open("data/eval_qr_dataset.json", "r") as file:
    eval_dataset_content = json.load(file)

# Print the content in JSON format
json_str = json.dumps(eval_dataset_content, indent=2)  # indent for pretty printing
eval_qs = eval_dataset.questions
ref_response_strs = [r for (_, r) in eval_dataset.qr_pairs]

### Step 3. Build index, query engine, and gather parameters ###
def _build_index(chunk_size, docs):
    index_out_path = f"./storage_{chunk_size}"
    if not os.path.exists(index_out_path):
        Path(index_out_path).mkdir(parents=True, exist_ok=True)
        
        # Using the new flattened interface for node parsing
        node_parser = SentenceSplitter(chunk_size=chunk_size)
        nodes = node_parser(docs)

        # build index
        index = VectorStoreIndex(nodes)

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

# contains the parameters that need to be tuned
param_dict = {"chunk_size": [256, 512, 1024], "top_k": [1, 2, 5]}

# contains parameters remaining fixed across all runs of the tuning process
fixed_param_dict = {
    "docs": documents,
    "eval_qs": eval_qs,
    "ref_response_strs": ref_response_strs,
}

### Step 4. Define EDD to measure the score for each parameter combination ###
def _get_eval_batch_runner_semantic_similarity():
    eval_service_context = ServiceContext.from_defaults(
        llm=OpenAI(model="gpt-4-1106-preview")
    )
    evaluator_s = SemanticSimilarityEvaluator(
        service_context=eval_service_context
    )
    eval_batch_runner = BatchEvalRunner(
        {"semantic_similarity": evaluator_s}, workers=2, show_progress=True
    )

    return eval_batch_runner

def objective_function_semantic_similarity(params_dict):
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
    eval_batch_runner = _get_eval_batch_runner_semantic_similarity()
    eval_results = eval_batch_runner.evaluate_responses(
        eval_qs, responses=pred_response_objs, reference=ref_response_strs
    )

    # get semantic similarity metric
    mean_score = np.array(
        [r.score for r in eval_results["semantic_similarity"]]
    ).mean()

    return RunResult(score=mean_score, params=params_dict)

### Step 5: Run ParamTuner ###
from llama_index.param_tuner import ParamTuner

param_tuner = ParamTuner(
    param_fn=objective_function_semantic_similarity,
    param_dict=param_dict,
    fixed_param_dict=fixed_param_dict,
    show_progress=True,
)

results = param_tuner.tune()

best_result = results.best_run_result
best_top_k = results.best_run_result.params["top_k"]
best_chunk_size = results.best_run_result.params["chunk_size"]

print("")
print(f"Semantic Similarity Score: {best_result.score}")
print(f"Top-k: {best_top_k}")
print(f"Chunk size: {best_chunk_size}")

import matplotlib.pyplot as plt
import numpy as np

# use the following lists to store data
scores = []
top_k_chunk_combos = []

for result in results.run_results:
    p = result.params
    score = result.score
    top_k = p["top_k"]
    chunk_size = p["chunk_size"]

    # Combine top_k and chunk_size for x-axis label
    top_k_chunk_combo = f"{top_k}_{chunk_size}"

    # Append values to the lists
    scores.append(score)
    top_k_chunk_combos.append(top_k_chunk_combo)

    print(f"Score: {score}, Top_k: {top_k}, Chunk_size: {chunk_size}")

# Create a bar chart with log scale for the y-axis
plt.bar(top_k_chunk_combos, scores, align='center', alpha=0.7)
plt.yscale('log')  # Set log scale for the y-axis
plt.xlabel('Top_k and Chunk_size')
plt.ylabel('Score (log scale)')
plt.title('Score vs Top_k and Chunk_size')
plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better visibility
plt.show()





