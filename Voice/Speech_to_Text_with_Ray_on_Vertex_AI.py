## From https://medium.com/google-cloud/whisper-goes-to-wall-street-scaling-speech-to-text-with-ray-on-vertex-ai-part-i-04c8ceb180c0

"""
# Requirement file
./requirements.txt
--extra-index-url https://download.pytorch.org/whl/cu118
ipython==8.22.2
torch==2.2.1
torchaudio==2.2.1
ray==2.10.0
ray[data]==2.10.0
ray[train]==2.10.0
datasets==2.17.0
transformers==4.39.0
evaluate==0.4.1
jiwer==3.0.0
accelerate==0.28.0
deepspeed==0.14.0
soundfile==0.12.1
librosa==0.10.0
pyarrow==15.0.2
fsspec==2023.10.0
gcsfs==2023.10.0
etils==1.7.0

# Dockerfile file
./Dockerfile
FROM us-docker.pkg.dev/vertex-ai/training/ray-gpu.2-9.py310:latest
ENV PIP_ROOT_USER_ACTION=ignore
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create a Docker image repository
! gcloud artifacts repositories create your-repo --repository-format=docker --location='your-region' --description="Tutorial repository"

# Build the image
! gcloud builds submit --region='your-region' --tag=your-region-docker.pkg.dev/your-project/your-repo/train --machine-type=your-build-machine --timeout=3600 ./

"""

from vertex_ray import NodeImages, Resources

HEAD_NODE_TYPE = Resources(
    machine_type='your-head-machine-type',
    node_count=1
)

WORKER_NODE_TYPES = [
    Resources(
        machine_type='your-worker-machine-type',
        node_count=3,
        accelerator_type='your-accelerator-type',
        accelerator_count=1
    )
]

CUSTOM_IMAGES = NodeImages(
    head='your-region-docker.pkg.dev/your-project/your-repo/train',
    worker='your-region-docker.pkg.dev/your-project/your-repo/train',
)

# create cluster
ray_cluster_name = vertex_ray.create_ray_cluster(
    head_node_type=HEAD_NODE_TYPE,
    worker_node_types=WORKER_NODE_TYPES,
    custom_images=CUSTOM_IMAGES,
    cluster_name=CLUSTER_NAME,
)



## for taining
import ray.train.huggingface.transformers
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

def main():

    # Initialize ray session
    ray.init()

    # Training config
    train_loop_config = {...}
    scaling_config = ScalingConfig(...)
    run_config = RunConfig(checkpoint_config=CheckpointConfig(...)

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_loop_config,
        run_config=run_config,
        scaling_config=scaling_config
    )
    result = trainer.fit()
    ray.shutdown()

# Initiate the client
client = JobSubmissionClient(
    address="vertex_ray://{}".format(ray_cluster.dashboard_address)
)

# Define train entrypoint
train_entrypoint=f"python3 trainer.py --experiment-name=your-experiment-name --num-workers=3 --use-gpu"
train_runtime_env={
        "working_dir": "./train_script_folder",
        "env_vars": {
            "HF_TOKEN": HF_TOKEN,
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "3"},
    }

# Submit the job
job_status = submit_and_monitor_job(
    client=client,
    submission_id=train_submission_id,
    entrypoint=train_entrypoint,
    runtime_env=train_runtime_env
)

# Get log in Vertex AI tensorboard
vertex_ai.upload_tb_log(
    tensorboard_id=tensorboard.name,
    tensorboard_experiment_name="your-tb-experiment",
    logdir="gs://your-bucket-name/your-log-dir")


from ray.tune import ExperimentAnalysis

experiment_analysis = ExperimentAnalysis('gs://your-bucket-name/experiments/train')
log_path = experiment_analysis.get_best_trial(metric="test_wer", mode="min")
model_checkpoint = experiment_analysis.get_best_checkpoint(log_path, metric="eval_wer", mode="min")


import evaluate 

# Load the WER metric
wer = evaluate.load("wer")

# Compute the metrics with base and tuned model
whisper_wer_metric = wer.compute(predictions=[trascriptions['whisper']],
                                 references=[trascriptions['reference']])

tuned_whisper_wer_metric = wer.compute(predictions=[trascriptions['tuned_whisper']],
                                       references=[trascriptions['reference']])

# Calculate error % difference 
error_difference = ((tuned_whisper_wer_metric - whisper_wer_metric) / whisper_wer_metric) * 100
print("Base Whisper vs Tuned Whisper - WER differences")
print(f"WER difference: {error_difference:.3f}%")

# WER difference: -31.345%





