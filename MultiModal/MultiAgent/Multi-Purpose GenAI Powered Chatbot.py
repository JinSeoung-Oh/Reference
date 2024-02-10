#From https://towardsdatascience.com/building-a-multi-purpose-genai-powered-chatbot-db20f1f81d90

import boto3
import sagemaker
import time
from tim import gmtime, strftime

## See : https://aws.amazon.com/ko/getting-started/hands-on/machine-learning-tutorial-deploy-model-to-real-time-inference-endpoint/
epc_name = f"{model_name}-endpoint-config"

#Setup
client = boto3.client(service_name="sagemaker")
runtime = boto3.client(service_name="sagemaker-runtime")
boto_session = boto3.session.Session()
s3 = boto_session.resource('s3')
region = boto_session.region_name
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.defualt_bucket()
role = sagemaker.get_execution_role()

#Client setup
s3_client = boto3.client("s3")
sm_client = boto3.client("sagemaker")
smr_client = boto3.client("sagemaker-runtime")

#Container Parameters, increase health check for LLM
variant_name = "AllTraffic"
instance_type = "ml.g5.12xlarge" #4 GPUs available per instance
model_data_download_timeout_in_seconds = 3600
container_startup_health_check_timeout_in_seconds = 3600

initial_instance_count = 1
max_instance_count = 2

## Endpoint config creation
endpoint_config_response = client.create_endpoint_config(
  EndpointConfigName=epc_name,
  ExecutionRoleArn=role,
  ProductionVariants=[
        {
            "VariantName": variant_name,
            "InstanceType": instance_type,
            "InitialInstanceCount": 1,
            "ModelDataDownloadTimeoutInSeconds": model_data_download_timeout_in_seconds,
            "ContainerStartupHealthCheckTimeoutInSeconds": container_startup_health_check_timeout_in_seconds,
            "ManagedInstanceScaling": {
                "Status": "ENABLED",
                "MinInstanceCount": initial_instance_count,
                "MaxInstanceCount": max_instance_count,
            },
            # can set to least outstanding or random: https://aws.amazon.com/blogs/machine-learning/minimize-real-time-inference-latency-by-using-amazon-sagemaker-routing-strategies/
            "RoutingConfig": {"RoutingStrategy": "LEAST_OUTSTANDING_REQUESTS"},
        }
    ],
)

## Endpoint Creation
endpoint_name = "ic-ep-chatbot" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
create_endpoint_response = client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=epc_name,
)

engine = MPI
option.model_id = TheBloke/Llama-2-7B-Chat-fp16
option.task = text-generation
option.trust_remote_code=true
option.tensor_parallel_degree=1
option.max_rolling_batch_size=32
option.rolling_batch=lmi-dist
option.dtype=fp16

##################################
%%sh
# create tarball
mkdir mymodel
rm mymodel.tar.gz
mv serving.properties mymodel/
mv model.py mymodel/
tar czvf mymodel.tar.gz mymodel/
rm -rf mymodel
###################################

image_uri = sagemaker.image_uris.retrieve(
  freamework = "djl-deepspeed",
  region = sagemaker_session.boto_session.region_name,
  version= "0.26.0"
)
  
## create sagemaker model object
from sagemaker.utils import name_from_base
llama_model_name = name_from_base(f"llama-7b-chat")
# See : https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker/client/create_model.html
create_model_response = sm_client.create_model(
    ModelName=llama_model_name,
    ExecutionRoleArn=role,
    PrimaryContainer={"Image": image_uri, "ModelDataUrl": code_artifact},
)
model_arn = create_model_response["ModelArn"]

llama7b_ic_name = "llama7b-chat-ic" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
variant_name = "AllTraffic"

## llama inference component reaction
create_llama_ic_response = sm_client.create_inference_component(
    InferenceComponentName=llama7b_ic_name,
    EndpointName=endpoint_name,
    VariantName=variant_name,
    Specification={
        "ModelName": llama_model_name,
        "ComputeResourceRequirements": {
            # need just one GPU for llama 7b chat
            "NumberOfAcceleratorDevicesRequired": 1,
            "NumberOfCpuCoresRequired": 1,
            "MinMemoryRequiredInMb": 1024,
        },
    },
    # can setup autoscaling for copies, each copy will retain the hardware you have allocated
    RuntimeConfig={"CopyCount": 1},
)

import json
content_type = "application/json"
chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I am software engineer looking to learn more about machine learning."},
]

payload = {"chat": chat, "parameters": {"max_tokens":256, "do_sample": True}}
response = smr_client.invoke_endpoint(
    EndpointName=endpoint_name,
    InferenceComponentName=llama7b_ic_name, #specify IC name
    ContentType=content_type,
    Body=json.dumps(payload),
    )
result = json.loads(response['Body'].read().decode())

## BART Summarization Inference Component Creation
bart_model_name = name_from_base(f"bart-summarization")
hf_transformers_image_uri = '763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04'
env = {'HF_MODEL_ID': 'knkarthick/MEETING_SUMMARY',
      'HF_TASK':'summarization',
      'SAGEMAKER_CONTAINER_LOG_LEVEL':'20',
      'SAGEMAKER_REGION':'us-east-1'}
create_model_response = sm_client.create_model(
    ModelName=bart_model_name,
    ExecutionRoleArn=role,
    # in this case no model data point directly towards HF Hub
    PrimaryContainer={"Image": hf_transformers_image_uri, 
                      "Environment": env},
)
model_arn = create_model_response["ModelArn"]
        
bart_ic_name = "bart-summarization-ic" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
variant_name = "AllTraffic"

# BART inference component reaction
create_bart_ic_response = sm_client.create_inference_component(
    InferenceComponentName=bart_ic_name,
    EndpointName=endpoint_name,
    VariantName=variant_name,
    Specification={
        "ModelName": bart_model_name,
        "ComputeResourceRequirements": {
            # will reserve one GPU
            "NumberOfAcceleratorDevicesRequired": 1,
            "NumberOfCpuCoresRequired": 8,
            "MinMemoryRequiredInMb": 1024,
        },
    },
    # can setup autoscaling for copies, each copy will retain the hardware you have allocated
    RuntimeConfig={"CopyCount": 1},
)

##  Streamlit UI Creation & Demo  
import os
import streamlit as st
from streamlit_chat import message

smr_client = boto3.client("sagemaker-runtime")
os.environ["endpoint_name"] = "enter endpoint name here"
os.environ["llama_ic_name"] = "enter llama IC name here"
os.environ["bart_ic_name"] = "enter bart IC name here"

# session state variables store user and model inputs
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# clear button
clear_button = st.sidebar.button("Clear Conversation", key="clear")
summarize_button = st.sidebar.button("Summarize Conversation", key="summarize")
# reset everything upon clear
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['chat_history'] = []

 if submit_button and user_input:
            st.session_state['past'].append(user_input)
            model_input = {"role": "user", "content": user_input}
            st.session_state['chat_history'].append(model_input)
            payload = {"chat": st.session_state['chat_history'], "parameters": {"max_tokens":400, "do_sample": True,
                                                                                "maxOutputTokens": 2000}}
            # invoke llama
            response = smr_client.invoke_endpoint(
                EndpointName=os.environ.get("endpoint_name"),
                InferenceComponentName=os.environ.get("llama_ic_name"), #specify IC name
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            full_output = json.loads(response['Body'].read().decode())
            print(full_output)
            display_output = full_output['content']
            print(display_output)
            st.session_state['chat_history'].append(full_output)
            st.session_state['generated'].append(display_output)

# for summarization
if summarize_button:
    st.header("Summary")
    st.write("Generating summary....")
    chat_history = st.session_state['chat_history']
    text = ''''''
    for resp in chat_history:
        if resp['role'] == "user":
            text += f"Ram: {resp['content']}\n"
        elif resp['role'] == "assistant":
            text += f"AI: {resp['content']}\n"
    summary_payload = {"inputs": text}
    summary_response = smr_client.invoke_endpoint(
        EndpointName=os.environ.get("endpoint_name"),
        InferenceComponentName=os.environ.get("bart_ic_name"), #specify IC name
        ContentType="application/json",
        Body=json.dumps(summary_payload),
    )
    summary_result = json.loads(summary_response['Body'].read().decode())
    summary = summary_result[0]['summary_text']
    st.write(summary)

