## From https://medium.com/@ingridwickstevens/quantization-of-llms-with-llama-cpp-9bbf59deda35

### Setting ###
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

### Donwload LLm model for quantization ###
git clone https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO nous-hermes-2-mistral-7B-DPO
mv nous-hermes-2-mistral-7B-DPO models/

### Convert the Model to a GGML FP16 format ###
see : https://medium.com/@tubelwj/introduction-to-ai-model-quantization-formats-dc643bfc335c
python3 convert.py models/nous-hermes-2-mistral-7B-DPO/
  
### quantize the model to 4-bits (using Q4_K_M method)
./quantize ./models/nous-hermes-2-mistral-7B-DPO/ggml-model-f16.gguf ./models/nous-hermes-2-mistral-7B-DPO/ggml-model-Q4_K_M.gguf Q4_K_M

### quantize the model to 3-bits (using Q3_K_M method)
./quantize ./models/nous-hermes-2-mistral-7B-DPO/ggml-model-f16.gguf ./models/nous-hermes-2-mistral-7B-DPO/ggml-model-Q3_K_M.gguf Q3_K_M

### quantize the model to 5-bits (using Q5_K_M method)
./quantize ./models/nous-hermes-2-mistral-7B-DPO/ggml-model-f16.gguf ./models/nous-hermes-2-mistral-7B-DPO/ggml-model-Q5_K_M.gguf Q5_K_M

### quantize the model to 2-bits (using Q2_K method)
./quantize ./models/nous-hermes-2-mistral-7B-DPO/ggml-model-f16.gguf ./models/nous-hermes-2-mistral-7B-DPO/ggml-model-Q2_K.gguf Q2_K

  
### Batched Bench ###
# Batched bench benchmarks the batched decoding performance of the llama.cpp library.
./batched-bench ./models/nous-hermes-2-mistral-7B-DPO/ggml-model-f16.gguf 2048 0 999 128,256,512 128,256 1,2,4,8,16,32
./batched-bench ./models/nous-hermes-2-mistral-7B-DPO/ggml-model-Q4_K_M.gguf 2048 0 999 128,256,512 128,256 1,2,4,8,16,32

  
### Evaluating Perplexity ###
# Calculate the perplexity of ggml-model-Q2_K.gguf
./perplexity -m ./models/nous-hermes-2-mistral-7B-DPO/ggml-model-Q2_K.gguf -f /Users/ingrid/Downloads/test-00000-of-00001.parquet


### Run the quantized model ###
# start inference on a gguf model
./main -m ./models/nous-hermes-2-mistral-7B-DPO/ggml-model-Q4_K_M.gguf -n 128
