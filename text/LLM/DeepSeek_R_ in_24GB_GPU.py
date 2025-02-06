### From https://medium.com/@sonamshrish1618/deepseek-r1-in-24gb-gpu-dynamic-quantization-by-unsloth-ai-for-a-671b-parameter-model-6b0cf85f9065

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake . -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build build --config Release -j --clean-first --target llama-quantize llama-cli llama-gguf-split

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="unsloth/DeepSeek-R1-GGUF",
    local_dir="DeepSeek-R1-GGUF",
    allow_patterns=["*UD-IQ1_S*"],  # For the 1.58-bit version
)

n_offload = floor((GPU_VRAM_GB / Model_FileSize_GB) * (Total_Layers - 4))

./build/bin/llama-cli \
    --model DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf \
    --cache-type-k q4_0 \
    --threads 16 \
    --prio 2 \
    --temp 0.6 \
    --ctx-size 8192 \
    --seed 3407 \
    --n-gpu-layers 7 \
    -no-cnv \
    --prompt "<|User|>Create a Flappy Bird game in Python.<|Assistant|>"
