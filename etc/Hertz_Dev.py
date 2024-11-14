### From https://levelup.gitconnected.com/breaking-latency-barriers-how-hertz-dev-makes-real-time-conversational-ai-with-open-source-power-ff96babb6ed8

"""
conda install gcc_linux-64 gxx_linux-64 -y
sudo apt-get install libportaudio2
git clone https://github.com/Standard-Intelligence/hertz-dev.git && cd hertz-dev
pip install -r requirements.txt
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
"""

%load_ext autoreload
%autoreload 2
import torch as T
import torchaudio
from utils import load_ckpt, print_colored
from tokenizer import make_tokenizer
from model import get_hertz_dev_config
import matplotlib.pyplot as plt
from IPython.display import Audio, display

# Set device for computation
device = 'cuda' if T.cuda.is_available() else 'cpu'
T.cuda.set_device(0)
print_colored(f"Using device: {device}", "grey")

# Initialize tokenizer and configure model settings
audio_tokenizer = make_tokenizer(device)
TWO_SPEAKER = False
USE_PURE_AUDIO_ABLATION = False
model_config = get_hertz_dev_config(is_split=TWO_SPEAKER, use_pure_audio_ablation=USE_PURE_AUDIO_ABLATION)
generator = model_config().eval().to(T.bfloat16).to(device)

def load_and_preprocess_audio(audio_path):
    print_colored("Loading and preprocessing audio...", "blue", bold=True)
    audio_tensor, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio_tensor = resampler(audio_tensor)
    audio_tensor = audio_tensor.mean(dim=0).unsqueeze(0) if audio_tensor.shape[0] == 2 else audio_tensor
    return audio_tensor.unsqueeze(0)

def get_completion(encoded_prompt_audio, prompt_len, gen_len=None):
    with T.autocast(device_type='cuda', dtype=T.bfloat16):
        completed_audio_batch = generator.completion(
            encoded_prompt_audio, temps=(.8, (0.5, 0.1)), use_cache=True, gen_len=gen_len)
        decoded_completion = audio_tokenizer.data_from_latent(completed_audio_batch.bfloat16())
    audio_tensor = decoded_completion.cpu().squeeze()
    return audio_

num_completions = 10
for _ in range(num_completions):
    completion = get_completion(encoded_prompt_audio, prompt_len, gen_len=20*8) 
    display_audio(completion)

def display_audio(audio_tensor):
    plt.figure(figsize=(4, 1))
    plt.plot(audio_tensor.numpy()[0], linewidth=0.5)
    plt.axis('off')
    plt.show()
    display(Audio(audio_tensor.numpy(), rate=16000))
