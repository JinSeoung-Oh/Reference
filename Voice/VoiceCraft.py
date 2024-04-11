"""
From https://levelup.gitconnected.com/voicecraft-ai-powered-speech-editing-and-crafting-e9f9a10618fb

VoiceCraft is an innovative neural codec language model designed for both speech editing and zero-shot text-to-speech tasks. 
It boasts state-of-the-art performance in these domains, making it a significant advancement in speech synthesis technology.

The core architecture of VoiceCraft comprises several key components:

1. Encodec Speech Tokenizer
   This module quantizes input speech waveforms into a sequence of discrete tokens using a Residual Vector Quantization (RVQ) approach.
   The output is a matrix representing temporal frames and RVQ codebooks.

2. Token Rearrangement
   A Two-Step Process
     a. Causal Masking
        During training, random spans of tokens are masked and shifted to the end of the sequence,
        allowing the model to condition on both past and future unmasked tokens during generation.
     b. Delayed Stacking
        The rearranged token matrix undergoes a delay pattern based on the codebook index, 
        ensuring that predictions of codebook k at time t are conditioned on the prediction of codebook k-1 from the same timestep.

3. Transformer Decoder
   This component models the rearranged token sequence autoregressively, conditioned on the speech transcript.
   It employs a multi-head self-attention mechanism to capture long-range dependencies and generate high-quality speech outputs.

During speech editing, VoiceCraft follows a series of steps:

1. Alignment
   Original and target transcripts are compared for editing, with word-level forced alignment used to identify token spans.
2. Masking
   Identified token spans are replaced with masks and moved to the end of the sequence.
3. Autoregressive Generation
   Masked spans are generated based on the target transcript and unmasked spans.
4. Waveform Synthesis
   Generated codec tokens are used to synthesize the edited speech waveform.

For zero-shot text-to-speech:

1. Prompt Preparation
   Voice prompt, its transcription, and the target transcript are concatenated.
2. Autoregressive Generation
   Codec tokens corresponding to the target transcript are generated based on the voice prompt and its transcription.
3. Waveform Synthesis
   Generated codec tokens are decoded to synthesize the final speech waveform.
VoiceCraft's mathematical formulations and modeling framework enable efficient token rearrangement and autoregressive prediction, 
contributing to its exceptional performance in speech synthesis tasks. 
This model represents a significant breakthrough in the field, offering advanced capabilities for speech editing and zero-shot text-to-speech applications.
"""

################# prepare enviroment #########################
!apt-get install -y git-core ffmpeg espeak-ng
!pip install -q condacolab

import condacolab
condacolab.install()
condacolab.check()
!echo -e "Grab a cup a coffee and a slice of pizza...\n\n"
!conda install -y -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068 && \
    pip install torch==2.1.0 && \
    pip install tensorboard==2.16.2 && \
    pip install phonemizer==3.2.1 && \
    pip install torchaudio==2.1.0 && \
    pip install datasets==2.16.0 && \
    pip install torchmetrics==0.11.1 && \
    pip install torchvision==0.16.0

!pip install -U git+https://git@github.com/facebookresearch/audiocraft#egg=audiocraft
!git clone https://github.com/jasonppy/VoiceCraft.git

!mfa model download dictionary english_us_arpa && \
mfa model download acoustic english_us_arpa
# simply installing audiocraft breaks due to no config, so move the default into site-packages
%cd /content/VoiceCraft
!git clone https://github.com/facebookresearch/audiocraft.git
!mv audiocraft/config /usr/local/lib/python3.10/site-packages/
!rm -rf audiocraft
##############################################################

import torch
import torchaudio
import os
import numpy as np
import random
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["USER"] = "YOUR_USERNAME" # TODO change this to your username

from data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)

from models import voicecraft

# hyperparameters for inference
left_margin = 0.08
right_margin = 0.08
codec_audio_sr = 16000
codec_sr = 50
top_k = 0
top_p = 0.8
temperature = 1
kvcache = 0
# NOTE: adjust the below three arguments if the generation is not as good
seed = 1 # random seed magic
silence_tokens = [1388,1898,131]
stop_repetition = -1 # if there are long silence in the generated audio, reduce the stop_repetition to 3, 2 or even 1
# what this will do to the model is that the model will run sample_batch_size examples of the same audio, and pick the one that's the shortest
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_everything(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# point to the original file or record the file
# write down the transcript for the file, or run whisper to get the transcript (and you can modify it if it's not accurate), save it as a .txt file
orig_audio = "./demo/84_121550_000074_000000.wav"
orig_transcript = "But when I had approached so near to them The common object, which the sense deceives, Lost not by distance any of its marks,"
# move the audio and transcript to temp folder
temp_folder = "./demo/temp"
os.makedirs(temp_folder, exist_ok=True)
os.system(f"cp {orig_audio} {temp_folder}")
filename = os.path.splitext(orig_audio.split("/")[-1])[0]
with open(f"{temp_folder}/{filename}.txt", "w") as f:
    f.write(orig_transcript)
# run MFA to get the alignment
align_temp = f"{temp_folder}/mfa_alignments"
os.makedirs(align_temp, exist_ok=True)
os.system(f"mfa align -j 1 --clean --output_format csv {temp_folder} english_us_arpa english_us_arpa {align_temp}")
# if it fail, it could be because the audio is too hard for the alignment model, increasing the beam size usually solves the issue
# os.system(f"mfa align -j 1 --clean --output_format csv {temp_folder} english_us_arpa english_us_arpa {align_temp} --beam 1000 --retry_beam 2000")
audio_fn = f"{temp_folder}/{filename}.wav"
transcript_fn = f"{temp_folder}/{filename}.txt"
align_fn = f"{align_temp}/{filename}.csv"
editTypes_set = set(['substitution', 'insertion', 'deletion'])
# propose what do you want the target modified transcript to be
target_transcript = "But when I saw the mirage of the lake in the distance, which the sense deceives, Lost not by distance any of its marks,"
edit_type = "substitution"
assert edit_type in editTypes_set, f"Invalid edit type {edit_type}. Must be one of {editTypes_set}."

# if you want to do a second modification on top of the first one, write down the second modification (target_transcript2, type_of_modification2)
# make sure the two modification do not overlap, if they do, you need to combine them into one modification

# run the script to turn user input to the format that the model can take
from edit_utils import get_span
orig_span, new_span = get_span(orig_transcript, target_transcript, edit_type)
if orig_span[0] > orig_span[1]:
    RuntimeError(f"example {audio_fn} failed")
if orig_span[0] == orig_span[1]:
    orig_span_save = [orig_span[0]]
else:
    orig_span_save = orig_span
if new_span[0] == new_span[1]:
    new_span_save = [new_span[0]]
else:
    new_span_save = new_span

orig_span_save = ",".join([str(item) for item in orig_span_save])
new_span_save = ",".join([str(item) for item in new_span_save])
from inference_speech_editing_scale import get_mask_interval

start, end = get_mask_interval(align_fn, orig_span_save, edit_type)
info = torchaudio.info(audio_fn)
audio_dur = info.num_frames / info.sample_rate
morphed_span = (max(start - left_margin, 1/codec_sr), min(end + right_margin, audio_dur)) # in seconds

# span in codec frames
mask_interval = [[round(morphed_span[0]*codec_sr), round(morphed_span[1]*codec_sr)]]
mask_interval = torch.LongTensor(mask_interval) # [M,2], M==1 for now

# load model, tokenizer, and other necessary files
voicecraft_name="giga330M.pth" # or giga830M.pth, or the newer models at https://huggingface.co/pyp1/VoiceCraft/tree/main
ckpt_fn =f"./pretrained_models/{voicecraft_name}"
encodec_fn = "./pretrained_models/encodec_4cb2048_giga.th"
if not os.path.exists(ckpt_fn):
    os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/{voicecraft_name}\?download\=true")
    os.system(f"mv {voicecraft_name}\?download\=true ./pretrained_models/{voicecraft_name}")
if not os.path.exists(encodec_fn):
    os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th")
    os.system(f"mv encodec_4cb2048_giga.th ./pretrained_models/encodec_4cb2048_giga.th")
ckpt = torch.load(ckpt_fn, map_location="cpu")
model = voicecraft.VoiceCraft(ckpt["config"])
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()

phn2num = ckpt['phn2num']

text_tokenizer = TextTokenizer(backend="espeak")
audio_tokenizer = AudioTokenizer(signature=encodec_fn) # will also put the neural codec model on gpu

# run the model to get the output
from inference_speech_editing_scale import inference_one_sample

decode_config = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 'stop_repetition': stop_repetition, 'kvcache': kvcache, "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr, "silence_tokens": silence_tokens}
orig_audio, new_audio = inference_one_sample(model, ckpt["config"], phn2num, text_tokenizer, audio_tokenizer, audio_fn, target_transcript, mask_interval, device, decode_config)

# save segments for comparison
orig_audio, new_audio = orig_audio[0].cpu(), new_audio[0].cpu()
# logging.info(f"length of the resynthesize orig audio: {orig_audio.shape}")

# display the audio
from IPython.display import Audio
print("original:")
display(Audio(orig_audio, rate=codec_audio_sr))

print("edited:")
display(Audio(new_audio, rate=codec_audio_sr))

################################ VoiceCraft Inference TTS ################################################
!apt-get install -y git-core ffmpeg espeak-ng
!pip install -q condacolab

import condacolab
condacolab.install()
condacolab.check()
!echo -e "Grab a cup a coffee and a slice of pizza...\n\n"
!conda install -y -c conda-forge montreal-forced-aligner=2.2.17 openfst=1.8.2 kaldi=5.5.1068 && \
    pip install torch==2.1.0 && \
    pip install tensorboard==2.16.2 && \
    pip install phonemizer==3.2.1 && \
    pip install torchaudio==2.1.0 && \
    pip install datasets==2.16.0 && \
    pip install torchmetrics==0.11.1 && \
    pip install torchvision==0.16.0

!pip install -U git+https://git@github.com/facebookresearch/audiocraft#egg=audiocraft
!git clone https://github.com/jasonppy/VoiceCraft.git
!mfa model download dictionary english_us_arpa && \
mfa model download acoustic english_us_arpa

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["USER"] = "YOUR_USERNAME" # TODO change this to your username

import torch
import torchaudio
import numpy as np
import random

from VoiceCraft.data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
# simply installing audiocraft breaks due to no config, so move the default into site-packages
%cd /content/VoiceCraft
!git clone https://github.com/facebookresearch/audiocraft.git
!mv audiocraft/config /usr/local/lib/python3.10/site-packages/
!rm -rf audiocraft
# load model, encodec, and phn2num
# # load model, tokenizer, and other necessary files
device = "cuda" if torch.cuda.is_available() else "cpu"
from VoiceCraft.models import voicecraft
# reload voicecraft
import importlib
importlib.reload(voicecraft)
from VoiceCraft.models import voicecraft
voicecraft_name="gigaHalfLibri330M_TTSEnhanced_max16s.pth" # or giga330M.pth, giga830M.pth
ckpt_fn =f"./pretrained_models/{voicecraft_name}"
encodec_fn = "./pretrained_models/encodec_4cb2048_giga.th"
if not os.path.exists(ckpt_fn):
    os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/{voicecraft_name}\?download\=true")
    os.system(f"mv {voicecraft_name}\?download\=true ./pretrained_models/{voicecraft_name}")
if not os.path.exists(encodec_fn):
    os.system(f"wget https://huggingface.co/pyp1/VoiceCraft/resolve/main/encodec_4cb2048_giga.th")
    os.system(f"mv encodec_4cb2048_giga.th ./pretrained_models/encodec_4cb2048_giga.th")

ckpt = torch.load(ckpt_fn, map_location="cpu")
model = voicecraft.VoiceCraft(ckpt["config"])
model.load_state_dict(ckpt["model"])
model.to(device)
model.eval()
phn2num = ckpt['phn2num']
text_tokenizer = TextTokenizer(backend="es2peak")
audio_tokenizer = AudioTokenizer(signature=encodec_fn, device=device) # will also put the neural codec model on gpu
# Prepare your audio
# point to the original audio whose speech you want to clone
# write down the transcript for the file, or run whisper to get the transcript (and you can modify it if it's not accurate), save it as a .txt file
orig_audio = "./demo/84_121550_000074_000000.wav"
orig_transcript = "But when I had approached so near to them The common object, which the sense deceives, Lost not by distance any of its marks,"
# move the audio and transcript to temp folder
temp_folder = "./demo/temp"
os.makedirs(temp_folder, exist_ok=True)
os.system(f"cp {orig_audio} {temp_folder}")
filename = os.path.splitext(orig_audio.split("/")[-1])[0]
with open(f"{temp_folder}/{filename}.txt", "w") as f:
    f.write(orig_transcript)
# run MFA to get the alignment
align_temp = f"{temp_folder}/mfa_alignments"
os.system(f"mfa align -j 1 --clean --output_format csv {temp_folder} english_us_arpa english_us_arpa {align_temp}")
# # if the above fails, it could be because the audio is too hard for the alignment model, increasing the beam size usually solves the issue
# os.system(f"mfa align -j 1 --clean --output_format csv {temp_folder} english_us_arpa english_us_arpa {align_temp} --beam 1000 --retry_beam 2000")
# take a look the csv file in VoiceCraft/demo/temp/mfa_alignment, decide which part of the audio to use as prompt
cut_off_sec = 3.01 # NOTE: according to forced-alignment file demo/temp/mfa_alignments/84_121550_000074_000000.csv, the word "common" stop as 3.01 sec, this should be different for different audio
target_transcript = "But when I had approached so near to them The common I cannot believe that the same model can also do text to speech synthesis as well!"
# NOTE: 3 sec of reference is generally enough for high quality voice cloning, but longer is generally better, try e.g. 3~6 sec.
audio_fn = f"{temp_folder}/{filename}.wav"
info = torchaudio.info(audio_fn)
audio_dur = info.num_frames / info.sample_rate

assert cut_off_sec < audio_dur, f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
prompt_end_frame = int(cut_off_sec * info.sample_rate)

# run the model to get the output
# hyperparameters for inference
codec_audio_sr = 16000
codec_sr = 50
top_k = 0
top_p = 0.8
temperature = 1
silence_tokens=[1388,1898,131]
kvcache = 1 # NOTE if OOM, change this to 0, or try the 330M model

# NOTE adjust the below three arguments if the generation is not as good
stop_repetition = 3 # NOTE if the model generate long silence, reduce the stop_repetition to 3, 2 or even 1
sample_batch_size = 2 # for gigaHalfLibri330M_TTSEnhanced_max16s.pth, 1 or 2 should be fine since the model is trained to do TTS, for the other two models, might need a higher number. NOTE: if the if there are long silence or unnaturally strecthed words, increase sample_batch_size to 5 or higher. What this will do to the model is that the model will run sample_batch_size examples of the same audio, and pick the one that's the shortest. So if the speech rate of the generated is too fast change it to a smaller number.
seed = 1 # change seed if you are still unhappy with the result

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_everything(seed)

decode_config = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature, 'stop_repetition': stop_repetition, 'kvcache': kvcache, "codec_audio_sr": codec_audio_sr, "codec_sr": codec_sr, "silence_tokens": silence_tokens, "sample_batch_size": sample_batch_size}
from VoiceCraft.inference_tts_scale import inference_one_sample
concated_audio, gen_audio = inference_one_sample(model, ckpt["config"], phn2num, text_tokenizer, audio_tokenizer, audio_fn, target_transcript, device, decode_config, prompt_end_frame)

# save segments for comparison
concated_audio, gen_audio = concated_audio[0].cpu(), gen_audio[0].cpu()
# logging.info(f"length of the resynthesize orig audio: {orig_audio.shape}")


# display the audio
from IPython.display import Audio
print("concatenate prompt and generated:")
display(Audio(concated_audio, rate=codec_audio_sr))

print("generated:")
display(Audio(gen_audio, rate=codec_audio_sr))




