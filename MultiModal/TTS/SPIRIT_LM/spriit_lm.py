### From https://ai.gopubby.com/meta-spirit-lm-a-complete-guide-to-multimodal-ai-for-text-and-speech-generation-ed0af74bc950

"""
SPIRIT-LM, Meta AI’s multimodal language model, exemplifies a powerful integration of text and speech, allowing for more natural and versatile language interactions.
Through a unique token-based architecture, SPIRIT-LM enables tasks that blend both modalities—like text-to-speech (TTS) and speech-to-text (STT)—while also retaining 
expressive qualities in speech, such as pitch and tone. 
This innovation not only aligns with the trends in large language models but also pushes them further into the realm of interactive, human-like communication.

SPIRIT-LM’s multimodal nature hinges on its ability to treat text and speech seamlessly within a unified model. 
-1. Multimodal Training
    SPIRIT-LM is trained on text-only, speech-only, and mixed sequences, adapting it to switch fluidly between modalities. 
    This versatile training allows the model to comprehend and generate language across diverse inputs, making it adaptable to a variety of real-world applications.
-2. Interleaved Token System
    By representing text and speech with distinct tokens and using special transition tokens, 
    SPIRIT-LM facilitates smooth switching between modalities within a single interaction. Text encoding uses Byte Pair Encoding (BPE),
    while speech is processed using a pre-processing model like HuBERT, converting spoken content into speech-unit tokens that can be directly integrated with 
    text tokens.
-3. Expressive Speech Generation
    The “Expressive” variant of SPIRIT-LM enriches generated speech with elements like pitch and style, 
    giving voice output a more human touch and creating opportunities for empathetic, context-aware responses in applications such as virtual assistants.

Architecture and Functional Mechanisms
SPIRIT-LM’s architecture is designed to bridge the gap between conventional LLMs and speech models, enhancing its applicability to complex tasks:
-1. Token-Based Integration
    The model uses speech tokens derived from pre-processed speech data, embedding them alongside text tokens within the same processing pipeline. 
    This approach allows SPIRIT-LM to operate in either mode or switch between them based on task requirements.
-2. Multimodal Token Reconstruction
    The decoder is capable of generating output in both text and speech forms, suited for applications that require direct audio output or written responses.
-3. Flexible Next-Token Prediction
    Following the approach of LLMs like GPT, SPIRIT-LM’s decoder predicts the next token from either the text or speech modality, 
    allowing it to maintain context seamlessly as it switches between types of data.

Potential Use Cases and Impact
SPIRIT-LM’s capabilities open up new possibilities in fields where multimodal communication is crucial:
-1. Conversational AI and Virtual Assistants
    SPIRIT-LM’s expressive version allows virtual assistants to interact in a more emotionally resonant way, with nuanced responses that make conversations 
    feel more natural.
-2. Cross-Modal Translation
    For tasks like real-time STT and TTS, SPIRIT-LM is well-equipped to handle both forms of language, providing smooth and 
    coherent translations between spoken and written formats.
-3. Accessibility Tools
    By combining text and speech, SPIRIT-LM can improve transcription services, enhancing accessibility for individuals with visual impairments or other disabilities.
-4. Multilingual Translation Systems
    The model’s text-speech fusion could serve multilingual applications by retaining sentiment and context, 
    ensuring that translations capture subtle language cues across modalities.

Conclusion
SPIRIT-LM’s blend of textual and spoken language in a single model sets a new standard for multimodal AI. 
By building upon the strengths of traditional language models and introducing a robust integration with speech,
Meta AI has created a model with transformative potential across diverse applications, from conversational agents to accessibility tools.
With SPIRIT-LM, the future of AI-assisted communication is one step closer to achieving a truly natural, multimodal user experience.
"""

git clone https://github.com/facebookresearch/spiritlm.git && cd spiritlm
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install gradio tempfile transformers numpy

conda create -n spiritlm-env python=3.9
conda activate spiritlm-env
pip install -e '.[eval]'

cd checkpoints/
mv /path/to/your/downloaded/spiritlm_all_checkpoints.zip .
unzip spiritlm_all_checkpoints.zip
rm spiritlm_all_checkpoints.zip
mv checkpoints/* .
ls -ltr checkpoints/

####### Basic Text Generation
from spiritlm.model.spiritlm_model import Spiritlm, OutputModality, GenerationInput, ContentType
from transformers import GenerationConfig

# Initialize the model
spirit_lm = Spiritlm("spirit-lm-base-7b")

# Generate a text output
output = spirit_lm.generate(
    interleaved_inputs=[
        GenerationInput(
            content="The largest country in the world is",
            content_type=ContentType.TEXT,
        )
    ],
    output_modality=OutputModality.TEXT,
    generation_config=GenerationConfig(
        max_new_tokens=20,
        do_sample=False,
    ),
)

print(output)

####### Speech-to-Text Generation
import IPython.display as ipd

# Load audio file for transcription
ipd.Audio("../audio/7143-88743-0029.flac")

# Generate text from speech input
output = spirit_lm.generate(
    interleaved_inputs=[('speech', "../audio/7143-88743-0029.flac")],
    output_modality=OutputModality.TEXT,
    generation_config=GenerationConfig(
        max_new_tokens=30,
        do_sample=True,
    ),
)

print(output)

######## Text-to-Speech Generation
outputs = spirit_lm.generate(
    interleaved_inputs=[('text', "One of the most beautiful cities in the world is Paris.")],
    output_modality='speech',
    generation_config=GenerationConfig(
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=200,
        do_sample=True,
    ),
)

# Play generated speech
ipd.display(ipd.Audio(outputs[0].content, rate=16_000))

########## Arbitrary Generation
interleaved_outputs = spirit_lm.generate(
    interleaved_inputs=[('speech', "../audio/7143-88743-0029.flac")],
    output_modality='arbitrary',
    generation_config=GenerationConfig(
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=200,
        do_sample=True,
    ),
)

# Display outputs
for output in interleaved_outputs:
    if output.content_type == ContentType.TEXT:
        print(output.content)
    else:
        ipd.display(ipd.Audio(output.content, rate=16_000))

############ Spirit LM Expressive
spirit_lm = Spiritlm("spirit-lm-expressive-7b")

outputs = spirit_lm.generate(
    interleaved_inputs=[('text', "I am so deeply saddened, it feels as if my heart is shattering into a million pieces and I can't hold back the tears that are streaming down my face.")],
    output_modality='speech',
    generation_config=GenerationConfig(
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=200,
        do_sample=True,
    ),
    speaker_id=1,
)
display_outputs(outputs)

outputs = spirit_lm.generate(
    interleaved_inputs=[('text', "Wow!!! Congratulations!!! I'm so excited that")],
    output_modality='speech',
    generation_config=GenerationConfig(
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=200,
        do_sample=True,
    ),
    speaker_id=1,
)
display_outputs(outputs)






