## From https://medium.com/@honeyricky1m3/black-forest-labs-flux-the-new-text-to-image-ai-thats-making-waves-2d9d490552f7

!git clone -b totoro3 https://github.com/camenduru/ComfyUI /content/TotoroUI
!pip install -q torchsde einops diffusers accelerate xformers==0.0.27
!apt -y install -qq aria2

!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.sft -d /content/TotoroUI/models/unet -o flux1-schnell.sft
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.sft -d /content/TotoroUI/models/vae -o ae.sft
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/clip_l.safetensors -d /content/TotoroUI/models/clip -o clip_l.safetensors
!aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/t5xxl_fp8_e4m3fn.safetensors -d /content/TotoroUI/models/clip -o t5xxl_fp8_e4m3fn.safetensors

import random
import torch
import numpy as np
from PIL import Image
import nodes
from nodes import NODE_CLASS_MAPPINGS
from totoro_extras import nodes_custom_sampler
from totoro import model_management

DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
RandomNoise = nodes_custom_sampler.NODE_CLASS_MAPPINGS["RandomNoise"]()
BasicGuider = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicGuider"]()
KSamplerSelect = nodes_custom_sampler.NODE_CLASS_MAPPINGS["KSamplerSelect"]()
BasicScheduler = nodes_custom_sampler.NODE_CLASS_MAPPINGS["BasicScheduler"]()
SamplerCustomAdvanced = nodes_custom_sampler.NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()

with torch.inference_mode():
    clip = DualCLIPLoader.load_clip("t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors", "flux")[0]
    unet = UNETLoader.load_unet("flux1-schnell.sft", "fp8_e4m3fn")[0]
    vae = VAELoader.load_vae("ae.sft")[0]

def closestNumber(n, m):
    q = int(n / m)
    n1 = m * q
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    if abs(n - n1) < abs(n - n2):
        return n1
    return n2
with torch.inference_mode():
    prompt = """Setting the Scene:

Prompt: Capture a highly realistic scene of a vibrant forest with real trees, lush green leaves, and colorful flowers. Include real animals such as birds, deer, and squirrels. At the center, position a real hare and a real tortoise facing each other, with curious and interested forest animals gathered around, as if preparing for a race.
The Challenge:

Prompt: Photograph a real tortoise with a serious expression as it appears to challenge a real hare, who looks amused and confident. Focus on capturing their contrasting expressions. In the background, include real forest animals reacting with surprise and curiosity to enhance the natural, live-action feel of the scene.
Start of the Race:

Prompt: Set up a starting line at a large tree in the forest. Capture the moment when the race starts, with the real hare poised to sprint and the real tortoise taking its first slow step. Surround them with real cheering forest animals, creating an authentic and lively atmosphere.
The Hare Dashing Ahead:

Prompt: Photograph the real hare speeding through the forest, leaving a trail of dust behind. Capture the hare in mid-leap, looking back mockingly at the distant tortoise. Use natural elements like blurred trees and plants to convey the hare’s speed, creating a dynamic and lifelike scene.
The Tortoise Plodding Along:

Prompt: Capture the real tortoise moving steadily along a forest path, with eyes focused ahead. Emphasize the tortoise’s slow but determined movement. Include background details of a peaceful forest path with real animals encouraging the tortoise, adding depth and authenticity to the scene.
The Hare Taking a Nap:

Prompt: Photograph the real hare confidently napping under a shady tree, with a relaxed, smug expression. Capture the hare curled up comfortably. Use natural sunlight filtering through the leaves to cast a serene shadow, enhancing the tranquility and realism of the scene.
The Tortoise Passing the Hare:

Prompt: Capture the real tortoise walking past the sleeping hare, focused and undeterred. Focus on the tortoise moving forward, with the hare asleep in the background. Include real forest animals watching silently, some whispering to each other, to create a sense of anticipation and excitement.
The Hare Waking Up:

Prompt: Photograph the real hare waking up, with eyes wide in shock as it sees the tortoise ahead. Focus on the hare's surprised and panicked expression. Include background details of the race path with the finish line in the distance, enhancing the urgency and realism of the moment.
The Hare Running to Catch Up:

Prompt: Capture the real hare sprinting at full speed, trying to catch up with the tortoise. Focus on the hare’s desperate and intense effort. Use natural elements like blurred trees and ground to convey the hare’s speed, creating a dynamic and urgent scene.
The Tortoise Nearing the Finish Line:

Prompt: Photograph the real tortoise slowly but steadily approaching the finish line, with determination in its eyes. Focus on the tortoise with the finish line clearly visible ahead. Include real animals cheering and looking amazed to add excitement and authenticity to the scene.
The Tortoise Winning:

Prompt: Capture the real tortoise crossing the finish line, winning the race, with a look of satisfaction. Focus on the tortoise just as it crosses the finish line. Include background details of the real hare in the background, looking defeated and regretful, while real forest animals cheer and celebrate the tortoise’s victory, creating a triumphant and realistic scene.
The Lesson Learned:

Prompt: Photograph the real hare and the real tortoise interacting in a way that shows mutual respect, with the hare acknowledging its mistake. Focus on the lesson learned. Include background details of real forest animals gathered around, smiling and nodding in approval, creating a harmonious and realistic scene. """
    positive_prompt = prompt
    width = 1024
    height = 1024
    seed = 0
    steps = 4
    sampler_name = "euler"
    scheduler = "simple"

    if seed == 0:
        seed = random.randint(0, 18446744073709551615)
    print(seed)

    cond, pooled = clip.encode_from_tokens(clip.tokenize(positive_prompt), return_pooled=True)
    cond = [[cond, {"pooled_output": pooled}]]
    noise = RandomNoise.get_noise(seed)[0] 
    guider = BasicGuider.get_guider(unet, cond)[0]
    sampler = KSamplerSelect.get_sampler(sampler_name)[0]
    sigmas = BasicScheduler.get_sigmas(unet, scheduler, steps, 1.0)[0]
    latent_image = EmptyLatentImage.generate(closestNumber(width, 16), closestNumber(height, 16))[0]
    sample, sample_denoised = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas, latent_image)
    model_management.soft_empty_cache()
    decoded = VAEDecode.decode(vae, sample)[0].detach()
    Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save("/content/flux.png")

Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0])



