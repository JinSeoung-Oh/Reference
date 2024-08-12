## From https://jimclydemonge.medium.com/flux-1-is-a-mind-blowing-open-weights-ai-image-generator-with-12b-parameters-5a138146bb51

"""
1. Flux.1 Pro
   This offers state-of-the-art performance in image generation, delivering top-notch prompt following,
   visual quality, image detail, and output diversity.
2. Flux.1 Dev
   This is an open-weight, guidance-distilled model designed for non-commercial use. 
   It is distilled from Flux.1 Pro, achieving similar quality and prompt adherence while being more efficient than a typical model of the same size.
3. Flux.1 Schnell
   This is their fastest model and is designed for local development and personal use. It is openly available under an Apache 2.0 license.
"""

import os
import requests

request = requests.post(
    'https://api.bfl.ml/v1/image',
    headers={
        'accept': 'application/json',
        'x-key': os.environ.get("BFL_API_KEY"),
        'Content-Type': 'application/json',
    },
    json={
        'prompt': 'A cat on its back legs running like a human is holding a big silver fish with its arms. The cat is running away from the shop owner and has a panicked look on his face. The scene is situated in a crowded market.',
        'width': 1024,
        'height': 1024,
    },
).json()
print(request)
request_id = request["id"]
