## From https://generativeai.pub/ai-image-generators-ideogram-2-0-is-a-game-changer-2791ef7556ac

import requests
# Generates images synchronously based on a given prompt and optional parameters. (POST /generate)
response = requests.post(
  "https://api.ideogram.ai/generate",
  headers={
    "Api-Key": "",
    "Content-Type": "application/json"
  },
  json={
    "image_request": {
      "prompt": "A serene tropical beach scene. Dominating the foreground are tall palm trees with lush green leaves, standing tall against a backdrop of a sandy beach. The beach leads to the azure waters of the sea, which gently kisses the shoreline. In the distance, there is an island or landmass with a silhouette of what appears to be a lighthouse or tower. The sky above is painted with fluffy white clouds, some of which are tinged with hues of pink and orange, suggesting either a sunrise or sunset.",
      "aspect_ratio": "ASPECT_10_16",
      "model": "V_2",
      "magic_prompt_option": "AUTO"
    }
  },
)
print(response.json())
