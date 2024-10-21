## From https://generativeai.pub/adding-custom-datasets-to-llama-factory-4aff22385c2f

"""
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -r requirements.txt
"""

### Create a JSON file for your custom dataset:
[
  {
    "messages": [
      {
        "content": "<image>What is this model?",
        "role": "user"
      },
      {
        "content": "This is a Convolutional Neural Network (CNN) used for image classification.",
        "role": "assistant"
      },
      {
        "content": "What is it classifying?",
        "role": "user"
      },
      {
        "content": "It's classifying images of handwritten digits from the MNIST dataset.",
        "role": "assistant"
      }
    ],
    "images": [
      "Example/cnn_mnist.jpg"
    ]
  },
  {
    "messages": [
      {
        "content": "<image>What does this graph represent?",
        "role": "user"
      },
      {
        "content": "This graph shows the loss function decreasing over time during model training.",
        "role": "assistant"
      },
      {
        "content": "Why is the loss decreasing?",
        "role": "user"
      },
      {
        "content": "Because the model is learning and improving its accuracy with each iteration.",
        "role": "assistant"
      }
    ],
    "images": [
      "Example/loss_curve.jpg"
    ]
  }
]

### Update the dataset_info.json file (located in the same folder) to register your dataset
