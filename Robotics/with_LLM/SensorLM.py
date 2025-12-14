### From https://medium.com/@bravekjh/sensorlm-a-language-model-for-sensor-data-95440b1e3225

"""
SensorLM is a transformer-based foundation model designed specifically to work with multivariate time-series sensor data,
such as that collected from IoT devices, industrial systems, HVAC, and other telemetry-heavy environments.
"""

!pip install transformers datasets torch



from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load SensorLM from Hugging Face Hub
model_name = "microsoft/sensorlm-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example sensor input: 10 sensors × 50 timesteps, normalized and flattened
# Shape: (10 sensors, 50 timesteps) → Flattened to 500
sensor_data = np.random.rand(10, 50).flatten()

# Convert to list of string tokens (SensorLM expects string inputs)
input_text = " ".join([str(round(val, 4)) for val in sensor_data])
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

print(f"Predicted Class: {predicted_class}")
