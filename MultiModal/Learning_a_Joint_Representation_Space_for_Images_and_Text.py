"""
From https://generativeai.pub/learning-a-joint-representation-space-for-images-and-text-9ac268fa1186

The text discusses the concept of Vision-Language Models (VLMs), which combine computer vision and natural language processing. 
It outlines the process of building a basic VLM in Python using libraries like Transformers and PyTorch, 
focusing on creating a joint representation space for images and text.

Key components of the VLM approach include:

1. Image and Text Encoders
   Separate encoders for images (e.g., VGG16 or ResNet) and text (e.g., BERT) project them into a common latent space.
2. Joint Representation Space
   Embeddings from image and text encoders are fed into a contrastive loss function to bring similar pairs closer and push dissimilar pairs apart.
3. Contrastive Loss
   The model learns to align image and text representations using a contrastive loss function like cosine similarity or InfoNCE loss.

The text provides Python code snippets for implementing the image and text encoders, defining the VLM model, 
and training it using a contrastive loss function. It also discusses dataset selection and preprocessing for real-world applications.

In summary, the article offers a simplified overview of VLMs and demonstrates how to build a basic model, 
acknowledging the complexity of real-world implementations and encouraging further exploration.
"""

## VLM Implementation in Pytorch
import torch
from torch import nn
from torchvision import models  # For pre-trained image models
from transformers import BertModel  # For pre-trained text model

# Image and Text Embeddings
class ImageEncoder(nn.Module):
  def __init__(self, pretrained=True):
    super(ImageEncoder, self).__init__()
    self.cnn = models.vgg16(pretrained=pretrained)
    # Remove classification layers from CNN
    self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])


class TextEncoder(nn.Module):
  def __init__(self, pretrained=True):
    super(TextEncoder, self).__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    # Remove classification layer from BERT
    self.bert = nn.Sequential(*list(self.bert.children())[:-1])

  def forward(self, x):
    # Pass text through BERT to get encoded representation
    output = self.bert(x)
    return output[0]  # Use the first hidden state (consider alternatives)

class VLModel(nn.Module):
  def __init__(self, image_encoder, text_encoder):
    super(CLIPModel, self).__init__()
    self.image_encoder = image_encoder
    self.text_encoder = text_encoder

  def forward(self, image, text):
    image_features = self.image_encoder(image)
    text_features = self.text_encoder(text)
    # Implement contrastive loss function here (cosine similarity, InfoNCE loss)
    return image_features, text_features  # For calculating similarity in loss function
  
  def forward(self, x):
    return self.cnn(x)

# Joint Representation: Contrastive Loss
def contrastive_loss(image_features, text_features):
  # Cosine similarity for contrastive loss
  similarity = nn.functional.cosine_similarity(image_features, text_features).squeeze(1)
  return similarity

# Negative Pair Generation
def generate_negative_pairs(positive_similarities, batch_size):
  # Simplified approach (replace with more advanced techniques)
  negative_similarity = torch.tensor(-1.0).to(device) * torch.ones(batch_size).to(device)
  return negative_similarity

# Training
def train_model(model, train_loader, optimizer, epochs):
  """
  Trains the VLM model on the provided data loader.
  """
  for epoch in range(epochs):
    model.train()
    for images, texts in train_loader:
      images, texts = images.to(device), texts.to(device)
      optimizer.zero_grad()

      # Get image and text features
      image_features, text_features = model(images, texts)

      # Calculate similarities (positive pairs
      positive_similarities = contrastive_loss(image_features, text_features)

      # Generate negative pairs (replace with more advanced techniques)
      negative_similarities = generate_negative_pairs(positive_similarities, images.shape[0])

      # Contrastive loss (using margin ranking loss for demonstration)
      loss = nn.functional.margin_ranking_loss(positive_similarities, negative_similarities, margin=1.0)
      loss.backward()
      optimizer.step()

      # Print training progress (optional)
      if (i+1) % 100 == 0:
        print(f"Epoch: [{epoch+1}/{epochs}], Step: [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    print(f"Epoch {epoch+1} complete!")
Selecting a Dataset

# Selecting a Dataset
def prepare_data(data_dir="./data", image_size=(224, 224)):
 

  # Import libraries
  from torchvision import transforms
  from PIL import Image  # For image loading

  # Define image transformations
  transform = transforms.Compose([
      transforms.Resize(image_size),  # Resize images
      transforms.ToTensor(),  # Convert to tensors
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
  ])

  # Text tokenizer (replace with your preferred tokenizer)
  tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

  # Load training and testing data (replace with your data loading logic)
  train_data = []  # List to store training data (image path, caption)
  test_data = []  # List to store testing data (image path, caption)
  # ... (your data loading logic to populate train_data and test_data)

  
  # Preprocess data
  for image_path, caption in train_data:
    image = Image.open(image_path).convert('RGB')  # Load and convert image to RGB
    image = transform(image)  # Apply image transformations

    # Tokenize text caption
    caption_tokens = tokenizer(caption, padding='max_length', truncation=True, return_tensors="pt")

    train_data.append((image, caption_tokens))

  for image_path, caption in test_data:
    image = Image.open(image_path).convert('RGB')
    image = transform(image)

    caption_tokens = tokenizer(caption, padding='max_length', truncation=True, return_tensors="pt")

    test_data.append((image, caption_tokens))

  # Create data loaders
  train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

  return train_loader, test_loader

# Start the Training
# Define device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model definition
image_encoder = ImageEncoder(pretrained=True)
text_encoder = TextEncoder(pretrained=True)
model = VLModel(image_encoder, text_encoder).to(device)

# Optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dataset and epochs
train_loader, test_loader = prepare_data()
epochs = 10

# Train the model
train_model(model, train_loader, optimizer, epochs)
