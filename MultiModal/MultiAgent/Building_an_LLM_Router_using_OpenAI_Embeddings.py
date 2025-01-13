### From https://itnext.io/building-an-llm-router-using-openai-embeddings-4d0e680afd44

### generate_dataset.py 
from dotenv import load_dotenv
import os
from pathlib import Path

import openai

import pandas as pd
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm  # For progress bar
import asyncio

# Load environment variables from a .env file
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
# EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_MODEL_NAME = "text-embedding-3-large"

# OUTPUT_DIR="./data/banking77"
# DATASET_ID = "legacy-datasets/banking77"
# TEXT_KEY = "text"
# TARGET_KEY = "label"
OUTPUT_DIR="./data/llm_router_dataset-synth"
DATASET_ID = "DevQuasar/llm_router_dataset-synth"
TEXT_KEY = "prompt"
TARGET_KEY = "label"

# Function to load datasets
def load_data():
    from datasets import load_dataset

    # Load dataset
    dataset = load_dataset(DATASET_ID)


    train_df = pd.DataFrame({
        'text': dataset['train'][TEXT_KEY],
        'target': dataset['train'][TARGET_KEY]
    })
    test_df = pd.DataFrame({
        'text': dataset['test'][TEXT_KEY],
        'target': dataset['test'][TARGET_KEY]
    })
    return train_df, test_df


# Async embedding function using LangChain
async def get_openai_embedding(text):
    embedding_model = OpenAIEmbeddings()
    return await embedding_model.aembed_query(text)


# Generate embeddings asynchronously with a progress bar
async def generate_embeddings(texts):
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, show_progress_bar=True,)
    return await embedding_model.aembed_documents(texts)


# Save embeddings to a CSV file
def save_embeddings_to_csv(embeddings, targets, filename):
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename

    df = pd.DataFrame(embeddings)
    df['target'] = targets
    df.to_csv(filepath, index=False)

async def main():
    print("Starting...")
    
     # Load the dataset
    train_df, test_df = load_data()
    

    # Split train and validation
    print("loading data...")
    train_texts, train_targets = train_df['text'].values, train_df['target'].values

    # in case the target column is not present
    # test_texts = test_df['text'].values
    # test_targets = pd.Series([None for _ in range(test_df.shape[0])])
    test_texts, test_targets = test_df['text'].values, test_df['target'].values

    print("Generating and saving embeddings...")
    train_embeddings = await generate_embeddings(train_texts)
    test_embeddings = await generate_embeddings(test_texts)

    print("Saving embeddings to CSV...")
    # Save embeddings to CSV
    save_embeddings_to_csv(train_embeddings, train_targets, f'train_embeddings__{EMBEDDING_MODEL_NAME}.csv')
    save_embeddings_to_csv(test_embeddings, test_targets, f'test_embeddings__{EMBEDDING_MODEL_NAME}.csv')
    print("Embeddings saved to CSV.")

if __name__ == "__main__":

    asyncio.run(main())


########### train_model.py 
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Hyperparameters
BATCH_SIZE = 32

# Dataset class for disaster tweets
class EmbeddingTextDataset(Dataset):
    def __init__(self, embeddings, targets):
        self.embeddings = embeddings
        self.targets = targets

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.targets[idx]
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# Load embeddings from a CSV file
def load_embeddings_from_csv(filepath):
    df = pd.read_csv(filepath)
    embeddings = df.drop(columns=['target']).values
    targets = df['target'].values
    return embeddings, targets




import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import f1_score

# PyTorch Lightning Module for Tweet Classification
class EmbeddingTextClassifier(pl.LightningModule):
    def __init__(self, embedding_dim: int, output_class_count: int=2, hidden_layer_size: int=20):
        super(EmbeddingTextClassifier, self).__init__()
        self.hidden = nn.Linear(embedding_dim, hidden_layer_size)  # Hidden layer with size hidden_emb_size for cosine similarities
        self.relu = nn.ReLU()  # Activation function
        self.fc = nn.Linear(hidden_layer_size, output_class_count)  # Output layer for binary classification

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)  # Apply ReLU activation
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        embeddings, labels = batch
        logits = self(embeddings)
        loss = nn.CrossEntropyLoss()(logits, labels)
        predictions = logits.argmax(dim=1)
        acc = (predictions == labels).float().mean()

        labels = labels.cpu().numpy()
        predictions = predictions.cpu().numpy()
        # source for f1 score: https://www.philschmid.de/fine-tune-modern-bert-in-2025
        score = f1_score(
            labels, predictions, labels=labels, pos_label=1, average="weighted"
        )

        
        # Log training loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_f1_score", score, prog_bar=False, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings, labels = batch
        logits = self(embeddings)
        loss = nn.CrossEntropyLoss()(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        
        # Log validation loss and accuracy
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        # Using weight decay for L2 regularization in Adam optimizer
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)



# train_csv_filepath = 'data/banking77/train_embeddings__text-embedding-3-large.csv'
# test_csv_filepath = 'data/banking77/test_embeddings__text-embedding-3-large.csv'
# output_class_count = 77
train_csv_filepath = 'data/llm_router_dataset-synth/train_embeddings__text-embedding-3-large.csv'
test_csv_filepath = 'data/llm_router_dataset-synth/test_embeddings__text-embedding-3-large.csv'
output_class_count = 2
train_embeddings_full, train_targets_full = load_embeddings_from_csv(train_csv_filepath)
train_embeddings, val_embeddings, train_targets, val_targets = train_test_split(
        train_embeddings_full, train_targets_full, test_size=0.2, random_state=42)

# Prepare datasets
train_dataset = EmbeddingTextDataset(train_embeddings, train_targets)
val_dataset = EmbeddingTextDataset(val_embeddings, val_targets)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Define model
embedding_dim = len(train_embeddings[0])  # OpenAI embedding size
model = EmbeddingTextClassifier(embedding_dim=embedding_dim, output_class_count=output_class_count)

# Trainer
lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(max_epochs=100, callbacks=[lr_monitor])
trainer.fit(model, train_loader, val_loader)

# %% Test model

# Load test embeddings
test_embeddings, test_targets = load_embeddings_from_csv(test_csv_filepath)
test_dataset = EmbeddingTextDataset(test_embeddings, test_targets)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

all_predictions = []
all_labels = []
for batch in test_loader:
    embeddings, labels = batch
    logits = model(embeddings)
    predictions = logits.argmax(dim=1)
    all_predictions.extend(predictions.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# Calculate F1 score
f1 = f1_score(all_labels, all_predictions, average='weighted')
print(f'F1 Score: {f1}')

# calculate accuracy
accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
print(f'Accuracy: {accuracy}')

# %% Save the model
filepath = train_csv_filepath.replace('train_embeddings', 'model').replace('.csv', '.onnx')
input_sample = torch.randn((1, embedding_dim))
model.to_onnx(filepath, input_sample, export_params=True)
