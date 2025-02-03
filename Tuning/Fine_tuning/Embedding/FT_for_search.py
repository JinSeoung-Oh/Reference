### From https://shawhin.medium.com/fine-tuning-text-embeddings-f913b882b11c

"""
1. Text Embedding Models and Their Use Cases
   -a. Text Embedding Models:
       -1. These models convert text into semantically meaningful vectors.
       -2. They are useful for various tasks such as retrieval, classification, etc.
   -b. Limitation of General-Purpose Models:
       -1. While versatile, these models often underperform on domain-specific tasks where specialized jargon or context is important.
   -c. Solution – Fine-Tuning:
       -1. Fine-tuning involves additional training on domain-specific data to adjust the model’s behavior.
       -2. This process tailors the embeddings to capture the nuances required for specific applications.

2. Retrieval Augmented Generation (RAG) and Embedding-Based Search
   -a. RAG Concept:
       -1. In systems built around large language models (LLMs), RAG retrieves relevant context (like FAQs) from a knowledge base 
           when given an input query.
   -b. Three-Step Process Using Embeddings:
       -1. Compute Embeddings for All Items:
           -1) Each item in the knowledge base is converted into an embedding vector.
       -2. Convert Input Text to an Embedding:
           -1) The input text (e.g., a customer query) is also transformed into a vector using the same embedding model.
       -3. Similarity Calculation and Retrieval:
           -1) The similarity (often computed via cosine similarity) between the input vector and each knowledge base vector is measured.
           -2) The items with the highest similarity scores are returned.
   -c. Problem with Simple Similarity Search:
       -1. Even if the query and a knowledge base item are similar in vector space (i.e., the angle between their embeddings is small), 
           that similarity does not guarantee the returned item is the most helpful answer.
       -2. For example, a query about updating a payment method might return a result about viewing payment history, which, 
           although semantically similar, does not directly answer the question.

3. Fine-Tuning Embeddings
   -a. Purpose:
       -1. Fine-tuning adapts the embedding model for a specific domain or task (e.g., matching customer questions to relevant FAQs 
           or job descriptions).
       -2. It helps the model capture domain-specific terms (such as “scaling” or “instances” in cloud computing) that might be poorly 
           represented by a general-purpose model.
   -b. Approach – Contrastive Learning:
       -1. The process uses pairs or triplets of data (e.g., query, positive match, negative match).
       -2. The training minimizes the distance between embeddings of matching pairs (positive examples) while maximizing the distance between 
           non-matching pairs (negative examples).
   -c. Five Key Steps for Fine-Tuning:
       -1. Gather Positive (and Negative) Pairs:
           -1) Collect domain-specific data and create matching (positive) pairs as well as non-matching (negative) pairs.
       -2. Pick a Pre-Trained Model:
           -1) Evaluate several models to choose one that performs best on your domain-specific task.
       -3. Select a Loss Function:
           -1) Choose an appropriate loss function based on your data; for triplet data, MultipleNegativesRankingLoss is often used.
       -4. Fine-Tune the Model:
           -1) Set hyperparameters (e.g., number of epochs, batch size, learning rate) and train the model.
       -5. Evaluate the Model:
           -1) Test the fine-tuned model on validation and test sets to ensure it meets performance goals, and optionally deploy it to a hub like 
               Hugging Face for easy access.

4. Fine-Tuning Example: AI Job Postings
   The text provides a concrete example of fine-tuning an embedding model to match job search queries with job descriptions (JDs).
   
   4.1 Data Collection and Preparation
       -a. Data Source:
           -1. Job descriptions for roles like Data Scientist, Data Engineer, and AI Engineer are extracted from the Hugging Face dataset
               datastax/linkedin_job_listings.
       -b. Generating Search Queries:
           -1. OpenAI’s Batch API in conjunction with GPT-4o-mini is used to generate human-like search queries corresponding to each job description.
           -2. This process is cost-efficient and takes about 24 hours to run.
       -c. Cleaning the Text:
           -1. Irrelevant parts of the job descriptions (those not related to qualifications) are removed because most text embedding models 
               have a token limit (typically 512 tokens).
       -d. Creating Positive Pairs:
           -1. The cleaned job descriptions (JDs) are paired with the generated queries, resulting in a set of 1012 positive pairs after removing duplicates.
       -e. Creating Negative Pairs:
           -1. Using a pre-trained embedding model (all-mpnet-base-v2), embeddings for all job descriptions are computed.
           -2. For each positive pair, the least similar job description (ensuring uniqueness) is selected as the negative pair.
       -f. Dataset Splitting and Upload:
           -1. The combined dataset is shuffled and split into train (80%), validation (10%), and test (10%) sets.
           -2. The final dataset is uploaded to the Hugging Face Hub, making it easily accessible with a single-line load command.
           
   4.2 Model Selection and Evaluation
       -a. Model Comparison:
           -1. Multiple pre-trained models are evaluated using a triplet-based evaluation method.
           -2. The evaluation uses the TripletEvaluator (comparing query, positive JD, and negative JD) to measure cosine accuracy.
       -b. Chosen Model:
           -1. “all-distilroberta-v1” is selected because it achieved the highest initial accuracy (~88.1%) on the validation set.
           
   4.3 Loss Function
       -a. Loss Function:
           -1. The MultipleNegativesRankingLoss is chosen since it fits the (anchor, positive, negative) triplet format well.
           
   4.4 Fine-Tuning the Model
       -a. Training Parameters:
           -1. Hyperparameters such as the number of epochs, batch size, learning rate, and warmup ratio are set.
           -2. A key point is that contrastive learning benefits from larger batch sizes and extended training times.
       -b. Trainer Setup:
           -1. The SentenceTransformerTrainer is used to fine-tune the model, incorporating the training dataset, validation dataset, loss function, 
               and evaluator.

   4.5 Model Evaluation and Deployment
       -a. Evaluation Results:
           -1. After fine-tuning, the model reaches about 99% accuracy on the validation set and 100% accuracy on the test set.
       -b. Deployment:
           -1. The fine-tuned model is pushed to the Hugging Face Hub for easy future access.
           -2. An example is provided showing how to encode a new query and compute similarities with job description embeddings.
"""

### 1. Load Data (Extract job descriptions from the Hugging Face dataset)
from datasets import load_dataset

# load data from HF hub
ds = load_dataset("datastax/linkedin_job_listings")

### 2. Create Negative Pairs: Compute JD embeddings and select least similar negative pairs
from sentence_transformers import SentenceTransformer
import numpy as np

# Load an embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# Encode all job descriptions
job_embeddings = model.encode(df['job_description_pos'].to_list())

# Compute similarities
similarities = model.similarity(job_embeddings, job_embeddings)
# Match least similar JDs to positive match as the negative match

# Get sorted indexes of similarities
similarities_argsorted = np.argsort(similarities.numpy(), axis=1)

# Initialize list to store negative pairs
negative_pair_index_list = []

for i in range(len(similarities)):
    # Start with the smallest similarity index for the current row
    j = 0
    index = int(similarities_argsorted[i][j])
    # Ensure the index is unique
    while index in negative_pair_index_list:
        j += 1  # Move to the next smallest index
        index = int(similarities_argsorted[i][j])  # Fetch next smallest index
    negative_pair_index_list.append(index)

# Add negative pairs to df
df['job_description_neg'] = df['job_description_pos'].iloc[negative_pair_index_list].values

### 3. Split Dataset and Upload to Hugging Face Hub
# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train, validation, and test sets (e.g., 80% train, 10% validation, 10% test)
train_frac = 0.8
valid_frac = 0.1
test_frac = 0.1

# Define train and validation size
train_size = int(train_frac * len(df))
valid_size = int(valid_frac * len(df))

# Create train, validation, and test datasets
df_train = df[:train_size]
df_valid = df[train_size:train_size + valid_size]
df_test = df[train_size + valid_size:]

from datasets import DatasetDict, Dataset

# Convert the pandas DataFrames back to Hugging Face Datasets
train_ds = Dataset.from_pandas(df_train)
valid_ds = Dataset.from_pandas(df_valid)
test_ds = Dataset.from_pandas(df_test)

# Combine into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_ds,
    'validation': valid_ds,
    'test': test_ds
})

# Push data to hub
dataset_dict.push_to_hub("shawhin/ai-job-embedding-finetuning")

### 4. Load the Dataset from Hugging Face Hub
from datasets import load_dataset

# Importing data
dataset = load_dataset("shawhin/ai-job-embedding-finetuning")

### 5. Select a Pre-Trained Model and Evaluate
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import TripletEvaluator

# Import model
model_name = "sentence-transformers/all-distilroberta-v1"
model = SentenceTransformer(model_name)

# Create evaluator
evaluator_valid = TripletEvaluator(
    anchors=dataset["validation"]["query"],
    positives=dataset["validation"]["job_description_pos"],
    negatives=dataset["validation"]["job_description_neg"],
    name="ai-job-validation",
)
evaluator_valid(model)

#>> {'ai-job-validation_cosine_accuracy': np.float64(0.8811881188118812)}

### 6. Select the Loss Function
from sentence_transformers.losses import MultipleNegativesRankingLoss

loss = MultipleNegativesRankingLoss(model)

### 7. Fine-Tuning: Set Training Arguments and Define the Trainer
from sentence_transformers import SentenceTransformerTrainingArguments

num_epochs = 1
batch_size = 16
lr = 2e-5
finetuned_model_name = "distilroberta-ai-job-embeddings"

train_args = SentenceTransformerTrainingArguments(
    output_dir=f"models/{finetuned_model_name}",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=lr,
    warmup_ratio=0.1,
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=100,
)

### 8. Fine-Tuning: Train the Model using the Trainer
from sentence_transformers import SentenceTransformerTrainer

trainer = SentenceTransformerTrainer(
    model=model,
    args=train_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    loss=loss,
    evaluator=evaluator_valid,
)
trainer.train()

### 9. Evaluate and Deploy the Model: Upload to Hugging Face Hub and Test with a New Query
# Push model to HF hub
model.push_to_hub(f"shawhin/{finetuned_model_name}")
# Import model
model = SentenceTransformer("shawhin/distilroberta-ai-job-embeddings")

# New query
query = "data scientist 6 year experience, LLMs, credit risk, content marketing"
query_embedding = model.encode(query)

# Encode job descriptions (JDs)
jd_embeddings = model.encode(dataset["test"]["job_description_pos"])

# Compute similarities
similarities = model.similarity(query_embedding, jd_embeddings)


