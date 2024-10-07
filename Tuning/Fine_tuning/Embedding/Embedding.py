## From https://blog.gopenai.com/fine-tuning-embeddings-for-specific-domains-a-comprehensive-guide-5e4298b42185

"""
1. Introduction: Why Fine-Tune for Specific Domains?
   When building a question-answering system in specialized domains (e.g., medicine, law, finance), generic embedding models often struggle 
   with the nuanced vocabulary and concepts unique to these fields. 
   Fine-tuning allows the model to understand the domain’s specific terminology and improve its performance in tasks like information retrieval, 
   question answering, and semantic similarity. By fine-tuning, you’re essentially tailoring the model to fit the context of the domain, 
   improving the accuracy and relevance of the retrieved results.

2. Understanding Embeddings
   Embeddings are vector representations of text (or other data types, like images or audio), where semantically similar concepts are closer in the embedding space. 
   They are crucial in many NLP tasks, such as:

   -1. Semantic Similarity: Measuring the similarity between two pieces of text.
   -2. Text Classification: Categorizing data based on meaning.
   -3. Question Answering: Retrieving the most relevant document or section for a given query.
   -4. Retrieval-Augmented Generation (RAG): Combining embedding models for retrieval with language models for text generation to improve result relevance and quality.

3. Matryoshka Representation Learning (MRL)
   MRL is a specialized technique for creating truncatable embeddings. Think of nested dolls where each successive layer (or embedding dimension) adds more detail. 
   This approach allows you to use only a portion of the embedding vector when needed, 
   helping reduce computational costs and storage without sacrificing the most important information, which is captured in the outer (earlier) dimensions. 
   MRL is especially helpful when dealing with large datasets where computational efficiency is important.

4. The Bge-base-en Model
   The BAAI/bge-base-en-v1.5 model, developed by the Beijing Academy of Artificial Intelligence (BAAI), is a powerful pre-trained text embedding model.
   It performs well on general benchmarks like MTEB and C-MTEB and is designed for tasks requiring limited computing resources. 
   For domains like medicine, it can serve as a strong foundation for further fine-tuning, allowing it to adapt to domain-specific nuances.

5. Why Fine-Tune Embedding Models?
   Fine-tuning an embedding model is essential for:
   -1. Optimizing Retrieval Systems: In tasks like RAG, fine-tuning helps the model understand domain-specific similarity patterns,
                                     improving its ability to retrieve the most relevant documents for a query.
   -2. Improving Relevance: Without fine-tuning, embeddings in a specialized field may fail to distinguish important terms. 
                            In the medical domain, for example, the model might confuse “diabetes” with unrelated terms 
                            if it isn’t fine-tuned to understand medical semantics.

6. Dataset Formats for Fine-Tuning
   Fine-tuning requires a well-structured dataset. Common formats include:

   -1. Positive Pair: Two related sentences, like question-answer pairs.
   -2. Triplets: Consisting of an anchor sentence, a positive (similar) sentence, and a negative (dissimilar) sentence.
   -3. Pair with Similarity Score: A pair of sentences along with a score indicating their semantic similarity.
   -4. Texts with Classes: Each sentence or document is labeled with a category or class.
  
   In this fine-tuning process, you would likely create a dataset of medical question-answer pairs to help the model understand the types of queries 
   it will encounter in your system.

7. Loss Functions for Fine-Tuning
   The right loss function guides the fine-tuning process by measuring the difference between the model’s predictions and the expected results.
   Key loss functions include:

   -1. Triplet Loss: Used with triplets (anchor, positive, negative). It encourages the model to place similar embeddings closer together
                     and dissimilar embeddings farther apart.
   -2. Contrastive Loss: Used with positive and negative pairs. Similar pairs are encouraged to be closer, while dissimilar ones are pushed apart.
   -3. Cosine Similarity Loss: Used when you have pairs with similarity scores. The goal is to align the cosine similarity of the embeddings
                               with the provided similarity scores.
   -4. Matryoshka Loss: This loss function is specific to MRL, encouraging truncatable embeddings, where the earlier parts of the embedding capture 
                        the most essential information.

8. Fine-Tuning Example
   Let’s say you want to fine-tune the bge-base-en-v1.5 model for the medical domain:

   -1. Dataset Creation: You first compile a dataset of question-answer pairs from medical literature. 
                         These pairs should cover a wide range of topics like symptoms, diagnoses, treatments, and medical conditions.
   -2. Training Setup: You would then set up the training process, choosing an appropriate loss function based on 
                       your dataset format (e.g., Contrastive Loss for question-answer pairs).
   -3. Optimization and Monitoring: During fine-tuning, the model adjusts its embeddings to better reflect the relationships in the medical data. 
                                    The process is guided by the loss function, and metrics like accuracy or retrieval precision are monitored to evaluate performance.

9. Expected Outcomes
   After fine-tuning, your embedding model will:
   Better understand medical vocabulary and nuances, improving its ability to retrieve relevant documents when presented with medical queries.
   Improve the performance of question-answer systems, allowing the model to surface more precise and contextually appropriate answers.

10. Benefits of Fine-Tuned Embeddings
    - Accuracy: By aligning the model’s understanding of similarity with domain-specific knowledge, you increase the accuracy of your system’s retrieval.
    - Efficiency: Fine-tuned embeddings can reduce the number of irrelevant documents retrieved, making your system faster and more efficient.
    - Domain Expertise: The model becomes an expert in the specific domain, handling specialized queries better than generic models.

11. Conclusion
    Fine-tuning embedding models is crucial for improving performance in specialized domains like medicine.
    By generating a custom dataset and choosing the appropriate loss functions, 
    you can tailor an embedding model to understand the unique vocabulary and patterns in your field. 
    With this optimized model, your question-answering system will be more accurate, efficient, and reliable for domain-specific NLP tasks.
"""
apt-get -qq install poppler-utils tesseract-ocr
pip install datasets sentence-transformers google-generativeai
pip install -q --user --upgrade pillow
pip install -q unstructured["all-docs"] pi_heif
pip install -q --upgrade unstructured
pip install --upgrade nltk

import nltk
import os 
from unstructured.partition.pdf import partition_pdf
from collections import Counter
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab') 

def process_pdfs_in_folder(folder_path):
    total_text = []  # To accumulate the text from all PDFs

    # Get list of all PDF files in the folder
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"Processing: {pdf_path}")

        # Apply the partition logic
        elements = partition_pdf(pdf_path, strategy="auto")

        # Display the types of elements
        display(Counter(type(element) for element in elements))

        # Join the elements to form text and add it to total_text list
        text = "\n\n".join([str(el) for el in elements])
        total_text.append(text)

    # Return the total concatenated text
    return "\n\n".join(total_text)


folder_path = "data"
all_text = process_pdfs_in_folder(folder_path)
#########################################################################################################
### Custom Text Chunking
import nltk

nltk.download('punkt')

def nltk_based_splitter(text: str, chunk_size: int, overlap: int) -> list:
    """
    Splits the input text into chunks of a specified size, with optional overlap between chunks.

    Parameters:
    - text: The input text to be split.
    - chunk_size: The maximum size of each chunk (in terms of characters).
    - overlap: The number of overlapping characters between consecutive chunks.

    Returns:
    - A list of text chunks, with or without overlap.
    """

    from nltk.tokenize import sent_tokenize

    # Tokenize the input text into individual sentences
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If the current chunk plus the next sentence doesn't exceed the chunk size, add the sentence to the chunk
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            # Otherwise, add the current chunk to the list of chunks and start a new chunk with the current sentence
            chunks.append(current_chunk.strip())  # Strip to remove leading spaces
            current_chunk = sentence

    # After the loop, if there is any leftover text in the current chunk, add it to the list of chunks
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Handle overlap if it's specified (overlap > 0)
    if overlap > 0:
        overlapping_chunks = []
        for i in range(len(chunks)):
            if i > 0:
                # Calculate the start index for overlap from the previous chunk
                start_overlap = max(0, len(chunks[i-1]) - overlap)
                # Combine the overlapping portion of the previous chunk with the current chunk
                chunk_with_overlap = chunks[i-1][start_overlap:] + " " + chunks[i]
                # Append the combined chunk, making sure it's not longer than chunk_size
                overlapping_chunks.append(chunk_with_overlap[:chunk_size])
            else:
                # For the first chunk, there's no previous chunk to overlap with
                overlapping_chunks.append(chunks[i][:chunk_size])

        return overlapping_chunks  # Return the list of chunks with overlap

    # If overlap is 0, return the non-overlapping chunks
    return chunks

chunks = nltk_based_splitter(text=all_text, 
                                  chunk_size=2048,
                                  overlap=0)
#########################################################################################################
### Dataset Generator
import google.generativeai as genai
import pandas as pd

# Replace with your valid Google API key
GOOGLE_API_KEY = "xxxxxxxxxxxx"

# Prompt generator with an explicit request for structured output
def prompt(text_chunk):
    return f"""
    Based on the following text, generate one Question and its corresponding Answer.
    Please format the output as follows:
    Question: [Your question]
    Answer: [Your answer]

    Text: {text_chunk}
    """
# Function to interact with Google's Gemini and return a QA pair
def generate_with_gemini(text_chunk:str, temperature:float, model_name:str):
    genai.configure(api_key=GOOGLE_API_KEY)
    generation_config = {"temperature": temperature}

    # Initialize the generative model
    gen_model = genai.GenerativeModel(model_name, generation_config=generation_config)

    # Generate response based on the prompt
    response = gen_model.generate_content(prompt(text_chunk))

    # Extract question and answer from response using keyword
    try:
        question, answer = response.text.split("Answer:", 1)
        question = question.replace("Question:", "").strip()
        answer = answer.strip()
    except ValueError:
        question, answer = "N/A", "N/A"  # Handle unexpected format in response

    return question, answer
#########################################################################################################
### Running Q&A Generation
def process_text_chunks(text_chunks:list, temperature:int, model_name=str):
    """
    Processes a list of text chunks to generate questions and answers using a specified model.

    Parameters:
    - text_chunks: A list of text chunks to process.
    - temperature: The sampling temperature to control randomness in the generated outputs.
    - model_name: The name of the model to use for generating questions and answers.

    Returns:
    - A Pandas DataFrame containing the text chunks, questions, and answers.
    """
    results = []

    # Iterate through each text chunk
    for chunk in text_chunks:
        question, answer = generate_with_gemini(chunk, temperature, model_name)
        results.append({"Text Chunk": chunk, "Question": question, "Answer": answer})

    # Convert results into a Pandas DataFrame
    df = pd.DataFrame(results)
    return df
# Process the text chunks and get the DataFrame
df_results = process_text_chunks(text_chunks=chunks, 
                                 temperature=0.7, 
                                 model_name="gemini-1.5-flash")
df_results.to_csv("generated_qa_pairs.csv", index=False)
#########################################################################################################
### Loading the Dataset
from datasets import load_dataset

# Load the CSV file into a Hugging Face Dataset
dataset = load_dataset('csv', data_files='generated_qa_pairs.csv')

def process_example(example, idx):
    return {
        "id": idx,  # Add unique ID based on the index
        "anchor": example["Question"],
        "positive": example["Answer"]
    }
dataset = dataset.map(process_example,
                      with_indices=True , 
                      remove_columns=["Text Chunk", "Question", "Answer"])
#########################################################################################################
### Loading the Model
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.util import cos_sim
from datasets import load_dataset, concatenate_datasets
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss


model_id = "BAAI/bge-base-en-v1.5" 

# Load a model
model = SentenceTransformer(
    model_id, device="cuda" if torch.cuda.is_available() else "cpu"
)

# Important: large to small
matryoshka_dimensions = [768, 512, 256, 128, 64] 
inner_train_loss = MultipleNegativesRankingLoss(model)
train_loss = MatryoshkaLoss(
    model, inner_train_loss, matryoshka_dims=matryoshka_dimensions
)
#########################################################################################################
#### Defining Training Arguments
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers

# define training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="bge-finetuned",                 # output directory and hugging face model ID
    num_train_epochs=1,                         # number of epochs
    per_device_train_batch_size=4,              # train batch size
    gradient_accumulation_steps=16,             # for a global batch size of 512
    per_device_eval_batch_size=16,              # evaluation batch size
    warmup_ratio=0.1,                           # warmup ratio
    learning_rate=2e-5,                         # learning rate, 2e-5 is a good value
    lr_scheduler_type="cosine",                 # use constant learning rate scheduler
    optim="adamw_torch_fused",                  # use fused adamw optimizer
    tf32=True,                                  # use tf32 precision
    bf16=True,                                  # use bf16 precision
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    eval_strategy="epoch",                      # evaluate after each epoch
    save_strategy="epoch",                      # save after each epoch
    logging_steps=10,                           # log every 10 steps
    save_total_limit=3,                         # save only the last 3 models
    load_best_model_at_end=True,                # load the best model when training ends
    metric_for_best_model="eval_dim_128_cosine_ndcg@10",  # Optimizing for the best ndcg@10 score for the 128 dimension
)
#########################################################################################################
#### Creating the Evaluator
corpus = dict(
    zip(dataset['train']['id'], 
        dataset['train']['positive'])
)  # Our corpus (cid => document)

queries = dict(
    zip(dataset['train']['id'], 
        dataset['train']['anchor'])
)  # Our queries (qid => question)

# Create a mapping of relevant document (1 in our case) for each query
relevant_docs = {}  # Query ID to relevant documents (qid => set([relevant_cids])
for q_id in queries:
    relevant_docs[q_id] = [q_id]

matryoshka_evaluators = []
# Iterate over the different dimensions
for dim in matryoshka_dimensions:
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=f"dim_{dim}",
        truncate_dim=dim,  # Truncate the embeddings to a certain dimension
        score_functions={"cosine": cos_sim},
    )
    matryoshka_evaluators.append(ir_evaluator)

# Create a sequential evaluator
evaluator = SequentialEvaluator(matryoshka_evaluators)
#########################################################################################################
#### Evaluating the Model Before Fine-tuning
results = evaluator(model)

for dim in matryoshka_dimensions:
    key = f"dim_{dim}_cosine_ndcg@10"
    print(f"{key}: {results[key]}")
#########################################################################################################
#### Defining the Trainer
from sentence_transformers import SentenceTransformerTrainer

trainer = SentenceTransformerTrainer(
    model=model, # our embedding model
    args=args,  # training arguments we defined above
    train_dataset=dataset.select_columns(
        ["positive", "anchor"]
    ),
    loss=train_loss, # Matryoshka loss
    evaluator=evaluator, # Sequential Evaluator
)
#########################################################################################################
#### Starting Fine-tuning
trainer.train()
# save the best model
trainer.save_model()
#########################################################################################################
#### Evaluating After Fine-tuning
from sentence_transformers import SentenceTransformer

fine_tuned_model = SentenceTransformer(
    args.output_dir, device="cuda" if torch.cuda.is_available() else "cpu"
)
# Evaluate the model
results = evaluator(fine_tuned_model)

# Print the main score
for dim in matryoshka_dimensions:
    key = f"dim_{dim}_cosine_ndcg@10"
    print(f"{key}: {results[key]}")
















