### From https://ai.gopubby.com/enhancing-ai-storytelling-with-multimodal-data-extracting-knowledge-from-videos-and-images-6725d77ae403

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def describe_image(image_path):
    """Generate a text description of an image (poster) using Qwen2-VL."""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # Define the input for the model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Process the input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=None, padding=True, return_tensors="pt").to("cuda")

    # Generate the description
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    description = processor.batch_decode(
        [out[len(inputs.input_ids[0]):] for out in generated_ids], skip_special_tokens=True
    )[0]

    return description

--------------------------------------------------------------------------------
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def describe_video(video_path):
    """Generate a text description of a video (trailer) using Qwen2-VL."""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # Define the input for the model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0},
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    # Process the input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=None, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

    # Generate the description
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    description = processor.batch_decode(
        [out[len(inputs.input_ids[0]):] for out in generated_ids], skip_special_tokens=True
    )[0]

    return description

------------------------------------------------------------------------------
from langchain.embeddings import OpenAIEmbeddings

def generate_embeddings(text):
    """Generate embeddings from a given text using OpenAI Embeddings."""
    embedding_model = OpenAIEmbeddings()
    return embedding_model.embed_text(text)

------------------------------------------------------------------------------
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from chromadb.config import Settings

def process_movies(csv_path, chroma_db_path):
    """
    Process movies listed in a CSV file, generate text descriptions and embeddings,
    and store them in ChromaDB.

    Args:
        csv_path (str): Path to the CSV file containing movie details.
        chroma_db_path (str): Path to persist the ChromaDB storage.
    """
    # Load CSV file
    movies = pd.read_csv(csv_path)

    # Initialize ChromaDB
    chroma_db = Chroma(
        collection_name="movies",
        client_settings=Settings(persist_directory=chroma_db_path),
    )

    for _, row in movies.iterrows():
        title = row["Title"]
        rating = row["Rating"]
        poster_path = row["Path to Poster"]
        trailer_path = row["Path to Trailer"]

        # Generate text descriptions
        poster_description = describe_image(poster_path)
        trailer_description = describe_video(trailer_path)

        # Concatenate all text into a single string
        combined_text = (
            f"Title: {title}\n"
            f"Rate: {rating}\n"
            f"Poster: {poster_description}\n"
            f"Trailer: {trailer_description}"
        )

        # Generate embeddings for the combined text
        embedding = generate_embeddings(combined_text)

        # Add the data to ChromaDB
        try:
            chroma_db.add_texts(
                texts=[combined_text],
                metadatas=[{"Title": title, "Rate": rating}],
                embeddings=[embedding],
            )
        except ValueError as e:
            print(f"Error processing movie '{title}': {e}")

    print("Processing complete. All data has been added to ChromaDB.")

---------------------------------------------------------------------------------------
csv_path = "path/to/movies.csv"
chroma_db_path = "path/to/chroma/db"
process_movies(csv_path, chroma_db_path)

retrieved_movies = retrieval_chain.run({"user_input": "Create a sci-fi story with gore and suspense"})
print("Retrieved Movies:", retrieved_movies)
