### From https://pub.towardsai.net/designing-multimodal-ai-search-engines-for-smarter-online-retail-43bafa996238

"""
1. The Need for Multimodal Search and Its Challenges
   Modern search engines have moved beyond text-only retrieval, now supporting multimodal search that combines text with images.
   For example, Google’s Circle to Search allows users to search with both text and visuals, 
   helping them find what they need faster and more accurately. However, building such systems is far from trivial.
   The Shein e-commerce dataset illustrates this well: each product contains heterogeneous data such as textual descriptions, 
   images, structured attributes (color, category, brand), and price. 
   This variety requires a search system capable of understanding and indexing text, visuals, and structured fields simultaneously.
  
   Additional challenges include:
   -a. Query Ambiguity: A query like “comfy top” is vague—does “comfy” mean cotton, oversized, sleeveless? 
                        Dataset attributes may not align with user language, creating a semantic gap.
   -b. Scalability: Even filtered datasets have tens of thousands of SKUs, while production systems handle millions. 
                    Vector indexes, hybrid models, and efficient filtering pipelines are essential for millisecond-level retrieval.
   -c. Personalization: Two users searching “floral dress” may have different preferences (boho maxi vs. skater style). 
                        Search engines must integrate user behavior and preferences in real time.
   -d. Metadata Filtering: E-commerce requires filters (price, size, color, material, sleeve length). 
                           Structured attributes must support seamless faceted search without slowing retrieval.
   -e. Real-Time Requirements: Queries must run under 200ms, even at scale, across multimodal data, filtering, and optional reranking.

2. Vector Search and Embeddings
   Modern platforms use vector search (semantic search) to capture meaning beyond keyword matching.
   -a. Vector Embeddings: Product text and images are converted into high-dimensional vectors that encode semantic meaning. 
                          Similar products are placed close in the embedding space.
       -1. Example: “Red satin evening gown” and “Black sleeveless cocktail dress” cluster as formal wear, despite different keywords.
   -b. Image Embeddings: Models like CLIP map both text and images into the same embedding space, enabling cross-modal search. 
                         An uploaded shoe photo and the phrase “red high-heeled shoe” map to similar vectors.
   -c. Similarity Metric: Cosine similarity measures closeness between vectors; scores near 1 indicate high semantic similarity.

3. Sparse Vectors
   -a. Need: Dense vectors are strong for semantic similarity but less effective for precise keyword matching in short queries.
   -b. Features: High-dimensional representations with mostly zero values, where a few active dimensions correspond to keywords 
                 (e.g., “white,” “floral,” “dress”).
   -c. Applications: Methods like BM25 and SPLADE preserve term-level precision. BM25 combines:
       -1. Term Frequency (TF): how often a query word appears in a document.
       -2. Inverse Document Frequency (IDF): how rare a word is across the corpus.
   -e. Formula:
       BM25(D, Q) = Σ(IDF(q) * ((TF(q, D) * (k1+1)) / (TF(q, D) + k1*(1-b+b*|D|/avgdl))))

4. Metadata Filtering (Payload & Metadata)
   User queries often involve structured constraints, e.g., “Red sneakers under ₹2000” or “Cotton tops from H&M in size M.”
   -a. Structured Attributes: brand, color, category, price, size, material, etc., stored as payloads in vector databases.
   -b. Process: Semantic retrieval finds candidates, then filters apply hard constraints.
       -1. Example payload (Qdrant format):
           {
             "id": "SKU123",
             "vector": [0.12, 0.75, -0.33, ...],
             "payload": {
               "brand": "SHEIN",
               "color": "White",
               "category": "Dress",
               "price": 1299,
               "size": ["M", "L"],
               "material": "Cotton"
             }
           }

       -2. Example filter: category = Dress; price < 1500; size = M
      This ensures semantic similarity plus precise rule-based matching.

5. Reranking
   Vector similarity retrieves top-k candidates efficiently but may miss subtle intent or contextual relevance.
   -a. Method: Feed the query and top-k candidates into a stronger model (e.g., cross-encoder, transformer reranker) that
               evaluates each pair jointly.
   -b. Effect: Produces refined relevance scores that better capture user intent and domain-specific priorities, 
               improving final ranking quality.
"""

! docker pull qdrant/qdrant
! docker run -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant

! pip install qdrant-client datasets fastembed transformers qdrant-client[fastembed] openai

from qdrant_client import models, QdrantClient
from google.colab import userdata
import pandas as pd
import os
import urllib
import pandas as pd
import numpy as np
import json
from typing import Optional
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding, ImageEmbedding
from sklearn.decomposition import PCA
from qdrant_client.models import Distance, VectorParams, models
from qdrant_client.models import PayloadSchemaType
from qdrant_client.models import PointStruct, SparseVector, Document

client = QdrantClient(
    url="YOUR_QDRANT_CLOUD_INSTANCE_URL",
    api_key=userdata.get('qdrant_api_key'),
)

path = "https://raw.githubusercontent.com/luminati-io/eCommerce-dataset-samples/main/shein-products.csv"
df = pd.read_csv(path)
df = df.dropna(subset=['color'])
df['description'] = df['description'].str.replace('Free Returns ✓ Free Shipping✓.', '', regex=False).str.strip()

documents = []
for _, row in df.iterrows():
    doc_text = ""
    if pd.notna(row.get('product_name')):
        doc_text += str(row['product_name'])
    if pd.notna(row.get('description')):
        if doc_text:
            doc_text += " " + str(row['description'])
        else:
            doc_text = str(row['description'])
    if pd.notna(row.get('category')):
        doc_text += " " + str(row['category'])


    documents.append(doc_text)

def download_images_for_row(row, base_folder="data/images") -> Optional[str]:
    folder_name = str(row.name)
    folder_path = os.path.join(base_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)


    urls = []
    # Handle main_image
    main_image = row.get("main_image")
    if isinstance(main_image, str) and main_image.startswith("http"):
        urls.append(main_image)


    # Handle image_urls (should be a stringified list)
    image_urls_raw = row.get("image_urls")
    try:
        image_urls = json.loads(image_urls_raw) if isinstance(image_urls_raw, str) else []
        if isinstance(image_urls, list):
            urls.extend(image_urls)
    except Exception as e:
        print(f"Failed to parse image_urls: {e}")


    # Download images
    for i, url in enumerate(urls):
        try:
            ext = os.path.splitext(url)[-1].split("?")[0]
            filename = f"image_{i}{ext or '.jpg'}"
            filepath = os.path.join(folder_path, filename)
            if not os.path.exists(filepath):
                urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            print(f"Failed to download {url}: {e}")


    if os.listdir(folder_path):  # At least one image downloaded
        return folder_path
    return None


# Apply to each row
df["image_folder_path"] = df.apply(download_images_for_row, axis=1)


# Drop rows with failed downloads
df = df.dropna(subset=["image_folder_path"])


# Preview sample
display(df[["main_image", "image_folder_path"]].sample(5).T)

dense_embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
dense_embeddings = list(dense_embedding_model.embed(doc for doc in documents))

minicoil_embedding_model = SparseTextEmbedding("Qdrant/minicoil-v1")
minicoil_embeddings = list(minicoil_embedding_model.embed(doc for doc in documents))

clip_embedding_model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")


def get_average_image_embedding(folder_path: str) -> Optional[np.ndarray]:
    """Get average embedding of all images in a folder"""
    if not os.path.exists(folder_path):
        return None


    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'))]


    if not image_files:
        return None


    image_paths = [os.path.join(folder_path, f) for f in image_files]


    try:
        # Get embeddings for all images in the folder
        embeddings = list(clip_embedding_model.embed(image_paths))


        if embeddings:
            # Convert to numpy arrays and compute average
            embedding_arrays = [np.array(emb) for emb in embeddings]
            average_embedding = np.mean(embedding_arrays, axis=0)
            return average_embedding


    except Exception as e:
        print(f"Error processing images in {folder_path}: {e}")
        return None


    return None

image_embeddings = []
for _, row in df.iterrows():
    folder_path = row['image_folder_path']
    avg_embedding = get_average_image_embedding(folder_path)
    image_embeddings.append(avg_embedding)


# Filter out None values and keep track of valid indices
valid_indices = [i for i, emb in enumerate(image_embeddings) if emb is not None]
valid_image_embeddings = [image_embeddings[i] for i in valid_indices]

weighted_avg = (0.6 * main_emb + 0.2 * side_emb + 0.2 * detail_emb)

def get_concatenated_image_embedding(folder_path: str) -> Optional[np.ndarray]:
    """Concatenate embeddings of all images in a folder"""
    if not os.path.exists(folder_path):
        return None


    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'))]


    if not image_files:
        return None


    image_paths = [os.path.join(folder_path, f) for f in image_files]


    try:
        embeddings = list(clip_embedding_model.embed(image_paths))


        if embeddings:
            # Concatenate all embeddings into one long vector
            embedding_arrays = [np.array(emb) for emb in embeddings]
            concatenated_embedding = np.concatenate(embedding_arrays, axis=0)
            return concatenated_embedding


    except Exception as e:
        print(f"Error processing images in {folder_path}: {e}")
        return None


    return None

concatenated_embeddings = []
for _, row in df.iterrows():
    folder_path = row['image_folder_path']
    concat_emb = get_concatenated_image_embedding(folder_path)
    concatenated_embeddings.append(concat_emb)


valid_indices = [i for i, emb in enumerate(concatenated_embeddings) if emb is not None]
valid_concat_embeddings = [concatenated_embeddings[i] for i in valid_indices]


pca = PCA(n_components=512)  # Choose your target dimension
reduced_embeddings = pca.fit_transform(valid_concat_embeddings)

late_interaction_embedding_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")
late_interaction_embeddings = list(late_interaction_embedding_model.embed(doc for doc in documents))

df["dense_embedding"] = dense_embeddings
df["image_embedding"] = image_embeddings
df["sparse_embedding"] = minicoil_embeddings
df["late_interaction_embedding"] = late_interaction_embeddings

client.recreate_collection(
    "shein_products",
    vectors_config={
        "all-MiniLM-L6-v2": models.VectorParams(
            size=len(dense_embeddings[0]),
            distance=models.Distance.COSINE,
        ),
        "colbertv2.0": models.VectorParams(
            size=len(late_interaction_embeddings[0][0]),
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0)  # Disable HNSW for reranking
        ),
        "clip": VectorParams(size=512, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "minicoil": models.SparseVectorParams(
            modifier=models.Modifier.IDF
        )
    },
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        ),
    ),
)

client.create_payload_index(
    collection_name="shein_products",
    field_name="color",
    field_schema=PayloadSchemaType.KEYWORD  # keyword (for string match)
)

client.create_payload_index(
    collection_name="shein_products",
    field_name="final_price",
    field_schema=PayloadSchemaType.FLOAT # float (for range queries)
)

client.create_payload_index(
    collection_name="shein_products",
    field_name="category",
    field_schema=models.TextIndexParams(
        type="text",
        tokenizer=models.TokenizerType.WORD,
        min_token_len=2,
        max_token_len=10,
        lowercase=True,
    ),
)


client.create_payload_index("shein_products", "rating", PayloadSchemaType.FLOAT)
client.create_payload_index("shein_products", "brand", PayloadSchemaType.KEYWORD)
client.create_payload_index("shein_products", "category", PayloadSchemaType.KEYWORD)
client.create_payload_index("shein_products", "product_name", PayloadSchemaType.KEYWORD)
client.create_payload_index("shein_products", "currency", PayloadSchemaType.KEYWORD)

def upload_points_in_batches(df, documents, batch_size=20):
    """Upload points in small batches to avoid payload size limits"""
    # Calculate average length only for documents that correspond to rows in the filtered df
    # This requires mapping back to the original documents
    original_indices = df.index.tolist()
    relevant_documents = [documents[i] for i in original_indices if i < len(documents)]
    avg_documents_length = sum(len(document.split()) for document in relevant_documents) / len(relevant_documents) if relevant_documents else 0

    total_uploaded = 0
    batch_points = []

    # Use enumerate to get a continuous index for accessing documents list
    for enum_idx, (df_idx, row) in enumerate(df.iterrows()):
        if row['image_embedding'] is None:
            continue

        # Use the original dataframe index to access the correct document
        original_doc_idx = df_idx
        if original_doc_idx >= len(documents):
            print(f"Warning: Original index {original_doc_idx} out of bounds for documents list. Skipping.")
            continue
        dense_emb = row['dense_embedding'].tolist() if isinstance(row['dense_embedding'], np.ndarray) else row['dense_embedding']
        late_interaction_emb = row['late_interaction_embedding'].tolist() if isinstance(row['late_interaction_embedding'], np.ndarray) else row['late_interaction_embedding']
        image_emb = row['image_embedding'].tolist() if isinstance(row['image_embedding'], np.ndarray) else row['image_embedding']
        minicoil_doc = Document(
            text=documents[original_doc_idx], # Use original index for the correct document text
            model="Qdrant/minicoil-v1",
            options={"avg_len": avg_documents_length}
        )

        point = PointStruct(
            id=original_doc_idx, # Use the original df index as the point ID
            vector={
                "all-MiniLM-L6-v2": dense_emb,
                "minicoil": minicoil_doc,
                "colbertv2.0": late_interaction_emb,
                "clip": image_emb,
            },
            payload={
                "document": documents[original_doc_idx], # Use original index for payload document
                "product_name": str(row.get('product_name', '')),
                "final_price": float(row.get('final_price', 0)) if pd.notna(row.get('final_price')) else 0.0,
                "currency": str(row.get('currency', ''))[:10],
                "rating": float(row.get('rating', 0)) if pd.notna(row.get('rating')) else 0.0,
                "category": str(row.get('category', ''))[:100],
                "brand": str(row.get('brand', ''))[:100],
                "image_path": str(row.get('main_image', '')),
                "color":  str(row.get('color', '')),
                "image_url": str(row.get('main_image', ''))
            }
        )
        batch_points.append(point)
        # Upload when batch is full
        if len(batch_points) >= batch_size:
            client.upsert(collection_name="shein_products", points=batch_points, wait=True) # Added wait=True for robustness
            total_uploaded += len(batch_points)
            print(f"Uploaded batch: {total_uploaded} points")
            batch_points = []

    # Upload remaining points
    if batch_points:
        client.upsert(collection_name="shein_products", points=batch_points, wait=True) 
        total_uploaded += len(batch_points)
        print(f"Final batch uploaded: {total_uploaded} total points")

upload_points_in_batches(df, documents, batch_size=20)



query="Women's running shoes"
dense_vectors = list(dense_embedding_model.query_embed([query]))[0]

    prefetch = [
        models.Prefetch(
            query=dense_vectors,
            using="all-MiniLM-L6-v2",
            limit=limit,
        ),
        models.Prefetch(
            query=models.Document(
                text=query,
                model="Qdrant/minicoil-v1"
            ),
            using="minicoil",
            limit=limit,
        ),
    ]

    results = client.query_points(
        collection_name="shein_products",
        query=dense_vectors,
        prefetch=prefetch,
        with_payload=True,
        limit=limit,
        using="all-MiniLM-L6-v2",  
    )

dense_vectors = list(dense_embedding_model.query_embed([query]))[0]
    late_vectors = list(late_interaction_embedding_model.query_embed([query]))[0]
    prefetch = [
        models.Prefetch(
            query=dense_vectors,
            using="all-MiniLM-L6-v2",
            limit=limit * 2,
        ),
        models.Prefetch(
            query=models.Document(
                text=query,
                model="Qdrant/minicoil-v1"
            ),
            using="minicoil",
            limit=limit * 2,
        ),
    ]

    # Final reranking with late interaction
    results = client.query_points(
        "shein_products",
        prefetch=prefetch,
        query=late_vectors,
        using="colbertv2.0",
        with_payload=True,
        limit=limit,
    )

query_image_path="/content/data/images/2/image_5.jpg"
image_vectors = list(clip_embedding_model.embed([query_image_path]))[0]
    # Direct image similarity search (no prefetch needed)
    results = client.query_points(
        "shein_products",
        query=image_vectors.tolist(),
        using="clip",
        with_payload=True,
        limit=limit,
    )

query="blue shoes",
query_image_path="/content/data/images/45/image_2.jpg"
dense_vectors = list(dense_embedding_model.query_embed([query]))[0]
image_vectors = list(clip_embedding_model.embed([query_image_path]))[0]
prefetch = [
    models.Prefetch(
        query=dense_vectors,
        using="all-MiniLM-L6-v2",
        limit=limit * 2,
    ),
    models.Prefetch(
        query=models.Document(
            text=query,
            model="Qdrant/minicoil-v1"
        ),
        using="minicoil",
        limit=limit * 2,
    ),
    models.Prefetch(
        query=image_vectors.tolist(),
        using="clip",
        limit=limit * 2,
    ),
]

# Use late interaction embeddings for final reranking
late_vectors = list(late_interaction_embedding_model.query_embed([query]))[0]

results = client.query_points(
    "shein_products",
    prefetch=prefetch,
    query=late_vectors,
    using="colbertv2.0",
    with_payload=True,
    limit=limit,
)
