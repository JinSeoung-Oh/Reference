### From https://medium.com/@benitomartin/building-a-multimodal-llm-application-with-pymupdf4llm-59753cb44483

!pip install -qq pymupdf4llm 
!pip install -qq llama-index 
!pip install -qq llama-index-vector-stores-qdrant 
!pip install -qq git+https://github.com/openai/CLIP.git
!pip install -qq llama-index-embeddings-clip 
!pip install -qq llama-index qdrant-client 

os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')

# Perform the markdown conversion
docs = pymupdf4llm.to_markdown(doc="/content/document.pdf",
                                  page_chunks = True,
                                  write_images = True,
                                  image_path = "/content/images",
                                  image_format = "jpg")

llama_documents = []

for document in docs:
    # Extract just the 'metadata' field and convert certain elements as needed
    metadata = {
        "file_path": document["metadata"].get("file_path"),
        "page": str(document["metadata"].get("page")),
        "images": str(document.get("images")),
        "toc_items": str(document.get("toc_items")),
        
    }

    # Create a Document object with just the text and the cleaned metadata
    llama_document = Document(
        text=document["text"],
        metadata=metadata,  
        text_template="Metadata: {metadata_str}\n-----\nContent: {content}",  
    )

    llama_documents.append(llama_document)

# Initialize Qdrant client
client = qdrant_client.QdrantClient(location=":memory:")

# Create a collection for text data
client.create_collection(
    collection_name="text_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Create a collection for image data
client.create_collection(
    collection_name="image_collection",
    vectors_config=VectorParams(size=512, distance=Distance.COSINE)
)

# Initialize Collections
text_store = QdrantVectorStore(
    client=client, collection_name="text_collection"
)

image_store = QdrantVectorStore(
    client=client, collection_name="image_collection"
)

storage_context = StorageContext.from_defaults(
    vector_store=text_store, image_store=image_store
)

# context images
image_path = "/content/images"
image_documents = SimpleDirectoryReader(image_path).load_data()

index = MultiModalVectorStoreIndex.from_documents(
    llama_documents + image_documents,
    storage_context=storage_context)

# Set query and retriever
query = "Could you provide an image of the Multi-Head Attention?"

retriever = index.as_retriever(similarity_top_k=1, image_similarity_top_k=1) 

retrieval_results = retriever.retrieve(query)

import matplotlib.pyplot as plt
from PIL import Image

def plot_images(image_paths):
    images_shown = 0
    plt.figure(figsize=(16, 9))
    for img_path in image_paths:
        if os.path.isfile(img_path):
            image = Image.open(img_path)

            plt.subplot(2, 3, images_shown + 1)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])

            images_shown += 1
            if images_shown >= 9:
                break


retrieved_image = []

for res_node in retrieval_results:
    if isinstance(res_node.node, ImageNode):
        print("Highest Scored ImageNode")
        print("-----------------------")
        retrieved_image.append(res_node.node.metadata["file_path"])
    else:
        print("Highest Scored TextNode")
        print("-----------------------")

        display_source_node(res_node, source_length=200)
