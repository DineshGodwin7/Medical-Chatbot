


from src.helper import load_pdf, text_split, download_hugging_face_embeddings 
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_pinecone import PineconeVectorStore  
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Retrieve Pinecone API key
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Load and process data
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

# Download Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone client
pc_client = PineconeClient(api_key=PINECONE_API_KEY)
index_name = "quickstart"

# Define the embedding dimension based on your model
EMBEDDING_DIMENSION = 384

# Check if the index exists and create it if it doesn't
existing_indexes = pc_client.list_indexes()
if index_name not in existing_indexes: 
    try:
        pc_client.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"Index '{index_name}' created.")
    except Exception as e:
        print(f"An error occurred while creating the index: {e}")
else:
    print(f"Index '{index_name}' already exists. Skipping creation.")

# Retrieve the index object
index = pc_client.Index(index_name)

# Initialize PineconeVectorStore with the required embedding function
pinecone_store = PineconeVectorStore(
    index=index,
    embedding=embeddings  # Use 'embedding' instead of 'embedding_function'
)

# Upsert vectors
index.upsert(
    vectors=[
        {
            "id": "vec1", 
            "values": [0.1] * EMBEDDING_DIMENSION, 
            "metadata": {"genre": "drama"}
        }, 
        {
            "id": "vec2", 
            "values": [0.2] * EMBEDDING_DIMENSION, 
            "metadata": {"genre": "action"}
        }, 
        {
            "id": "vec3", 
            "values": [0.3] * EMBEDDING_DIMENSION, 
            "metadata": {"genre": "drama"}
        }, 
        {
            "id": "vec4", 
            "values": [0.4] * EMBEDDING_DIMENSION, 
            "metadata": {"genre": "action"}
        }
    ],
    namespace="ns1"
)

# Query the vectors
query_results = index.query(
    namespace="ns1",
    vector=[0.1] * EMBEDDING_DIMENSION,
    top_k=2,
    include_values=True,
    include_metadata=True,
    filter={"genre": {"$eq": "action"}}
)

# Print the query results
print(query_results)







