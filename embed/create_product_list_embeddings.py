import pandas as pd
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import time
import requests

PRODUCT_CSV_PATH = "data/product_list.csv"
DESCRIPTION_COLUMN = "Description"
CHUNK_SIZE = 32

PINECONE_API_KEY = ""
HUGGINGFACE_TOKEN = ""  
INDEX_NAME = "products"

client = OpenAI(
    base_url="",
    api_key=""
)

def get_embedding(text: str) -> list:
    try:
        response = client.embeddings.create(
            model="sentence-transformers/all-MiniLM-L6-v2",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error from HuggingFace API: {e}")
        return []

def create_vector_db(csv_path: str = PRODUCT_CSV_PATH, chunk_size: int = CHUNK_SIZE):
    """Create Pinecone vector database from product CSV"""
    print(f"Creating vector database from: {csv_path}")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if it doesn't exist
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # all-MiniLM-L6-v2 embedding dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Created new Pinecone index")
    
    index = pc.Index(INDEX_NAME)
    
    # Process CSV in chunks
    chunk_id = 0
    for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunk_size), desc="Processing chunks"):
        descriptions = chunk[DESCRIPTION_COLUMN].fillna("").astype(str).tolist()
        
        # Get embeddings and prepare vectors
        vectors = []
        for i, desc in enumerate(descriptions):
            if desc.strip():  # Skip empty descriptions
                embedding = get_embedding(desc)
                if embedding:  
                    vectors.append({
                        "id": f"product_{chunk_id}_{i}",
                        "values": embedding,
                        "metadata": {"description": desc}
                    })
            
        # Upsert to Pinecone
        if vectors:
            index.upsert(vectors)
        
        chunk_id += 1
    
    print(f"Vector database created with {index.describe_index_stats()['total_vector_count']} products")

def query_vector_db(query_text: str, top_k: int = 1) -> list:
    """Query the Pinecone vector database"""
    print(f"Querying vector DB for: '{query_text}'")
    
    # Initialize Pinecone and get index
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    # Get query embedding
    query_embedding = get_embedding(query_text)
    
    # Search
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Extract descriptions
    matches = [match['metadata']['description'] for match in results['matches']]
    return matches

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-db", action="store_true", help="Create vector DB from product CSV")
    parser.add_argument("--query", type=str, help="Query the product vector DB")
    
    args = parser.parse_args()
    
    if args.create_db:
        create_vector_db()
    
    if args.query:
        matches = query_vector_db(args.query)
        print("Best match:")
        print(matches[0])