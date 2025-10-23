from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import json
import os

def load_data_from_file(
    directory_name: str = './embeddings_data/', 
    filename: str = 'embeddings.json'
) -> list[dict]:
    file_path = os.path.join(directory_name, filename)
    embeddings_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            embeddings_data.extend(json.load(f))
        print(f"Read {len(embeddings_data)} embeddings from {file_path}")
    except:
        print(f"Error reading embeddings from file.")
    finally:
        return embeddings_data

def get_qdrant_client(
    host: str = 'localhost', 
    port: int = 6333, 
    collection_name: str = 'CSE291A-RAG-Project-Phase1'
) -> QdrantClient:
    qdrant_client = None
    try:
        qdrant_client = QdrantClient(host=host, port=port)
        # Create collection
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print('Successfully connected to Qdrant client.')
    except:
        print('Error connecting to Qdrant client.')
    finally:
        return qdrant_client

def persist_chunks_to_qdrant(data: list[dict], collection_name: str = 'CSE291A-RAG-Project-Phase1') -> None:
    try:
        qdrant_client = get_qdrant_client()
    
        points = [
            PointStruct(
                id=datum["id"], 
                vector=datum["embedding"], 
                payload={"text": datum["chunk"]} # add other metadata that we might need to store (recency, etc..)
            ) for datum in data
        ]
    
        qdrant_client.upsert(collection_name=collection_name, points=points)
        print(f"Stored {len(points)} embeddings in Qdrant vector store.")
    except:
        print('Error saving embeddings to Qdrant.')

def main():
    directory = './embeddings_data/'
    filename = 'embeddings.json'
    embeddings_list = load_data_from_file(directory, filename)
    persist_chunks_to_qdrant(embeddings_list)

if __name__=='__main__':
    main()
