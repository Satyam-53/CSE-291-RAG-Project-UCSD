from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import json
import os

def load_data_from_directory(directory_name: str = './embeddings_data/') -> list[dict]:
    """
    Load embedding data from all JSON files in the specified directory.

    Args:
        directory_name (str): Path to the directory containing JSON files. Defaults to './embeddings_data/'.

    Returns:
        List[Dict]: A list of dictionaries containing the combined data from all JSON files.
    
    Notes:
        - Only files with a '.json' extension are processed.
        - If a file fails to load, an error message is printed and the function continues with the next file.
        - The function always returns the accumulated data, even if some files fail to load.
    """
    embeddings_data = []
    try:
        for filename in os.listdir(directory_name):
            if filename.endswith('.json'):
                file_path = os.path.join(directory_name, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            embeddings_data.extend(data)
                        else:
                            embeddings_data.append(data)
                    print(f"Read {len(data)} embeddings from {file_path}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    except Exception as e:
        print(f"Error accessing directory {directory_name}: {e}")
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
                id=int(datum["id"]), 
                vector=datum["embedding"], 
                payload={"text": datum["chunk"]} # add other metadata that we might need to store (recency, etc..)
            ) for datum in data
        ]
    
        for point in points:
            qdrant_client.upsert(collection_name=collection_name, points=[point])
        print(f"Stored {len(points)} embeddings in Qdrant vector store.")
    except Exception as e:
        print(e)

def main():
    directory = './embeddings_data/'
    embeddings_list = load_data_from_directory(directory)
    persist_chunks_to_qdrant(embeddings_list)

if __name__=='__main__':
    main()
