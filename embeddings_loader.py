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

def get_model_dimension(model_name: str) -> int:
    """
    Returns the embedding dimension for a given model.
    MiniLM: 384, BioBERT/PubMedBERT/SciBERT/BlueBERT: 768
    """
    model_name = model_name.lower()
    if model_name == "minilm":
        return 384
    else:  # biobert, pubmedbert, scibert, bluebert
        return 768

def get_qdrant_client(
    collection_name: str = 'CSE291A-RAG-Project-Phase1',
    host: str = 'localhost', 
    port: int = 6333,
    vector_size: int = 384
) -> QdrantClient:
    qdrant_client = None
    try:
        qdrant_client = QdrantClient(host=host, port=port)
        # Create collection with specified vector dimension
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f'Successfully connected to Qdrant client. Collection created with vector size: {vector_size}')
    except:
        print('Error connecting to Qdrant client.')
    finally:
        return qdrant_client

def persist_chunks_to_qdrant(data: list[dict], collection_name: str = 'CSE291A-RAG-Project-Phase1', model_name: str = 'minilm') -> None:
    try:
        # Get the correct vector dimension for the model
        vector_size = get_model_dimension(model_name)
        qdrant_client = get_qdrant_client(collection_name, vector_size=vector_size)
    
        points = [
            PointStruct(
                id=int(datum["id"]), 
                vector=datum["embedding"], 
                payload={"text": datum["chunk"], "fname": datum["fname"]} # add other metadata that we might need to store (recency, etc..)
            ) for datum in data
        ]
    
        for point in points:
            qdrant_client.upsert(collection_name=collection_name, points=[point])
        print(f"Stored {len(points)} embeddings in Qdrant vector store.")
    except Exception as e:
        print(e)

def main():
    model_name = 'neupubmedbert' #[default = minilm, biobert, pubmedbert, scibert, bluebert, neupubmedbert]
    chunking_strategy = 'overlapping_token_chunks' #['overlapping_token_chunks', overlapping_sentence_chunks, sentence_chunks]

    directory = f"./embeddings_data/{model_name}_{chunking_strategy}"
    embeddings_list = load_data_from_directory(directory)

    collection_name =  f"CSE291A-RAG-Project-Phase1_{model_name}_{chunking_strategy}"
    persist_chunks_to_qdrant(embeddings_list, collection_name, model_name)

if __name__=='__main__':
    main()
