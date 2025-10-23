from sentence_transformers import SentenceTransformer
import os
import re
import json


def process_all_processed_file(directory_path: str = './processed_dataset/') -> list[str]:
    all_chunks = []
    try:
        # Iterate over all files in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory_path, filename)
                file_content = read_file(file_path)
                file_chunks = create_chunks(file_content)
                all_chunks.extend(file_chunks)
        print("Successfully read data from all processed files.")
    except:
        print(f"Error encountered while accessing directory: {directory_name}")
    finally:
        return all_chunks

def read_file(file_path: str) -> str:
    content = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except:
        print(f"Error encountered while reading file: {file_name}")
    finally:
        return content

def create_chunks(content: str) -> list[str]:
    chunks = []
    try:
        # Split content into paragraphs (assuming double newline separates paragraphs)
        paragraphs = content.split('\n')
        for paragraph in paragraphs:
            # Strip leading/trailing whitespace
            paragraph = paragraph.strip()
            if paragraph:
                # Split paragraph into sentences using regex
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                chunks.extend(sentences)
        print("Created chunks successfully from processed data sources.")
    except:
        print("Error encountered while creating chunks.")
    finally:
        return chunks

def get_embedding_model() -> SentenceTransformer:
    # Initialize embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def create_embeddings(chunks: list[str]) -> list[int | float]:
    embeddings = []
    try:
        model = get_embedding_model()
        embeddings.extend(model.encode(chunks, show_progress_bar=True).tolist())
        print("Created embeddings successfully.")
    except:
        print("Error encountered while creating embeddings.")
    finally:
        return embeddings

def persist_embeddings_to_file(
    chunks: list[str], 
    embeddings: list[int | float], 
    directory_name: str = './embeddings_data/',
    filename: str = 'embeddings.json'
) -> None:
    try:
        # Save id, chunks (sentences) and corresponding embeddings to a json file
        data = [
            {
                "id": str(i), 
                "chunk": s, 
                "embedding": embedding
            } for i, (s, embedding) in enumerate(zip(chunks, embeddings))
        ]
        
        with open(directory_name+filename, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print("Successfully saved embeddings to embeddings.json file.")
    except:
        print("Error encountered while saving embeddings to file.")

def main():
    input_directory = './processed_dataset/'
    output_directory = './embeddings_data/'
    output_filename = 'embeddings.json'

    chunks = process_all_processed_file(input_directory)
    embeddings = create_embeddings(chunks)
    persist_embeddings_to_file(chunks, embeddings, output_directory, output_filename)

if __name__=='__main__':
    main()
