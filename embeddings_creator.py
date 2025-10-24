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

def create_embeddings(chunks: list[str]) -> list[float]:
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
    embeddings: list[list[float]], 
    directory_name: str = './embeddings_data/',
    base_filename: str = 'embeddings'
) -> None:
    """
    Saves text chunks and their corresponding embeddings to multiple JSON files, 
    each capped at approximately 100 MB in size.

    Parameters:
        chunks (list[str]): List of text segments or sentences.
        embeddings (list[list[float]]): List of embedding vectors corresponding to each chunk.
        directory_name (str): Directory where JSON files will be saved. Defaults to './embeddings_data/'.
        base_filename (str): Base name for output files. Files will be named as base_filename_0.json, base_filename_1.json, etc.

    Returns:
        None
    """
    try:
        # Ensure the output directory exists
        os.makedirs(directory_name, exist_ok=True)

        max_file_size = 90 * 1024 * 1024  # 100 MB in bytes
        batch = []        # Current batch of entries
        batch_size = 0    # Size of current batch in bytes
        file_index = 0    # Index for naming output files

        # Iterate through chunks and embeddings
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            entry = {
                "id": str(i),
                "chunk": chunk,
                "embedding": embedding
            }

            # Estimate size of entry in bytes
            entry_json = json.dumps(entry)
            entry_size = len(entry_json.encode('utf-8'))

            # If adding this entry exceeds the size limit, write current batch to file
            if batch_size + entry_size > max_file_size:
                filename = f"{base_filename}_{file_index}.json"
                with open(os.path.join(directory_name, filename), "w", encoding="utf-8") as f:
                    json.dump(batch, f)
                print(f"Saved {filename} with {len(batch)} entries.")

                # Reset batch and counters
                file_index += 1
                batch = []
                batch_size = 0

            # Add entry to batch
            batch.append(entry)
            batch_size += entry_size

        # Save any remaining entries
        if batch:
            filename = f"{base_filename}_{file_index}.json"
            with open(os.path.join(directory_name, filename), "w", encoding="utf-8") as f:
                json.dump(batch, f)
            print(f"Saved {filename} with {len(batch)} entries.")

    except Exception as e:
        print(f"Error encountered while saving embeddings: {e}")

def main():
    input_directory = './processed_dataset/'
    output_directory = './embeddings_data/'
    output_filename = 'embeddings.json'

    chunks = process_all_processed_file(input_directory)
    embeddings = create_embeddings(chunks)
    persist_embeddings_to_file(chunks, embeddings, output_directory, output_filename)

if __name__=='__main__':
    main()
