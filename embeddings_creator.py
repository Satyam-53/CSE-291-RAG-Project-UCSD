import math

from utils import checkdir

import torch
from sentence_transformers import SentenceTransformer
import os
import re
import json
import unicodedata

torch.set_grad_enabled(False) # Since we are not doing any training.

def preprocess(text):
    text = unicodedata.normalize("NFKC", text)
    return text

def process_all_processed_file(chunking_strategy, c, overlap, directory_path: str = './processed_dataset/'):
    all_chunks = []
    file_names = []
    try:
        # Iterate over all files in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory_path, filename)
                file_content = read_file(file_path)
                file_content = preprocess(file_content)
                # file_chunks = create_sentence_chunks(file_content)
                file_chunks = []
                if chunking_strategy == 'overlapping_token_chunks':
                    file_chunks = create_overlapping_token_chunks(file_content, c, overlap)
                elif chunking_strategy == 'overlapping_sentence_chunks':
                    file_chunks = create_overlapping_sentence_chunks(file_content, c, overlap)
                elif chunking_strategy == 'sentence_chunks':
                    file_chunks = create_sentence_chunks(file_content)
                else:
                    print(f"Error encountered when chunking file {filename}")
                all_chunks.extend(file_chunks)
                file_names.extend([filename] * len(file_chunks))
        print("Successfully read data from all processed files.")
    except:
        print(f"Error encountered while accessing directory: {directory_path}")
    finally:
        return all_chunks, file_names

def read_file(file_path: str) -> str:
    content = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except:
        print(f"Error encountered while reading file: {file_path}")
    finally:
        return content

# Sentence-based chunking
def create_sentence_chunks(content: str) -> list[str]:
    chunks = []
    try:
        # Split content into paragraphs (assuming newline separates paragraphs in cleaned data)
        paragraphs = content.split('\n')
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                # Split paragraph into sentences using regex
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                chunks.extend(sentences)
        print("Created sentence-based chunks successfully.")
    except Exception as e:
        print(f"Error encountered while creating sentence chunks: {e}")
    finally:
        return chunks

# Overlapping sentence-based chunking
def create_overlapping_sentence_chunks(
    content: str, c: int = 5, overlap: int = 2
) -> list[str]:
    chunks = []
    try:
        # Split into sentences
        sentences = list(map(str.strip, re.split(r'(?<=[.!?])\s+', content)))
        start = 0
        while start < len(sentences):
            end = start + c
            chunk = " ".join(sentences[start:min(end, len(sentences))])
            chunks.append(chunk)
            # Move forward but keep overlap
            start += c - overlap
        print("Created overlapping sentence-based chunks successfully.")
    except Exception as e:
        print(f"Error encountered while creating overlapping sentence chunks: {e}")
    finally:
        return chunks

# Hybrid chunking (fixed-size + overlap)
def create_overlapping_token_chunks(
        content: str, 
        c: int = 500,
        overlap: int = 50, 
        similarity_threshold: float = 0.75) -> list[str]:
    chunks = []
    try:
        # Step 1: Fixed-size chunking with overlap
        words = content.split()
        fixed_chunks = []
        start = 0
        while start < len(words):
            end = start + c
            chunk = " ".join(words[start:min(end, len(words))])
            fixed_chunks.append(chunk)
            start = end - overlap  # overlap ensures context continuity

        # # Step 2: Semantic refinement
        # refined_chunks = []
        # model = get_embedding_model()
        # embeddings = model.encode(fixed_chunks, convert_to_tensor=True)

        # current_chunk = [fixed_chunks[0]]
        # for i in range(1, len(fixed_chunks)):
        #     sim = util.cos_sim(embeddings[i-1], embeddings[i]).item()
        #     if sim > similarity_threshold:
        #         current_chunk.append(fixed_chunks[i])
        #     else:
        #         refined_chunks.append(" ".join(current_chunk))
        #         current_chunk = [fixed_chunks[i]]

        # if current_chunk:
        #     refined_chunks.append(" ".join(current_chunk))

        chunks = fixed_chunks
        print("Created hybrid chunks successfully.")
    except Exception as e:
        print(f"Error encountered while creating hybrid chunks: {e}")
    finally:
        return chunks

# -------------------------------------------------------------------------
# Embedding Approaches:
#
# 1. SentenceTransformer Wrapper:
#    - Uses the SentenceTransformer library to load HuggingFace models.
#    - Provides a simple .encode() interface that handles tokenization,
#      batching, and mean pooling automatically.
#    - Advantage: Easy integration with existing code (semantic/hybrid
#      chunking functions work unchanged). Minimal overhead compared to
#      manual HuggingFace usage.
#    - Limitation: Pooling strategy is fixed (usually mean pooling).
#
# 2. HuggingFace AutoModel + Tokenizer (manual pooling):
#    - Directly loads models via transformers.AutoModel and AutoTokenizer.
#    - Requires manual pooling (CLS token, mean, max, etc.) to create
#      sentence-level embeddings.
#    - Advantage: Full control over pooling strategy and embedding details.
#    - Limitation: More boilerplate code, must adapt downstream functions
#      to accept embeddings created manually.
#
# For RAG pipelines, SentenceTransformer is usually preferred for simplicity
# and consistency. HuggingFace manual pooling is useful if you want to
# experiment with alternative pooling strategies or fine-tune embeddings.
# -------------------------------------------------------------------------

# --- Individual model loaders ---
def load_minilm():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") #Already used this in phase 1.

def load_biobert():
    return SentenceTransformer("dmis-lab/biobert-base-cased-v1.1")

def load_pubmedbert():
    return SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract") #Best as per literature.

def load_neuMlpubmedbert():
    return SentenceTransformer("NeuML/pubmedbert-base-embeddings") #Best as per literature.

def load_scibert():
    return SentenceTransformer("allenai/scibert_scivocab_uncased") #Not very great.

def load_bluebert():
    return SentenceTransformer("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12") #Some hybrid model.

# --- Dispatcher function ---
def get_embedding_model(model_name: str = "minilm") -> SentenceTransformer:
    """
    Returns a SentenceTransformer model based on the given name.
    Options: 'minilm', 'biobert', 'pubmedbert', 'scibert', 'bluebert'
    """
    model_name = model_name.lower() 
    if model_name == "minilm":
        return load_minilm()
    elif model_name == "biobert":
        return load_biobert()
    elif model_name == "pubmedbert":
        return load_pubmedbert()
    elif model_name == "scibert":
        return load_scibert()
    elif model_name == "bluebert":
        return load_bluebert()
    elif model_name == "neupubmedbert":
        return load_neuMlpubmedbert()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def create_embeddings(chunks: list[str], model_name = 'minilm') -> list[float]:
    embeddings = []
    try:
        model = get_embedding_model(model_name)

        # Move model to GPU if available
        if torch.cuda.is_available():
            print("Using GPU for embeddings.")
            model = model.to('cuda')
        else:
            print("GPU not available. Using CPU.")

        # embeddings.extend(model.encode(chunks, show_progress_bar=True).tolist())
        embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu').tolist()
        print("Created embeddings successfully.")
    except:
        print("Error encountered while creating embeddings.")
    finally:
        return embeddings

def persist_embeddings_to_file(
    chunks: list[str], 
    embeddings: list[float],
    fnames,
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
        for i, (chunk, embedding, fname) in enumerate(zip(chunks, embeddings, fnames)):
            entry = {
                "id": str(i),
                "chunk": chunk,
                "embedding": embedding,
                "fname": fname
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

    model_name = 'neupubmedbert' #[default = minilm, biobert, pubmedbert, scibert, bluebert, neupubmedbert]

    input_directory = './processed_dataset/'
    output_filename = 'embeddings.json'

    test_param = [
        # {
        #     'chunking_strategy': 'overlapping_sentence_chunks',
        #     'c': [3, 5, 7],
        #     'overlap_ratio': 3
        # },
        # {
        #     'chunking_strategy': 'overlapping_token_chunks',
        #     'c': [300, 400, 500, 600, 700],
        #     'overlap_ratio': 10
        # },
        {
            'chunking_strategy': 'overlapping_token_chunks',
            'c': [500],
            'overlap_ratio': 10
        }
    ]
    for d in test_param:
        chunking_strategy = d['chunking_strategy']  #['overlapping_token_chunks', overlapping_sentence_chunks, sentence_chunks]

        for c in d['c']:
            overlap_ratio = d['overlap_ratio']
            output_directory = f"./embeddings_data/{model_name}_{chunking_strategy}_{c}_{overlap_ratio}"
            overlap = int(math.ceil(c / d['overlap_ratio']))

            checkdir(output_directory)
            chunks, file_names = process_all_processed_file(chunking_strategy, c, overlap, input_directory)
            embeddings = create_embeddings(chunks, model_name)
            persist_embeddings_to_file(chunks, embeddings, file_names, output_directory, output_filename)

if __name__=='__main__':
    main()