from difflib import SequenceMatcher

from utils import checkdir

from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import os
import psutil
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

def load_evaluation_data_from_file(
    directory_name: str = './metrics_evaluation_data/', 
    filename: str = 'evaluation_input_data.json'
) -> list[dict]:
    file_path = os.path.join(directory_name, filename)
    evaluation_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            evaluation_data.extend(json.load(f))
        print(f"Read {len(evaluation_data)} records from {file_path} for Evaluation.")
    except:
        print("Error reading evaluation input data from file.")
    finally:
        return evaluation_data

def persist_evaluation_result_to_output_file(
    evaluation_results: list[dict],
    directory_name: str = './metrics_evaluation_data/',
    filename: str = 'evaluation_metrics_result.json'
) -> None:
    file_path = os.path.join(directory_name, filename)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f)
        print("Successfully saved evaluation results to file.")
    except Exception as e:
        print(e)
        print("Error writing evaluation results data to file.")

# def get_embedding_model() -> SentenceTransformer:
#     # Initialize embedding model
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     return model

# --- Dispatcher function ---
def get_embedding_model(model_name: str = "minilm") -> SentenceTransformer:
    """
    Returns a SentenceTransformer model based on the given name.
    Options: 'minilm', 'biobert', 'pubmedbert', 'scibert', 'bluebert'
    """
    model_name = model_name.lower() 
    if model_name == "minilm":
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") #Already used this in phase 1.
    elif model_name == "biobert":
        return SentenceTransformer("dmis-lab/biobert-base-cased-v1.1")
    elif model_name == "pubmedbert":
        return SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract") #Best as per literature.
    elif model_name == "scibert":
        return SentenceTransformer("allenai/scibert_scivocab_uncased") #Not very great.
    elif model_name == "bluebert":
        return SentenceTransformer("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12") #Some hybrid model.
    elif model_name == "neupubmedbert":
        return SentenceTransformer("NeuML/pubmedbert-base-embeddings") #Best as per literature.
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_embedding_vector(model: SentenceTransformer, query: str) -> list:
    query_vector = []
    try:
        query_vector.extend(model.encode(query).tolist())
    except:
        print(f"Error encountered while creating embedding from query: {query}")
    finally:
        return query_vector

def get_qdrant_client(
    host: str = 'localhost', 
    port: int = 6333, 
) -> QdrantClient:
    qdrant_client = None
    try:
        qdrant_client = QdrantClient(host=host, port=port)
        print('Successfully connected to Qdrant.')
    except:
        print('Error connecting to Qdrant.')
    finally:
        return qdrant_client

def get_rag_retrieved_chunks(
    qdrant_client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    top_k = 15
):
    retrieved_chunks = []
    try:
        retrieved_chunks.extend(
            qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                search_params=SearchParams(hnsw_ef=128)
            )
        )
    except Exception as e:
        print(f"Error encountered while retrieving context chunks: {e}")
        print(f"Collection name: {collection_name}")
        print(f"Query vector dimension: {len(query_vector) if query_vector else 'None'}")
    finally:
        return retrieved_chunks



def rerank_with_cross_encoder(query, candidates, top_n = 10):
    if not candidates:
        return []

    # Build list of (query, candidate_text) pairs for the cross-encoder
    texts = [p.payload["text"] for p in candidates]

    # Optionally skip empty texts
    pairs = [(query, t) for t in texts]
    # Get relevance scores
    cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3") # cross-encoder/ms-marco-MiniLM-L-6-v2
    scores = cross_encoder.predict(pairs)

    # Attach scores back to candidates
    scored = list(zip(candidates, scores))

    # Sort by cross-encoder score desc
    scored.sort(key=lambda x: float(x[1]), reverse=True)

    # Take top_n
    reranked = scored[:top_n]

    # Convert to a nicer structure for returning
    results = [cand for cand, _ in reranked]
    return results

def print_metrics_average(file_path):
    # Reading the file content
    with open(file_path, 'r') as f:
        file_content = f.read()

    # Loading the JSON content using json.loads
    json_data = json.loads(file_content)

    # Extracting precision values
    # Assuming the JSON structure is a list of dictionaries like [{"precision": 0.85}, ...]
    recall_values = [item['metrics']['retrieval_metrics']['recall@k'] for item in json_data]
    precision_values = [item['metrics']['retrieval_metrics']['precision@k'] for item in json_data]
    mrr_values = [item['metrics']['retrieval_metrics']['mrr'] for item in json_data]
    ndcg_values = [item['metrics']['retrieval_metrics']['ndcg'] for item in json_data]

    # Calculating the average precision score
    if recall_values:
        average_recall = sum(recall_values) / len(recall_values)
        print(f"Average recall score: {average_recall}")
    else:
        print("No precision values found.")

    if precision_values:
        average_precision = sum(precision_values) / len(precision_values)
        print(f"Average precision score: {average_precision}")
    else:
        print("No recall values found.")

    if mrr_values:
        average_mrr = sum(mrr_values) / len(mrr_values)
        print(f"Average mrr score: {average_mrr}")
    else:
        print("No mrr values found.")

    if ndcg_values:
        average_ndcg = sum(ndcg_values) / len(ndcg_values)
        print(f"Average ndcg score: {average_ndcg}")
    else:
        print("No ndcg values found.")

def evaluate_metrics(
    evaluation_input_data: list[dict], model_name, qdrant_collection_name, rerank, k
) -> list[dict]:
    result_metrics_data = []
    try:
        embedding_model = get_embedding_model(model_name)
        qdrant_client = get_qdrant_client()
        
        for input_data in evaluation_input_data:
            query_category = input_data["category"]
            query = input_data["question"]
            manually_retrieved_chunks = input_data["manually_retrieved_chunks"]

            query_embedding = get_embedding_vector(embedding_model, query)
            
            # Check if query embedding was created successfully
            if not query_embedding or len(query_embedding) == 0:
                print(f"Warning: Empty query embedding for query: {query}")
                continue

            # Start timing and memory (in MB)
            start_time = time.time()
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 ** 2

            # Run retrieval
            number_of_chunks_to_retrieve = k
            rag_retrieved_chunks = get_rag_retrieved_chunks(qdrant_client, qdrant_collection_name, query_embedding, number_of_chunks_to_retrieve)
            if rerank:
                rag_retrieved_chunks = rerank_with_cross_encoder(query, rag_retrieved_chunks, number_of_chunks_to_retrieve)

            rag_retrieved_chunks_fnames = [point.payload['fname'] for point in rag_retrieved_chunks]
            rag_retrieved_chunks = [point.payload['text'] for point in rag_retrieved_chunks]

            # End timing and memory (in MB)
            mem_after = process.memory_info().rss / 1024 ** 2
            end_time = time.time()
            
            retrieval_metrics = get_retrieval_metrics(manually_retrieved_chunks, rag_retrieved_chunks, embedding_model)
            efficiency_metrics = get_efficiency_metrics(start_time, end_time, mem_before, mem_after)

            result_metrics_data.append(
                {
                    "category": query_category,
                    "question": query,
                    "manually_retrieved_chunks": manually_retrieved_chunks,
                    "rag_retrieved_chunks": rag_retrieved_chunks,
                    "fnames": rag_retrieved_chunks_fnames,
                    "metrics": {
                        "retrieval_metrics": retrieval_metrics,
                        "efficiency_metrics": efficiency_metrics
                    }
                }
            )
    except:
        print("Encountered error while evaluating metrics.")
    finally:
        return result_metrics_data

def sequence_match_ratio(s1, s2, threshold=0.9):
    # Split into words
    words1 = s1.split()
    words2 = s2.split()

    # Use SequenceMatcher on word lists (not characters)
    matcher = SequenceMatcher(None, words1, words2)

    # Find longest matching block
    longest_match = max(matcher.get_matching_blocks(), key=lambda m: m.size)

    # Calculate ratio based on string1 length
    ratio = longest_match.size / len(words1)

    return ratio, ratio >= threshold, longest_match

def get_retrieval_metrics(expected_chunks, retrieved_chunks, embedding_model, k=15):
    """
    expected_chunks: list of relevant chunk texts (ground truth)
    retrieved_chunks: list of retrieved chunk texts (top-k)
    k: number of retrieved chunks to evaluate

    Returns: dict of retrieval metrics
    """
    precision_at_k, recall_at_k, hit_ratio_at_k, mrr, ndcg = 0.0, 0.0, 0.0, 0.0, 0.0
    try:

        expected_lower = [e.lower() for e in expected_chunks]
        retrieved_lower = [r.lower() for r in retrieved_chunks]

        # Precision = How many retrieved chunks are correct (i.e. matches ground truth)
        # Precision = #(Matching retrieved chunks) / k

        # Recall = How many Ground Truth chunks were retrieved out of total ground truth chunks
        # Recall = #(Matching ground truth chunks) / #(All ground truth chunks)

        matched_ground_truth, matched_retrieved = set(), set()
        for e in expected_lower:
            max_semantic_score = 0

            for r in retrieved_lower:
                # Condition 1: If full ground truth exists fully within RAG retrieved chunk
                # condition1 = (e in r) or (r in e)

                # Condition 2: sequence similarity >= 0.8
                # To add: from difflib import SequenceMatcher
                similarity_ratio, condition_2, _ = sequence_match_ratio(e, r)

                embed_e = embedding_model.encode(e).tolist()
                embed_r = embedding_model.encode(r).tolist()
                max_semantic_score = max(max_semantic_score, cosine_similarity([embed_e], [embed_r]))

                if condition_2:
                    matched_ground_truth.add(e)
                    matched_retrieved.add(r)
                    break  # count each expected item only once

            # print(max_semantic_score)

        # Precision@k
        precision_at_k = min(len(matched_retrieved) / k, 1.0)

        # Recall@k
        recall_at_k = len(matched_ground_truth) / len(expected_lower) if expected_lower else 0

        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for idx, chunk in enumerate(retrieved_lower, start=1):
            if chunk in matched_retrieved:
                mrr = 1.0 / idx
                break

        # nDCG@k
        dcg = 0.0
        for i, chunk in enumerate(retrieved_lower[:k], start=1):
            rel_i = 1 if chunk in matched_retrieved else 0
            dcg += rel_i / np.log2(i + 1)
        idcg = sum([1 / np.log2(i + 1) for i in range(1, min(len(expected_lower), k) + 1)])
        ndcg = dcg / idcg if idcg > 0 else 0.0

        print("----------- RETRIEVAL METRICS -----------")
        print("Precision @ K  : ", round(precision_at_k, 3))
        print("Recall @ K     : ", round(recall_at_k, 3))
        print("MRR            : ", round(mrr, 3))
        print("NDCG           : ", round(ndcg, 3))
    except:
        print("Error encountered while calculating retrieval metrics.")
    finally:
        return {
            "precision@k": round(precision_at_k, 3),
            "recall@k": round(recall_at_k, 3),
            "mrr": round(mrr, 3),
            "ndcg": round(ndcg, 3)
        }

def get_efficiency_metrics(start_time: time, end_time: time, start_memory: float, end_memory: float) -> dict:
    latency, throughput, memory_used = 0.0, 0.0, 0.0
    try:
        # Efficiency metrics
        latency = end_time - start_time
        throughput = 1 / latency if latency > 0 else 0
        memory_used = end_memory - start_memory

        print("----------- EFFICIENCY METRICS -----------")
        print("Latency(sec)  : ", round(latency, 2))
        print("Throughput(qps): ", round(throughput, 2))
        print("Memory Used(MB): ", round(memory_used, 2))
    except:
        print("Error encountered while calculating efficiency metrics.")
    finally:
        return {
            "latency": round(latency, 2),
            "throughput": round(throughput, 2),
            "memory_used": round(memory_used, 2),
        }

def main():
    model_name = 'neupubmedbert'
    rerank = True
    input_filename = 'evaluation_input_data.json'
    output_filename = 'evaluation_metrics_result.json'

    test_param = [
        # {
        #     'chunking_strategy': 'overlapping_sentence_chunks',
        #     'c': [3, 5, 7],
        #     'k': [5, 10, 15, 20]
        # },
        # {
        #     'chunking_strategy': 'overlapping_token_chunks',
        #     'c': [300, 400, 500, 600, 700],
        #     'overlap_ratio': 10,
        #     'k': [5, 10, 15, 20]
        # },
        {
            'chunking_strategy': 'overlapping_token_chunks',
            'c': [600],
            'overlap_ratio': 10,
            'k': [20]
        }
    ]

    for d in test_param:
        chunking_strategy = d['chunking_strategy'] #['overlapping_token_chunks', overlapping_sentence_chunks, sentence_chunks]

        for c in d['c']:
            for k in d['k']:
                overlap_ratio = d['overlap_ratio']
                collection_name = f"CSE291A-RAG-Project-Phase1_{model_name}_{chunking_strategy}_{c}_{overlap_ratio}"  # Name of the collection in qdrant (matches embeddings_loader.py format).
                input_directory_name = f"./metrics_evaluation_data/"
                output_directory_name = f"./metrics_evaluation_data/{model_name}_{chunking_strategy}_{c}_{overlap_ratio}_{k}{'_with_rerank' if rerank else ''}"

                checkdir(output_directory_name)

                input_evaluation_data = load_evaluation_data_from_file(input_directory_name, input_filename)
                output_evaluation_data = evaluate_metrics(input_evaluation_data, model_name, collection_name, rerank, k)
                persist_evaluation_result_to_output_file(output_evaluation_data, output_directory_name, output_filename)

                print(f'Chunking Strategy: {chunking_strategy}, c: {c}, k: {k}')
                print_metrics_average(os.path.join(output_directory_name, output_filename))

if __name__=='__main__':
    main()


# INPUT FILE json structure:
# [
#   {
#     category: "",
#     question: "",
#     manually_retrieved_chunks: [
#                   "...",
#                   "...",
#               ]
#   }
# ]

# OUTPUT FILE json structure:
# [
#   {
#     category: "",
#     question: "",
#     manually_retrieved_chunks: [
#                   "...",
#                   "...",
#               ],
#     rag_retrieved_chunks: [
#                   "...",
#                   "...",
#               ],
#     metrics: {
#         retrival_metrics: {
#             recall@5: ,
#             precision@5: ,
#             hit_ratio@5: ,
#             mrr: ,
#             ndcg: 
#         },
#         efficiency_metrics: {
#             lateny_in_sec: ,
#             throughput_in_qps: ,
#             memory_usage_in_mb: ,
#         }
#     }
#   }
# ]
