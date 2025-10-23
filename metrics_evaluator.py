from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, SearchParams
from sentence_transformers import SentenceTransformer
import json
import os

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
    directory_name: str = './metrics_evaluation_data/',
    filename: str = 'evaluation_metrics_result.json',
    evaluation_results: list[dict]
) -> None:
    file_path = os.path.join(directory_name, filename)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print("Successfully saved evaluation results to file.")
    except:
        print("Error writing evaluation results data to file.")

def get_embedding_model() -> SentenceTransformer:
    # Initialize embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def get_embedding_vector(model: SentenceTransformer, query: str) -> list
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
    collection_name: str = 'CSE291A_RAG_Project_Phase1'
) -> QdrantClient | None:
    qdrant_client = None
    try:
        qdrant_client = QdrantClient(host=host, port=port)
        # Create collection
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print('Successfully connected to Qdrant.')
    except:
        print('Error connecting to Qdrant.')
    finally:
        return qdrant_client

def get_rag_retrieved_chunks(
    qdrant_client: QdrantClient,
    collection_name: str,
    embedding_vector: list[int | float],
    top_k = 5
) -> dict:
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
    except:
        print("Error encountered while retrieving context chunks.")
    finally:
        return retrieved_chunks

def evaluate_metrics(
    evaluation_input_data: list[dict]
) -> list[dict]:
    result_metrics_data = []
    try:
        embedding_model = get_embedding_model()
        qdrant_collection_name = 'CSE291A_RAG_Project_Phase1'
        qdrant_client = get_qdrant_client()
        
        for input_data in evaluation_input_data:
            query_category = input_data["category"]
            query = input_data["question"]
            manually_retrieved_chunks = input_data["manually_retrieved_chunks"]

            query_embedding = get_embedding_vector(embedding_model, query)

            # Start timing and memory (in MB)
            start_time = time.time()
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 ** 2

            # Run retrieval
            number_of_chunks_to_retrieve = len(manually_retrieved_chunks)
            rag_retrieved_chunks = get_rag_retrieved_chunks(model, qdrant_client, qdrant_collection_name, query, number_of_chunks_to_retrieve)

            # End timing and memory (in MB)
            mem_after = process.memory_info().rss / 1024 ** 2
            end_time = time.time()
            
            retrieval_metrics = get_retrieval_metrics(manually_retrieved_chunks, rag_retrieved_chunks)
            efficiency_metrics = get_efficiency_metrics(start_time, end_time, start_memory, end_memory)

            result_metrics_data.append(
                {
                    "category": query_category,
                    "question": query,
                    "manually_retrieved_chunks": manually_retrieved_chunks,
                    "rag_retrieved_chunks": rag_retrieved_chunks,
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

def get_retrieval_metrics(expected_chunks, retrieved_chunks, k=5):
    """
    expected_chunks: list of relevant chunk texts (ground truth)
    retrieved_chunks: list of retrieved chunk texts (top-k)
    k: number of retrieved chunks to evaluate

    Returns: dict of retrieval metrics
    """
    precision_at_k, recall_at_k, hit_ratio_at_k, mrr, ndcg = 0.0, 0.0, 0.0, 0.0, 0.0
    try:
        expected_set = set(expected_chunks[:k])
        retrieved_set = set(retrieved_chunks[:k])

        # Precision@k
        precision_at_k = len(expected_set & retrieved_set) / k

        # Recall@k
        recall_at_k = len(expected_set & retrieved_set) / len(expected_set) if expected_set else 0

        # Hit Ratio@k
        hit_ratio_at_k = 1 if expected_set & retrieved_set else 0

        # MRR (Mean Reciprocal Rank)
        ranks = [i + 1 for i, chunk in enumerate(retrieved_chunks[:k]) if chunk in expected_set]
        mrr = 1 / ranks[0] if ranks else 0

        # DCG and nDCG
        relevance_scores = [1 if chunk in expected_set else 0 for chunk in retrieved_chunks[:k]]
        dcg = sum([score / np.log2(i + 2) for i, score in enumerate(relevance_scores)])
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = sum([score / np.log2(i + 2) for i, score in enumerate(ideal_scores)])
        ndcg = dcg / idcg if idcg > 0 else 0

        # print("----------- RETRIEVAL METRICS -----------")
        # print("Precision @ K  : ", retrieval_metrics["precision@k"])
        # print("Recall @ K     : ", retrieval_metrics["recall@k"])
        # print("Hit Ratio @ K  : ", retrieval_metrics["hit_ratio@k"])
        # print("MRR            : ", retrieval_metrics["mrr"])
        # print("NDCG           : ", retrieval_metrics["ndcg"])
    except:
        print("Error encountered while calculating retrieval metrics.")
    finally:
        return {
            "precision@k": round(precision_at_k, 3),
            "recall@k": round(recall_at_k, 3),
            "hit_ratio@k": hit_ratio_at_k,
            "mrr": round(mrr, 3),
            "ndcg": round(ndcg, 3)
        }

def get_efficiency_metrics(start_time: time, end_time: time, start_memory: int | float, end_memory: int | float) -> dict:
    latency, throughput, memory_used = 0.0, 0.0, 0.0
    try:
        # Efficiency metrics
        latency = end_time - start_time
        throughput = 1 / latency if latency > 0 else 0
        memory_used = mem_after - mem_before

        # print("----------- EFFICIENCY METRICS -----------")
        # print("Latency(sec)  : ", latency)
        # print("Throughput(qps): ", throughput)
        # print("Memory Used(MB): ", memory_used)
    except:
        print("Error encountered while calculating efficiency metrics.")
    finally:
        return {
            "latency": round(latency, 2),
            "throughput": round(throughput, 2),
            "memory_used": round(memory_used, 2),
        }

def main():
    directory_name = './metrics_evaluation_data/'
    input_filename = 'evaluation_input_data.json'
    output_filename = 'evaluation_metrics_result.json'

    input_evaluation_data = load_evaluation_data_from_file(directory_name, input_filename)
    output_evaluation_data = evaluate_metrics(input_evaluation_data)
    persist_evaluation_result_to_output_file(directory_name, output_filename, output_evaluation_data)

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
