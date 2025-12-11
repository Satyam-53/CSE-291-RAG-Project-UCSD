# CSE-291-RAG-Project-UCSD (Project Option 2 Group 9)

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to process unstructured and structured research data, create embeddings, store them in a **Qdrant** vector database, and evaluate retrieval performance using standard information retrieval metrics.

---

## 📁 Project Overview

The pipeline takes research papers, articles, and structured data (e.g., CSV, TXT) as input, preprocesses them into clean text, embeds the content using **sentence-transformers**, and stores these embeddings in a **Docker-hosted Qdrant** database.  
Finally, it evaluates the retrieval quality using metrics such as **Precision@K**, **Recall@K**, **MRR**, and **nDCG**.

---

## ⚙️ Pipeline Architecture

### 1. **Qdrant Vector Database (Docker)**

Runs the **Qdrant** vector store for efficient embedding storage and retrieval.
```bash
docker-compose up
```

---

### 2. **Preprocessing — `preprocess.py`**

**Purpose:** Clean and standardize textual data for embedding.

- **Input:**  
  - Unstructured PDFs (research papers, articles)  
  - Semi-structured or structured data files (TXT, CSV/TSV)

- **Output:**  
  - Cleaned text files saved in a designated folder.

- **Key operations:**
  - Extracts text content.
  - Removes inline citations and unnecessary whitespace.
  - Normalizes text for downstream embedding creation.

---

### 3. **Embedding Creation — `embeddings_creator.py`**

**Purpose:** Generate semantic vector embeddings for text chunks.

- **Input:**  
  - Preprocessed text files.

- **Output:**  
  - Embedding vectors stored in `embeddings_data/{embedding-model}_{chunking-strategy}_{chunk-size}_{overlap}/`.
    Example folder structure is `embeddings_data/neupubmedbert_overlapping_token_chunks_500_10/`

- **Key operations:**
  - In Phase 1, the embedding model [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) was used for creating vector embeddings along with single-sentence chunking strategy for semantic coherence.
  - In this step, we experimented with different embedding models (like BioBERT, PubMedBERT, BlueBERT, SciBERT, NeuMLPubMedBert) and chunking strategies (overlapping-sentence and overlapping-word chunks)
  - Finally, after exploring their effects on evaluation metrics such as precision and recall, we chose NeuMLPubMedBERT embedding model along with overlapping-word-based chunking (chunk size in words = 500 & overlapping words = 10%).
  - Saves embeddings in a structured JSON format (id, chunk text, embedding vector & its source file name) for ingestion into Qdrant.

- **Note:**
  - All steps until here can be skipped since an embeddings checkpoint is already available in the `embeddings_data/` folder. This can be re-used in the next steps by unzipping `neupubmedbert_overlapping_token_chunks_500_10.zip`

---

### 4. **Embedding Loader — `embedding_loader.py`**

**Purpose:** Load embedding vectors into the Qdrant vector database.

- **Input:**  
  - Generated embeddings in `embeddings_data/{embedding-model}_{chunking-strategy}_{chunk-size}_{overlap}/`.

- **Output:**  
  - Populated collection with name `{embedding-model}_{chunking-strategy}_{chunk-size}_{overlap}` in Qdrant vector store.

- **Key operations:**
  - Connects to the running Qdrant instance.
  - Creates or updates the relevant collection.
  - Inserts embedding vectors with metadata (chunk text and its source file name).

> **Note:** Ensure Qdrant is running before executing this step:
> ```bash
> docker-compose up
> ```

---

### 5. **Metrics Evaluation — `metrics_evaluator.py`**

**Purpose:** Evaluate retrieval quality using manually retrieved ground truth contexts and RAG-retrieved results.

- **Input:**  
  - `metrics_evaluation_data/evaluation_input_data.json` containing:
    - Queries and their categories (Factual/Synthesis/Hybrid)
    - Manually verified ground truth chunks

- **Output:**  
  - `metrics_evaluation_data/{embedding-model}_{chunking-strategy}_{chunk-size}_{overlap}_{k}/evaluation_metrics_result.json` containing evaluation scores.

**Evaluation Metrics:**

***A. Retrieval Metrics:***
- **Precision@K** — Fraction of top-K retrieved chunks that are relevant  
- **Recall@K** — Fraction of relevant chunks retrieved
- **MRR (Mean Reciprocal Rank)** — Rank of first relevant retrieval  
- **nDCG (Normalized Discounted Cumulative Gain)** — Position-sensitive ranking quality metric

***B. Efficiency Metrics:***
- **Latency** — Time taken to serve one query (in seconds)
- **Throughput** — Queries processed per second
- **Memory Used** - Amount of memory used for processing queries (in MB)

---

## Environment Setup
- Install Docker (for Qdrant)
- Install Python 3.9.x
- Clone this GIT repository
- Create a python environment (optional) and install necessary packages with command: `pip install -r requirements.txt`

---

## 📊 Evaluation Workflow for Test Set

1. Define evaluation queries and their manually retrieved ground truth in:`metrics_evaluation_data/evaluation_input_data.json`  
(Here, we already have 12 test questions and manually retrieved ground truth. You can replace it with any test set of your choice - please stick to the existing JSON format).


2. Preprocess raw data (pdf, tsv, csv) into text using:  
```bash 
python preprocess.py
```

This would give us processed text files under `processed_dataset/` directory.  
Note: You can skip running the script and use the pre-existing processed set of data for next steps.


3. From processed data, generate text chunks and their vector embeddings using 
```bash
python embeddings_creator.py
```  
This would give us embeddings in JSON files stored inside `embeddings_data/{embedding-model}_{chunking-strategy}_{chunk-size}_{overlap}/` folder.  
Note: You can skip running the script and use the pre-existing vector embeddings by unzipping `embeddings_data/neupubmedbert_overlapping_token_chunks_500_10.zip`


4. Spin-up Qdrant vector db on a docker container using
```bash
docker-compose up
```


5. Load the embeddings data onto the qdrant vector store using:
```bash
python embeddings_loader.py
```


6. Run the evaluation script:
```bash
python metrics_evaluator.py
```
This will read questions and their ground truth chunks from `metrics_evaluation_data/evaluation_input_data.json`, generate evaluation metrics and save them at `metrics_evaluation_data/{embedding-model}_{chunking-strategy}_{chunk-size}_{overlap}_{k}/evaluation_metrics_result.json`.  
Note: Based on our final hyperparameter configs, the evaluation output JSON will be stored at `metrics_evaluation_data/neupubmedbert_overlapping_token_chunks_500_10_10/evaluation_metrics_result.json`

---

## Output JSON

| # | Category   | Question (Summary)                                                                                                              | Precision@K | Recall@K | MRR   | nDCG  | Latency (s) | Throughput | Memory (MB) |
|---|-------------|-------------------------------------------------------------------------------------------------------------------------------|--------------|-----------|-------|-------|--------------|-------------|--------------|
| 1 | Factual     | Most frequent TP53 missense mutations and their predicted impacts (SIFT & PolyPhen)                                           | 0.0          | 0.0       | 0.0   | 0.0   | 0.01         | 128.77      | 0.22         |
| 2 | Factual     | Proteins interacting with CCAR2 and their confidence scores                                                                   | 0.6          | 0.15      | 0.5   | 0.712 | 0.01         | 195.58      | 0.0          |
| 3 | Factual     | Mechanism of p53-dependent mitochondrial apoptosis and alternative pathway in p53-null cells                                 | 0.4          | 0.143     | 0.333 | 0.544 | 0.0          | 237.53      | 0.03         |
| 4 | Hybrid      | Low-penetrance TP53 variants and founder mutation segregation (p.R337H)                                                      | 0.2          | 0.1       | 0.2   | 0.387 | 0.0          | 296.4       | 0.0          |
| 5 | Synthesis   | Aneuploidy facilitating mutant p53 gain-of-function and chromosomal instability pathway                                       | 0.0          | 0.0       | 0.0   | 0.0   | 0.0          | 235.69      | 0.05         |
| 6 | Synthesis   | TP53 germline variant clusters and p53’s role in male fertility                                                               | 0.2          | 0.1       | 1.0   | 1.0   | 0.0          | 293.7       | 0.03         |
| 7 | Synthesis   | TP53 mutations in aggressive B-cell lymphomas and therapeutic implications                                                   | 0.0          | 0.0       | 0.0   | 0.0   | 0.0          | 278.25      | 0.0          |
| 8 | Hybrid      | p53-mediated cancer metabolism reversal and TP53 p.K164E variant effect                                                     | 0.0          | 0.0       | 0.0   | 0.0   | 0.0          | 320.81      | 0.02         |
| 9 | Factual     | Somatic TP53 mutations and patient prognosis/survival in breast cancer                                                       | 0.0          | 0.0       | 0.0   | 0.0   | 0.0          | 289.34      | 0.0          |
|10 | Synthesis   | Prognostic impact of TP53 mutations across breast cancer molecular subtypes (HR+/HER2− vs TNBC)                              | 0.0          | 0.0       | 0.0   | 0.0   | 0.0          | 304.77      | 0.0          |
|11 | Factual     | Therapeutic strategies to target mutant p53                                                                                  | 0.0          | 0.0       | 0.0   | 0.0   | 0.0          | 317.46      | 0.02         |
|12 | Hybrid      | Dual role of p53 in tumor suppression and antiviral defense                                                                  | 0.2          | 0.167     | 1.0   | 1.0   | 0.0          | 315.24      | 0.03         |
