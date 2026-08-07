# scripts/06_hybrid_search.py
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

corpus = (df["title"] + " " + df["abstract"]).apply(lambda x: x.lower().split()).tolist()
bm25 = BM25Okapi(corpus)

def search_bm25(query_text, top_k=TOP_K):
    tokenized_query = query_text.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_indices):
        results.append({"index": int(idx), "score": float(scores[idx]), "rank": rank + 1})
    return results

def search_vector(query_text, top_k=TOP_K):
    q_emb = model.encode(query_text, normalize_embeddings=True)
    res = index.query(vector=q_emb.tolist(), top_k=top_k, include_metadata=True)
    results = []
    for rank, match in enumerate(res["matches"]):
        idx = int(match["id"].split("_")[1])
        results.append({"index": idx, "score": float(match["score"]), "rank": rank + 1})
    return results

def reciprocal_rank_fusion(bm25_results, vector_results, k=60, top_k=5):
    rrf_scores = {}
    def add_ranks(results):
        for item in results:
            idx = item["index"]
            rank = item["rank"]
            if idx not in rrf_scores:
                rrf_scores[idx] = 0.0
            rrf_scores[idx] += 1.0 / (k + rank)
    add_ranks(bm25_results)
    add_ranks(vector_results)
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    final_results = []
    for rank, (idx, score) in enumerate(sorted_docs[:top_k]):
        final_results.append({"index": idx, "rrf_score": score, "rank": rank + 1})
    return final_results

queries = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text"
]

for q in queries:
    print(f"\n==================== ЗАПИТ: '{q}' ====================")
    b_res = search_bm25(q, top_k=10)
    v_res = search_vector(q, top_k=10)
    hybrid_res = reciprocal_rank_fusion(b_res, v_res, k=60, top_k=5)

    print("\n--- ТОП-5 BM25 ---")
    for item in b_res[:5]:
        row = df.iloc[item["index"]]
        print(f"[{item['rank']}] {row['title']} (Score: {item['score']:.2f})")

    print("\n--- ТОП-5 ВЕКТОРНОГО ПОШУКУ ---")
    for item in v_res[:5]:
        row = df.iloc[item["index"]]
        print(f"[{item['rank']}] {row['title']} (Score: {item['score']:.4f})")

    print("\n--- ТОП-5 ГІБРИДНОГО ПОШУКУ (RRF) ---")
    for item in hybrid_res:
        row = df.iloc[item["index"]]
        print(f"[{item['rank']}] {row['title']} (RRF Score: {item['rrf_score']:.4f})")