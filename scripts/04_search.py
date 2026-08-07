# scripts/04_search.py
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

def get_query_embedding(query_text):
    return model.encode(query_text, normalize_embeddings=True)

# 1. Чистий семантичний пошук
query = "teaching machines to recognize objects in pictures"
q_emb = get_query_embedding(query)

print(f"=== Чистий семантичний пошук за запитом: '{query}' ===")
results = index.query(vector=q_emb.tolist(), top_k=TOP_K, include_metadata=True)
for match in results["matches"]:
    meta = match["metadata"]
    print(f"- [{meta['category']}] ({meta['year']}) {meta['title']} (Score: {match['score']:.4f})")
    print(f"  Абстракт: {meta['abstract'][:200]}...\n")

# 2. Пошук з фільтрацією
print("=== Пошук з фільтрацією (Приклад A: RL за останні 5 років, cs.LG) ===")
filter_a = {"category": {"$eq": "cs.LG"}, "year": {"$gte": 2021}}
res_a = index.query(vector=get_query_embedding("reinforcement learning policies").tolist(), top_k=TOP_K, include_metadata=True, filter=filter_a)
for match in res_a["matches"]:
    meta = match["metadata"]
    print(f"- [{meta['category']}] ({meta['year']}) {meta['title']} (Score: {match['score']:.4f})")

# 3. Порівняння метрик на локальних ембеддингах
all_embeddings = np.load("embeddings/embeddings.npy")
local_q_emb = get_query_embedding(query)

cosine_sims = np.dot(all_embeddings, local_q_emb)
top_cosine = np.argsort(cosine_sims)[::-1][:TOP_K]

dot_prods = np.dot(all_embeddings, local_q_emb)
top_dot = np.argsort(dot_prods)[::-1][:TOP_K]

l2_dists = np.linalg.norm(all_embeddings - local_q_emb, axis=1)
top_l2 = np.argsort(l2_dists)[:TOP_K]

print("\n=== Порівняння метрик на локальних ембеддингах ===")
print("Топ-5 Cosine Similarity індекси:", top_cosine)
print("Топ-5 Dot Product індекси:", top_dot)
print("Топ-5 L2 Distance індекси:", top_l2)