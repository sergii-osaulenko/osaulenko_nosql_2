# scripts/05_chunking.py
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")

df["abstract_len"] = df["abstract"].apply(lambda x: len(x.split()))
longest_df = df.sort_values(by="abstract_len", ascending=False).head(30).reset_index(drop=True)

def fixed_size_chunking(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def semantic_chunking(text, max_words=100):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    for sent in sentences:
        words = sent.split()
        if current_word_count + len(words) > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_word_count = len(words)
        else:
            current_chunk.append(sent)
            current_word_count += len(words)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

for idx_name in ["arxiv-chunks-fixed", "arxiv-chunks-semantic"]:
    if idx_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(name=idx_name, dimension=VECTOR_DIM, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))

index_fixed = pc.Index("arxiv-chunks-fixed")
index_semantic = pc.Index("arxiv-chunks-semantic")

fixed_vectors = []
semantic_vectors = []

for row_idx, row in longest_df.iterrows():
    base_id = str(row["id"])
    
    f_chunks = fixed_size_chunking(row["abstract"])
    f_embs = model.encode(f_chunks, normalize_embeddings=True)
    for c_idx, (chunk, emb) in enumerate(zip(f_chunks, f_embs)):
        fixed_vectors.append({
            "id": f"paper_{base_id}_fixed_{c_idx}",
            "values": emb.tolist(),
            "metadata": {
                "arxiv_id": base_id,
                "title": str(row["title"]),
                "chunk_text": chunk[:500],
                "chunk_index": c_idx,
                "year": int(row["year"]),
                "category": str(row["category"])
            }
        })

    s_chunks = semantic_chunking(row["abstract"])
    s_embs = model.encode(s_chunks, normalize_embeddings=True)
    for c_idx, (chunk, emb) in enumerate(zip(s_chunks, s_embs)):
        semantic_vectors.append({
            "id": f"paper_{base_id}_sem_{c_idx}",
            "values": emb.tolist(),
            "metadata": {
                "arxiv_id": base_id,
                "title": str(row["title"]),
                "chunk_text": chunk[:500],
                "chunk_index": c_idx,
                "year": int(row["year"]),
                "category": str(row["category"])
            }
        })

def upload_in_batches(idx_obj, vectors):
    for i in range(0, len(vectors), 200):
        idx_obj.upsert(vectors=vectors[i:i+200])

upload_in_batches(index_fixed, fixed_vectors)
upload_in_batches(index_semantic, semantic_vectors)
print("Чанки успішно завантажено в індекси Pinecone.")