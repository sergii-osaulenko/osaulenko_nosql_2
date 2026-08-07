# scripts/03_load_to_pinecone.py
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200  # Pinecone рекомендує батчі до 200 векторів

# Ініціалізація клієнта
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Створюємо індекс (якщо не існує)
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)

# Завантажуємо дані та ембеддинги
df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

print(f"Завантаження даних у Pinecone індекс '{INDEX_NAME}'...")
for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Завантаження батчів"):
    batch_df = df.iloc[i : i + BATCH_SIZE]
    batch_embeddings = embeddings[i : i + BATCH_SIZE]

    vectors = []
    for idx_row, row, emb in zip(
        range(i, i + len(batch_df)),
        batch_df.itertuples(),
        batch_embeddings,
    ):
        vectors.append(
            {
                "id": f"paper_{idx_row}",
                "values": emb.tolist(),
                "metadata": {
                    "arxiv_id": str(row.id),
                    "title": str(row.title),
                    "abstract": str(row.abstract)[:500],
                    "authors": str(row.authors)[:200],
                    "year": int(row.year),
                    "category": str(row.category),
                },
            }
        )
    index.upsert(vectors=vectors)

stats = index.describe_index_stats()
print(
    f"\nЗагальна кількість векторів в індексі: {stats.get('total_vector_count', 0)}"
)