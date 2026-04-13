import faiss
import numpy as np
import os
import json
import pickle
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text: str, chunk_size: int = 40, overlap: int = 10) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_index(report_id: str, text: str):
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(f"data/indexes/{report_id}", exist_ok=True)
    faiss.write_index(index, f"data/indexes/{report_id}/index.faiss")

    with open(f"data/indexes/{report_id}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index built — {len(chunks)} chunks stored for report '{report_id}'")


def retrieve(report_id: str, query: str, top_k: int = 3) -> list:
    index = faiss.read_index(f"data/indexes/{report_id}/index.faiss")

    with open(f"data/indexes/{report_id}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            results.append({
                "chunk": chunks[idx],
                "score": round(float(distances[0][i]), 4)
            })
    return results


def build_index_from_pdf(report_id: str, pdf_path: str):
    from pipeline.pdf_parser import read_pdf
    text = read_pdf(pdf_path)
    build_index(report_id, text)
    print(f"Index built from PDF: {pdf_path}")


if __name__ == "__main__":
    build_index_from_pdf("test_report", "/Users/adismacbook/Desktop/medical-report-explainer/data/uploads/CBC-test-report-format-example-sample-template-Drlogy-lab-report.pdf")

    print("\nTest queries:")
    queries = [
        "what is the hemoglobin value?",
        "is the platelet count normal?",
        "what does the interpretation say?"
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        results = retrieve("test_report", q, top_k=2)
        for r in results:
            print(f"  → {r['chunk'][:120]}...")