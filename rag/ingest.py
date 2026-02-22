import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from chunker import chunk_text

BASE_PATH = "/content/drive/MyDrive/instruction-tuned-gpt2-rag"
RAW_PATH = os.path.join(BASE_PATH, "data", "raw2")
INDEX_PATH = os.path.join(BASE_PATH, "rag", "faiss_index")

EMBED_MODEL = "all-MiniLM-L6-v2"


def load_documents_recursive(path):
    documents = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    documents.append({
                        "text": f.read(),
                        "source": file
                    })
    return documents


def main():
    docs = load_documents_recursive(RAW_PATH)
    if not docs:
        raise ValueError("❌ No .txt files found in data/raw2")

    all_chunks = []
    metadata = []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for c in chunks:
            all_chunks.append(c)
            metadata.append(doc["source"])

    print(f"📄 Files ingested: {len(docs)}")
    print(f"🔹 Total chunks: {len(all_chunks)}")

    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(
        all_chunks,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(INDEX_PATH, exist_ok=True)

    faiss.write_index(index, os.path.join(INDEX_PATH, "index.faiss"))

    with open(os.path.join(INDEX_PATH, "chunks.pkl"), "wb") as f:
        pickle.dump(all_chunks, f)

    with open(os.path.join(INDEX_PATH, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print("✅ Ingestion complete")


if __name__ == "__main__":
    main()