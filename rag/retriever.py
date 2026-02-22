# import faiss
# import pickle
# from sentence_transformers import SentenceTransformer


# class Retriever:
#     def __init__(self, index_path):
#         self.index = faiss.read_index(f"{index_path}/index.faiss")

#         with open(f"{index_path}/chunks.pkl", "rb") as f:
#             self.chunks = pickle.load(f)

#         self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

#     def retrieve(self, query):
#         q_emb = self.embedder.encode(
#             [query],
#             normalize_embeddings=True
#         )

#         scores, indices = self.index.search(q_emb, 1)

#         best_idx = indices[0][0]

#         # FAISS always returns something for top-1
#         return [self.chunks[best_idx]]


import faiss
import pickle
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, index_path):
        self.index = faiss.read_index(f"{index_path}/index.faiss")

        with open(f"{index_path}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query):
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        _, indices = self.index.search(q_emb, 1)
        return [self.chunks[indices[0][0]]]