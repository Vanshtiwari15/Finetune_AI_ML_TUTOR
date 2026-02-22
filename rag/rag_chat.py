# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from retriever import Retriever
# from prompt_builder import build_prompt

# BASE_PATH = "/content/drive/MyDrive/instruction-tuned-gpt2-rag"
# MODEL_PATH = f"{BASE_PATH}/model/stage3-final"
# INDEX_PATH = f"{BASE_PATH}/rag/faiss_index"

# # Load model
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
# model.eval()

# retriever = Retriever(INDEX_PATH)

# def is_relevant(question, chunks):
#     q_words = set(question.lower().split())
#     text = " ".join(chunks).lower()
#     overlap = sum(1 for w in q_words if w in text)
#     return overlap >= 2   # simple, effective


# def answer(question: str) -> str:
#     q = question.strip()

#     # greetings (optional, still fine)
#     if q.lower() in ["hi", "hello", "hey", "good morning", "good evening"]:
#         return "Hello! You can ask questions related to my knowledge base."

#     context_chunks = retriever.retrieve(q)

#     if not context_chunks or not is_relevant(q, context_chunks):
#       return (
#           "I cannot answer this question because it is not present "
#           "in my knowledge base."
#       )

#     # ✅ IN SCOPE → RAG + stage3
#     prompt = build_prompt(context_chunks, q)

#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=1024
#     )

#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=150,
#             do_sample=False,
#             repetition_penalty=1.2,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id,
#         )

#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     if "Answer:" in decoded:
#         decoded = decoded.split("Answer:")[-1].strip()

#     return decoded


# if __name__ == "__main__":
#     while True:
#         q = input("You: ").strip()
#         if not q:
#             print("Bot: Please ask a question.\n")
#             continue

#         print("\nBot:", answer(q), "\n")


from retriever import Retriever

BASE_PATH = "/content/drive/MyDrive/instruction-tuned-gpt2-rag"
INDEX_PATH = f"{BASE_PATH}/rag/faiss_index"

retriever = Retriever(INDEX_PATH)

def answer(question: str) -> str:
    q = question.strip()

    if q.lower() in ["hi", "hello", "hey"]:
        return "Hello! You can ask questions related to my knowledge base."

    chunks = retriever.retrieve(q)

    if not chunks:
        return "I cannot answer this question because it is not present in my knowledge base."

    # 🔑 Return retrieved knowledge directly
    return chunks[0]


if __name__ == "__main__":
    while True:
        q = input("You: ").strip()
        if not q:
            print("Bot: Please ask a question.\n")
            continue

        print("\nBot:", answer(q), "\n")