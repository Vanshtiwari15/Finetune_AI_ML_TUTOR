def build_prompt(context_chunks, question):
    context = "\n".join(context_chunks)

    return f"""
You are an assistant answering questions for a college project.

Answer ONLY using information explicitly stated in the context.
Do NOT add explanations.
Do NOT add extra details.
If the answer is not clearly present, say:
"The answer is not explicitly stated in the knowledge base."

Context:
{context}

Question:
{question}

Answer (use only the context):
"""