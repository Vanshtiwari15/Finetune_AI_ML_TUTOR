import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware

# =========================
# MODEL PATH
# =========================
MODEL_PATH = "../model/stage4-final"

# =========================
# LOAD TOKENIZER (IMPORTANT)
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False        # REQUIRED for GPT-2 fine-tunes
)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# LOAD MODEL
# =========================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,   # CPU safe
    device_map="auto"
)
model.eval()

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="AI & ML Tutor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # OK for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =========================
# REQUEST SCHEMA
# =========================
class ChatRequest(BaseModel):
    question: str

# =========================
# SIMPLE GREETING CHECK
# =========================
def is_greeting(text: str) -> bool:
    return text.lower().strip() in {"hi", "hello", "hey"}

# =========================
# GENERATION
# =========================
def generate_answer(question: str) -> str:
    prompt = f"""### Instruction:
{question}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()

# =========================
# ROUTES
# =========================
@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/chat")
def chat(req: ChatRequest):
    if is_greeting(req.question):
        return {"answer": "Hello! Ask me anything about AI or Machine Learning."}

    return {"answer": generate_answer(req.question)}