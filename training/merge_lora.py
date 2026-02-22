import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel

# Correct project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

BASE_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "model/stage2-final/checkpoint-3750"
)

TOKENIZER_PATH = os.path.join(
    PROJECT_ROOT, "model/stage2-final"
)

LORA_PATH = os.path.join(
    PROJECT_ROOT, "model/lora_adapter"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "model/stage3-final"
)

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(
    TOKENIZER_PATH,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model (CPU is safer for merge)
base_model = GPT2LMHeadModel.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cpu",
    local_files_only=True
)

# Load LoRA and merge
model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    local_files_only=True
)

model = model.merge_and_unload()

# Save merged model
model.save_pretrained(
    OUTPUT_PATH,
    safe_serialization=True
)
tokenizer.save_pretrained(OUTPUT_PATH)

print("✅ Merge complete")
print(f"✅ Stage-3 model saved to: {OUTPUT_PATH}")