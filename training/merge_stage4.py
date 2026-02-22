import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# PATHS
BASE_MODEL_PATH = "/content/drive/MyDrive/instruction-tuned-gpt2-rag/model/stage3-final"
LORA_MODEL_PATH = "/content/drive/MyDrive/instruction-tuned-gpt2-rag/model/ai_ml_stage4"
OUTPUT_MODEL_PATH = "/content/drive/MyDrive/instruction-tuned-gpt2-rag/model/stage4-final"

# LOAD TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(OUTPUT_MODEL_PATH)

# LOAD BASE MODEL

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cpu"   # safer for merging
)

# LOAD LORA & MERGE
model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
model = model.merge_and_unload()

# SAVE FINAL MERGED MODEL
model.save_pretrained(OUTPUT_MODEL_PATH)

print("✅ Stage-4 FINAL model saved at:", OUTPUT_MODEL_PATH)