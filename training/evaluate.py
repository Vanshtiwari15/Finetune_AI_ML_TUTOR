from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "models/stage2_final"
LORA_PATH = "models/lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base, LORA_PATH)
model.eval()

prompt = "### Instruction:\nExplain LoRA\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

out = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(out[0], skip_special_tokens=True))
