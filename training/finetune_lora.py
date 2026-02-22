"""
finetune_lora.py

Stage-3 LoRA fine-tuning on top of Stage-2 GPT-2 checkpoint.
Folder layout expected:

project-root/
├── model/
│   └── stage2-final/
│       └── checkpoint-3750/
│           └── model.safetensors
├── training/
│   └── dataset.py
└── data/
    └── lora_dataset.jsonl
"""

import os
import sys
import torch

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from training.dataset import InstructionDataset


# PATHS (USING `model/`, NOT `models/`)

BASE_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model/stage2-final/checkpoint-3750"
)

TOKENIZER_PATH = os.path.join(
    PROJECT_ROOT,
    "model/stage2-final"
)

LORA_OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "model/lora_adapter"
)

LORA_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data/lora_dataset.jsonl"
)

MAX_LENGTH = 512


# Safety checks

assert os.path.isdir(BASE_MODEL_PATH), f"Missing model dir: {BASE_MODEL_PATH}"
assert os.path.isfile(os.path.join(BASE_MODEL_PATH, "model.safetensors")), \
    "model.safetensors not found in checkpoint"
assert os.path.isdir(TOKENIZER_PATH), f"Missing tokenizer dir: {TOKENIZER_PATH}"


# Load tokenizer

tokenizer = GPT2Tokenizer.from_pretrained(
    TOKENIZER_PATH,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# -------------------------------------------------
# Load Stage-2 model
# -------------------------------------------------

model = GPT2LMHeadModel.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)


# Attach LoRA

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn"],   # GPT-2 specific
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# Dataset
train_dataset = InstructionDataset(
    data_path=LORA_DATA_PATH,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
)


# Training args

training_args = TrainingArguments(
    output_dir=LORA_OUTPUT_PATH,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=2,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
)

# Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)


# Train
trainer.train()


# Save ONLY LoRA adapter
model.save_pretrained(LORA_OUTPUT_PATH)
tokenizer.save_pretrained(LORA_OUTPUT_PATH)

print("\n✅ LoRA training complete.")
print(f"Adapters saved to: {LORA_OUTPUT_PATH}")