"""
finetune.py

Runs instruction fine-tuning (Stage-3) for GPT-2 Medium using
cleaned pretrained weights and cleaned instruction data.

This script assumes:
- Pretrained GPT-2 Medium is already cleaned and stored locally
- Instruction data is available in JSONL format (train / val)
- Training is executed on a GPU-enabled environment (Colab)
"""

import os

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

from .dataset import InstructionDataset

# -----------------------------
# Paths (adjust if needed)
# -----------------------------

PROJECT_ROOT = "/content/drive/MyDrive/instruction-tuned-gpt2-rag"

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "pretrained/gpt2-medium/cleaned"
)

TRAIN_DATA_PATH = os.path.join(
    PROJECT_ROOT, "data/cleaned/train.jsonl"
)

VAL_DATA_PATH = os.path.join(
    PROJECT_ROOT, "data/cleaned/val.jsonl"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "model/checkpoints/gpt2-medium-instruct"
)

# Load tokenizer and model

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)


# Load datasets

train_dataset = InstructionDataset(
    data_path=TRAIN_DATA_PATH,
    tokenizer=tokenizer,
    max_length=512,
)

MAX_TRAIN_SAMPLES = 20000

if len(train_dataset) > MAX_TRAIN_SAMPLES:
    train_dataset.samples = train_dataset.samples[:MAX_TRAIN_SAMPLES]

    
val_dataset = InstructionDataset(
    data_path=VAL_DATA_PATH,
    tokenizer=tokenizer,
    max_length=512,
)

# Training configuration

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True, ## change to false

    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    learning_rate=2e-5,
    warmup_steps=100,

    fp16=True,
    logging_steps=50,

    eval_strategy="epoch",
    save_strategy="epoch",

    save_total_limit=2,
    load_best_model_at_end=True,

    report_to="none",
)

# Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Run training

trainer.train(
    resume_from_checkpoint=
    "/content/drive/MyDrive/instruction-tuned-gpt2-rag/model/checkpoints/gpt2-medium-instruct/checkpoint-1250"
)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete. Model saved.")