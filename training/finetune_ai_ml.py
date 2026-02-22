import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# PATHS (CHANGE ONLY IF NEEDED)
BASE_MODEL_PATH = "/content/drive/MyDrive/instruction-tuned-gpt2-rag/model/stage3-final"
DATA_PATH = "/content/drive/MyDrive/instruction-tuned-gpt2-rag/data/ai_ml_basics.jsonl"
LORA_SAVE_PATH = "/content/drive/MyDrive/instruction-tuned-gpt2-rag/model/ai_ml_stage4"

# LOAD TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

# LOAD BASE MODEL (STAGE 3)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)


# LORA CONFIG (SAFE, STAGE 4)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# LOAD DATASET
dataset = load_dataset("json", data_files=DATA_PATH)["train"]

def format_example(example):
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256
    )

dataset = dataset.map(format_example, remove_columns=dataset.column_names)


# DATA COLLATOR

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="./ai_ml_logs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,        # DO NOT INCREASE
    learning_rate=1e-4,        # SAFE FOR STAGE 4
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none"
)

# TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# TRAIN
trainer.train()

# SAVE STAGE-4 LORA
model.save_pretrained(LORA_SAVE_PATH)
tokenizer.save_pretrained(LORA_SAVE_PATH)

print("Stage-4 LoRA adapters saved to:", LORA_SAVE_PATH)