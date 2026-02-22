# Purpose:

# Define how instruction + response are combined

# This directly affects model learning quality

# Even if you already tokenized in a cell, this file documents it.

"""
dataset.py

Defines the dataset used for instruction fine-tuning of GPT-2.

This dataset:
- Loads cleaned instruction-response JSONL files
- Formats them into a single prompt string
- Tokenizes inputs for causal language modeling

The design is intentionally simple and explicit so it is easy
to reason about model behavior and debugging.
"""

import json
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class InstructionDataset(Dataset):
    """
    PyTorch Dataset for instruction fine-tuning.

    Each training example is converted into a single text sequence:
        Instruction + Response

    Labels are the same as input_ids (causal LM objective).
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: GPT2Tokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_data(data_path)

    def _load_data(self, path: str) -> List[Dict[str, str]]:
        """
        Loads JSONL file containing cleaned instruction-response pairs.
        """
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def _format_prompt(self, instruction: str, response: str) -> str:
        """
        Formats instruction and response into a single prompt.

        This format helps GPT-2 learn to:
        - Understand task boundaries
        - Generate structured answers
        """
        return (
            "Instruction:\n"
            f"{instruction}\n\n"
            "Response:\n"
            f"{response}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        prompt = self._format_prompt(
            instruction=sample["instruction"],
            response=sample["response"],
        )

        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }