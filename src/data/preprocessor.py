"""
src/data/preprocessor.py

Converts SQLExample objects into tokenized HuggingFace datasets
ready for training with the Trainer API.

The preprocessor:
    1. Formats examples using SQLPromptFormatter
    2. Tokenizes text using the model's tokenizer
    3. Creates labels (what the model should predict)
    4. Returns a HuggingFace Dataset object

Why labels matter:
    During training, we don't want the model to predict
    the entire prompt — only the SQL answer.
    We mask the prompt tokens with -100 so the loss
    is only computed on the SQL part.

    Input:  [prompt tokens] [SQL tokens]
    Labels: [-100, -100...] [SQL tokens]

This is called "completion-only training" and is much more
efficient than training on the full sequence.

Usage:
    from src.data.preprocessor import DataPreprocessor
    preprocessor = DataPreprocessor(tokenizer)
    dataset      = preprocessor.prepare(train_examples)
"""

import torch
from datasets import Dataset
from src.data.dataset_loader import SQLExample
from src.data.prompt_formatter import SQLPromptFormatter
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Tokenizes and prepares SQLExample objects for training.

    Key concept — Label masking:
        We only compute loss on SQL tokens, not prompt tokens.
        This focuses training on what matters: generating correct SQL.

        The -100 label value is special in PyTorch — CrossEntropyLoss
        ignores positions where label == -100.
    """

    def __init__(self, tokenizer, max_length: int = None):
        self.tokenizer  = tokenizer
        self.max_length = max_length or settings.model.max_length
        self.formatter  = SQLPromptFormatter()

        # Ensure tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _tokenize_example(self, example: SQLExample) -> dict:
        """
        Tokenize one example with label masking.

        Steps:
            1. Format full training prompt (prompt + SQL)
            2. Format inference prompt (prompt only)
            3. Tokenize both
            4. Create labels: -100 for prompt tokens, SQL token ids for SQL
            5. Return input_ids, attention_mask, labels
        """
        full_text      = self.formatter.format_training(example)
        prompt_text    = self.formatter.format_inference(example)

        # Tokenize full text
        full_tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize prompt only to find where SQL starts
        prompt_tokens = self.tokenizer(
            prompt_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = full_tokens["input_ids"][0]
        attention_mask = full_tokens["attention_mask"][0]
        prompt_length  = prompt_tokens["input_ids"].shape[1]

        # Create labels — mask prompt with -100
        labels = input_ids.clone()
        labels[:prompt_length] = -100   # ignore prompt in loss

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

    def prepare(self, examples: list[SQLExample]) -> Dataset:
        """
        Prepare a list of SQLExamples as a HuggingFace Dataset.

        Args:
            examples: List of SQLExample objects.

        Returns:
            HuggingFace Dataset ready for Trainer API.
        """
        logger.info(f"Preprocessing {len(examples)} examples")

        tokenized = []
        skipped   = 0

        for ex in examples:
            try:
                tokens = self._tokenize_example(ex)
                tokenized.append({
                    "input_ids":      tokens["input_ids"].tolist(),
                    "attention_mask": tokens["attention_mask"].tolist(),
                    "labels":         tokens["labels"].tolist(),
                })
            except Exception as e:
                skipped += 1
                logger.debug(f"Skipped example", extra={"error": str(e)})

        if skipped:
            logger.warning(f"Skipped {skipped} examples during preprocessing")

        logger.info(f"Preprocessing complete", extra={
            "total":     len(tokenized),
            "skipped":   skipped,
        })

        return Dataset.from_list(tokenized)