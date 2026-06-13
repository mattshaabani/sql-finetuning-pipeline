"""
src/training/trainer.py

Main fine-tuning pipeline using HuggingFace Trainer + PEFT LoRA.

This file orchestrates the complete training process:
    1. Load base model with 4-bit quantization (QLoRA)
    2. Apply LoRA adapters
    3. Load and preprocess Spider dataset
    4. Configure and run HuggingFace Trainer
    5. Save model and push to HuggingFace Hub

Designed to run on Google Colab free T4 GPU.

Usage:
    from src.training.trainer import SQLFineTuner
    finetuner = SQLFineTuner()
    finetuner.train()
"""

import os
import torch
from pathlib import Path
from src.utils.config import settings
from src.utils.logger import get_logger
from src.training.lora_config import (
    get_lora_config,
    get_bnb_config,
    get_trainable_params_info,
)
from src.training.callbacks import MLflowCallback, ProgressCallback
from src.data.dataset_loader import SpiderDatasetLoader
from src.data.preprocessor import DataPreprocessor

logger = get_logger(__name__)


class SQLFineTuner:
    """
    Complete fine-tuning pipeline for SQL generation.

    Architecture:
        Base model: Phi-3-mini-4k-instruct (3.8B params)
        Method:     QLoRA (4-bit quantization + LoRA adapters)
        Task:       Instruction tuning for text-to-SQL

    Why Phi-3-mini?
        - Small enough for free Colab GPU
        - Strong instruction following
        - Good at code/SQL generation
        - Microsoft's most efficient small model
    """

    def __init__(self):
        self.model_name = settings.model.base_model
        self.output_dir = Path(settings.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model     = None
        self.tokenizer = None

        logger.info(f"Initialized SQLFineTuner", extra={
            "model":      self.model_name,
            "output_dir": str(self.output_dir),
        })

    def load_model(self):
        """
        Load base model with 4-bit quantization.

        Steps:
            1. Load tokenizer
            2. Create BitsAndBytes 4-bit config
            3. Load model with quantization
            4. Apply LoRA adapters with PEFT
            5. Print trainable parameter stats
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, prepare_model_for_kbit_training

        logger.info(f"Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading model with 4-bit quantization")
        bnb_config = get_bnb_config()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare model for k-bit training
        # This enables gradient checkpointing and casts
        # layer norms to float32 for stability
        self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA adapters
        lora_config = get_lora_config()
        self.model  = get_peft_model(self.model, lora_config)

        # Show parameter breakdown
        get_trainable_params_info(self.model)

        logger.info(f"Model loaded and LoRA applied")
        return self

    def prepare_data(self):
        """
        Load Spider dataset and tokenize for training.

        Returns:
            (train_dataset, eval_dataset) as HuggingFace Datasets
        """
        logger.info(f"Loading and preprocessing dataset")

        loader = SpiderDatasetLoader()
        train_examples, eval_examples = loader.load()

        preprocessor    = DataPreprocessor(self.tokenizer)
        train_dataset   = preprocessor.prepare(train_examples)
        eval_dataset    = preprocessor.prepare(eval_examples)

        logger.info(f"Datasets ready", extra={
            "train": len(train_dataset),
            "eval":  len(eval_dataset),
        })

        return train_dataset, eval_dataset

    def train(self):
        """
        Run the complete fine-tuning pipeline.

        Training loop (handled by HuggingFace Trainer):
            for each epoch:
                for each batch:
                    1. Forward pass: model predicts next tokens
                    2. Compute cross-entropy loss on SQL tokens only
                    3. Backward pass: compute gradients
                    4. Update only LoRA parameters (A and B matrices)
                    5. Log metrics to MLflow
        """
        from transformers import TrainingArguments, Trainer

        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        # Prepare data
        train_dataset, eval_dataset = self.prepare_data()

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=settings.training.num_epochs,
            per_device_train_batch_size=settings.training.per_device_train_batch_size,
            per_device_eval_batch_size=settings.training.per_device_eval_batch_size,
            gradient_accumulation_steps=settings.training.gradient_accumulation_steps,
            learning_rate=settings.training.learning_rate,
            warmup_steps=settings.training.warmup_steps,
            logging_steps=settings.training.logging_steps,
            eval_steps=settings.training.eval_steps,
            save_steps=settings.training.save_steps,
            fp16=settings.training.fp16,
            bf16=settings.training.bf16,
            dataloader_num_workers=settings.training.dataloader_num_workers,
            remove_unused_columns=settings.training.remove_unused_columns,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            report_to="none",           # we handle logging via callbacks
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[
                MLflowCallback(),
                ProgressCallback(),
            ],
        )

        logger.info(f"Starting training")
        trainer.train()

        # Save final model
        self.save_model()

        return trainer

    def save_model(self):
        """Save LoRA adapter weights and tokenizer."""
        save_path = self.output_dir / "final"
        save_path.mkdir(exist_ok=True)

        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))

        logger.info(f"Model saved", extra={"path": str(save_path)})

    def push_to_hub(self):
        """Push fine-tuned model to HuggingFace Hub."""
        repo_id = settings.evaluation.hub_repo_id

        logger.info(f"Pushing to HuggingFace Hub", extra={"repo": repo_id})

        self.model.push_to_hub(repo_id, token=settings.env.hf_token)
        self.tokenizer.push_to_hub(repo_id, token=settings.env.hf_token)

        logger.info(f"Model pushed to Hub", extra={"repo": repo_id})
        print(f"\nModel available at: https://huggingface.co/{repo_id}")