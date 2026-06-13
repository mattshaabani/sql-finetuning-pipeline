"""
src/training/lora_config.py

LoRA and quantization configuration for fine-tuning.

This file translates our YAML config into actual
HuggingFace PEFT and BitsAndBytes config objects
ready to pass to the model loader.

Usage:
    from src.training.lora_config import get_lora_config, get_bnb_config
    bnb_config  = get_bnb_config()
    lora_config = get_lora_config()
"""

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_bnb_config():
    """
    Create BitsAndBytes 4-bit quantization config for QLoRA.

    This tells the model loader to:
    1. Load weights in 4-bit instead of 32-bit
    2. Use NF4 quantization type (optimal for normally distributed weights)
    3. Use float16 for computation (faster than float32 on GPU)
    4. Apply double quantization (quantize the quantization constants)

    Memory savings:
        Without quantization: ~15GB for Phi-3-mini
        With 4-bit QLoRA:     ~4GB for Phi-3-mini
    """
    try:
        import torch
        from transformers import BitsAndBytesConfig

        compute_dtype = (
            torch.float16
            if settings.quantization.bnb_4bit_compute_dtype == "float16"
            else torch.bfloat16
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=settings.quantization.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=settings.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=settings.quantization.bnb_4bit_use_double_quant,
        )

        logger.info(f"Created BitsAndBytes config", extra={
            "load_in_4bit":    settings.quantization.load_in_4bit,
            "quant_type":      settings.quantization.bnb_4bit_quant_type,
            "double_quant":    settings.quantization.bnb_4bit_use_double_quant,
        })

        return bnb_config

    except ImportError:
        logger.warning("BitsAndBytes not available — running without quantization")
        return None


def get_lora_config():
    """
    Create PEFT LoRA configuration.

    target_modules specifies which layers to apply LoRA to.
    For transformer models these are the attention projection matrices:
        q_proj: query projection
        k_proj: key projection
        v_proj: value projection
        o_proj: output projection

    Why these layers?
        The attention mechanism is where the model learns
        relationships between tokens. Fine-tuning these layers
        is most effective for task adaptation.

    LoRA math recap:
        new_weight = original_weight + B×A × (alpha/r)
        Trainable params: only A and B matrices
        Memory reduction: ~128x per layer
    """
    try:
        from peft import LoraConfig, TaskType

        lora_config = LoraConfig(
            r=settings.lora.r,
            lora_alpha=settings.lora.lora_alpha,
            lora_dropout=settings.lora.lora_dropout,
            bias=settings.lora.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=settings.lora.target_modules,
        )

        logger.info(f"Created LoRA config", extra={
            "r":              settings.lora.r,
            "lora_alpha":     settings.lora.lora_alpha,
            "target_modules": settings.lora.target_modules,
            "scale":          settings.lora.lora_alpha / settings.lora.r,
        })

        return lora_config

    except ImportError:
        raise ImportError("Run: pip install peft")


def get_trainable_params_info(model) -> dict:
    """
    Calculate and return trainable parameter statistics.

    After applying LoRA, most parameters are frozen.
    This function shows exactly how many we're training.

    Example output:
        total_params:     3,821,079,552
        trainable_params:    20,185,088
        trainable_percent:        0.53%
    """
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        "total_params":       total_params,
        "trainable_params":   trainable_params,
        "frozen_params":      total_params - trainable_params,
        "trainable_percent":  round(100 * trainable_params / total_params, 4),
    }

    logger.info(f"Model parameter breakdown", extra=info)
    print(f"\nModel Parameters:")
    print(f"  Total:      {total_params:>15,}")
    print(f"  Trainable:  {trainable_params:>15,}  ({info['trainable_percent']}%)")
    print(f"  Frozen:     {info['frozen_params']:>15,}")

    return info