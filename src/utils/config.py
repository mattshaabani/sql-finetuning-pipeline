"""
src/utils/config.py

Central configuration loader for the SQL fine-tuning pipeline.
Loads settings from:
  - .env file (secrets, tokens)
  - configs/training_config.yaml
  - configs/lora_config.yaml
  - configs/eval_config.yaml
"""

from pathlib import Path
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────
# 1. Project root
# ─────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent.parent.parent


# ─────────────────────────────────────────────
# 2. YAML loader
# ─────────────────────────────────────────────

def load_yaml(filename: str) -> dict:
    path = ROOT_DIR / "configs" / filename
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# 3. Load all config files
# ─────────────────────────────────────────────

_train_cfg = load_yaml("training_config.yaml")
_lora_cfg  = load_yaml("lora_config.yaml")
_eval_cfg  = load_yaml("eval_config.yaml")


# ─────────────────────────────────────────────
# 4. Model config
# ─────────────────────────────────────────────

class ModelConfig:
    base_model:  str = _train_cfg["model"]["base_model"]
    model_type:  str = _train_cfg["model"]["model_type"]
    max_length:  int = _train_cfg["model"]["max_length"]


# ─────────────────────────────────────────────
# 5. Training config
# ─────────────────────────────────────────────

class TrainingConfig:
    output_dir:                      str   = _train_cfg["training"]["output_dir"]
    num_epochs:                      int   = _train_cfg["training"]["num_epochs"]
    per_device_train_batch_size:     int   = _train_cfg["training"]["per_device_train_batch_size"]
    per_device_eval_batch_size:      int   = _train_cfg["training"]["per_device_eval_batch_size"]
    gradient_accumulation_steps:     int   = _train_cfg["training"]["gradient_accumulation_steps"]
    learning_rate:                   float = _train_cfg["training"]["learning_rate"]
    warmup_steps:                    int   = _train_cfg["training"]["warmup_steps"]
    logging_steps:                   int   = _train_cfg["training"]["logging_steps"]
    eval_steps:                      int   = _train_cfg["training"]["eval_steps"]
    save_steps:                      int   = _train_cfg["training"]["save_steps"]
    fp16:                            bool  = _train_cfg["training"]["fp16"]
    bf16:                            bool  = _train_cfg["training"]["bf16"]
    dataloader_num_workers:          int   = _train_cfg["training"]["dataloader_num_workers"]
    remove_unused_columns:           bool  = _train_cfg["training"]["remove_unused_columns"]


# ─────────────────────────────────────────────
# 6. Data config
# ─────────────────────────────────────────────

class DataConfig:
    dataset_name:       str   = _train_cfg["data"]["dataset_name"]
    train_split:        str   = _train_cfg["data"]["train_split"]
    eval_split:         str   = _train_cfg["data"]["eval_split"]
    max_train_samples:  int   = _train_cfg["data"]["max_train_samples"]
    max_eval_samples:   int   = _train_cfg["data"]["max_eval_samples"]
    test_size:          float = _train_cfg["data"]["test_size"]


# ─────────────────────────────────────────────
# 7. LoRA config
# ─────────────────────────────────────────────

class LoRAConfig:
    r:               int   = _lora_cfg["lora"]["r"]
    lora_alpha:      int   = _lora_cfg["lora"]["lora_alpha"]
    lora_dropout:    float = _lora_cfg["lora"]["lora_dropout"]
    bias:            str   = _lora_cfg["lora"]["bias"]
    task_type:       str   = _lora_cfg["lora"]["task_type"]
    target_modules:  list  = _lora_cfg["lora"]["target_modules"]


# ─────────────────────────────────────────────
# 8. Quantization config
# ─────────────────────────────────────────────

class QuantizationConfig:
    load_in_4bit:              bool = _lora_cfg["quantization"]["load_in_4bit"]
    bnb_4bit_compute_dtype:    str  = _lora_cfg["quantization"]["bnb_4bit_compute_dtype"]
    bnb_4bit_quant_type:       str  = _lora_cfg["quantization"]["bnb_4bit_quant_type"]
    bnb_4bit_use_double_quant: bool = _lora_cfg["quantization"]["bnb_4bit_use_double_quant"]


# ─────────────────────────────────────────────
# 9. Evaluation config
# ─────────────────────────────────────────────

class EvaluationConfig:
    metrics:              list  = _eval_cfg["evaluation"]["metrics"]
    exact_match_threshold: float = _eval_cfg["evaluation"]["thresholds"]["exact_match"]
    execution_threshold:   float = _eval_cfg["evaluation"]["thresholds"]["execution_accuracy"]
    hub_repo_id:           str   = _eval_cfg["hub"]["repo_id"]
    hub_private:           bool  = _eval_cfg["hub"]["private"]


# ─────────────────────────────────────────────
# 10. MLflow config
# ─────────────────────────────────────────────

class MLflowConfig:
    experiment_name: str = _train_cfg["mlflow"]["experiment_name"]
    tracking_uri:    str = _train_cfg["mlflow"]["tracking_uri"]


# ─────────────────────────────────────────────
# 11. Environment settings
# ─────────────────────────────────────────────

class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Anthropic
    anthropic_api_key: str = Field(default="")

    # HuggingFace
    hf_token:    str = Field(default="")
    hf_username: str = Field(default="mattiinn")

    # App
    app_env:   str = Field(default="development")
    log_level: str = Field(default="INFO")


# ─────────────────────────────────────────────
# 12. Master settings object
# ─────────────────────────────────────────────

class Settings:
    """
    Single object that holds everything.

    Usage:
        from src.utils.config import settings
        print(settings.lora.r)              # 16
        print(settings.training.num_epochs) # 3
        print(settings.env.hf_token)        # hf_...
    """
    model:        ModelConfig        = ModelConfig()
    training:     TrainingConfig     = TrainingConfig()
    data:         DataConfig         = DataConfig()
    lora:         LoRAConfig         = LoRAConfig()
    quantization: QuantizationConfig = QuantizationConfig()
    evaluation:   EvaluationConfig   = EvaluationConfig()
    mlflow:       MLflowConfig       = MLflowConfig()
    env:          EnvSettings        = EnvSettings()
    root_dir:     Path               = ROOT_DIR


settings = Settings()