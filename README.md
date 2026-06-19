# SQL Generation Fine-tuning Pipeline

Fine-tuning TinyLlama-1.1B with LoRA for text-to-SQL generation on the Spider dataset, with full MLOps tooling: experiment tracking, evaluation framework, and HuggingFace Hub deployment.

---

## Overview

This project fine-tunes a small open-source LLM to convert natural language questions into SQL queries, using parameter-efficient fine-tuning (LoRA) on free Google Colab GPU compute.

**This is an honest portfolio project** — it documents real results including genuine limitations, not just successes. See [Evaluation Results](#evaluation-results) below.

### Key Features

- **LoRA fine-tuning** from scratch — full math explanation included
- **Spider dataset** — industry-standard text-to-SQL benchmark
- **Free GPU training** — runs entirely on Google Colab's free T4 GPU
- **Three-tier evaluation** — exact match, execution accuracy, BLEU
- **MLflow experiment tracking**
- **Model hosted on HuggingFace Hub** — publicly usable
- **Honest failure analysis** — documents overfitting and limitations

---

## Architecture

    Spider Dataset (HuggingFace)
            |
       DatasetLoader + SchemaFormatter
            |
       PromptFormatter (instruction-tuning format)
            |
       DataPreprocessor (tokenize + label masking)
            |
       TinyLlama-1.1B + LoRA Adapters (r=16, alpha=32)
            |
       HuggingFace Trainer (Google Colab T4 GPU)
            |
       SQLModelEvaluator (exact match, execution, BLEU)
            |
       HuggingFace Hub (mattiinn/sql-generation-tinyllama)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Base model | TinyLlama-1.1B-Chat-v1.0 |
| Fine-tuning method | LoRA (PEFT) |
| Dataset | Spider (xlangai/spider) |
| Training compute | Google Colab T4 GPU (free) |
| Experiment tracking | MLflow |
| Model hosting | HuggingFace Hub |
| Environment | Conda + ipykernel |

---

## Project Structure

    sql-finetuning-pipeline/
    |-- src/
    |   |-- data/
    |   |   |-- dataset_loader.py     # Spider dataset + schema formatting
    |   |   |-- prompt_formatter.py   # instruction-tuning prompt templates
    |   |   └-- preprocessor.py       # tokenization + label masking
    |   |-- training/
    |   |   |-- lora_config.py        # LoRA + quantization configs
    |   |   |-- trainer.py            # fine-tuning orchestration
    |   |   └-- callbacks.py          # MLflow logging callbacks
    |   |-- evaluation/
    |   |   |-- metrics.py            # exact match, execution, BLEU
    |   |   |-- sql_evaluator.py      # model evaluation pipeline
    |   |   └-- benchmarks.py         # base vs fine-tuned comparison
    |   |-- api/
    |   |   |-- main.py
    |   |   |-- routes.py
    |   |   └-- schemas.py
    |   └-- utils/
    |       |-- config.py
    |       └-- logger.py
    |-- notebooks/
    |   |-- 01_dataset_exploration.ipynb
    |   |-- 02_training_colab.ipynb       # run this on Google Colab
    |   |-- 03_model_evaluation.ipynb
    |   └-- 04_inference_demo.ipynb
    |-- configs/
    |   |-- training_config.yaml
    |   |-- lora_config.yaml
    |   └-- eval_config.yaml
    |-- data/
    |   └-- eval/                          # evaluation artifacts, plots
    |-- environment.yml
    |-- requirements.txt
    └-- .env.example

---

## Quick Start

**1. Clone and set up environment**

    git clone https://github.com/MattShaabani/sql-finetuning-pipeline.git
    cd sql-finetuning-pipeline

    conda env create -f environment.yml
    conda activate sql-finetune
    pip install -e .
    python -m ipykernel install --user --name sql-finetune --display-name "Python (sql-finetune)"

**2. Configure environment variables**

    cp .env.example .env

Edit .env and add your HuggingFace token.

**3. Explore the dataset**

    jupyter notebook notebooks/01_dataset_exploration.ipynb

**4. Train on Google Colab (free GPU required)**

Open `notebooks/02_training_colab.ipynb` in Google Colab:
File → Open notebook → GitHub → `MattShaabani/sql-finetuning-pipeline`

Runtime → Change runtime type → T4 GPU → Run all cells.

**5. Try the fine-tuned model**

    jupyter notebook notebooks/04_inference_demo.ipynb

Loads the model directly from HuggingFace Hub — no local training needed.

---

## Evaluation Results

Fine-tuned TinyLlama-1.1B on 1000 Spider training examples for 3 epochs using LoRA (r=16, alpha=32) on a Google Colab T4 GPU.

### Training Loss

| Step | Training Loss | Validation Loss |
|---|---|---|
| 10 | 16.93 | - |
| 40 | 0.487 | - |
| 100 | 0.042 | 0.0799 |
| 189 (final) | 0.023 | 0.0857 |

Training loss crashed to near-zero within ~40 steps while validation loss plateaued and **increased slightly** between step 100 and 189 — a clear overfitting signal on this small dataset.

### Metrics (50 held-out examples)

| Metric | Score |
|---|---|
| Exact Match | 12% |
| Execution Accuracy | 0%* |
| BLEU | 36.95% |

*\*Known evaluator limitation: the execution accuracy check runs against an empty in-memory SQLite database without the example's actual schema loaded, so this number does not reliably reflect true query correctness. Documented as a known issue rather than hidden.*

### Before vs After — Qualitative Examples

| Question | Base Model | Fine-tuned Model |
|---|---|---|
| How many singers do we have? | `\`\`\`SELECT COUNT(*) FROM concert_singer;\`\`\`` (markdown-wrapped, wrong table) | `SELECT count(*) FROM Concert_Singers` (clean SQL, still wrong table) |
| Order singers by age, oldest first | `SELECT singer.name, singer.country, singer.age FROM singer ORDER BY singer.age DESC;` (correct logic) | `SELECT name, country, age FROM singer ORDER BY age ASC LIMIT 1` (wrong direction, hallucinated LIMIT) |

### Honest Conclusions

**What worked:** The fine-tune successfully taught output formatting — clean raw SQL instead of markdown-wrapped explanations. Loss curves show the model rapidly learned the task's surface structure.

**What didn't work:** With only 1000 training examples and 3 epochs, the model overfit rather than generalized. Table-name hallucination persisted, and in one case fine-tuning actually **regressed** correct query logic from the base model.

**Root cause:** Insufficient training data relative to epochs for a 1.1B parameter model to learn robust schema grounding.

**Next steps to improve:**
1. Train on the full 7000-example Spider set instead of 1000
2. Add early stopping on validation loss (would have stopped around step 100)
3. Fix the execution accuracy evaluator to load real schema SQL
4. Try LoRA rank 32 for additional model capacity
5. Reduce epochs to 2 to avoid the overfitting window observed in epoch 3

---

## The Math Behind LoRA

Standard fine-tuning updates the full weight matrix W. LoRA instead freezes W and learns a low-rank decomposition:

    new_output = W·x + (B·A)·x × (alpha/r)

Where A is `[d × r]`, B is `[r × d]`, and r << d. For a 4096×4096 matrix with r=16:

    Full fine-tune: 16,777,216 trainable parameters
    LoRA:              131,072 trainable parameters (128x fewer)

B is initialized to zero so training starts exactly at the pretrained weights — `B·A = 0` at step zero.

---

## Model on HuggingFace Hub

The fine-tuned LoRA adapter is publicly available:

**https://huggingface.co/mattiinn/sql-generation-tinyllama**

Load it directly:

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = PeftModel.from_pretrained(base, "mattiinn/sql-generation-tinyllama")

---

## Lessons Learned: Colab Dependency Management

This project hit significant version conflicts between transformers, peft, and bitsandbytes on Google Colab's Python 3.12 environment. The resolution: **work with Colab's pre-installed package versions rather than force-installing pinned versions**, and skip 4-bit quantization (bitsandbytes) in favor of float16 LoRA when running into CUDA binary mismatches. This is documented in `notebooks/02_training_colab.ipynb`.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| HF_TOKEN | HuggingFace API token | required |
| HF_USERNAME | HuggingFace username | mattiinn |
| LOG_LEVEL | Logging level | INFO |

---

## License

MIT