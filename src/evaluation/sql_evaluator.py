"""
src/evaluation/sql_evaluator.py

Evaluates a fine-tuned SQL generation model on the Spider dataset.
Compares base model vs fine-tuned model performance.

Usage:
    from src.evaluation.sql_evaluator import SQLModelEvaluator
    evaluator = SQLModelEvaluator(model, tokenizer)
    results   = evaluator.evaluate(eval_examples)
    evaluator.print_report(results)
"""

import time
import mlflow
from src.data.dataset_loader import SQLExample
from src.data.prompt_formatter import SQLPromptFormatter
from src.evaluation.metrics import SQLMetrics
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SQLModelEvaluator:
    """
    Evaluates SQL generation quality of a fine-tuned model.

    Runs inference on eval examples and measures:
    - Exact match rate
    - Execution accuracy
    - BLEU score
    - Inference latency
    """

    def __init__(self, model, tokenizer):
        self.model     = model
        self.tokenizer = tokenizer
        self.formatter = SQLPromptFormatter()
        self.metrics   = SQLMetrics()

    def generate_sql(self, example: SQLExample) -> str:
        """
        Generate SQL for one example using the model.

        Steps:
            1. Format inference prompt (no SQL answer)
            2. Tokenize
            3. Generate with greedy decoding
            4. Extract SQL from generated text
        """
        import torch

        prompt = self.formatter.format_inference(example)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=settings.model.max_length,
            truncation=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=False,      # greedy decoding for SQL
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return self.formatter.extract_sql(generated_text)

    def evaluate(
        self,
        examples:    list[SQLExample],
        max_samples: int = 100,
        log_mlflow:  bool = True,
    ) -> dict:
        """
        Evaluate model on a list of SQL examples.

        Args:
            examples:    List of SQLExample objects.
            max_samples: Limit evaluation for speed.
            log_mlflow:  Whether to log results to MLflow.

        Returns:
            Dict with averaged metrics and per-example results.
        """
        examples = examples[:max_samples]

        logger.info(f"Starting evaluation", extra={
            "n_examples": len(examples)
        })

        all_metrics    = []
        per_example    = []
        total_latency  = 0

        for i, example in enumerate(examples):
            start_time    = time.time()
            generated_sql = self.generate_sql(example)
            latency_ms    = (time.time() - start_time) * 1000
            total_latency += latency_ms

            scores = self.metrics.compute_all(
                generated=generated_sql,
                reference=example.sql,
            )

            all_metrics.append(scores)
            per_example.append({
                "question":     example.question,
                "reference":    example.sql,
                "generated":    generated_sql,
                "metrics":      scores,
                "latency_ms":   latency_ms,
            })

            if (i + 1) % 10 == 0:
                avg_em = sum(m["exact_match"] for m in all_metrics) / len(all_metrics)
                logger.info(f"Evaluated {i+1}/{len(examples)}", extra={
                    "avg_exact_match": round(avg_em, 4)
                })

        # Average all metrics
        avg_metrics = {
            metric: sum(m[metric] for m in all_metrics) / len(all_metrics)
            for metric in all_metrics[0].keys()
        }
        avg_metrics["avg_latency_ms"] = total_latency / len(examples)

        # Log to MLflow
        if log_mlflow:
            try:
                import os
                os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
                mlflow.set_tracking_uri("sqlite:///mlflow.db")
                with mlflow.start_run(run_name="evaluation"):
                    for name, value in avg_metrics.items():
                        mlflow.log_metric(name, round(value, 4))
            except Exception as e:
                logger.warning(f"MLflow logging failed", extra={"error": str(e)})

        logger.info(f"Evaluation complete", extra=avg_metrics)

        return {
            "avg_metrics":   avg_metrics,
            "per_example":   per_example,
            "n_evaluated":   len(examples),
        }

    def print_report(self, results: dict) -> None:
        """Print a formatted evaluation report."""
        avg = results["avg_metrics"]
        n   = results["n_evaluated"]

        print("\n" + "="*55)
        print("SQL GENERATION EVALUATION REPORT")
        print("="*55)
        print(f"  Examples evaluated: {n}")
        print(f"  Avg latency:        {avg.get('avg_latency_ms', 0):.0f}ms")
        print("-"*55)

        metrics = ["exact_match", "execution_accuracy", "bleu"]
        for metric in metrics:
            value = avg.get(metric, 0)
            bar   = "█" * int(value * 30)
            print(f"  {metric:<22} {value:.4f}  {bar}")

        print("="*55)

        # Show best and worst examples
        per_ex = results["per_example"]
        best   = max(per_ex, key=lambda x: x["metrics"]["exact_match"])
        worst  = min(per_ex, key=lambda x: x["metrics"]["exact_match"])

        print("\nBEST EXAMPLE:")
        print(f"  Q: {best['question']}")
        print(f"  Expected:  {best['reference']}")
        print(f"  Generated: {best['generated']}")

        print("\nWORST EXAMPLE:")
        print(f"  Q: {worst['question']}")
        print(f"  Expected:  {worst['reference']}")
        print(f"  Generated: {worst['generated']}")
        print("="*55)