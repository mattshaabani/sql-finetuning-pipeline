"""
src/evaluation/benchmarks.py

Benchmark comparison between base model and fine-tuned model.
Logs everything to MLflow for visual comparison.

Usage:
    from src.evaluation.benchmarks import ModelBenchmark
    benchmark = ModelBenchmark()
    benchmark.compare(base_model, finetuned_model, eval_examples)
"""

import mlflow
import os
from src.data.dataset_loader import SQLExample
from src.evaluation.sql_evaluator import SQLModelEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelBenchmark:
    """
    Compares base model vs fine-tuned model on SQL generation.
    Results logged to MLflow for side-by-side comparison.
    """

    def __init__(self):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("sql-model-comparison")

    def evaluate_model(
        self,
        model,
        tokenizer,
        examples:   list[SQLExample],
        run_name:   str,
        max_samples: int = 50,
    ) -> dict:
        """Evaluate one model and log to MLflow."""

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_type", run_name)
            mlflow.log_param("n_samples",  max_samples)

            evaluator = SQLModelEvaluator(model, tokenizer)
            results   = evaluator.evaluate(
                examples,
                max_samples=max_samples,
                log_mlflow=False,
            )

            for name, value in results["avg_metrics"].items():
                mlflow.log_metric(name, round(float(value), 4))

        return results

    def compare(
        self,
        base_model,
        base_tokenizer,
        finetuned_model,
        finetuned_tokenizer,
        eval_examples: list[SQLExample],
        max_samples:   int = 50,
    ) -> dict:
        """
        Run both models on same examples and compare.

        Returns:
            Dict with results for both models and improvement metrics.
        """
        logger.info(f"Running benchmark comparison")

        print("Evaluating base model...")
        base_results = self.evaluate_model(
            base_model, base_tokenizer,
            eval_examples, "base_model", max_samples
        )

        print("Evaluating fine-tuned model...")
        ft_results = self.evaluate_model(
            finetuned_model, finetuned_tokenizer,
            eval_examples, "finetuned_model", max_samples
        )

        # Compute improvements
        base_avg = base_results["avg_metrics"]
        ft_avg   = ft_results["avg_metrics"]

        improvements = {
            metric: round(ft_avg[metric] - base_avg[metric], 4)
            for metric in ["exact_match", "bleu", "execution_accuracy"]
            if metric in base_avg and metric in ft_avg
        }

        # Print comparison
        print("\n" + "="*60)
        print("BASE MODEL vs FINE-TUNED MODEL COMPARISON")
        print("="*60)
        print(f"{'Metric':<25} {'Base':>10} {'Fine-tuned':>12} {'Δ':>8}")
        print("-"*60)

        for metric in ["exact_match", "execution_accuracy", "bleu"]:
            base_val = base_avg.get(metric, 0)
            ft_val   = ft_avg.get(metric, 0)
            delta    = improvements.get(metric, 0)
            arrow    = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"{metric:<25} {base_val:>10.4f} {ft_val:>12.4f} {arrow}{abs(delta):>6.4f}")

        print("="*60)
        print("\nOpen MLflow UI to see visual comparison:")
        print("  mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db")

        return {
            "base":         base_results,
            "finetuned":    ft_results,
            "improvements": improvements,
        }