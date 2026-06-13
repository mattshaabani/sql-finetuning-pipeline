"""
src/training/callbacks.py

Custom training callbacks for MLflow logging and early stopping.

HuggingFace Trainer calls these hooks at specific points:
    on_train_begin    — before training starts
    on_epoch_end      — after each epoch
    on_log            — when metrics are logged
    on_train_end      — after training completes

We use these to:
    1. Log all metrics to MLflow in real time
    2. Save best model checkpoint
    3. Print readable progress during training

Usage:
    from src.training.callbacks import MLflowCallback
    trainer = Trainer(callbacks=[MLflowCallback()])
"""

import mlflow
from transformers import TrainerCallback, TrainerState, TrainerControl
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MLflowCallback(TrainerCallback):
    """
    Logs training metrics to MLflow in real time.

    Every time the Trainer logs metrics (every logging_steps),
    this callback forwards them to MLflow so you can watch
    training progress in the MLflow UI live.
    """

    def __init__(self, experiment_name: str = "sql-generation-finetuning"):
        self.experiment_name = experiment_name
        self.run             = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Called once before training starts. Sets up MLflow run."""
        import os
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(self.experiment_name)

        self.run = mlflow.start_run()

        # Log all training hyperparameters
        mlflow.log_params({
            "model":                      args.output_dir,
            "num_epochs":                 args.num_train_epochs,
            "learning_rate":              args.learning_rate,
            "batch_size":                 args.per_device_train_batch_size,
            "gradient_accumulation":      args.gradient_accumulation_steps,
            "warmup_steps":               args.warmup_steps,
        })

        logger.info(f"MLflow run started", extra={
            "run_id":     self.run.info.run_id,
            "experiment": self.experiment_name,
        })

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called every logging_steps. Forwards metrics to MLflow."""
        if logs is None or self.run is None:
            return

        # Filter to numeric metrics only
        numeric_logs = {
            k: v for k, v in logs.items()
            if isinstance(v, (int, float))
        }

        if numeric_logs:
            mlflow.log_metrics(numeric_logs, step=state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called after each epoch ends."""
        logger.info(f"Epoch complete", extra={
            "epoch":       state.epoch,
            "global_step": state.global_step,
        })

    def on_train_end(self, args, state, control, **kwargs):
        """Called when training finishes. Closes MLflow run."""
        if self.run:
            mlflow.end_run()
            logger.info(f"MLflow run ended", extra={
                "run_id": self.run.info.run_id,
            })


class ProgressCallback(TrainerCallback):
    """
    Prints clean, readable training progress to terminal.
    The default HuggingFace progress output is verbose — this is cleaner.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        step  = state.global_step
        total = state.max_steps
        pct   = round(100 * step / total) if total else 0

        loss = logs.get("loss", logs.get("train_loss", "?"))
        lr   = logs.get("learning_rate", "?")

        if isinstance(loss, float):
            loss = f"{loss:.4f}"
        if isinstance(lr, float):
            lr = f"{lr:.2e}"

        print(f"  Step {step}/{total} ({pct}%) | loss={loss} | lr={lr}")