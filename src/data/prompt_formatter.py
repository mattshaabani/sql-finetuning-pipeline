"""
src/data/prompt_formatter.py

Formats SQLExample objects into training prompts.

The prompt format is critical for fine-tuning success.
We use instruction tuning format — the model learns to
follow a consistent template.

Two formats:
    1. Training format  — includes the SQL answer
    2. Inference format — excludes the answer (model must generate it)

Usage:
    from src.data.prompt_formatter import SQLPromptFormatter
    formatter = SQLPromptFormatter()
    prompt    = formatter.format_training(example)
    inference = formatter.format_inference(example)
"""

from src.data.dataset_loader import SQLExample
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SQLPromptFormatter:
    """
    Formats SQL examples into instruction-tuning prompts.

    Why prompt format matters:
        The model learns patterns from training data.
        A consistent, clear format makes it easier to learn
        the input-output relationship.

        Bad format:  "question: {q} answer: {sql}"
        Good format: Structured template with clear sections

    The format we use:
        ### Task
        Convert the natural language question to SQL.

        ### Database Schema
        {schema}

        ### Question
        {question}

        ### SQL
        {sql}  ← only in training, not inference
    """

    TRAINING_TEMPLATE = """### Task
Convert the following natural language question to a SQL query.

### Database Schema
{schema}

### Question
{question}

### SQL
{sql}"""

    INFERENCE_TEMPLATE = """### Task
Convert the following natural language question to a SQL query.

### Database Schema
{schema}

### Question
{question}

### SQL
"""

    def format_training(self, example: SQLExample) -> str:
        """
        Format example for training — includes SQL answer.
        The model learns to predict everything after '### SQL'.
        """
        return self.TRAINING_TEMPLATE.format(
            schema=example.schema,
            question=example.question,
            sql=example.sql,
        )

    def format_inference(self, example: SQLExample) -> str:
        """
        Format example for inference — excludes SQL answer.
        The model must generate the SQL from scratch.
        """
        return self.INFERENCE_TEMPLATE.format(
            schema=example.schema,
            question=example.question,
        )

    def format_batch(
        self,
        examples:   list[SQLExample],
        for_training: bool = True,
    ) -> list[str]:
        """Format a batch of examples."""
        formatter = self.format_training if for_training else self.format_inference
        return [formatter(ex) for ex in examples]

    def extract_sql(self, generated_text: str) -> str:
        """
        Extract just the SQL from a model's generated output.

        The model generates the full prompt + SQL continuation.
        We need to extract only the SQL part.

        Example generated text:
            "### Task\n...\n### SQL\nSELECT count(*) FROM singer"

        Returns:
            "SELECT count(*) FROM singer"
        """
        marker = "### SQL\n"
        if marker in generated_text:
            sql = generated_text.split(marker)[-1].strip()
            # Stop at next section marker if present
            if "###" in sql:
                sql = sql.split("###")[0].strip()
            return sql
        return generated_text.strip()