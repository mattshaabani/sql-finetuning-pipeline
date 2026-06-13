"""
src/data/dataset_loader.py

Loads and preprocesses the Spider text-to-SQL dataset.

Spider dataset structure:
    Each example contains:
    - question:  Natural language question
    - query:     Ground truth SQL query
    - db_id:     Database identifier
    - db_schema: Database tables and columns

The loader downloads Spider from HuggingFace datasets,
extracts schema information, and returns clean examples
ready for prompt formatting.

Usage:
    from src.data.dataset_loader import SpiderDatasetLoader
    loader = SpiderDatasetLoader()
    train, eval = loader.load()
    print(train[0])
"""

from dataclasses import dataclass, field
from datasets import load_dataset
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# 1. Data containers
# ─────────────────────────────────────────────

@dataclass
class SQLExample:
    """
    A single text-to-SQL training example.
    Contains everything needed to format a training prompt.
    """
    question:   str
    sql:        str
    db_id:      str
    schema:     str                    # formatted schema string
    difficulty: str = "unknown"        # easy/medium/hard/extra
    metadata:   dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"SQLExample(\n"
            f"  question='{self.question[:60]}...'\n"
            f"  sql='{self.sql[:60]}...'\n"
            f"  db='{self.db_id}'\n"
            f"  difficulty='{self.difficulty}'\n"
            f")"
        )


# ─────────────────────────────────────────────
# 2. Schema formatter
# ─────────────────────────────────────────────

def format_schema(tables: list[dict]) -> str:
    """
    Convert Spider's table structure into a readable schema string.

    Spider stores schemas as:
        {
            "table_names_original": ["singer", "concert"],
            "column_names_original": [[-1, "*"], [0, "singer_id"], [0, "name"], ...],
            "column_types": ["text", "number", "text", ...]
        }

    We convert this to:
        Table: singer (singer_id: number, name: text, country: text)
        Table: concert (concert_id: number, concert_name: text)

    This format is much easier for a language model to understand.
    """
    if not tables:
        return "Schema not available"

    table_names   = tables.get("table_names_original", [])
    column_names  = tables.get("column_names_original", [])
    column_types  = tables.get("column_types", [])

    # Group columns by table
    table_columns: dict[int, list[str]] = {i: [] for i in range(len(table_names))}

    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx == -1:
            continue   # skip the wildcard column "*"
        col_type = column_types[col_idx] if col_idx < len(column_types) else "text"
        table_columns[table_idx].append(f"{col_name}: {col_type}")

    # Format each table
    lines = []
    for table_idx, table_name in enumerate(table_names):
        cols = ", ".join(table_columns.get(table_idx, []))
        lines.append(f"Table: {table_name} ({cols})")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# 3. Dataset loader
# ─────────────────────────────────────────────

class SpiderDatasetLoader:
    """
    Loads the Spider text-to-SQL dataset from HuggingFace.

    Spider is hosted at: xlangai/spider on HuggingFace Hub
    It contains train (7000 examples) and validation (1034 examples).

    We limit samples via config to keep training fast on free hardware.
    """

    DATASET_NAME = "xlangai/spider"

    def __init__(self):
        self.max_train = settings.data.max_train_samples
        self.max_eval  = settings.data.max_eval_samples

        logger.info(f"Initializing Spider dataset loader", extra={
            "max_train": self.max_train,
            "max_eval":  self.max_eval,
        })

    def _convert_example(self, example: dict) -> SQLExample:
        """Convert a raw Spider example to SQLExample."""
        # Extract schema if available
        schema = ""
        if "db_table_names" in example:
            # Some versions of Spider include schema inline
            tables = {
                "table_names_original": example.get("db_table_names", []),
                "column_names_original": list(enumerate(
                    example.get("db_column_names", {}).get("column_name", [])
                )),
                "column_types": example.get("db_column_types", []),
            }
            schema = format_schema(tables)
        else:
            schema = f"Database: {example.get('db_id', 'unknown')}"

        return SQLExample(
            question=example.get("question", ""),
            sql=example.get("query", ""),
            db_id=example.get("db_id", "unknown"),
            schema=schema,
            difficulty=example.get("difficulty", "unknown"),
            metadata={"source": "spider"},
        )

    def load(self) -> tuple[list[SQLExample], list[SQLExample]]:
        """
        Load Spider dataset and return train and eval splits.

        Returns:
            (train_examples, eval_examples)
        """
        logger.info(f"Loading Spider dataset from HuggingFace")

        try:
            dataset = load_dataset(
                self.DATASET_NAME,
            )
        except Exception as e:
            logger.error(f"Failed to load dataset", extra={"error": str(e)})
            raise

        # Convert train split
        train_data = dataset["train"]
        if self.max_train:
            train_data = train_data.select(range(
                min(self.max_train, len(train_data))
            ))

        train_examples = [
            self._convert_example(ex) for ex in train_data
        ]

        # Convert validation split
        eval_data = dataset["validation"]
        if self.max_eval:
            eval_data = eval_data.select(range(
                min(self.max_eval, len(eval_data))
            ))

        eval_examples = [
            self._convert_example(ex) for ex in eval_data
        ]

        logger.info(f"Dataset loaded", extra={
            "train": len(train_examples),
            "eval":  len(eval_examples),
        })

        return train_examples, eval_examples

    def get_statistics(self, examples: list[SQLExample]) -> dict:
        """
        Compute dataset statistics for exploration.
        Useful for the dataset exploration notebook.
        """
        difficulties = {}
        for ex in examples:
            d = ex.difficulty
            difficulties[d] = difficulties.get(d, 0) + 1

        sql_lengths    = [len(ex.sql.split()) for ex in examples]
        question_lengths = [len(ex.question.split()) for ex in examples]

        import numpy as np
        return {
            "total":               len(examples),
            "difficulties":        difficulties,
            "avg_sql_length":      float(np.mean(sql_lengths)),
            "avg_question_length": float(np.mean(question_lengths)),
            "unique_databases":    len(set(ex.db_id for ex in examples)),
            "sql_keywords":        self._count_sql_keywords(examples),
        }

    def _count_sql_keywords(self, examples: list[SQLExample]) -> dict:
        """Count frequency of SQL keywords to understand dataset complexity."""
        keywords = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY",
                   "ORDER BY", "HAVING", "LIMIT", "UNION", "INTERSECT"]
        counts = {}
        for kw in keywords:
            counts[kw] = sum(
                1 for ex in examples
                if kw.upper() in ex.sql.upper()
            )
        return counts