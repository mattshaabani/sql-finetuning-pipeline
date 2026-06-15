"""
src/evaluation/metrics.py

SQL evaluation metrics implementation.

Three metrics:
    1. exact_match        — string comparison after normalization
    2. execution_accuracy — compare query results on real database
    3. bleu               — n-gram overlap score

Usage:
    from src.evaluation.metrics import SQLMetrics
    metrics = SQLMetrics()
    scores  = metrics.compute_all(generated="SELECT...", reference="SELECT...")
"""

import re
import sqlite3
import tempfile
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SQLMetrics:
    """
    Computes evaluation metrics for generated SQL queries.
    """

    def normalize_sql(self, sql: str) -> str:
        """
        Normalize SQL for fair comparison.

        Normalization steps:
            1. Lowercase everything
            2. Collapse multiple spaces into one
            3. Remove trailing semicolons
            4. Strip whitespace

        This prevents penalizing correct SQL for cosmetic differences.
        """
        sql = sql.lower().strip()
        sql = re.sub(r'\s+', ' ', sql)
        sql = sql.rstrip(';').strip()
        return sql

    def exact_match(self, generated: str, reference: str) -> float:
        """
        Exact match after normalization.
        Returns 1.0 if match, 0.0 if not.

        Lenient exact match — normalizes whitespace and case
        but still requires identical structure.
        """
        gen = self.normalize_sql(generated)
        ref = self.normalize_sql(reference)
        score = float(gen == ref)

        logger.debug(f"Exact match", extra={
            "generated": gen[:50],
            "reference": ref[:50],
            "score":     score,
        })

        return score

    def bleu_score(self, generated: str, reference: str) -> float:
        """
        BLEU score for SQL.

        BLEU (Bilingual Evaluation Understudy) measures n-gram overlap.

        For each n-gram size (1,2,3,4):
            precision_n = matching n-grams / total n-grams in generated

        BLEU = brevity_penalty × geometric_mean(precisions)

        Range: 0 to 1. Higher is better.
        Note: BLEU was designed for machine translation.
              For SQL it's a rough approximation.
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            import nltk
            nltk.download('punkt', quiet=True)

            gen_tokens = self.normalize_sql(generated).split()
            ref_tokens = self.normalize_sql(reference).split()

            smoothing = SmoothingFunction().method1
            score = sentence_bleu(
                [ref_tokens],
                gen_tokens,
                smoothing_function=smoothing,
            )
            return float(score)

        except ImportError:
            logger.warning("nltk not available — skipping BLEU")
            return 0.0

    def execution_accuracy(
        self,
        generated:  str,
        reference:  str,
        schema_sql: str = "",
    ) -> float:
        """
        Execute both queries on a temporary SQLite database
        and compare results.

        Steps:
            1. Create temporary in-memory SQLite database
            2. Create tables from schema SQL
            3. Execute generated SQL
            4. Execute reference SQL
            5. Compare results

        Returns 1.0 if results match, 0.0 otherwise.

        Note: requires valid schema SQL to create tables.
        Without schema, falls back to structural comparison.
        """
        try:
            conn = sqlite3.connect(":memory:")

            # Create schema if provided
            if schema_sql:
                try:
                    conn.executescript(schema_sql)
                except Exception as e:
                    logger.debug(f"Schema creation failed", extra={"error": str(e)})

            # Execute both queries
            try:
                gen_result = conn.execute(generated).fetchall()
            except Exception:
                gen_result = None

            try:
                ref_result = conn.execute(reference).fetchall()
            except Exception:
                ref_result = None

            conn.close()

            if gen_result is None or ref_result is None:
                return 0.0

            # Compare results (sort for order-independence)
            return float(sorted(gen_result) == sorted(ref_result))

        except Exception as e:
            logger.debug(f"Execution accuracy failed", extra={"error": str(e)})
            return 0.0

    def compute_all(
        self,
        generated:  str,
        reference:  str,
        schema_sql: str = "",
    ) -> dict[str, float]:
        """
        Compute all metrics for one generated SQL query.

        Args:
            generated:  Model's generated SQL
            reference:  Ground truth SQL
            schema_sql: Optional CREATE TABLE statements for execution

        Returns:
            Dict of metric_name → score
        """
        return {
            "exact_match":          self.exact_match(generated, reference),
            "bleu":                 self.bleu_score(generated, reference),
            "execution_accuracy":   self.execution_accuracy(
                generated, reference, schema_sql
            ),
        }