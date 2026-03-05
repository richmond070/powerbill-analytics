"""
Observability Contract
Extends the bronze_ingestion_contract.json with per-dataset observability rules
as described in bronze_observability.md §8.

Each dataset can optionally carry an `observability` block:

    {
      "dataset_name": "billing_payments",
      ...
      "observability": {
        "alert_on_zero_rows":       true,
        "max_expected_duration_sec": 120,
        "expected_min_rows":         1000
      }
    }

If the block is absent, conservative defaults are applied so every dataset
gets at least zero-row protection.

`ObservabilityRuleEvaluator.evaluate()` checks the completed run against
the rules and returns a list of `RuleViolation` objects.  Alerting itself
is Phase 2 scope — here we only detect and log violations.

Referenced by: bronze_orchestrator.py
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from uuid import UUID

from .bronze_logger import BronzeLogger

logger = logging.getLogger("bronze.observability")


# ---------------------------------------------------------------------------
# Rule config dataclass
# ---------------------------------------------------------------------------

@dataclass
class DatasetObservabilityRules:
    """
    Parsed observability rules for a single dataset.
    Populated from the optional `observability` block in the ingestion contract,
    with safe defaults applied for missing keys.

    Attributes:
        dataset_name:              Dataset these rules apply to.
        alert_on_zero_rows:        Trigger a warning when row_count == 0.
        max_expected_duration_sec: Warn if ingestion exceeds this threshold.
        expected_min_rows:         Warn if row_count falls below this value.
    """
    dataset_name:               str
    alert_on_zero_rows:         bool  = True
    max_expected_duration_sec:  int   = 300   # 5-minute default
    expected_min_rows:          int   = 1     # at least one row expected


@dataclass
class RuleViolation:
    """
    Represents a single observability rule that was violated during a run.

    Attributes:
        rule:   Short machine-readable rule identifier.
        detail: Human-readable explanation for the log / alert body.
    """
    rule:   str
    detail: str


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ObservabilityContractParser:
    """
    Extracts `DatasetObservabilityRules` from the ingestion contract JSON.
    Works with the existing contract format — the `observability` block is
    purely additive; missing blocks fall back to defaults.
    """

    @staticmethod
    def parse(dataset_metadata: Dict) -> DatasetObservabilityRules:
        """
        Parse observability rules for a single dataset.

        Args:
            dataset_metadata: One element from contract['datasets'].

        Returns:
            DatasetObservabilityRules with defaults filled in for absent keys.
        """
        name = dataset_metadata["dataset_name"]
        obs  = dataset_metadata.get("observability", {})

        return DatasetObservabilityRules(
            dataset_name              = name,
            alert_on_zero_rows        = obs.get("alert_on_zero_rows",        True),
            max_expected_duration_sec = obs.get("max_expected_duration_sec",  300),
            expected_min_rows         = obs.get("expected_min_rows",          1),
        )

    @staticmethod
    def parse_all(contract: Dict) -> Dict[str, DatasetObservabilityRules]:
        """
        Parse observability rules for every dataset in the contract.

        Args:
            contract: Full parsed content of bronze_ingestion_contract.json.

        Returns:
            Mapping of dataset_name → DatasetObservabilityRules.
        """
        return {
            d["dataset_name"]: ObservabilityContractParser.parse(d)
            for d in contract.get("datasets", [])
        }


# ---------------------------------------------------------------------------
# Rule evaluator
# ---------------------------------------------------------------------------

class ObservabilityRuleEvaluator:
    """
    Evaluates a completed ingestion run against the dataset's observability rules.
    Violations are logged via BronzeLogger and returned as a list for callers
    that want to act on them (e.g. trigger an alert in Phase 2).
    """

    def __init__(self, rules: DatasetObservabilityRules) -> None:
        self.rules  = rules
        self._blog  = BronzeLogger(rules.dataset_name)

    def evaluate(
        self,
        trace_id:    UUID,
        row_count:   Optional[int],
        duration_ms: int,
    ) -> List[RuleViolation]:
        """
        Evaluate all observability rules against the run outcome.

        Args:
            trace_id:    Run correlation UUID (flows into log entries).
            row_count:   Rows ingested (None is treated as 0).
            duration_ms: Total wall-clock time in milliseconds.

        Returns:
            List of RuleViolation instances (empty means all rules passed).
        """
        violations: List[RuleViolation] = []
        rows           = row_count or 0
        duration_sec   = duration_ms / 1000.0

        # --- Rule 1: zero rows when not expected ---
        if self.rules.alert_on_zero_rows and rows == 0:
            v = RuleViolation(
                rule   = "zero_rows",
                detail = (
                    f"Dataset '{self.rules.dataset_name}' ingested 0 rows. "
                    "alert_on_zero_rows is enabled."
                ),
            )
            violations.append(v)
            self._blog.log_observability_warning(trace_id, v.rule, v.detail)

        # --- Rule 2: rows below expected minimum ---
        elif rows < self.rules.expected_min_rows:
            v = RuleViolation(
                rule   = "low_row_count",
                detail = (
                    f"Dataset '{self.rules.dataset_name}' ingested {rows:,} rows, "
                    f"below expected minimum of {self.rules.expected_min_rows:,}."
                ),
            )
            violations.append(v)
            self._blog.log_observability_warning(trace_id, v.rule, v.detail)

        # --- Rule 3: duration exceeded maximum ---
        if duration_sec > self.rules.max_expected_duration_sec:
            v = RuleViolation(
                rule   = "max_duration_exceeded",
                detail = (
                    f"Dataset '{self.rules.dataset_name}' took {duration_sec:.1f}s, "
                    f"exceeding limit of {self.rules.max_expected_duration_sec}s."
                ),
            )
            violations.append(v)
            self._blog.log_observability_warning(trace_id, v.rule, v.detail)

        if not violations:
            logger.debug(
                "All observability rules passed",
                extra={
                    "event":        "bronze_rules_passed",
                    "trace_id":     str(trace_id),
                    "dataset_name": self.rules.dataset_name,
                    "row_count":    rows,
                    "duration_sec": round(duration_sec, 3),
                },
            )

        return violations
