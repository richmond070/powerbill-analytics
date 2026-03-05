"""
bronze_observability
====================
Lightweight, metadata-driven observability layer for the Bronze ingestion pipeline.

Public surface:
    - configure_logging()               : set up JSON stdout logging (call once at startup)
    - ensure_observability_tables()     : idempotent Postgres DDL bootstrap
    - BronzeLogger                      : structured JSON event emitter
    - AuditWriter                       : lifecycle writes to bronze_ingestion_audit
    - MetricsAggregator                 : daily upserts into bronze_ingestion_metrics
    - ObservabilityContractParser       : parse per-dataset rules from contract JSON
    - ObservabilityRuleEvaluator        : evaluate rules after each run
    - DatasetObservabilityRules         : typed rule config dataclass
    - close_pool()                      : graceful shutdown of Postgres connection pool
"""

from .bronze_logger          import BronzeLogger, configure_logging
from .observability_schema   import ensure_observability_tables
from .audit_writer           import AuditWriter
from .metrics_aggregator     import MetricsAggregator
from .observability_contract import (
    ObservabilityContractParser,
    ObservabilityRuleEvaluator,
    DatasetObservabilityRules,
    RuleViolation,
)
from .db_pool import close_pool

__all__ = [
    "configure_logging",
    "ensure_observability_tables",
    "BronzeLogger",
    "AuditWriter",
    "MetricsAggregator",
    "ObservabilityContractParser",
    "ObservabilityRuleEvaluator",
    "DatasetObservabilityRules",
    "RuleViolation",
    "close_pool",
]
