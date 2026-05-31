BRONZE_SCHEDULE = "@daily"
SILVER_SCHEDULE = None          # triggered after bronze
GOLD_SCHEDULE   = None          # triggered after silver
FULL_PIPELINE_SCHEDULE = "@daily"

SCHEDULES = {
    "bronze_ingestion_dag":      BRONZE_SCHEDULE,
    "silver_transformation_dag": SILVER_SCHEDULE,
    "gold_dbt_dag":              GOLD_SCHEDULE,
    "full_pipeline_dag":         FULL_PIPELINE_SCHEDULE,
}