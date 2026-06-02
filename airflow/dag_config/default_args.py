from datetime import timedelta

DEFAULT_ARGS = {
    "owner": "data-platform",
    "depends_on_past": False,
    "email_on_failure": False,   # configure alert backend later
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

BRONZE_DEFAULT_ARGS = {
    **DEFAULT_ARGS, 
    "owner": "bronze_ingestion", 
    "retry_delay": timedelta(minutes=5), 
    "execution_timeout": timedelta(minutes=30),
}
SILVER_DEFAULT_ARGS = {
    **DEFAULT_ARGS, "retries": 2, 
    "retry_delay": timedelta(minutes=3), 
    "execution_timeout": timedelta(minutes=20)
}
GOLD_DEFAULT_ARGS   = {
    **DEFAULT_ARGS, "retries": 1, 
    "retry_delay": timedelta(minutes=2), 
    "execution_timeout": timedelta(minutes=15)    
}