FROM apache/airflow:2.9.0-python3.11

USER root

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt

RUN pip install -- no-cache-dir --default-timeout=300 -r  /requirements.txt


ENV PYTHONPATH=/opt/airflow \
    AIRFLOW__CORE__LOAD_EXAMPLES=false \
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=true \
    DBT_PROFILES_DIR=/opt/airflow/dbt_project
