# =============================================================================
# airflow/Dockerfile
# Location: airflow/Dockerfile
#
# Multi-stage build for the Airflow image used by this project.
#
# WHY A CUSTOM DOCKERFILE INSTEAD OF image: apache/airflow directly?
#   The previous docker-compose ran `pip install -r /requirements.txt` inside
#   the container startup command. That means every time the container
#   restarts, pip re-downloads and reinstalls ~150MB of packages from scratch.
#   By baking dependencies into the image at BUILD time, they become a cached
#   Docker layer. They only reinstall when requirements.txt actually changes.
#
# STAGE OVERVIEW:
#   Stage 1 (builder) -> installs all Python dependencies into an isolated prefix
#   Stage 2 (runtime) -> copies only the compiled site-packages from builder,
#                        no build tools, no pip cache, no compiler artifacts
#
# LAYER CACHING STRATEGY (Tip 3):
#   COPY requirements.txt          <- layer A
#   RUN pip install                <- layer B (only invalidated if A changes)
#   [app code comes via volumes]   <- no COPY needed, mounted at runtime
#
# This means editing bronze/, dbt_project/, or airflow/dags/ never causes
# a pip reinstall. Only changing requirements.txt triggers a rebuild.
# =============================================================================


# =============================================================================
# STAGE 1 — builder
# Purpose: install all Python dependencies cleanly into /install prefix.
# This entire stage is discarded after build. Its only output is /install.
# =============================================================================
FROM apache/airflow:2.9.0-python3.11 AS builder

WORKDIR /build

# --- Tip 3: Copy requirements.txt FIRST, before any application code ---
# Docker hashes this file. If it hasn't changed since last build, Docker
# serves the next RUN layer from cache — zero network calls, zero pip work.
COPY requirements.txt .

# --- Tips 5 + 6: Install into isolated prefix, single RUN layer ---
#
# --prefix=/install
#     Installs packages into /install rather than system site-packages.
#     Lets us COPY just this one folder into the runtime stage cleanly.
#
# --no-cache-dir
#     Tells pip not to write its HTTP cache to disk.
#     Saves ~50-80MB of cache files that would otherwise sit in the image.
#
# --no-warn-script-location
#     Suppresses noisy warnings about scripts installed outside PATH.
#     Not an error — just cleans up build output.
#
# All chained with && in one RUN instruction = one single Docker layer (Tip 6).
RUN pip install \
    --prefix=/install \
    --no-cache-dir \
    --no-warn-script-location \
    -r requirements.txt


# =============================================================================
# STAGE 2 — runtime
# Purpose: the actual image that runs in docker-compose.
# Starts fresh from the same base — no build tools, no pip cache carried over.
# =============================================================================
FROM apache/airflow:2.9.0-python3.11 AS runtime

# --- Tip 1: Copy ONLY the compiled packages from the builder stage ---
# /install contains exactly what pip built — nothing else from stage 1.
# Copying into /usr/local puts packages where Python's sys.path already looks.
COPY --from=builder /install /usr/local

# --- Tip 6: Set all environment defaults in one ENV instruction = one layer ---
# These are build-time defaults. Every value can be overridden by the
# environment: block in docker-compose.yaml at runtime — no rebuild needed.
ENV PYTHONPATH=/opt/airflow \
    AIRFLOW__CORE__LOAD_EXAMPLES=false \
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=true \
    DBT_PROFILES_DIR=/opt/airflow/dbt_project

# --- No COPY of application code ---
# bronze/, dbt_project/, airflow/dags/, runners/ are all bind-mounted by
# docker-compose volumes. Code changes are live without any image rebuild.
# The ONLY reason to run `docker-compose build` is when requirements.txt changes.

# The Airflow base image already sets the correct ENTRYPOINT and default USER.
# We intentionally do not override either.