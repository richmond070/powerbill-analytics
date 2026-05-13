-- Silver: customers_enriched
-- Source:  main.silver.billing_payments (depends_on: billing_payments)
--          main.bronze.bronze_customers_complaint
-- Target:  main.silver.customers_enriched
--
-- Transformations applied:
--   1. TIMESTAMP casts on created_time and resolved_time (stored as STRING in bronze)
--   2. COALESCE all complaint-derived columns → 0 / 0.0 when customer has no complaints
--      (prevents NULL rows for customers who have never filed a complaint)
--   3. Complaint category pivot: count per category type so analysts can see
--      exactly what kinds of problems each customer raises, not just a flat total
--   4. Aggregate billing metrics per customer from silver.billing_payments
CREATE OR REPLACE TABLE main.silver.customers_enriched USING DELTA AS WITH complaints_base AS (
        -- Cast string timestamps to TIMESTAMP before any time arithmetic
        SELECT customer_id,
            disco,
            category,
            sla_met,
            outcome,
            CAST(created_time AS TIMESTAMP) AS created_ts,
            CAST(resolved_time AS TIMESTAMP) AS resolved_ts
        FROM main.bronze.bronze_customers_complaint
    ),
    complaint_summary AS (
        -- Overall complaint metrics per customer
        SELECT customer_id,
            COUNT(*) AS total_complaints,
            AVG(
                CASE
                    WHEN sla_met THEN 1.0
                    ELSE 0.0
                END
            ) AS sla_met_rate,
            AVG(TIMESTAMPDIFF(HOUR, created_ts, resolved_ts)) AS avg_resolution_hours
        FROM complaints_base
        GROUP BY customer_id
    ),
    complaint_by_category AS (
        -- Pivot complaint counts per category so downstream can filter/rank by type.
        -- Categories come from bronze_customers_complaint.category (free-text string).
        -- Using conditional aggregation keeps this SQL-only with no hardcoded enum list.
        SELECT customer_id,
            COUNT_IF(
                LOWER(category) LIKE '%billing%'
                OR LOWER(category) LIKE '%invoice%'
            ) AS billing_complaints,
            COUNT_IF(
                LOWER(category) LIKE '%outage%'
                OR LOWER(category) LIKE '%blackout%'
                OR LOWER(category) LIKE '%supply%'
            ) AS outage_complaints,
            COUNT_IF(LOWER(category) LIKE '%meter%') AS metering_complaints,
            COUNT_IF(
                LOWER(category) LIKE '%connect%'
                OR LOWER(category) LIKE '%disconnect%'
            ) AS connection_complaints,
            COUNT_IF(
                LOWER(category) LIKE '%refund%'
                OR LOWER(category) LIKE '%overcharge%'
                OR LOWER(category) LIKE '%credit%'
            ) AS overcharge_complaints,
            -- Catch-all: anything not matched by the categories above
            COUNT_IF(
                NOT (
                    LOWER(category) LIKE '%billing%'
                    OR LOWER(category) LIKE '%invoice%'
                    OR LOWER(category) LIKE '%outage%'
                    OR LOWER(category) LIKE '%blackout%'
                    OR LOWER(category) LIKE '%supply%'
                    OR LOWER(category) LIKE '%meter%'
                    OR LOWER(category) LIKE '%connect%'
                    OR LOWER(category) LIKE '%disconnect%'
                    OR LOWER(category) LIKE '%refund%'
                    OR LOWER(category) LIKE '%overcharge%'
                    OR LOWER(category) LIKE '%credit%'
                )
            ) AS other_complaints
        FROM complaints_base
        GROUP BY customer_id
    ),
    billing AS (
        SELECT customer_id,
            disco,
            COUNT(*) AS total_billing_cycles,
            SUM(amount_billed_ngn) AS total_billed,
            SUM(amount_paid_ngn) AS total_paid,
            SUM(payment_gap_ngn) AS total_outstanding,
            AVG(collection_rate) AS avg_collection_rate
        FROM main.silver.billing_payments
        GROUP BY customer_id,
            disco
    )
SELECT b.customer_id,
    b.disco,
    -- Billing metrics
    b.total_billing_cycles,
    b.total_billed,
    b.total_paid,
    b.total_outstanding,
    b.avg_collection_rate,
    -- Overall complaint metrics  (0 when no complaints — no NULLs)
    COALESCE(s.total_complaints, 0) AS total_complaints,
    COALESCE(s.sla_met_rate, 0.0) AS sla_met_rate,
    COALESCE(s.avg_resolution_hours, 0.0) AS avg_resolution_hours,
    -- Per-category complaint counts  (0 when no complaints in that category)
    COALESCE(cat.billing_complaints, 0) AS billing_complaints,
    COALESCE(cat.outage_complaints, 0) AS outage_complaints,
    COALESCE(cat.metering_complaints, 0) AS metering_complaints,
    COALESCE(cat.connection_complaints, 0) AS connection_complaints,
    COALESCE(cat.overcharge_complaints, 0) AS overcharge_complaints,
    COALESCE(cat.other_complaints, 0) AS other_complaints
FROM billing b
    LEFT JOIN complaint_summary s ON b.customer_id = s.customer_id
    LEFT JOIN complaint_by_category cat ON b.customer_id = cat.customer_id;