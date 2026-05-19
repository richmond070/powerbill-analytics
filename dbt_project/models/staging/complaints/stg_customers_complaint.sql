-- staging: stg_customers_complaint
-- Source:  main.bronze.bronze_customers_complaint
--
-- Note: the Silver layer produces customers_enriched (aggregated) and
--       grid_stress_complaints (one row per ticket + grid context).
--       This staging model reads the raw ticket-level data from Bronze
--       for marts that need ticket-grain detail (e.g. mart_payment_behavior
--       complaint flags, mart_customer_lifetime_value complaint count).
--
-- Transformations applied here:
--   - TIMESTAMP casts on created_time and resolved_time
--   - Derived resolution_hours for convenience
--   - Filter out rows where ticket_id is null (structural garbage)
WITH source AS (
    SELECT ticket_id,
        customer_id,
        disco,
        category,
        channel,
        outcome,
        sla_met,
        -- BOOLEAN
        CAST(created_time AS TIMESTAMP) AS created_at,
        CAST(resolved_time AS TIMESTAMP) AS resolved_at,
        TIMESTAMPDIFF(
            HOUR,
            CAST(created_time AS TIMESTAMP),
            CAST(resolved_time AS TIMESTAMP)
        ) AS resolution_hours
    FROM { { source('bronze', 'bronze_customers_complaint') } }
    WHERE ticket_id IS NOT NULL
)
SELECT *
FROM source