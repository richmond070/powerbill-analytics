-- staging: stg_grid_stress_complaints
-- Source:  main.silver.grid_stress_complaints
--
-- Purpose: exposes the complaint + grid-context enriched table for marts.
--          Silver already joined complaints to hourly grid load and power
--          flow data and computed the stress flags. This staging model
--          adds no new logic — it selects all columns and filters nulls
--          on the primary identifier so downstream marts work cleanly.
--
-- Downstream marts that read this:
--   - mart_grid_reliability  (primary consumer)
--   - mart_customer_lifetime_value  (stress flag count per customer)
WITH source AS (
    SELECT ticket_id,
        customer_id,
        disco,
        category,
        channel,
        outcome,
        sla_met,
        complaint_date,
        -- DATE partition column
        created_ts,
        -- TIMESTAMP
        resolved_ts,
        -- TIMESTAMP
        -- Grid load metrics at time of complaint
        avg_load_mw,
        avg_forecast_mw,
        load_vs_forecast,
        avg_frequency_hz,
        freq_deviation_hz,
        peak_load_mw,
        -- Power flow metrics at time of complaint
        avg_line_pf,
        avg_active_power_mw,
        avg_reactive_power_mvar,
        active_line_count,
        -- Stress flags (BOOLEAN)
        grid_overloaded,
        grid_unstable,
        line_stressed,
        any_stress_at_complaint_time
    FROM {{ source('silver', 'grid_stress_complaints') }}
    WHERE ticket_id IS NOT NULL
        AND complaint_date IS NOT NULL
)
SELECT *
FROM source