-- Silver: grid_stress_complaints
-- Sources: main.bronze.bronze_customers_complaint
--          main.bronze.bronze_grid_load
--          main.bronze.bronze_power_flow
-- Target:  main.silver.grid_stress_complaints
-- Depends: customers_enriched (ensures billing + complaint enrichment ran first)
--
-- Business question answered:
--   Do complaints spike during periods of grid stress?
--   Grid stress = load_mw significantly exceeds forecast_mw  (overload)
--                 OR frequency_hz deviates from nominal 50 Hz  (instability)
--                 OR power factor on a transmission line is poor  (line stress)
--
-- What this table contains:
--   One row per complaint, enriched with the grid conditions at that DISCO
--   in the hour the complaint was raised.
--   This lets analysts run: "complaints where grid was stressed at the time"
--   and correlate complaint category (outage, billing, metering) with grid state.
--
-- Join strategy:
--   complaints.disco  = grid_load.disco
--   DATE_TRUNC('hour', complaint.created_ts) = DATE_TRUNC('hour', grid_load.timestamp)
--   This is a point-in-time join at hourly granularity — coarse enough to be
--   robust to minor timestamp skew between the two datasets.
--
--   For power_flow we aggregate line-level stress to DISCO level (avg pf per disco
--   per hour) since power_flow has no direct customer_id linkage.
CREATE OR REPLACE TABLE main.silver.grid_stress_complaints USING DELTA PARTITIONED BY (complaint_date) AS WITH complaints_ts AS (
        SELECT ticket_id,
            customer_id,
            disco,
            category,
            channel,
            outcome,
            sla_met,
            CAST(created_time AS TIMESTAMP) AS created_ts,
            CAST(resolved_time AS TIMESTAMP) AS resolved_ts,
            CAST(created_time AS DATE) AS complaint_date,
            DATE_TRUNC('hour', CAST(created_time AS TIMESTAMP)) AS complaint_hour
        FROM main.bronze.bronze_customers_complaint
        WHERE created_time IS NOT NULL
    ),
    grid_hourly AS (
        -- Aggregate grid load metrics to DISCO + hour level
        -- load_vs_forecast > 1.0 means actual load exceeded the forecast (stressed grid)
        -- freq_deviation measures how far from nominal 50 Hz the grid was running
        SELECT disco,
            DATE_TRUNC('hour', CAST(timestamp AS TIMESTAMP)) AS grid_hour,
            AVG(load_mw) AS avg_load_mw,
            AVG(forecast_mw) AS avg_forecast_mw,
            AVG(load_mw / NULLIF(forecast_mw, 0)) AS load_vs_forecast,
            AVG(frequency_hz) AS avg_frequency_hz,
            ABS(AVG(frequency_hz) - 50.0) AS freq_deviation_hz,
            MAX(load_mw) AS peak_load_mw
        FROM main.bronze.bronze_grid_load
        WHERE timestamp IS NOT NULL
        GROUP BY disco,
            DATE_TRUNC('hour', CAST(timestamp AS TIMESTAMP))
    ),
    power_flow_hourly AS (
        -- Aggregate transmission line health to DISCO + hour level
        -- Low avg_pf means reactive power problems on the lines serving that DISCO
        SELECT disco,
            DATE_TRUNC('hour', CAST(timestamp AS TIMESTAMP)) AS flow_hour,
            AVG(pf) AS avg_line_pf,
            AVG(p_mw) AS avg_active_power_mw,
            AVG(q_mvar) AS avg_reactive_power_mvar,
            COUNT(DISTINCT line_id) AS active_line_count
        FROM main.bronze.bronze_power_flow
        WHERE timestamp IS NOT NULL
        GROUP BY disco,
            DATE_TRUNC('hour', CAST(timestamp AS TIMESTAMP))
    ),
    grid_stress_flag AS (
        -- Combine grid load and power flow into a single grid health record per DISCO/hour
        -- Stress flags: overloaded = load > 105% of forecast
        --               unstable   = frequency deviation > 0.5 Hz from 50 Hz
        --               line_stress = average power factor < 0.85
        SELECT g.disco,
            g.grid_hour,
            g.avg_load_mw,
            g.avg_forecast_mw,
            g.load_vs_forecast,
            g.avg_frequency_hz,
            g.freq_deviation_hz,
            g.peak_load_mw,
            p.avg_line_pf,
            p.avg_active_power_mw,
            p.avg_reactive_power_mvar,
            p.active_line_count,
            -- Derived stress flags
            CASE
                WHEN g.load_vs_forecast > 1.05 THEN true
                ELSE false
            END AS grid_overloaded,
            CASE
                WHEN g.freq_deviation_hz > 0.5 THEN true
                ELSE false
            END AS grid_unstable,
            CASE
                WHEN p.avg_line_pf < 0.85 THEN true
                ELSE false
            END AS line_stressed,
            CASE
                WHEN g.load_vs_forecast > 1.05
                OR g.freq_deviation_hz > 0.5
                OR p.avg_line_pf < 0.85 THEN true
                ELSE false
            END AS any_stress
        FROM grid_hourly g
            LEFT JOIN power_flow_hourly p ON g.disco = p.disco
            AND g.grid_hour = p.flow_hour
    )
SELECT -- Complaint identifiers
    c.ticket_id,
    c.customer_id,
    c.disco,
    c.category,
    c.channel,
    c.outcome,
    c.sla_met,
    c.complaint_date,
    c.created_ts,
    c.resolved_ts,
    -- Grid conditions at the time of complaint
    gs.avg_load_mw,
    gs.avg_forecast_mw,
    gs.load_vs_forecast,
    gs.avg_frequency_hz,
    gs.freq_deviation_hz,
    gs.peak_load_mw,
    gs.avg_line_pf,
    gs.avg_active_power_mw,
    gs.avg_reactive_power_mvar,
    gs.active_line_count,
    -- Stress flags at the time of complaint
    COALESCE(gs.grid_overloaded, false) AS grid_overloaded,
    COALESCE(gs.grid_unstable, false) AS grid_unstable,
    COALESCE(gs.line_stressed, false) AS line_stressed,
    COALESCE(gs.any_stress, false) AS any_stress_at_complaint_time
FROM complaints_ts c
    LEFT JOIN grid_stress_flag gs ON c.disco = gs.disco
    AND c.complaint_hour = gs.grid_hour;