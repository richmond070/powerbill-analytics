-- mart: mart_grid_reliability
-- Domain:  operations
-- Reads:   stg_grid_stress_complaints
-- Answers: By DISCO and month, what share of complaints occurred during grid stress?
--          Which stress type drives the most complaints — overload, instability, or line stress?
--          Is the grid becoming more or less reliable over time?
--          Does poor SLA performance correlate with stressed grid periods?
--
-- Grain: one row per (disco, complaint_month)
-- This is the primary surface for the operational reliability business objective.
-- Feed this into a BI dashboard to give DISCOs a monthly grid health scorecard.
WITH stress AS (
    SELECT *
    FROM {{ ref('stg_grid_stress_complaints') }}
),
monthly_disco AS (
    SELECT disco,
        DATE_TRUNC('month', complaint_date) AS complaint_month,
        -- ── Total complaint volume ────────────────────────────────────────────
        COUNT(*) AS total_complaints,
        COUNT(DISTINCT customer_id) AS unique_customers_complaining,
        -- ── Stress-correlated complaint counts ───────────────────────────────
        -- Overall: any stress type active at time of complaint
        SUM(
            CASE
                WHEN any_stress_at_complaint_time THEN 1
                ELSE 0
            END
        ) AS stress_complaints,
        COUNT(
            DISTINCT CASE
                WHEN any_stress_at_complaint_time THEN customer_id
            END
        ) AS unique_customers_stress_complaints,
        -- Individual stress type breakdown
        -- Overload: grid was carrying more than 105% of forecast load
        SUM(
            CASE
                WHEN grid_overloaded THEN 1
                ELSE 0
            END
        ) AS overload_complaints,
        -- Instability: grid frequency deviated more than 0.5 Hz from 50 Hz
        SUM(
            CASE
                WHEN grid_unstable THEN 1
                ELSE 0
            END
        ) AS instability_complaints,
        -- Line stress: average power factor on transmission lines below 0.85
        SUM(
            CASE
                WHEN line_stressed THEN 1
                ELSE 0
            END
        ) AS line_stress_complaints,
        -- ── Complaint category breakdown during stress events ────────────────
        -- These columns help answer: "when the grid is stressed, what do customers complain about?"
        SUM(
            CASE
                WHEN any_stress_at_complaint_time
                AND LOWER(category) LIKE '%outage%' THEN 1
                ELSE 0
            END
        ) AS stress_outage_complaints,
        SUM(
            CASE
                WHEN any_stress_at_complaint_time
                AND LOWER(category) LIKE '%billing%' THEN 1
                ELSE 0
            END
        ) AS stress_billing_complaints,
        SUM(
            CASE
                WHEN any_stress_at_complaint_time
                AND LOWER(category) LIKE '%meter%' THEN 1
                ELSE 0
            END
        ) AS stress_metering_complaints,
        SUM(
            CASE
                WHEN any_stress_at_complaint_time
                AND LOWER(category) LIKE '%connect%' THEN 1
                ELSE 0
            END
        ) AS stress_connection_complaints,
        -- ── SLA performance: overall vs during stress ────────────────────────
        -- If SLA drops significantly during stress, it means the DISCO's support
        -- team is overwhelmed by grid-driven complaint spikes
        ROUND(
            AVG(
                CASE
                    WHEN sla_met THEN 1.0
                    ELSE 0.0
                END
            ) * 100,
            2
        ) AS overall_sla_met_rate_pct,
        ROUND(
            AVG(
                CASE
                    WHEN any_stress_at_complaint_time THEN CASE
                        WHEN sla_met THEN 1.0
                        ELSE 0.0
                    END
                END
            ) * 100,
            2
        ) AS sla_met_rate_during_stress_pct,
        ROUND(
            AVG(
                CASE
                    WHEN NOT any_stress_at_complaint_time THEN CASE
                        WHEN sla_met THEN 1.0
                        ELSE 0.0
                    END
                END
            ) * 100,
            2
        ) AS sla_met_rate_no_stress_pct,
        -- ── Grid condition averages across all complaint moments ─────────────
        -- These summarise the grid state at the times when customers called in
        ROUND(AVG(load_vs_forecast), 3) AS avg_load_vs_forecast,
        ROUND(AVG(avg_frequency_hz), 3) AS avg_frequency_hz,
        ROUND(AVG(freq_deviation_hz), 3) AS avg_freq_deviation_hz,
        ROUND(AVG(avg_line_pf), 3) AS avg_line_pf,
        ROUND(MAX(peak_load_mw), 2) AS max_peak_load_mw
    FROM stress
    GROUP BY disco,
        DATE_TRUNC('month', complaint_date)
),
-- ── Month-over-month trend computation ──────────────────────────────────────
with_trends AS (
    SELECT *,
        -- Stress complaint share: what fraction of all complaints this month
        -- happened when the grid was under stress
        ROUND(
            100.0 * stress_complaints / NULLIF(total_complaints, 0),
            2
        ) AS stress_complaint_share_pct,
        -- MoM change in total complaint volume
        total_complaints - LAG(total_complaints) OVER (
            PARTITION BY disco
            ORDER BY complaint_month
        ) AS complaint_mom_change,
        -- MoM change in stress share (percentage points)
        -- Negative = fewer complaints happening during stress = grid improving
        -- Positive = more complaints clustering around stress events = grid worsening
        ROUND(
            100.0 * stress_complaints / NULLIF(total_complaints, 0),
            2
        ) - LAG(
            ROUND(
                100.0 * stress_complaints / NULLIF(total_complaints, 0),
                2
            )
        ) OVER (
            PARTITION BY disco
            ORDER BY complaint_month
        ) AS stress_share_mom_change_ppt,
        -- Rolling 3-month average stress share (smooths out single-month spikes)
        ROUND(
            AVG(
                100.0 * stress_complaints / NULLIF(total_complaints, 0)
            ) OVER (
                PARTITION BY disco
                ORDER BY complaint_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ),
            2
        ) AS stress_share_3m_avg_pct,
        -- Grid reliability tier for the month based on stress share
        -- This gives operations a simple RAG status per DISCO per month
        CASE
            WHEN 100.0 * stress_complaints / NULLIF(total_complaints, 0) >= 60 THEN 'critical' -- majority of complaints during stress
            WHEN 100.0 * stress_complaints / NULLIF(total_complaints, 0) >= 40 THEN 'degraded' -- significant stress correlation
            WHEN 100.0 * stress_complaints / NULLIF(total_complaints, 0) >= 20 THEN 'moderate' -- some stress correlation
            ELSE 'stable' -- most complaints not grid-driven
        END AS grid_reliability_status
    FROM monthly_disco
)
SELECT disco,
    complaint_month,
    -- Complaint volumes
    total_complaints,
    stress_complaints,
    stress_complaint_share_pct,
    complaint_mom_change,
    stress_share_mom_change_ppt,
    stress_share_3m_avg_pct,
    grid_reliability_status,
    -- Stress type breakdown
    overload_complaints,
    instability_complaints,
    line_stress_complaints,
    -- Category breakdown during stress
    stress_outage_complaints,
    stress_billing_complaints,
    stress_metering_complaints,
    stress_connection_complaints,
    -- Customer reach
    unique_customers_complaining,
    unique_customers_stress_complaints,
    -- SLA performance split
    overall_sla_met_rate_pct,
    sla_met_rate_during_stress_pct,
    sla_met_rate_no_stress_pct,
    -- SLA gap: how much worse does SLA get during stress?
    ROUND(
        COALESCE(sla_met_rate_no_stress_pct, 0) - COALESCE(sla_met_rate_during_stress_pct, 0),
        2
    ) AS sla_stress_gap_ppt,
    -- Grid condition context
    avg_load_vs_forecast,
    avg_frequency_hz,
    avg_freq_deviation_hz,
    avg_line_pf,
    max_peak_load_mw
FROM with_trends
ORDER BY disco,
    complaint_month