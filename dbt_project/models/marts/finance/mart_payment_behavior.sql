-- mart: mart_payment_behavior
-- Reads:  stg_billing_payments, stg_customers_complaint
-- Answers: How do customers pay? Who pays on time, who is chronically late?
--          Does complaint history correlate with payment behaviour?
--
-- Grain: one row per customer — lifetime payment profile
WITH billing AS (
    SELECT *
    FROM {{ ref('stg_billing_payments') }}
),
complaints AS (
    SELECT customer_id,
        COUNT(*) AS total_complaints,
        SUM(
            CASE
                WHEN sla_met THEN 1
                ELSE 0
            END
        ) AS sla_met_count,
        AVG(resolution_hours) AS avg_resolution_hours
    FROM {{ ref('stg_customers_complaint') }}
    GROUP BY customer_id
),
customer_billing AS (
    SELECT customer_id,
        disco,
        -- Payment volume
        COUNT(*) AS total_billing_cycles,
        SUM(amount_billed_ngn) AS total_billed,
        SUM(amount_paid_ngn) AS total_paid,
        SUM(payment_gap_ngn) AS total_payment_gap,
        SUM(arrears_ngn) AS total_arrears,
        -- On-time behaviour
        SUM(
            CASE
                WHEN paid_on_time THEN 1
                ELSE 0
            END
        ) AS on_time_count,
        COUNT(*) AS bill_count,
        AVG(
            CASE
                WHEN paid_on_time THEN 1.0
                ELSE 0.0
            END
        ) AS on_time_rate,
        -- Collection efficiency
        AVG(collection_rate) AS avg_collection_rate,
        MIN(collection_rate) AS min_collection_rate,
        MAX(collection_rate) AS max_collection_rate,
        -- Payment gap distribution (spread of individual bill gaps)
        AVG(payment_gap_ngn) AS avg_payment_gap_ngn,
        MAX(payment_gap_ngn) AS max_payment_gap_ngn,
        -- Date range
        MIN(billing_month) AS first_billing_month,
        MAX(billing_month) AS last_billing_month
    FROM billing
    GROUP BY customer_id,
        disco
),
-- Segment customers by payment behaviour
-- Chronic late payer: on_time_rate < 50%
-- Occasional late:    50% <= on_time_rate < 80%
-- Reliable payer:     on_time_rate >= 80%
segmented AS (
    SELECT cb.*,
        COALESCE(c.total_complaints, 0) AS total_complaints,
        COALESCE(c.sla_met_count, 0) AS complaint_sla_met_count,
        COALESCE(c.avg_resolution_hours, 0.0) AS avg_complaint_resolution_hours,
        CASE
            WHEN cb.on_time_rate >= 0.80 THEN 'reliable'
            WHEN cb.on_time_rate >= 0.50 THEN 'occasional_late'
            ELSE 'chronic_late'
        END AS payment_segment,
        CASE
            WHEN cb.avg_collection_rate >= 0.95 THEN 'high'
            WHEN cb.avg_collection_rate >= 0.75 THEN 'medium'
            ELSE 'low'
        END AS collection_tier
    FROM customer_billing cb
        LEFT JOIN complaints c ON cb.customer_id = c.customer_id
)
SELECT customer_id,
    disco,
    payment_segment,
    collection_tier,
    total_billing_cycles,
    total_billed,
    total_paid,
    total_payment_gap,
    total_arrears,
    on_time_count,
    bill_count,
    ROUND(on_time_rate * 100, 2) AS on_time_rate_pct,
    ROUND(avg_collection_rate * 100, 2) AS avg_collection_rate_pct,
    ROUND(min_collection_rate * 100, 2) AS min_collection_rate_pct,
    ROUND(max_collection_rate * 100, 2) AS max_collection_rate_pct,
    avg_payment_gap_ngn,
    max_payment_gap_ngn,
    first_billing_month,
    last_billing_month,
    total_complaints,
    complaint_sla_met_count,
    avg_complaint_resolution_hours
FROM segmented
ORDER BY total_arrears DESC