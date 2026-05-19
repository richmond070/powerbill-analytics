-- mart: mart_customer_lifetime_value
-- Reads:  stg_billing_payments, stg_customers_complaint, stg_grid_stress_complaints
-- Answers: What is the lifetime value of each customer?
--          Which customers are high-value AND low-risk?
--          Does exposure to grid stress events affect a customer's payment reliability?
--
-- Grain: one row per customer — lifetime summary
--
-- CLV Score methodology:
--   The CLV score is a composite 0-100 index built from three components:
--   1. Revenue component (50% weight): total_paid / max(total_paid across all customers)
--      — normalised so the highest-paying customer scores 50 on this component
--   2. Reliability component (30% weight): avg on_time_rate * 30
--      — a customer who always pays on time scores 30
--   3. Low-friction component (20% weight): penalises complaint volume
--      — 20 * (1 - complaints / max_complaints) — fewer complaints = higher score
--
-- This is intentionally simple and transparent. It can be recalibrated with
-- real business weights once the data is validated.
WITH billing AS (
    SELECT *
    FROM { { ref('stg_billing_payments') } }
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
        COUNT(DISTINCT category) AS distinct_complaint_categories
    FROM { { ref('stg_customers_complaint') } }
    GROUP BY customer_id
),
grid_stress_exposure AS (
    -- How many times was a customer's complaint raised during a grid stress event?
    -- This tells us whether the customer's problems are grid-driven vs billing-driven.
    SELECT customer_id,
        COUNT(*) AS stress_period_complaints,
        SUM(
            CASE
                WHEN grid_overloaded THEN 1
                ELSE 0
            END
        ) AS overload_period_complaints,
        SUM(
            CASE
                WHEN grid_unstable THEN 1
                ELSE 0
            END
        ) AS unstable_period_complaints,
        SUM(
            CASE
                WHEN line_stressed THEN 1
                ELSE 0
            END
        ) AS line_stress_period_complaints,
        SUM(
            CASE
                WHEN any_stress_at_complaint_time THEN 1
                ELSE 0
            END
        ) AS any_stress_complaints
    FROM { { ref('stg_grid_stress_complaints') } }
    GROUP BY customer_id
),
customer_billing_summary AS (
    SELECT customer_id,
        disco,
        COUNT(*) AS total_billing_cycles,
        SUM(amount_billed_ngn) AS total_billed,
        SUM(amount_paid_ngn) AS total_paid,
        SUM(payment_gap_ngn) AS total_outstanding,
        SUM(arrears_ngn) AS total_arrears,
        AVG(collection_rate) AS avg_collection_rate,
        AVG(
            CASE
                WHEN paid_on_time THEN 1.0
                ELSE 0.0
            END
        ) AS on_time_rate,
        MIN(billing_month) AS first_billing_month,
        MAX(billing_month) AS last_billing_month,
        -- Tenure in months (approximate)
        MONTHS_BETWEEN(MAX(billing_month), MIN(billing_month)) AS tenure_months
    FROM billing
    GROUP BY customer_id,
        disco
),
joined AS (
    SELECT b.customer_id,
        b.disco,
        b.total_billing_cycles,
        b.total_billed,
        b.total_paid,
        b.total_outstanding,
        b.total_arrears,
        b.avg_collection_rate,
        b.on_time_rate,
        b.first_billing_month,
        b.last_billing_month,
        b.tenure_months,
        COALESCE(c.total_complaints, 0) AS total_complaints,
        COALESCE(c.sla_met_count, 0) AS complaint_sla_met_count,
        COALESCE(c.distinct_complaint_categories, 0) AS distinct_complaint_categories,
        COALESCE(g.stress_period_complaints, 0) AS stress_period_complaints,
        COALESCE(g.overload_period_complaints, 0) AS overload_period_complaints,
        COALESCE(g.unstable_period_complaints, 0) AS unstable_period_complaints,
        COALESCE(g.line_stress_period_complaints, 0) AS line_stress_period_complaints,
        COALESCE(g.any_stress_complaints, 0) AS any_stress_complaints
    FROM customer_billing_summary b
        LEFT JOIN complaints c ON b.customer_id = c.customer_id
        LEFT JOIN grid_stress_exposure g ON b.customer_id = g.customer_id
),
-- Normalise for CLV scoring
normalised AS (
    SELECT *,
        MAX(total_paid) OVER () AS max_total_paid,
        MAX(total_complaints) OVER () AS max_total_complaints
    FROM joined
),
clv_scored AS (
    SELECT *,
        -- Revenue component: 0-50
        ROUND(50.0 * total_paid / NULLIF(max_total_paid, 0), 2) AS clv_revenue_score,
        -- Reliability component: 0-30
        ROUND(30.0 * on_time_rate, 2) AS clv_reliability_score,
        -- Low-friction component: 0-20 (fewer complaints = higher score)
        ROUND(
            20.0 * (
                1.0 - total_complaints / NULLIF(max_total_complaints, 0)
            ),
            2
        ) AS clv_friction_score
    FROM normalised
)
SELECT customer_id,
    disco,
    total_billing_cycles,
    tenure_months,
    first_billing_month,
    last_billing_month,
    -- Financial summary
    total_billed,
    total_paid,
    total_outstanding,
    total_arrears,
    ROUND(avg_collection_rate * 100, 2) AS avg_collection_rate_pct,
    ROUND(on_time_rate * 100, 2) AS on_time_rate_pct,
    -- Complaint profile
    total_complaints,
    complaint_sla_met_count,
    distinct_complaint_categories,
    -- Grid stress exposure
    stress_period_complaints,
    overload_period_complaints,
    unstable_period_complaints,
    line_stress_period_complaints,
    any_stress_complaints,
    -- What share of this customer's complaints coincided with grid stress?
    ROUND(
        100.0 * any_stress_complaints / NULLIF(total_complaints, 0),
        2
    ) AS pct_complaints_during_stress,
    -- CLV components
    clv_revenue_score,
    clv_reliability_score,
    clv_friction_score,
    ROUND(
        clv_revenue_score + clv_reliability_score + clv_friction_score,
        2
    ) AS clv_score,
    -- CLV tier
    CASE
        WHEN (
            clv_revenue_score + clv_reliability_score + clv_friction_score
        ) >= 75 THEN 'platinum'
        WHEN (
            clv_revenue_score + clv_reliability_score + clv_friction_score
        ) >= 50 THEN 'gold'
        WHEN (
            clv_revenue_score + clv_reliability_score + clv_friction_score
        ) >= 25 THEN 'silver'
        ELSE 'bronze'
    END AS clv_tier
FROM clv_scored
ORDER BY clv_score DESC