-- mart: mart_outstanding_balances
-- Domain:  customers
-- Reads:   stg_billing_payments
-- Answers: What is the current and rolling outstanding balance per customer and DISCO?
--          Which customers are accumulating arrears month-on-month?
--          Who is at risk of becoming a bad debt case?
--
-- Grain: one row per (customer_id, billing_month)
-- This is a time-series mart — every month is a snapshot of where the customer stood.
-- Use the latest billing_month per customer_id for a current balance view.
WITH billing AS (
    SELECT *
    FROM { { ref('stg_billing_payments') } }
),
monthly_balance AS (
    SELECT customer_id,
        disco,
        tariff_band,
        billing_month,
        amount_billed_ngn,
        amount_paid_ngn,
        payment_gap_ngn,
        arrears_ngn,
        collection_rate,
        paid_on_time,
        -- ── Cumulative running totals per customer ordered by month ──────────
        -- These give a "balance sheet" view at any point in time
        SUM(amount_billed_ngn) OVER (
            PARTITION BY customer_id
            ORDER BY billing_month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_billed,
        SUM(amount_paid_ngn) OVER (
            PARTITION BY customer_id
            ORDER BY billing_month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_paid,
        SUM(payment_gap_ngn) OVER (
            PARTITION BY customer_id
            ORDER BY billing_month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_outstanding,
        SUM(arrears_ngn) OVER (
            PARTITION BY customer_id
            ORDER BY billing_month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_arrears,
        -- ── Count of months where customer did not pay in full ───────────────
        SUM(
            CASE
                WHEN payment_gap_ngn > 0 THEN 1
                ELSE 0
            END
        ) OVER (
            PARTITION BY customer_id
            ORDER BY billing_month ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS months_with_gap_cumulative,
        -- ── Month-over-month arrears movement ────────────────────────────────
        -- Positive = arrears grew (bad), Negative = customer is catching up (good)
        arrears_ngn - LAG(arrears_ngn) OVER (
            PARTITION BY customer_id
            ORDER BY billing_month
        ) AS arrears_mom_change,
        -- ── Previous month values for trend context ──────────────────────────
        LAG(arrears_ngn) OVER (
            PARTITION BY customer_id
            ORDER BY billing_month
        ) AS prev_month_arrears,
        LAG(payment_gap_ngn) OVER (
            PARTITION BY customer_id
            ORDER BY billing_month
        ) AS prev_month_payment_gap
    FROM billing
),
-- ── Risk flagging ────────────────────────────────────────────────────────────
-- arrears_growing_flag: arrears increased for two or more consecutive months
-- This is the early warning signal — one bad month can be a blip,
-- two consecutive is a pattern that collections should act on
with_risk_flags AS (
    SELECT *,
        CASE
            WHEN arrears_mom_change > 0
            AND LAG(arrears_mom_change) OVER (
                PARTITION BY customer_id
                ORDER BY billing_month
            ) > 0 THEN true
            ELSE false
        END AS arrears_growing_flag,
        -- High risk: cumulative outstanding > 3x average monthly bill
        CASE
            WHEN cumulative_outstanding > (
                cumulative_billed / NULLIF(
                    ROW_NUMBER() OVER (
                        PARTITION BY customer_id
                        ORDER BY billing_month
                    ),
                    0
                ) * 3
            ) THEN true
            ELSE false
        END AS high_outstanding_flag
    FROM monthly_balance
)
SELECT customer_id,
    disco,
    tariff_band,
    billing_month,
    -- Monthly figures
    amount_billed_ngn,
    amount_paid_ngn,
    payment_gap_ngn,
    arrears_ngn,
    prev_month_arrears,
    arrears_mom_change,
    prev_month_payment_gap,
    ROUND(collection_rate * 100, 2) AS collection_rate_pct,
    paid_on_time,
    -- Cumulative running balance
    cumulative_billed,
    cumulative_paid,
    cumulative_outstanding,
    cumulative_arrears,
    months_with_gap_cumulative,
    -- Risk indicators
    arrears_growing_flag,
    high_outstanding_flag,
    -- Derived: what percentage of all billed amount is still outstanding?
    ROUND(
        100.0 * cumulative_outstanding / NULLIF(cumulative_billed, 0),
        2
    ) AS outstanding_pct_of_total_billed
FROM with_risk_flags
ORDER BY customer_id,
    billing_month