-- Silver: billing_payments
-- Source:  main.bronze.bronze_billing_payments
-- Target:  main.silver.billing_payments
--
-- Transformations applied:
--   1. Deduplicate on (customer_id, billing_month) keeping the row with
--      the highest arrears_ngn as tiebreaker (same month billed twice → keep latest charge)
--   2. Cast billing_month STRING → DATE
--   3. Derive payment_gap_ngn  = amount_billed - amount_paid
--   4. Derive collection_rate  = amount_paid / amount_billed  (0 when billed = 0)
CREATE OR REPLACE TABLE main.silver.billing_payments USING DELTA PARTITIONED BY (billing_month) AS WITH deduplicated AS (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY customer_id,
                billing_month
                ORDER BY arrears_ngn DESC
            ) AS rn
        FROM main.bronze.bronze_billing_payments
    ),
    cleaned AS (
        SELECT customer_id,
            disco,
            CAST(billing_month AS DATE) AS billing_month,
            tariff_band,
            kwh,
            price_ngn_kwh,
            amount_billed_ngn,
            amount_paid_ngn,
            paid_on_time,
            arrears_ngn,
            -- Derived metrics
            (amount_billed_ngn - amount_paid_ngn) AS payment_gap_ngn,
            CASE
                WHEN amount_billed_ngn = 0 THEN 0.0
                ELSE amount_paid_ngn / amount_billed_ngn
            END AS collection_rate
        FROM deduplicated
        WHERE rn = 1
    )
SELECT *
FROM cleaned;