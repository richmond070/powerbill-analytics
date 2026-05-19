-- mart: mart_revenue_trends
-- Reads:  stg_billing_payments, stg_retail_tariffs
-- Answers: How is revenue trending month-over-month per DISCO and tariff band?
--          Which bands are growing, which are declining?
--
-- Grain: one row per (billing_month, disco, tariff_band)
WITH billing AS (
    SELECT *
    FROM { { ref('stg_billing_payments') } }
),
tariffs AS (
    SELECT *
    FROM { { ref('stg_retail_tariffs') } }
),
-- Get the effective tariff price for each (disco, tariff_band, billing_month).
-- We match on the most recent tariff as_of_date that is <= the billing_month.
-- This is a point-in-time tariff lookup so revenue reflects actual rates in effect.
tariff_effective AS (
    SELECT t.disco,
        t.tariff_band,
        t.as_of_date,
        AVG(t.price_ngn_kwh) AS avg_price_ngn_kwh -- average across hours of day
    FROM tariffs t
    GROUP BY t.disco,
        t.tariff_band,
        t.as_of_date
),
billing_with_tariff AS (
    SELECT b.billing_month,
        b.disco,
        b.tariff_band,
        b.customer_id,
        b.kwh,
        b.amount_billed_ngn,
        b.amount_paid_ngn,
        b.payment_gap_ngn,
        b.collection_rate,
        b.paid_on_time,
        -- Effective tariff price: latest as_of_date <= billing_month for this disco+band
        t_eff.avg_price_ngn_kwh AS effective_price_ngn_kwh
    FROM billing b
        LEFT JOIN LATERAL (
            SELECT AVG(price_ngn_kwh) AS avg_price_ngn_kwh
            FROM tariffs t
            WHERE t.disco = b.disco
                AND t.tariff_band = b.tariff_band
                AND t.as_of_date <= b.billing_month
            GROUP BY t.disco,
                t.tariff_band,
                t.as_of_date
            ORDER BY t.as_of_date DESC
            LIMIT 1
        ) t_eff ON true
),
monthly_revenue AS (
    SELECT billing_month,
        disco,
        tariff_band,
        -- Volume metrics
        COUNT(DISTINCT customer_id) AS unique_customers,
        SUM(kwh) AS total_kwh_billed,
        -- Revenue metrics
        SUM(amount_billed_ngn) AS total_revenue_billed,
        SUM(amount_paid_ngn) AS total_revenue_collected,
        SUM(payment_gap_ngn) AS total_revenue_uncollected,
        -- Collection efficiency
        AVG(collection_rate) AS avg_collection_rate,
        SUM(
            CASE
                WHEN paid_on_time THEN 1
                ELSE 0
            END
        ) AS on_time_payments,
        COUNT(*) AS total_bills,
        -- Effective tariff
        AVG(effective_price_ngn_kwh) AS avg_effective_tariff_ngn_kwh
    FROM billing_with_tariff
    GROUP BY billing_month,
        disco,
        tariff_band
),
-- Month-over-month revenue change using window functions
with_mom_change AS (
    SELECT *,
        LAG(total_revenue_billed) OVER (
            PARTITION BY disco,
            tariff_band
            ORDER BY billing_month
        ) AS prev_month_revenue_billed,
        total_revenue_billed - LAG(total_revenue_billed) OVER (
            PARTITION BY disco,
            tariff_band
            ORDER BY billing_month
        ) AS mom_revenue_change,
        ROUND(
            100.0 * (
                total_revenue_billed - LAG(total_revenue_billed) OVER (
                    PARTITION BY disco,
                    tariff_band
                    ORDER BY billing_month
                )
            ) / NULLIF(
                LAG(total_revenue_billed) OVER (
                    PARTITION BY disco,
                    tariff_band
                    ORDER BY billing_month
                ),
                0
            ),
            2
        ) AS mom_revenue_change_pct
    FROM monthly_revenue
)
SELECT billing_month,
    disco,
    tariff_band,
    unique_customers,
    total_kwh_billed,
    total_revenue_billed,
    total_revenue_collected,
    total_revenue_uncollected,
    ROUND(avg_collection_rate * 100, 2) AS collection_rate_pct,
    on_time_payments,
    total_bills,
    ROUND(
        100.0 * on_time_payments / NULLIF(total_bills, 0),
        2
    ) AS on_time_payment_rate_pct,
    avg_effective_tariff_ngn_kwh,
    prev_month_revenue_billed,
    mom_revenue_change,
    mom_revenue_change_pct
FROM with_mom_change
ORDER BY billing_month,
    disco,
    tariff_band