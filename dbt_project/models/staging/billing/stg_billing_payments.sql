-- staging: stg_billing_payments
-- Source:  main.silver.billing_payments
--
-- Purpose: thin staging layer — confirms types are correct coming out of Silver,
--          renames nothing (Silver names are already clean), adds no business logic.
--          Downstream marts read from this model, never directly from Silver.
WITH source AS (
    SELECT customer_id,
        disco,
        billing_month,
        -- DATE (cast done in Silver)
        tariff_band,
        kwh,
        price_ngn_kwh,
        amount_billed_ngn,
        amount_paid_ngn,
        paid_on_time,
        -- BOOLEAN
        arrears_ngn,
        payment_gap_ngn,
        -- derived in Silver
        collection_rate -- derived in Silver
    FROM { { source('silver', 'billing_payments') } }
    WHERE customer_id IS NOT NULL
        AND billing_month IS NOT NULL
)
SELECT *
FROM source