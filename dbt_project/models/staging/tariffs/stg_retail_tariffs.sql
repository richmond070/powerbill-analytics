-- staging: stg_retail_tariffs
-- Source:  main.silver.tariff_reference
--
-- Purpose: exposes the deduplicated, date-cast tariff reference table
--          for mart joins. No additional transformation needed —
--          Silver already handled deduplication and date casting.
WITH source AS (
    SELECT as_of_date,
        -- DATE (cast done in Silver)
        disco,
        customer_class,
        tariff_band,
        hour,
        price_ngn_kwh
    FROM { { source('silver', 'tariff_reference') } }
    WHERE as_of_date IS NOT NULL
        AND disco IS NOT NULL
)
SELECT *
FROM source