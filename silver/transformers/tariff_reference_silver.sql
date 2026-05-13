-- Silver: tariff_reference
-- Source:  main.bronze.bronze_retail_tariffs
-- Target:  main.silver.tariff_reference
--
-- Transformations applied:
--   1. Deduplicate on (disco, customer_class, tariff_band, hour, as_of_date)
--      keeping the latest record per natural key
--   2. Cast as_of_date STRING → DATE
--   3. Select only the columns needed downstream
CREATE OR REPLACE TABLE main.silver.tariff_reference USING DELTA PARTITIONED BY (as_of_date) AS WITH deduplicated AS (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY disco,
                customer_class,
                tariff_band,
                hour,
                as_of_date
                ORDER BY as_of_date DESC
            ) AS rn
        FROM main.bronze.bronze_retail_tariffs
    )
SELECT CAST(as_of_date AS DATE) AS as_of_date,
    disco,
    customer_class,
    tariff_band,
    hour,
    price_ngn_kwh
FROM deduplicated
WHERE rn = 1;