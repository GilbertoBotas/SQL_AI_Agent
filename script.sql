SELECT
    i.Name AS InsuranceName,
    COUNT(f.Id) * 100.0 / (
        SELECT
            COUNT(*)
        FROM
            dbo.fact_insurance_claim
        WHERE
            Date LIKE '2024%'
    ) AS ClaimPercentage
FROM
    dbo.fact_insurance_claim f
    JOIN dbo.dim_insurance i ON f.Insurance_Id = i.Id
WHERE
    f.Date LIKE '2024%'
GROUP BY
    i.Name