SELECT 'inpatient' AS tbl, SEGMENT, COUNT(*) AS n_rows,
       COUNT(CLM_PMT_AMT) AS rows_with_payment, SUM(CLM_PMT_AMT) AS total_paid
FROM inpatient_claims GROUP BY SEGMENT
UNION ALL
SELECT 'outpatient', SEGMENT, COUNT(*), COUNT(CLM_PMT_AMT), SUM(CLM_PMT_AMT)
FROM outpatient_claims GROUP BY SEGMENT
ORDER BY tbl, SEGMENT;