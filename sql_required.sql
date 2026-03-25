create database CLTV;
use CLTV;
CREATE TABLE online_retail (
    InvoiceNo VARCHAR(20),
    StockCode VARCHAR(20),
    Description TEXT,
    Quantity INT,
    InvoiceDate DATETIME,
    UnitPrice DECIMAL(10,2),
    CustomerID INT,
    Country VARCHAR(50)
);

USE cltv;

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/online_retail_II.csv'
INTO TABLE online_retail
CHARACTER SET utf8mb4
FIELDS 
    TERMINATED BY ',' 
    OPTIONALLY ENCLOSED BY '"' 
    ESCAPED BY '"'
LINES 
    TERMINATED BY '\r\n' 
IGNORE 1 LINES
(InvoiceNo, StockCode, Description, Quantity, @InvoiceDate, UnitPrice, @CustomerID, Country, @dummy) -- @dummy catches extra columns
SET 
    InvoiceDate = STR_TO_DATE(@InvoiceDate, '%m/%d/%Y %H:%i'),
    CustomerID = NULLIF(TRIM(@CustomerID), '');





select * from online_retail;

CREATE TABLE cleaned_retail AS
SELECT 
    InvoiceNo, 
    StockCode, 
    Description, 
    Quantity, 
    InvoiceDate, 
    UnitPrice, 
    CustomerID, 
    Country,
    (Quantity * UnitPrice) AS TotalPrice -- Pre-calculating revenue
FROM online_retail
WHERE CustomerID IS NOT NULL 
  AND Quantity > 0 
  AND UnitPrice > 0
  AND InvoiceNo NOT LIKE 'C%'; -- Filters out cancellations
  
  select * from cleaned_retail;
  
  CREATE TABLE customer_summary AS
SELECT 
    CustomerID,
    -- Recency: Days since last purchase
    DATEDIFF((SELECT MAX(InvoiceDate) FROM cleaned_retail), MAX(InvoiceDate)) AS Recency,
    -- Frequency: Total number of unique orders
    COUNT(DISTINCT InvoiceNo) AS Frequency,
    -- Monetary: Total revenue from this customer
    SUM(TotalPrice) AS Monetary,
    -- Tenure: Days between first and last purchase
    DATEDIFF(MAX(InvoiceDate), MIN(InvoiceDate)) AS Tenure,
    -- Avg Basket Value
    AVG(TotalPrice) AS AvgOrderValue,
    -- Unique Items: Diversity of purchase
    COUNT(DISTINCT StockCode) AS UniqueItems
FROM cleaned_retail
GROUP BY CustomerID;

DROP TABLE IF EXISTS customer_summary;

CREATE TABLE customer_summary AS
SELECT 
    CustomerID,
    -- Recency: Days since last purchase
    DATEDIFF((SELECT MAX(InvoiceDate) FROM cleaned_retail), MAX(InvoiceDate)) AS Recency,
    -- Frequency: Total number of unique orders
    COUNT(DISTINCT InvoiceNo) AS Frequency,
    -- Monetary: Total revenue from this customer
    CAST(SUM(TotalPrice) AS DECIMAL(15,2)) AS Monetary,
    -- Tenure: Days between first and last purchase
    DATEDIFF(MAX(InvoiceDate), MIN(InvoiceDate)) AS Tenure,
    -- Explicitly cast AvgOrderValue to handle decimals correctly
    CAST(AVG(TotalPrice) AS DECIMAL(15,4)) AS AvgOrderValue,
    -- Unique Items: Diversity of purchase
    COUNT(DISTINCT StockCode) AS UniqueItems
FROM cleaned_retail
GROUP BY CustomerID;

CREATE TABLE cltv_features AS
SELECT 
    CustomerID,
    -- Recency relative to our 'cutoff' date of Sept 1st
    DATEDIFF('2010-09-01', MAX(InvoiceDate)) AS Recency,
    COUNT(DISTINCT InvoiceNo) AS Frequency,
    CAST(SUM(TotalPrice) AS DECIMAL(15,2)) AS Monetary_History,
    DATEDIFF(MAX(InvoiceDate), MIN(InvoiceDate)) AS Tenure,
    CAST(AVG(TotalPrice) AS DECIMAL(15,4)) AS AvgOrderValue_History
FROM cleaned_retail
WHERE InvoiceDate < '2010-09-01'
GROUP BY CustomerID;

CREATE TABLE cltv_target AS
SELECT 
    CustomerID,
    CAST(SUM(TotalPrice) AS DECIMAL(15,2)) AS Target_Revenue_90Days
FROM cleaned_retail
WHERE InvoiceDate >= '2010-09-01'
GROUP BY CustomerID;

CREATE TABLE cltv_final_ml_data AS
SELECT 
    f.*,
    COALESCE(t.Target_Revenue_90Days, 0) AS Target_Spend
FROM cltv_features f
LEFT JOIN cltv_target t ON f.CustomerID = t.CustomerID;

select * from  cltv_final_ml_data;