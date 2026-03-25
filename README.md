# Customer Churn & Customer Lifetime Value (CLTV)

Project using the **Online Retail II (2009–2010)** transactional dataset to:

- **Engineer RFM features** (Recency, Frequency, Monetary) plus additional behavior features
- **Predict churn** (non-contractual definition: no purchase in last 90 days)
- If a customer is predicted to churn, classify them into one of:
  - **High Value High Risk**
  - **High Value Low Risk**
  - **Low Value High Risk**
  - **Low Value Low Risk**
- (Optional) **Build a CLTV-style supervised dataset** (history-window features → future-window spend target) via SQL

---

## What’s in this repository

### Notebooks

- `Untitled.ipynb`
  - Loads the raw dataset, performs **data cleaning**, basic EDA, and exports `cleaned_online_retail.csv`
  - Builds **RFM features**
  - Creates **churn label**: churned if `Recency > 90`
  - Runs **K-Means segmentation** on scaled RFM (4 clusters) and maps them to business-friendly labels
  - Trains a **RandomForestClassifier** for churn prediction (excludes `Recency` to reduce leakage)
  - Produces a final feature table with **churn probability**
- `1st_ML_Project.ipynb`
  - Another notebook version of the same workflow (RFM → churn labeling → ML modeling)

> Note: `Info_project.txt` exists but is currently empty.

### SQL pipeline (optional)

- `sql_required.sql`
  - Creates a MySQL database/table, loads online retail CSV into `online_retail`
  - Builds a cleaned transactional table `cleaned_retail`
  - Aggregates customer-level features (recency/frequency/monetary/tenure/AOV)
  - Creates a **feature window** (before `2010-09-01`) and a **target window** (on/after `2010-09-01`)
  - Outputs `cltv_final_ml_data` with `Target_Spend` (= revenue in next 90 days)

---

## Data files

### Raw inputs

- `online_retail_II 2009-2010.csv` (primary raw input used by the notebooks)
- `online_retail_II.xlsx` and various copies (`online_retail_II - Copy*.xlsx/.csv`)

### Generated / intermediate outputs

- `cleaned_online_retail.csv`
  - Cleaned transactional data with `TotalPrice` and `MonthYear`
  - Columns (sample): `Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country, TotalPrice, MonthYear`
- `rfm_features.csv`
  - Customer-level RFM + churn label + AOV
  - Columns: `Customer ID, Recency, Frequency, Monetary, Churn, AvgOrderValue`
- `final_data_with_all_features.csv`
  - Customer-level feature table (engineered features + churn probability)
  - Columns (sample): `FirstPurchaseDate, Frequency, Monetary, Tenure, Variety, PurchaseVelocity, AvgTicketSize, Recency, Churn, Churn_Probability`
- `cltv_final_ml_data.csv`
  - Customer-level supervised dataset for CLTV-style modeling (history features → future spend)
  - Columns: `CustomerID, Recency, Frequency, Monetary_History, Tenure, AvgOrderValue_History, Target_Spend`

---

## How churn is defined (project convention)

This project treats churn as a **behavioral churn** definition for non-contractual retail:

- **Churn = 1** if a customer’s **Recency > 90 days**
- Otherwise **Churn = 0**

---

## Value/Risk buckets (project convention)

The runnable pipeline (`predict.py`) produces these columns:

- **Value**:
  - `Value_Level = High Value` if `Monetary` is above the chosen quantile (default: median, `--value-quantile 0.5`)
  - else `Low Value`
- **Risk**:
  - `Risk_Level = High Risk` if `Churn_Probability >= --churn-threshold` (default `0.5`)
  - else `Low Risk`
- **Bucket**:
  - `Value_Risk_Bucket` is the combination of the two (the 4 categories listed above)

---

## How to run

### Option A: Run the churn pipeline (fully functional)

This is the **recommended, reproducible** way to run the project and generate:

- `customer_predictions.csv` (churn + 4 value/risk buckets)
- `bucket_summary.csv` (counts by bucket)
- `models/churn_model.joblib` (trained model)
- `reports/classification_report.txt` (evaluation on a holdout split)

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the pipeline (train + predict):

```bash
python predict.py --train
```

Optional flags:

- `--churn-threshold 0.5`: threshold used for churn prediction and risk level
- `--value-quantile 0.5`: value split based on `Monetary` (0.5 = median)
- `--data "online_retail_II 2009-2010.csv"`: raw dataset path

### Option A2: Run the Streamlit website (upload RFM CSV)

This app lets you upload a CSV containing **RFM features** and returns:

- churn prediction + probability
- the 4 categories: **High/Low Value** × **High/Low Risk**

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the website:

```bash
streamlit run streamlit_app.py
```

3. Upload a CSV containing these columns:

- `Customer ID` (optional but recommended)
- `Frequency`
- `Monetary`
- `AvgOrderValue`

The app will automatically train a model from `rfm_features.csv` if `models/rfm_churn_model.joblib` is missing.

### Option B: Run the notebooks (exploration)

1. Open `Untitled.ipynb` (or `1st_ML_Project.ipynb`) in Jupyter / VS Code.
2. Ensure the raw dataset file is present in the repo root:
   - `online_retail_II 2009-2010.csv`
3. Run cells top-to-bottom.

Expected outputs created in the repo root:

- `cleaned_online_retail.csv`
- `rfm_features.csv` (if you export it in your notebook run)
- `final_data_with_all_features.csv` (if you export it in your notebook run)

### Option B: Build the CLTV modeling table via MySQL

1. Install/configure MySQL and enable file import (`LOAD DATA INFILE`) as needed.
2. Update the file path inside `sql_required.sql` if your CSV path differs:
   - The script currently references:
     - `C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/online_retail_II.csv`
3. Run `sql_required.sql` in MySQL Workbench (or your preferred client).
4. Export the resulting `cltv_final_ml_data` table to CSV (this repo already includes `cltv_final_ml_data.csv`).

---

## Repository layout (high level)

- `*.ipynb`: notebooks (data prep, EDA, RFM features, churn modeling)
- `sql_required.sql`: optional SQL ETL + CLTV dataset construction
- `*.csv`: raw + cleaned + engineered feature datasets
- `*.xlsx`: raw dataset copies

---

## Notes / assumptions captured from the implementation

- **Data cleaning** (notebook): removes invalid/empty customers and computes `TotalPrice = Quantity * Price`; also builds `MonthYear` for seasonality analysis.
- **Segmentation**: K-Means on scaled RFM with 4 clusters; clusters mapped to business labels.
- **Churn model**: Random Forest trained on `Frequency`, `Monetary`, `AvgOrderValue` (explicitly excluding `Recency` to reduce leakage), producing `Churn_Probability`.

