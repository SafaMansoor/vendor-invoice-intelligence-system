# Invoice Intelligence System

A Streamlit application for **freight cost estimation** and **invoice risk review**. It combines pre-trained scikit-learn models with an interactive interface so users can explore invoice inputs and identify records that may need human review.

> Risk predictions are decision-support signals only. They are not proof of fraud, non-compliance, or wrongdoing.

## Features

- **Freight cost prediction:** estimates freight from an invoice amount.
- **Invoice risk analysis:** returns a risk probability and a low/high risk classification.
- **Interactive inputs:** accepts invoice and purchase summary values without requiring a database at app runtime.

## Risk model inputs

The risk classifier expects these seven inputs, in this order:

1. `invoice_quantity`
2. `invoice_dollars`
3. `Freight`
4. `days_to_pay`
5. `total_brands`
6. `total_item_quantity`
7. `total_item_dollars`

The app applies the saved `StandardScaler` before passing the values to the classifier. Keep the model and scaler files together in `models/`; they must be trained with the same feature names and order.

## How risk labels are created

The training workflow uses an Isolation Forest to flag a portion of the dataset as anomalous (the default contamination is 10%). Welch's t-tests are then used to identify features that differ between flagged and unflagged records. A supervised classifier is trained to reproduce those generated labels.

These are **anomaly-derived labels**, not independently verified fraud findings or confirmed audit outcomes. Evaluation scores therefore describe agreement with the generated labels, not real-world fraud-detection performance.

## Project layout

```text
.
├── app.py
├── requirements.txt
├── models/
│   ├── invoice_risk_model.pkl
│   ├── predict_freight_model.pkl
│   └── risk_scaler.pkl
├── notebook/
│   ├── 01_eda_invoice_risk.ipynb
│   ├── 02_data_preprocessing_invoice_risk.ipynb
│   ├── 03_model_training_invoice_risk.ipynb
│   ├── Eda_freight_cost.ipynb
│   ├── data_preprocessing_freightcost.ipynb
│   ├── model_training_freight_cost.ipynb
│   └── processed_invoices.csv
└── src/
    ├── data_preprocessing_freight_cost.py
    └── data_preprocessing_invoice_risk.py
```

The pre-trained model files are required to run the application. The local SQLite database (`data/inventory.db`) is not required for predictions and is not included in the repository.

## Run locally

### Requirements

- Python 3.10 or later
- pip

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) when the application is ready.

## Training and notebooks

The notebooks in `notebook/` document exploratory analysis, preprocessing, and model experiments. Re-running the workflows may require the original SQLite dataset and the paths or settings used in the notebooks. The database is intentionally excluded from this repository; provide an authorized local copy before attempting to reproduce training.

The Python preprocessing modules are in `src/`. The Streamlit app loads the existing model artifacts from `models/` and does not train models at startup.

## Data and responsible use

- Do not publish the source database, confidential invoice records, or identifying business data.
- Review `notebook/processed_invoices.csv` before sharing the repository. Remove it or replace it with an authorized, anonymized sample if it contains sensitive information.
- Treat model outputs as prompts for human review, not as final decisions.
- Validate model performance on representative, independently reviewed data before using it for operational decisions.

## Dependencies

Dependencies are listed in [`requirements.txt`](requirements.txt). The project uses Streamlit, pandas, NumPy, SciPy, scikit-learn, Joblib, Matplotlib, and Seaborn.

## Maintainer

**Safa Mansoor** · [GitHub](https://github.com/SafaMansoor)
