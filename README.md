# 📊 Vendor Invoice Intelligence System
> **An AI-Powered Freight Cost Prediction & Invoice Risk Detection System**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-ff4b4b.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression%20%7C%20Classification-orange.svg)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Library-f7931e.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243.svg)
![Joblib](https://img.shields.io/badge/Joblib-Model%20Serialization-green.svg)
![Status](https://img.shields.io/badge/Project-Completed%20Development-brightgreen.svg)

---

# Vendor Invoice Intelligence System

### Freight Cost Prediction + Invoice Risk Detection


# 📌 Project Overview

The **Vendor Invoice Intelligence System** is an end-to-end Machine Learning-based application designed to assist businesses in analyzing vendor invoices, estimating freight expenses, and identifying [...]

The system focuses on two major business problems:

- Predicting the expected freight cost associated with an invoice.
- Identifying invoices that may require additional verification or manual review.

The project integrates data preprocessing, exploratory data analysis, feature engineering, Machine Learning algorithms, model evaluation, and deployment through an interactive Streamlit application.

### Core Components

- Data Understanding and Exploratory Data Analysis
- Data Preprocessing and Feature Engineering
- Freight Cost Prediction
- Invoice Risk Detection
- Streamlit Web Application

---

# 🎯 Business Problem

Businesses process numerous invoices as part of their regular purchasing and financial operations.

Manual invoice analysis can become time-consuming and inefficient, particularly when dealing with large volumes of transaction records.

Some common challenges include:

- Difficulty estimating expected freight expenses.
- Identifying unusual invoice characteristics.
- Detecting potentially inconsistent invoice information.
- Increasing workload for finance and verification teams.
- Lack of automated decision-support systems.

The Vendor Invoice Intelligence System addresses these challenges using Machine Learning to provide automated predictions and risk indicators.

---

# 🚀 Business Objectives

The primary objectives are:

- Automate freight cost estimation.
- Identify potentially risky invoices.
- Reduce repetitive manual verification.
- Support invoice validation workflows.
- Provide data-driven insights for business operations.
- Develop an interactive application for real-time model inference.

---

# 🖥️ Streamlit Web Application

The application is developed using Streamlit and provides two independent Machine Learning modules.

## Application Features

- Freight Cost Prediction
- Invoice Risk Detection
- Interactive invoice input interface
- Real-time model predictions
- Risk probability visualization
- Automated invoice risk flagging
- Model information and prediction results

---

## Module 1 — Freight Cost Prediction 🚚

This module uses a trained regression model to estimate the expected freight cost associated with an invoice.

### Input Features
- Invoice Quantity
- Invoice Dollars

### Output
- Predicted Freight Cost

### How It Works

1. The user enters invoice information.
2. The application prepares the input features.
3. The trained regression model processes the information.
4. The predicted freight cost is displayed.


## Module 2 — Invoice Risk Detection 🚨

This module uses a trained classification model to identify invoices that may require additional verification.

It evaluates invoice-related financial and operational characteristics to estimate the likelihood of an invoice being flagged.

### Input Features
- Invoice Quantities
- Invoice Dollars
- Freight Cost
- Delay Features

### Output
- Risk Probability (%)
- Normal Invoice
- Manual Review Required

### How It Works

1. The user enters the invoice information.
2. The application processes the selected features.
3. The trained classification model generates a prediction.
4. The application displays the risk probability.
5. The invoice is classified according to the decision threshold.


# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning algorithms |
| Streamlit | Interactive web application |
| Joblib | Model serialization and loading |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| SQLite | Relational database operations |

---

# 📂 Dataset and Data Understanding

The project uses invoice-related transactional data containing financial, purchasing, quantity, and operational information.

The data is used to understand invoice characteristics and develop predictive features for the two Machine Learning modules.

### Data Understanding Includes

- Database and table inspection
- Data type identification
- Missing value analysis
- Duplicate detection
- Statistical summaries
- Distribution analysis
- Correlation analysis
- Identification of inconsistent records

---

# 🧹 Data Preprocessing and Feature Engineering

Data preprocessing is performed to prepare the transactional information for Machine Learning.

### Preprocessing Operations

- Handling missing values
- Identifying duplicate records
- Data type conversion
- Date validation
- Handling inconsistent records
- Feature selection
- Feature engineering
- Preparing model-ready datasets

### Engineered Features

The system uses invoice-related financial and operational information, including:

- Invoice Quantity
- Invoice Dollars
- Freight Cost
- Total Item Quantity
- Total Item Dollars
- Receiving Delay
- Purchase Order to Invoice Delay
- Payment Delay

The final features used by each model are selected according to its prediction objective.

---

# 🤖 Machine Learning Models

The project implements two predictive Machine Learning workflows.

## 1. Freight Cost Prediction

**Problem Type:** Regression

The objective is to predict a continuous numerical value representing the expected freight cost.

| Model | Purpose |
|---|---|
| Linear Regression | Baseline regression model |

## 2. Invoice Risk Detection

**Problem Type:** Classification

The objective is to classify invoices into risk categories to assist manual verification.

| Model | Purpose |
|---|---|
|Fine Tuned Random Forest Classifier | Invoice risk classification |

---

# 📊 Model Evaluation

Both Machine Learning workflows are evaluated using metrics appropriate to their prediction tasks.

### Regression Evaluation

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Classification Evaluation

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix


---

# 💾 Model Serialization

The trained Machine Learning models are saved using Joblib.

This allows the Streamlit application to load the trained models directly without retraining them every time the application runs.

### Saved Models

- `freight_cost_model.pkl`
- `invoice_risk_model.pkl`

---

# 🏗️ Project Architecture

The system follows a structured Machine Learning workflow:

1. Data Collection
2. Data Understanding and EDA
3. Data Preprocessing
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Model Serialization
8. Streamlit Deployment
9. Real-Time Prediction

---

# 📁 Project Structure

```bash
vendor-invoice-intelligence-system/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── freight_cost_model.pkl
│   └── invoice_risk_model.pkl
│
├── data/
│
├── notebooks/
│   ├── 01_eda_invoice_risk
│   ├── 02_data_preprocessing_invoice_risk
│   ├── 03_model_training_invoice_risk.ipynb
│   ├── eda_freightcost.ipynb
│   └── model_training_freight_cost.ipynb
│
└── src/
    ├── data_preprocessing_invoice_risk.py
    └── data_preprocessing_freight_cost.py
```

### Directory Description

| File / Directory | Description |
|---|---|
| `app.py` | Main Streamlit application |
| `requirements.txt` | Project dependencies |
| `README.md` | Project documentation |
| `models/` | Saved trained Machine Learning models |
| `freight_cost_model.pkl` | Trained freight prediction model |
| `invoice_risk_model.pkl` | Trained invoice risk classifier |
| `data/` | Dataset and processed data files |
| `notebooks/` | Jupyter notebooks for EDA and model training |
| `01_eda_invoice_risk` | Invoice risk exploratory data analysis notebook |
| `02_data_preprocessing_invoice_risk` | Invoice risk data preprocessing notebook |
| `03_model_training_invoice_risk.ipynb` | Invoice risk model training notebook |
| `eda_freightcost.ipynb` | Freight cost exploratory data analysis notebook |
| `model_training_freight_cost.ipynb` | Freight cost model training notebook |
| `src/` | Source code for data preprocessing |
| `data_preprocessing_invoice_risk.py` | Invoice risk data preprocessing module |
| `data_preprocessing_freight_cost.py` | Freight cost data preprocessing module |

---

# ⚙️ Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/SafaMansoor/vendor-invoice-intelligence-system.git
```

### 2. Navigate to the Project Directory

```bash
cd vendor-invoice-intelligence-system
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

---

# 📈 Key Technical Learnings

This project provided practical experience in:

- Data preprocessing and validation
- Exploratory Data Analysis
- Feature engineering
- Regression algorithms
- Classification algorithms
- Model evaluation
- Model serialization
- Machine Learning inference
- Streamlit application development
- End-to-end ML project development

---

# 🔮 Future Improvements

Potential future enhancements include:

- Advanced anomaly detection techniques
- Additional Machine Learning algorithms
- REST API integration
- Cloud deployment
- Automated model retraining
- Integration with enterprise financial systems
- Enhanced invoice validation workflows

---

# 🏁 Conclusion

The **Vendor Invoice Intelligence System** demonstrates how Machine Learning can be applied to invoice-related financial and operational workflows.

By combining freight cost prediction and invoice risk detection, the system provides an interactive decision-support solution that helps businesses estimate freight expenses and identify invoices that[...]

The project brings together data analysis, preprocessing, predictive modeling, and application deployment into a single end-to-end Machine Learning system.

---

## 👩‍💻 Author

**Safa Mansoor**
GitHub: [SafaMansoor](https://github.com/SafaMansoor)

---
