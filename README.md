# 💳 Financial Fraud Detection Using Machine Learning
### A Modern, Interactive Streamlit Web Dashboard

---

## 📌 Project Overview
This project presents an interactive, full-stack machine learning web dashboard built with **Streamlit**, **Scikit-Learn**, and **Plotly** to detect and analyze fraudulent financial transactions. 

The application automatically loads financial transaction datasets, visualizes critical metrics, detects high-risk fraud patterns, provides interactive filtering, and performs real-time fraud predictions using a pre-trained **Random Forest Classifier**.

---

## 🚀 Key Features

### 1. 📊 Key Statistics Metric Cards
- **Total Transactions:** Total record count loaded dynamically.
- **Fraudulent Transactions:** Count of confirmed fraudulent activities.
- **Legitimate Transactions:** Count of verified safe customer transactions.
- **Fraud Percentage:** Fraud rate with comparative delta indicators.

### 2. 🧭 Sidebar Navigation
- **📊 Dashboard:** High-level executive overview with donut charts, transaction amount distributions, and multi-dimensional breakdowns by Transaction Type, Access Channel, and Account Category.
- **🔍 Data Explorer:** Interactive dataframe viewer with multi-select filters (Type, Country, Channel, Account Type, Fraud Status, Amount slider) and filtered CSV export.
- **📈 Fraud Analysis:** Deep-dive analytics exploring country-level fraud hotspots, amount behavior metrics (mean/median differences), Pearson correlation heatmaps, and bivariate scatter plots.
- **🤖 Fraud Prediction:** Real-time prediction interface with quick-fill presets for instant testing, clear verdict banners (🟢 Legitimate / 🔴 Fraudulent), an animated probability meter, and model feature importance charts.

### 3. 🤖 Machine Learning Pipeline
- **Automatic Preprocessing:** Automatically detects and strips irrelevant identifiers (e.g., `Transaction ID`, `Customer ID`, `Customer Name`, coordinates).
- **Categorical Encoding:** One-Hot Encoding with `handle_unknown='ignore'` via Scikit-Learn `ColumnTransformer`.
- **Handling Class Imbalance:** Employs balanced class weighting (`class_weight='balanced'`) to effectively learn patterns from skewed fraud distributions (~1.5% fraud rate).
- **Algorithm:** Random Forest Classifier (120 estimators, max depth 12) achieving **>0.92 ROC-AUC**.
- **Model Caching:** Uses Streamlit's `@st.cache_resource` and persists model weights to `fraud_model.joblib` for instant load times.

---

## 📂 Project Structure

```text
financial-fraud-detection/
│
├── app.py                      # Main Streamlit web application & ML pipeline
├── financial_transactions.csv  # Financial transactions dataset
├── requirements.txt            # Python dependencies
├── fraud_model.joblib          # Serialized Random Forest model (auto-generated)
└── README.md                   # Project documentation and guide
```

---

## 💻 Installation & Quickstart

### Prerequisites
- Python 3.9, 3.10, 3.11, 3.12, or 3.13 installed on your system.

### Step 1: Open Terminal / Command Prompt
Navigate to the project directory:
```bash
cd "C:\Users\rohit\OneDrive\Desktop\New folder (2)"
```

### Step 2: Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Dashboard
Launch the web dashboard:
```bash
streamlit run app.py
```

Once launched, Streamlit will display local and network URLs:
```text
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```
Open your web browser at **`http://localhost:8501`** to view and interact with the application.

---

## 🧪 Testing the Live Prediction Page

The **Fraud Prediction** section includes two instant-fill buttons to demonstrate both outcomes to professors and examiners during project demos:
1. **🟢 Load Normal Legitimate Sample:** Fills the form with typical spending behavior ($145 online payment, high credit score, established account). Output: **🟢 Legitimate Transaction** with low fraud risk (~5%).
2. **🔴 Load Suspicious / Fraud Sample:** Fills the form with high-risk characteristics (large debit amount, brand new account, newly created credentials, elevated hourly velocity). Output: **🔴 Fraudulent Transaction Detected!** with high risk score (>90%).

---

## 🛠️ Technologies Used
- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Interactive Visualizations:** [Plotly Express & Graph Objects](https://plotly.com/python/)
- **Machine Learning:** [Scikit-Learn](https://scikit-learn.org/) (Random Forest Classifier, Pipelines, ColumnTransformers)
- **Model Persistence:** [Joblib](https://joblib.readthedocs.io/)
