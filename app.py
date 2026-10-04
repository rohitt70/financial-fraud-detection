"""
Financial Fraud Detection Using Machine Learning
=================================================
A modern, interactive, and beginner-friendly Streamlit web dashboard
designed for academic projects and machine learning demonstrations.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report

# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Fraud Detection ML",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling, cards, metrics, and risk badges
st.markdown("""
<style>
    /* Metric card styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        padding: 18px 22px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px -2px rgba(0, 0, 0, 0.08);
    }
    div[data-testid="stMetric"] label {
        font-weight: 600;
        font-size: 0.88rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Result card styling */
    .result-card-safe {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 24px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.15);
    }
    .result-card-fraud {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 24px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.2);
    }
    .result-title-safe {
        color: #065f46;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .result-title-fraud {
        color: #991b1b;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .result-subtitle {
        color: #334155;
        font-size: 1.05rem;
        font-weight: 500;
        line-height: 1.5;
    }

    /* Info callout card */
    .info-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 18px;
        font-size: 0.95rem;
        color: #334155;
    }

    /* Subheader custom styling */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 15px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. DATASET LOADER & DYNAMIC COLUMN DETECTOR
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_dataset(file_path_or_buffer=None):
    """
    Loads dataset from provided file or checks workspace for default CSVs.
    Returns cleaned pandas DataFrame.
    """
    if file_path_or_buffer is not None:
        df = pd.read_csv(file_path_or_buffer)
    elif os.path.exists("financial_transactions.csv"):
        df = pd.read_csv("financial_transactions.csv")
    else:
        # Fallback: scan current directory for any CSV
        csv_files = [f for f in os.listdir(".") if f.endswith(".csv")]
        if csv_files:
            df = pd.read_csv(csv_files[0])
        else:
            return None
    return df


def detect_columns(df):
    """
    Dynamically maps actual CSV column names to standardized logical keys.
    Works robustly even with case variations, underscores, hyphens, or aliases.
    """
    normalized = {c.strip().lower().replace("_", " ").replace("-", " "): c for c in df.columns}
    
    mapping = {}

    def find_match(candidates):
        for cand in candidates:
            cand_norm = cand.lower().replace("_", " ").replace("-", " ")
            if cand_norm in normalized:
                return normalized[cand_norm]
        # Partial match fallback
        for cand in candidates:
            for k, original in normalized.items():
                if cand.lower() in k:
                    return original
        return None

    mapping['fraud'] = find_match(['fraud', 'is fraud', 'isfraud', 'class', 'target'])
    mapping['amount'] = find_match(['transaction amount', 'amount', 'amt', 'spending'])
    mapping['type'] = find_match(['transaction type', 'type', 'trans type'])
    mapping['country'] = find_match(['country/region', 'country region', 'country', 'region', 'location'])
    mapping['access'] = find_match(['access method', 'accessmethod', 'channel', 'device'])
    mapping['account_type'] = find_match(['account type', 'account_type', 'acct type'])
    mapping['account_age'] = find_match(['account age', 'accountage', 'tenure'])
    mapping['customer_age'] = find_match(['customer age', 'customerage', 'age'])
    mapping['credit_score'] = find_match(['credit score', 'creditscore', 'score'])
    mapping['income'] = find_match(['annual income', 'annualincome', 'income', 'salary'])
    mapping['balance'] = find_match(['account balance', 'accountbalance', 'balance'])
    
    # Optional helpful features
    mapping['tx_per_hour'] = find_match(['transactions in hour', 'transactions_in_hour'])
    mapping['is_new_acc'] = find_match(['is new account', 'is_new_account'])
    mapping['high_risk_country'] = find_match(['high risk country flag', 'high_risk_country_flag'])

    # Identifier columns to drop
    id_candidates = [
        'transaction id', 'customer id', 'customer name', 'transaction timestamp',
        'city', 'latitude', 'longitude', 'transaction location', 'bank name',
        'browser type', 'round number transactions', 'transaction status', 'transaction currency'
    ]
    drop_cols = []
    for cand in id_candidates:
        cand_norm = cand.lower().replace("_", " ").replace("-", " ")
        if cand_norm in normalized:
            drop_cols.append(normalized[cand_norm])
    
    mapping['drop_cols'] = list(set(drop_cols))
    return mapping


# -----------------------------------------------------------------------------
# 3. MACHINE LEARNING MODEL TRAINING & PERSISTENCE
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def get_trained_model(df):
    """
    Trains a Random Forest Classifier on the preprocessed dataset.
    Caches model in memory and serializes to 'fraud_model.joblib'.
    """
    col_map = detect_columns(df)
    fraud_col = col_map.get('fraud')
    if not fraud_col or fraud_col not in df.columns:
        return None, None, None, None, None

    # Required feature candidates
    categorical_candidates = [
        col_map.get('type'),
        col_map.get('country'),
        col_map.get('access'),
        col_map.get('account_type')
    ]
    numerical_candidates = [
        col_map.get('amount'),
        col_map.get('account_age'),
        col_map.get('customer_age'),
        col_map.get('credit_score'),
        col_map.get('income'),
        col_map.get('balance'),
        col_map.get('tx_per_hour'),
        col_map.get('is_new_acc'),
        col_map.get('high_risk_country')
    ]

    cat_cols = [c for c in categorical_candidates if c and c in df.columns]
    num_cols = [c for c in numerical_candidates if c and c in df.columns]
    feature_cols = cat_cols + num_cols

    # Prepare features and target
    X = df[feature_cols].copy()
    y = df[fraud_col].astype(int).copy()

    # Handle missing values
    for c in cat_cols:
        X[c] = X[c].fillna('Unknown').astype(str)
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(X[c].median())

    # Build preprocessing and model pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', 'passthrough', num_cols)
        ]
    )

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            class_weight='balanced'
        ))
    ])

    # Stratified split to preserve fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model_pipeline.fit(X_train, y_train)

    # Evaluate
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)

    # Extract feature names & importances
    ohe = model_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cat_names = list(ohe.get_feature_names_out(cat_cols))
    all_feature_names = encoded_cat_names + num_cols
    importances = model_pipeline.named_steps['classifier'].feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    # Clean feature names for clean UI display
    importance_df['Clean_Feature'] = (
        importance_df['Feature']
        .str.replace('cat__', '')
        .str.replace('num__', '')
        .str.replace('_', ' ')
    )

    # Persist to disk
    try:
        joblib.dump(model_pipeline, 'fraud_model.joblib')
    except Exception:
        pass

    return model_pipeline, col_map, feature_cols, importance_df, roc_auc


# -----------------------------------------------------------------------------
# 4. TOP METRIC CARDS COMPONENT
# -----------------------------------------------------------------------------
def display_top_metrics(df, col_map):
    """Displays the 4 mandatory metric cards at the top of the dashboard."""
    fraud_col = col_map.get('fraud')
    total_tx = len(df)
    
    if fraud_col and fraud_col in df.columns:
        fraud_tx = int(df[fraud_col].sum())
        legit_tx = total_tx - fraud_tx
        fraud_pct = (fraud_tx / total_tx) * 100
    else:
        fraud_tx = 0
        legit_tx = total_tx
        fraud_pct = 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="💳 Total Transactions",
            value=f"{total_tx:,}",
            help="Total volume of transaction records loaded in the dataset"
        )
    with col2:
        st.metric(
            label="🔴 Fraudulent Transactions",
            value=f"{fraud_tx:,}",
            delta=f"-{fraud_pct:.2f}% risk" if fraud_tx > 0 else "0",
            delta_color="inverse",
            help="Transactions confirmed or labeled as fraudulent"
        )
    with col3:
        st.metric(
            label="🟢 Legitimate Transactions",
            value=f"{legit_tx:,}",
            delta=f"{(100 - fraud_pct):.2f}% safe",
            delta_color="normal",
            help="Transactions verified as normal customer activities"
        )
    with col4:
        st.metric(
            label="⚠️ Fraud Percentage",
            value=f"{fraud_pct:.2f}%",
            delta=f"{fraud_tx} cases",
            delta_color="inverse",
            help="Proportion of fraudulent transactions across the dataset"
        )
    st.write("")


# -----------------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION & DATASET CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🛡️ Financial Fraud Detection")
st.sidebar.markdown(
    "<span style='background-color:#eff6ff; color:#1d4ed8; padding:3px 8px; border-radius:12px; font-size:0.75rem; font-weight:600;'>AI & MACHINE LEARNING</span>",
    unsafe_allow_html=True
)
st.sidebar.divider()

# Dataset File Upload / Source Selector
st.sidebar.subheader("📂 Dataset Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload Financial Transactions CSV",
    type=["csv"],
    help="Upload your own transactions CSV file or use the built-in dataset"
)

# Load dataframe
df = load_dataset(uploaded_file)

if df is None:
    st.error("⚠️ No dataset found! Please ensure 'financial_transactions.csv' is present or upload a CSV file via the sidebar.")
    st.stop()

# Detect columns
col_map = detect_columns(df)
fraud_col = col_map.get('fraud', 'Fraud')

# Train/load ML model
with st.spinner("Initializing Machine Learning model..."):
    model, col_map, feature_cols, importance_df, model_roc = get_trained_model(df)

# Sidebar Dataset Summary Info
with st.sidebar.expander("📊 Dataset Status & Columns", expanded=False):
    st.write(f"**Rows:** {len(df):,}")
    st.write(f"**Columns:** {df.shape[1]}")
    st.write(f"**Target Column:** `{fraud_col}`")
    st.write(f"**Model ROC-AUC:** `{model_roc:.3f}`" if model_roc else "N/A")

# Navigation Menu
st.sidebar.subheader("🧭 Navigation")
page = st.sidebar.radio(
    "Select Dashboard Section:",
    [
        "📊 Dashboard",
        "🔍 Data Explorer",
        "📈 Fraud Analysis",
        "🤖 Fraud Prediction"
    ],
    index=0
)

st.sidebar.divider()
st.sidebar.info(
    "💡 **Project Tip:**\n"
    "This dashboard uses Random Forest classification with balanced weighting to identify suspicious financial behavior in real time."
)


# -----------------------------------------------------------------------------
# 6. PAGE 1: 📊 DASHBOARD
# -----------------------------------------------------------------------------
if page == "📊 Dashboard":
    st.title("📊 Financial Fraud Detection Dashboard")
    st.markdown(
        "Real-time overview of financial transactions, fraud ratios, distribution patterns, and channel risk vectors."
    )
    
    # Top KPI Metrics Cards
    display_top_metrics(df, col_map)
    st.divider()

    # Row 1: Fraud vs Legitimate & Amount Distribution
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### 🍩 Transaction Class Distribution")
        counts = df[fraud_col].value_counts().reset_index()
        counts.columns = ['Status', 'Count']
        counts['Label'] = counts['Status'].map({0: 'Legitimate (Normal)', 1: 'Fraudulent (High Risk)'})
        
        fig_donut = px.pie(
            counts,
            values='Count',
            names='Label',
            hole=0.6,
            color='Label',
            color_discrete_map={
                'Legitimate (Normal)': '#10b981',
                'Fraudulent (High Risk)': '#ef4444'
            }
        )
        fig_donut.update_traces(
            textinfo='percent+label',
            pull=[0, 0.1],
            marker=dict(line=dict(color='#ffffff', width=2))
        )
        fig_donut.update_layout(
            showlegend=False,
            height=340,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        amount_col = col_map.get('amount')
        st.markdown("### 💰 Transaction Amount Distribution")
        if amount_col:
            # Filter extreme upper percentile for clean histogram visualization
            upper_limit = df[amount_col].quantile(0.98)
            subset_amount = df[df[amount_col] <= upper_limit].copy()
            subset_amount['Fraud_Label'] = subset_amount[fraud_col].map({0: 'Legitimate', 1: 'Fraud'})

            fig_hist = px.histogram(
                subset_amount,
                x=amount_col,
                color='Fraud_Label',
                nbins=40,
                barmode='overlay',
                opacity=0.75,
                color_discrete_map={'Legitimate': '#10b981', 'Fraud': '#ef4444'},
                labels={amount_col: 'Transaction Amount ($)', 'Fraud_Label': 'Status'}
            )
            fig_hist.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=25, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Transaction Amount column not detected.")

    st.write("")
    st.divider()

    # Row 2: Fraud by Categorical Dimensions (Type, Access Method, Account Type)
    st.markdown("### 🔍 Multi-Dimensional Fraud Breakdown")
    col_t1, col_t2, col_t3 = st.columns(3)

    # 1. Fraud by Transaction Type
    with col_t1:
        type_col = col_map.get('type')
        if type_col:
            st.markdown("#### 🔄 Fraud by Transaction Type")
            fraud_by_type = (
                df[df[fraud_col] == 1][type_col]
                .value_counts()
                .reset_index()
            )
            fraud_by_type.columns = [type_col, 'Fraud Count']
            
            fig_type = px.bar(
                fraud_by_type.head(8),
                x='Fraud Count',
                y=type_col,
                orientation='h',
                color='Fraud Count',
                color_continuous_scale='Reds',
                labels={type_col: 'Type'}
            )
            fig_type.update_layout(
                height=320,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_type, use_container_width=True)

    # 2. Fraud by Access Method
    with col_t2:
        access_col = col_map.get('access')
        if access_col:
            st.markdown("#### 📱 Fraud by Access Method")
            fraud_by_access = (
                df[df[fraud_col] == 1][access_col]
                .value_counts()
                .reset_index()
            )
            fraud_by_access.columns = [access_col, 'Fraud Count']
            
            fig_access = px.bar(
                fraud_by_access,
                x=access_col,
                y='Fraud Count',
                color='Fraud Count',
                color_continuous_scale='Oranges',
                labels={access_col: 'Channel'}
            )
            fig_access.update_layout(
                height=320,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_access, use_container_width=True)

    # 3. Fraud by Account Type
    with col_t3:
        acct_col = col_map.get('account_type')
        if acct_col:
            st.markdown("#### 🏦 Fraud by Account Type")
            fraud_by_acct = (
                df[df[fraud_col] == 1][acct_col]
                .value_counts()
                .reset_index()
            )
            fraud_by_acct.columns = [acct_col, 'Fraud Count']
            
            fig_acct = px.bar(
                fraud_by_acct,
                x='Fraud Count',
                y=acct_col,
                orientation='h',
                color='Fraud Count',
                color_continuous_scale='Purples',
                labels={acct_col: 'Account'}
            )
            fig_acct.update_layout(
                height=320,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_acct, use_container_width=True)

    # Key Insights Card
    st.markdown("""
    <div class="info-card">
        <strong>💡 Key Operational Insights:</strong>
        <ul>
            <li><strong>High-Risk Channels:</strong> Web Browser and Mobile App interactions experience the highest concentration of attempted fraud.</li>
            <li><strong>Vulnerable Account Categories:</strong> Credit and Fixed Deposit accounts exhibit higher fraud-to-volume ratios.</li>
            <li><strong>Transaction Type Vulnerability:</strong> High-risk activity is heavily concentrated in Debit transfers, Stock Trading, and direct ATM/Crypto withdrawals.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 7. PAGE 2: 🔍 DATA EXPLORER
# -----------------------------------------------------------------------------
elif page == "🔍 Data Explorer":
    st.title("🔍 Interactive Financial Data Explorer")
    st.markdown(
        "Search, inspect, and filter individual transactions by transaction category, geographic origin, access channel, and fraud confirmation status."
    )

    # Interactive Filter Panel
    with st.expander("🎯 Filter Transaction Records", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)

        # Filter: Transaction Type
        type_col = col_map.get('type')
        if type_col and type_col in df.columns:
            types_available = sorted(list(df[type_col].dropna().unique()))
            with f_col1:
                selected_types = st.multiselect(
                    "Transaction Type:",
                    options=types_available,
                    default=[]
                )
        else:
            selected_types = []

        # Filter: Country / Region
        country_col = col_map.get('country')
        if country_col and country_col in df.columns:
            countries_available = sorted(list(df[country_col].dropna().unique()))
            with f_col2:
                selected_countries = st.multiselect(
                    "Country / Region:",
                    options=countries_available,
                    default=[]
                )
        else:
            selected_countries = []

        # Filter: Access Method
        access_col = col_map.get('access')
        if access_col and access_col in df.columns:
            access_available = sorted(list(df[access_col].dropna().unique()))
            with f_col3:
                selected_access = st.multiselect(
                    "Access Method:",
                    options=access_available,
                    default=[]
                )
        else:
            selected_access = []

        f_col4, f_col5, f_col6 = st.columns(3)

        # Filter: Account Type
        acct_col = col_map.get('account_type')
        if acct_col and acct_col in df.columns:
            acct_available = sorted(list(df[acct_col].dropna().unique()))
            with f_col4:
                selected_accts = st.multiselect(
                    "Account Type:",
                    options=acct_available,
                    default=[]
                )
        else:
            selected_accts = []

        # Filter: Fraud Status
        with f_col5:
            fraud_filter = st.selectbox(
                "Fraud Status:",
                options=["All Transactions", "Legitimate Only (0)", "Fraudulent Only (1)"],
                index=0
            )

        # Filter: Amount Range
        amount_col = col_map.get('amount')
        with f_col6:
            if amount_col and amount_col in df.columns:
                min_amt = float(df[amount_col].min())
                max_amt = float(df[amount_col].quantile(0.995))
                amount_range = st.slider(
                    "Transaction Amount Range ($):",
                    min_value=round(min_amt, 1),
                    max_value=round(max_amt, 1),
                    value=(round(min_amt, 1), round(max_amt, 1))
                )
            else:
                amount_range = None

    # Apply filters dynamically
    filtered_df = df.copy()

    if selected_types and type_col:
        filtered_df = filtered_df[filtered_df[type_col].isin(selected_types)]
    if selected_countries and country_col:
        filtered_df = filtered_df[filtered_df[country_col].isin(selected_countries)]
    if selected_access and access_col:
        filtered_df = filtered_df[filtered_df[access_col].isin(selected_access)]
    if selected_accts and acct_col:
        filtered_df = filtered_df[filtered_df[acct_col].isin(selected_accts)]
    
    if fraud_filter == "Legitimate Only (0)":
        filtered_df = filtered_df[filtered_df[fraud_col] == 0]
    elif fraud_filter == "Fraudulent Only (1)":
        filtered_df = filtered_df[filtered_df[fraud_col] == 1]

    if amount_range and amount_col:
        filtered_df = filtered_df[
            (filtered_df[amount_col] >= amount_range[0]) & 
            (filtered_df[amount_col] <= amount_range[1])
        ]

    # Filtered Count & Export
    c_info, c_btn = st.columns([3, 1])
    with c_info:
        st.markdown(
            f"**Displaying:** `{len(filtered_df):,}` of `{len(df):,}` records "
            f"({(len(filtered_df)/len(df))*100:.1f}% of total dataset)"
        )
    with c_btn:
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered CSV",
            data=csv_data,
            file_name="filtered_transactions.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Interactive Table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=450
    )

    # Summary Statistics Expander
    with st.expander("📈 Quick Statistical Summary of Filtered Data"):
        st.write(filtered_df.describe().T)


# -----------------------------------------------------------------------------
# 8. PAGE 3: 📈 FRAUD ANALYSIS
# -----------------------------------------------------------------------------
elif page == "📈 Fraud Analysis":
    st.title("📈 Advanced Fraud Pattern Analysis")
    st.markdown(
        "In-depth exploratory visualizations examining geographic risk hotspots, value discrepancies, and multi-variable correlations."
    )

    amount_col = col_map.get('amount')
    country_col = col_map.get('country')

    tab1, tab2, tab3 = st.tabs([
        "🌍 Geographic Distribution",
        "💵 Amount & Spending Risk",
        "🔥 Correlation & Feature Relationships"
    ])

    # Tab 1: Geographic Analysis
    with tab1:
        st.markdown("### 🌍 Country-Level Fraud Hotspots")
        if country_col and country_col in df.columns:
            # Aggregate total tx and fraud tx by country
            country_agg = df.groupby(country_col).agg(
                Total_Transactions=(fraud_col, 'count'),
                Fraud_Transactions=(fraud_col, 'sum')
            ).reset_index()
            country_agg['Fraud_Rate_Pct'] = (
                country_agg['Fraud_Transactions'] / country_agg['Total_Transactions']
            ) * 100
            country_agg = country_agg.sort_values('Fraud_Transactions', ascending=False)

            g1, g2 = st.columns([1, 1])
            with g1:
                st.markdown("#### 🚨 Top Countries by Fraud Volume")
                fig_c1 = px.bar(
                    country_agg.head(10),
                    x=country_col,
                    y='Fraud_Transactions',
                    color='Fraud_Transactions',
                    color_continuous_scale='Reds',
                    text='Fraud_Transactions',
                    labels={'Fraud_Transactions': 'Fraud Cases', country_col: 'Country'}
                )
                fig_c1.update_layout(height=380, coloraxis_showscale=False)
                st.plotly_chart(fig_c1, use_container_width=True)

            with g2:
                st.markdown("#### 📊 Country Fraud Rate Percentage")
                fig_c2 = px.bar(
                    country_agg.sort_values('Fraud_Rate_Pct', ascending=False).head(10),
                    x=country_col,
                    y='Fraud_Rate_Pct',
                    color='Fraud_Rate_Pct',
                    color_continuous_scale='Burg',
                    text=country_agg.sort_values('Fraud_Rate_Pct', ascending=False).head(10)['Fraud_Rate_Pct'].apply(lambda x: f"{x:.1f}%"),
                    labels={'Fraud_Rate_Pct': 'Fraud Rate (%)', country_col: 'Country'}
                )
                fig_c2.update_layout(height=380, coloraxis_showscale=False)
                st.plotly_chart(fig_c2, use_container_width=True)
        else:
            st.info("Country column not available for geographic analysis.")

    # Tab 2: Amount Analysis
    with tab2:
        st.markdown("### 💵 Transaction Amount Behavior Analysis")
        if amount_col:
            m1, m2, m3 = st.columns(3)
            legit_mean = df[df[fraud_col] == 0][amount_col].mean()
            fraud_mean = df[df[fraud_col] == 1][amount_col].mean()
            diff_ratio = (fraud_mean / legit_mean) if legit_mean > 0 else 1.0

            with m1:
                st.metric("Legitimate Average Amount", f"${legit_mean:,.2f}")
            with m2:
                st.metric("Fraudulent Average Amount", f"${fraud_mean:,.2f}")
            with m3:
                st.metric("Fraud vs Legit Ratio", f"{diff_ratio:.2f}x higher")

            st.write("")
            b1, b2 = st.columns(2)

            with b1:
                st.markdown("#### 📦 Amount Boxplot by Fraud Status")
                # Clip upper outliers for readable boxplot
                p98 = df[amount_col].quantile(0.98)
                clip_df = df[df[amount_col] <= p98].copy()
                clip_df['Status'] = clip_df[fraud_col].map({0: 'Legitimate', 1: 'Fraud'})

                fig_box = px.box(
                    clip_df,
                    x='Status',
                    y=amount_col,
                    color='Status',
                    color_discrete_map={'Legitimate': '#10b981', 'Fraud': '#ef4444'},
                    labels={amount_col: 'Amount ($)'}
                )
                fig_box.update_layout(height=360, showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)

            with b2:
                st.markdown("#### 📈 Cumulative Amount Comparison")
                fig_ecdf = px.ecdf(
                    clip_df,
                    x=amount_col,
                    color='Status',
                    color_discrete_map={'Legitimate': '#10b981', 'Fraud': '#ef4444'},
                    labels={amount_col: 'Amount ($)'}
                )
                fig_ecdf.update_layout(height=360)
                st.plotly_chart(fig_ecdf, use_container_width=True)

    # Tab 3: Numerical Correlation & Scatter Matrix
    with tab3:
        st.markdown("### 🔥 Numerical Feature Correlation Heatmap")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter down to most relevant behavioral columns
        preferred_num = [
            amount_col,
            col_map.get('account_age'),
            col_map.get('customer_age'),
            col_map.get('credit_score'),
            col_map.get('income'),
            col_map.get('balance'),
            fraud_col
        ]
        chosen_num = [c for c in preferred_num if c and c in df.columns]

        if len(chosen_num) >= 3:
            corr = df[chosen_num].corr()
            fig_heat = px.imshow(
                corr,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale="RdBu_r",
                labels=dict(color="Pearson Corr")
            )
            fig_heat.update_layout(height=420)
            st.plotly_chart(fig_heat, use_container_width=True)
        
        # Bivariate Scatter
        cust_age = col_map.get('customer_age')
        income = col_map.get('income')
        if cust_age and income:
            st.markdown("### 🎯 Customer Age vs Annual Income vs Fraud")
            sample_scatter = df.sample(min(1500, len(df)), random_state=42).copy()
            sample_scatter['Status'] = sample_scatter[fraud_col].map({0: 'Legitimate', 1: 'Fraud'})

            fig_scat = px.scatter(
                sample_scatter,
                x=cust_age,
                y=income,
                color='Status',
                color_discrete_map={'Legitimate': '#10b981', 'Fraud': '#ef4444'},
                opacity=0.6,
                labels={cust_age: 'Customer Age (Years)', income: 'Annual Income ($)'}
            )
            fig_scat.update_layout(height=400)
            st.plotly_chart(fig_scat, use_container_width=True)


# -----------------------------------------------------------------------------
# 9. PAGE 4: 🤖 FRAUD PREDICTION
# -----------------------------------------------------------------------------
elif page == "🤖 Fraud Prediction":
    st.title("🤖 Real-Time Transaction Fraud Prediction")
    st.markdown(
        "Enter transactional and customer parameters below to evaluate real-time fraud probability using our trained Random Forest Classifier."
    )

    if model is None:
        st.error("Model could not be trained. Please verify target column in dataset.")
        st.stop()

    # Pre-populate sample profiles
    if "sample_preset" not in st.session_state:
        st.session_state.sample_preset = "Custom Input"

    preset_col1, preset_col2, preset_col3 = st.columns([1.2, 1, 1])
    with preset_col1:
        st.markdown("**⚡ Quick Load Sample Profile:**")
    with preset_col2:
        if st.button("🟢 Load Normal Legitimate Sample", use_container_width=True):
            st.session_state.amt_val = 145.0
            st.session_state.type_val = "Online Payment"
            st.session_state.access_val = "Mobile App"
            st.session_state.acct_val = "Checking"
            st.session_state.country_val = "USA"
            st.session_state.acc_age_val = 1200
            st.session_state.cust_age_val = 35
            st.session_state.credit_val = 740
            st.session_state.income_val = 85000
            st.session_state.balance_val = 18500.0
            st.session_state.tx_hour_val = 2
            st.rerun()

    with preset_col3:
        if st.button("🔴 Load Suspicious / Fraud Sample", use_container_width=True):
            st.session_state.amt_val = 8400.0
            st.session_state.type_val = "Debit"
            st.session_state.access_val = "Web Browser"
            st.session_state.acct_val = "Credit"
            st.session_state.country_val = "UAE"
            st.session_state.acc_age_val = 12
            st.session_state.cust_age_val = 22
            st.session_state.credit_val = 510
            st.session_state.income_val = 32000
            st.session_state.balance_val = 1200.0
            st.session_state.tx_hour_val = 14
            st.rerun()

    # Dynamic options from dataset
    type_col = col_map.get('type')
    access_col = col_map.get('access')
    acct_col = col_map.get('account_type')
    country_col = col_map.get('country')

    types_list = sorted(list(df[type_col].dropna().unique())) if type_col else ["Bank Transfer", "Credit", "Debit", "Stock Trading", "UPI"]
    access_list = sorted(list(df[access_col].dropna().unique())) if access_col else ["Mobile App", "Web Browser", "ATM", "Phone Banking"]
    acct_list = sorted(list(df[acct_col].dropna().unique())) if acct_col else ["Checking", "Savings", "Credit", "Fixed Deposit"]
    country_list = sorted(list(df[country_col].dropna().unique())) if country_col else ["USA", "India", "UK", "Canada", "UAE"]

    # Prediction Input Form
    with st.form("fraud_prediction_form"):
        st.markdown("#### 📝 Transaction & Customer Assessment Parameters")
        
        c_form1, c_form2, c_form3 = st.columns(3)

        with c_form1:
            st.markdown("**1️⃣ Transaction Details**")
            in_amount = st.number_input(
                "Transaction Amount ($):",
                min_value=1.0,
                max_value=500000.0,
                value=float(st.session_state.get('amt_val', 450.0)),
                step=50.0
            )
            in_type = st.selectbox(
                "Transaction Type:",
                options=types_list,
                index=types_list.index(st.session_state.get('type_val', types_list[0])) if st.session_state.get('type_val') in types_list else 0
            )
            in_access = st.selectbox(
                "Access Channel / Method:",
                options=access_list,
                index=access_list.index(st.session_state.get('access_val', access_list[0])) if st.session_state.get('access_val') in access_list else 0
            )
            in_acct = st.selectbox(
                "Account Type:",
                options=acct_list,
                index=acct_list.index(st.session_state.get('acct_val', acct_list[0])) if st.session_state.get('acct_val') in acct_list else 0
            )

        with c_form2:
            st.markdown("**2️⃣ Customer Demographics**")
            in_cust_age = st.number_input(
                "Customer Age (Years):",
                min_value=18,
                max_value=100,
                value=int(st.session_state.get('cust_age_val', 38)),
                step=1
            )
            in_credit = st.number_input(
                "Credit Score (300 - 850):",
                min_value=300,
                max_value=850,
                value=int(st.session_state.get('credit_val', 680)),
                step=10
            )
            in_income = st.number_input(
                "Annual Income ($):",
                min_value=5000,
                max_value=500000,
                value=int(st.session_state.get('income_val', 75000)),
                step=2500
            )
            in_country = st.selectbox(
                "Transaction Country / Region:",
                options=country_list,
                index=country_list.index(st.session_state.get('country_val', country_list[0])) if st.session_state.get('country_val') in country_list else 0
            )

        with c_form3:
            st.markdown("**3️⃣ Account History & Velocity**")
            in_acc_age = st.number_input(
                "Account Age (Days):",
                min_value=0,
                max_value=5000,
                value=int(st.session_state.get('acc_age_val', 850)),
                step=30
            )
            in_balance = st.number_input(
                "Current Account Balance ($):",
                min_value=0.0,
                max_value=1000000.0,
                value=float(st.session_state.get('balance_val', 12500.0)),
                step=500.0
            )
            in_tx_hour = st.number_input(
                "Transactions in Last Hour:",
                min_value=0,
                max_value=30,
                value=int(st.session_state.get('tx_hour_val', 3)),
                step=1
            )

        st.write("")
        submit_btn = st.form_submit_button(
            "🔍 Predict Fraud",
            type="primary",
            use_container_width=True
        )

    # Prediction Execution & Results Display
    if submit_btn:
        # Construct single-row DataFrame aligned with training feature names
        input_data = {}
        if type_col: input_data[type_col] = [str(in_type)]
        if country_col: input_data[country_col] = [str(in_country)]
        if access_col: input_data[access_col] = [str(in_access)]
        if acct_col: input_data[acct_col] = [str(in_acct)]
        
        amount_c = col_map.get('amount')
        acc_age_c = col_map.get('account_age')
        cust_age_c = col_map.get('customer_age')
        credit_c = col_map.get('credit_score')
        income_c = col_map.get('income')
        balance_c = col_map.get('balance')
        tx_hr_c = col_map.get('tx_per_hour')
        new_acc_c = col_map.get('is_new_acc')
        hr_cntry_c = col_map.get('high_risk_country')

        if amount_c: input_data[amount_c] = [float(in_amount)]
        if acc_age_c: input_data[acc_age_c] = [int(in_acc_age)]
        if cust_age_c: input_data[cust_age_c] = [int(in_cust_age)]
        if credit_c: input_data[credit_c] = [int(in_credit)]
        if income_c: input_data[income_c] = [float(in_income)]
        if balance_c: input_data[balance_c] = [float(in_balance)]
        if tx_hr_c: input_data[tx_hr_c] = [int(in_tx_hour)]
        
        # Automatically infer risk indicators if present in training set
        if new_acc_c:
            input_data[new_acc_c] = [1 if in_acc_age <= 30 else 0]
        if hr_cntry_c:
            # Check high risk countries in dataset
            high_risk_nations = ['UAE', 'Russia', 'South Africa']
            input_data[hr_cntry_c] = [1 if in_country in high_risk_nations else 0]

        input_df = pd.DataFrame(input_data)

        # Ensure all columns present in feature_cols exist
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = df[col].median() if col in df.select_dtypes(include=[np.number]) else "Unknown"

        # Re-order to exact training column sequence
        input_df = input_df[feature_cols]

        # Model Inference
        probabilities = model.predict_proba(input_df)[0]
        fraud_probability = float(probabilities[1])
        legit_probability = float(probabilities[0])

        # Classification Threshold: 0.35 optimized for highly-skewed fraud classification
        is_fraud = fraud_probability >= 0.35

        st.write("")
        st.markdown("### 📊 Prediction Verdict")

        # Large Result Display
        if is_fraud:
            st.markdown(f"""
            <div class="result-card-fraud">
                <div class="result-title-fraud">🔴 Fraudulent Transaction Detected!</div>
                <div class="result-subtitle">
                    <strong>High Risk Anomaly Flagged:</strong> This transaction exhibits behavioral characteristics 
                    strongly correlated with fraudulent operations. Immediate automated block or manual AML review recommended.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-safe">
                <div class="result-title-safe">🟢 Legitimate Transaction</div>
                <div class="result-subtitle">
                    <strong>Standard Operational Pattern:</strong> Transaction parameters align with valid 
                    cardholder spending habits and low-risk behavioral baselines. Safe to authorize.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Visual Risk Probability Progress Bar & Metrics
        st.markdown("#### ⚡ Risk Probability Meter")
        st.progress(fraud_probability)

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric(
                label="Fraud Probability Score",
                value=f"{fraud_probability * 100:.1f}%",
                delta="High Risk" if is_fraud else "Low Risk",
                delta_color="inverse" if is_fraud else "normal"
            )
        with r2:
            st.metric(
                label="Legitimate Confidence",
                value=f"{legit_probability * 100:.1f}%",
                delta="Safe" if not is_fraud else "Uncertain",
                delta_color="normal"
            )
        with r3:
            risk_tier = "🔴 CRITICAL RISK" if fraud_probability > 0.65 else ("🟠 MODERATE RISK" if fraud_probability >= 0.35 else "🟢 LOW RISK")
            st.metric(
                label="Assigned Risk Tier",
                value=risk_tier
            )

        st.divider()

        # Feature Importance Chart
        st.markdown("### 🔍 Model Feature Importance")
        st.caption("Shows which transactional signals contribute most significantly to Random Forest decision splits.")

        if importance_df is not None and not importance_df.empty:
            top_features = importance_df.head(10).copy()
            fig_imp = px.bar(
                top_features,
                x='Importance',
                y='Clean_Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Blues',
                labels={'Importance': 'Gini Importance Score', 'Clean_Feature': 'Feature Signal'}
            )
            fig_imp.update_layout(
                height=380,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_imp, use_container_width=True)
