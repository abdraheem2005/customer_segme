import streamlit as st
import pandas as pd
import plotly.express as px
import base64, time
from pathlib import Path

from utils.io import load_customer_file
from inference.batch_predictor import BatchPredictor

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Retail Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# ============================================================
# SESSION STATE
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "login"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ============================================================
# THEME + STYLING
# ============================================================
def apply_theme():
    st.markdown("""
    <style>
    html, body, [class*="css"] {
        color: white !important;
    }

    h1,h2,h3,h4,h5,h6,p,span,label,div {
        color: white !important;
    }

    input, textarea {
        color: white !important;
        background-color: #1e1e1e !important;
    }

    /* Pill buttons */
    button:not(div.modebar button) {
        background-color: black !important;
        color: white !important;
        border: 1px solid white !important;
        border-radius: 999px !important;
        padding: 10px 22px !important;
    }

    /* Pill tabs */
    div[data-baseweb="tab-list"] {
        gap: 12px;
    }

    div[data-baseweb="tab-list"] button {
        border-radius: 999px !important;
        padding: 10px 24px !important;
        background-color: #111 !important;
        color: white !important;
        border: 1px solid #555 !important;
    }

    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: black !important;
        border: 2px solid white !important;
    }

    div[data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Plotly toolbar normal */
    div.modebar button {
        border-radius: 6px !important;
        padding: 4px 6px !important;
        background-color: #222 !important;
    }

    .stApp {
        background-color: #0e1117;
    }
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# ============================================================
# BACKGROUND IMAGE
# ============================================================
def set_background(image_path: str, dim: float = 0.7):
    p = Path(image_path)
    if not p.exists():
        return
    encoded = base64.b64encode(p.read_bytes()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background: url("data:image/jpeg;base64,{encoded}") no-repeat center center fixed;
        background-size: cover;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,{dim});
        z-index: 0;
    }}
    .block-container {{
        position: relative;
        z-index: 1;
    }}
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# NAV BAR
# ============================================================
def nav_bar():
    col1, col2, col3 = st.columns([4,1,1])

    with col1:
        if st.button("🏠 Home"):
            st.session_state.page = "home"
            st.rerun()
        if st.button("📊 Predict"):
            st.session_state.page = "predict"
            st.rerun()

    with col3:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.rerun()

# ============================================================
# BUSINESS LABELING
# ============================================================
def get_segment_labels(df_results):
    cluster_stats = df_results.groupby("Cluster").agg({
        "Monetary": "mean",
        "Recency": "mean",
        "Frequency": "mean"
    }).reset_index()

    labels = {}

    loyal_cluster = cluster_stats.sort_values("Monetary", ascending=False).iloc[0]["Cluster"]
    labels[loyal_cluster] = "💎 High-Value Loyal"

    remaining = cluster_stats[cluster_stats["Cluster"] != loyal_cluster]
    at_risk_cluster = remaining.sort_values("Recency", ascending=False).iloc[0]["Cluster"]
    labels[at_risk_cluster] = "⚠️ At-Risk / Churning"

    remaining = remaining[remaining["Cluster"] != at_risk_cluster]
    recent_cluster = remaining.sort_values("Recency", ascending=True).iloc[0]["Cluster"]
    labels[recent_cluster] = "🌱 Recent Low Spenders"

    remaining = remaining[remaining["Cluster"] != recent_cluster]
    if not remaining.empty:
        bulk_cluster = remaining.iloc[0]["Cluster"]
        labels[bulk_cluster] = "🛒 Bulk/Average Buyers"

    return labels

# ============================================================
# LOGIN PAGE
# ============================================================
def login_page():
    set_background("bgl.jpg",0.4)

    st.markdown("<h1 style='text-align:center;'>🔐 Admin Login</h1>", unsafe_allow_html=True)

    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if u == "admin" and p == "12345":
                st.session_state.logged_in = True
                st.session_state.page = "home"
                st.success("Login successful")
                time.sleep(0.3)
                st.rerun()
            else:
                st.error("Invalid credentials")

# ============================================================
# HOME PAGE
# ============================================================
def home_page():
    set_background("bgh.jpg",0.3)
    nav_bar()

    st.markdown("""
    <h1 style="text-align:center; margin-top:120px;">
        🛍️ AI Customer Segmentation System
    </h1>
    <p style="text-align:center;">
        Upload your sales data and discover Loyal, At-Risk and New customers.
    </p>
    """, unsafe_allow_html=True)

# ============================================================
# PREDICT PAGE
# ============================================================
def predict_page():
    set_background("bgc.jpg",0.6)
    nav_bar()

    st.markdown("<h2 style='text-align:center;'>📊 Customer Segmentation Predictor</h2>", unsafe_allow_html=True)

    # REQUIRED COLUMNS INFO
    st.info("""
    ### 📂 Required Columns in Uploaded File
    Your CSV/Excel file **must contain** the following columns:

    - **InvoiceNo** – Invoice ID  
    - **StockCode** – Product ID  
    - **Description** – Product Name  
    - **Quantity** – Units sold  
    - **InvoiceDate** – Transaction date  
    - **UnitPrice** – Price per unit  
    - **CustomerID** – Unique customer ID  
    """)

    uploaded_file = st.file_uploader("Upload Transaction History", type=["csv","xlsx","xls"])

    if uploaded_file is None:
        st.info("👆 Please upload your transaction file to begin.")
        return

    raw_df = load_customer_file(uploaded_file)

    st.write(f"**Data Loaded:** {raw_df.shape[0]:,} rows")

    with st.expander("🔍 Preview Raw Data"):
        st.dataframe(raw_df.head())

    if st.button("🚀 Analyze Segments"):
        with st.spinner("Analyzing data..."):
            predictor = BatchPredictor()
            results_df = predictor.predict(raw_df)

            segment_map = get_segment_labels(results_df)
            results_df["Segment Name"] = results_df["Cluster"].map(segment_map)

        st.success("Analysis Complete!")

        tab1, tab2, tab3 = st.tabs(["📊 Business Insights","📈 Visualizations","📥 Export Data"])

        with tab1:
            summary = results_df.groupby("Segment Name").agg({
                "CustomerID":"count",
                "Monetary":"mean",
                "Recency":"mean",
                "Frequency":"mean"
            }).rename(columns={"CustomerID":"Count"})
            st.dataframe(summary)

            st.markdown("""
            ### 💡 Interpretation Guide
            - **💎 High-Value Loyal:** VIP customers, high spend & frequent visits  
            - **🛒 Bulk/Average:** Regular shoppers  
            - **🌱 Recent Low Spenders:** New or small buyers  
            - **⚠️ At-Risk:** Long time since last purchase
            """)

        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                fig1 = px.scatter(
                    results_df,
                    x="Monetary",
                    y="Frequency",
                    color="Segment Name",
                    hover_data=["CustomerID"],
                    title="Customer Value vs Frequency"
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                fig2 = px.box(
                    results_df,
                    x="Segment Name",
                    y="Recency",
                    color="Segment Name",
                    title="Recency by Segment"
                )
                st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Full Report (CSV)",
                csv,
                "segmented_customers.csv"
            )

# ============================================================
# ROUTER
# ============================================================
if not st.session_state.logged_in:
    login_page()
elif st.session_state.page == "home":
    home_page()
elif st.session_state.page == "predict":
    predict_page()
else:
    home_page()
