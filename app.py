import streamlit as st
import pandas as pd
import plotly.express as px
import base64
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
# THEME
# ============================================================
def apply_theme():
    st.markdown("""
    <style>
    html, body, [class*="css"] { color: white !important; }
    h1,h2,h3,h4,h5,h6,p,span,label,div { color: white !important; }

    input {
        color: white !important;
        background-color: #1e1e1e !important;
    }

    button:not(div.modebar button) {
        background-color: black !important;
        color: white !important;
        border: 1px solid white !important;
        border-radius: 999px !important;
        padding: 10px 22px !important;
    }

    .stApp { background-color: #0e1117; }
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# ============================================================
# BACKGROUND
# ============================================================
def set_background(image_path: str, dim: float = 0.6):
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
# NAVIGATION
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
        if st.button("🔮 Simulator"):
            st.session_state.page = "simulator"
            st.rerun()

    with col3:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.rerun()

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
        🛍️ AI-Powered Customer Intelligence System
    </h1>
    """, unsafe_allow_html=True)

# ============================================================
# PREDICT PAGE
# ============================================================
def predict_page():
    set_background("bgc.jpg",0.6)
    nav_bar()

    st.markdown("<h2 style='text-align:center;'>📊 Portfolio-Level Segmentation Analysis</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Customer Transaction Dataset", type=["csv","xlsx","xls"])

    if uploaded_file is None:
        return

    raw_df = load_customer_file(uploaded_file)

    if st.button("🚀 Run Segmentation Analysis"):
        with st.spinner("Processing dataset..."):
            predictor = BatchPredictor()
            results_df = predictor.predict(raw_df)

        st.success("Segmentation Completed Successfully!")
        st.dataframe(results_df.head())

        st.markdown("## 📈 Customer Distribution Overview")
        segment_counts = results_df["Segment"].value_counts()
        st.write(segment_counts)

        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Segment Composition"
        )
        st.plotly_chart(fig)

# ============================================================
# SIMULATOR PAGE
# ============================================================
def simulator_page():
    set_background("bgc.jpg",0.6)
    nav_bar()

    st.markdown("<h2 style='text-align:center;'>🔮 Individual Customer Scenario Simulator</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        recency = st.number_input("Days Since Last Purchase", min_value=0, value=30)
        frequency = st.number_input("Total Transactions", min_value=1, value=5)
        monetary = st.number_input("Total Revenue Generated (₹)", min_value=0.0, value=1000.0)
        total_quantity = st.number_input("Total Units Purchased", min_value=0, value=20)
        unique_products = st.number_input("Distinct Products Purchased", min_value=0, value=5)

        predict_btn = st.button("🔍 Evaluate Customer Profile")

    with col2:
        st.markdown("""
        ### 📘 Metric Explanation
        - Recency → Engagement freshness  
        - Frequency → Purchase consistency  
        - Monetary → Revenue contribution  
        - Quantity → Purchase volume  
        - Unique products → Buying diversity  
        """)

    if predict_btn:
        with st.spinner("Evaluating profile..."):
            predictor = BatchPredictor()

            input_df = pd.DataFrame({
                "Recency": [recency],
                "Frequency": [frequency],
                "Monetary": [monetary],
                "TotalQuantity": [total_quantity],
                "UniqueProducts": [unique_products]
            })

            scaled = predictor.scaler.transform(input_df)
            cluster = predictor.kmeans.predict(scaled)[0]

        segment_map = {
            0: "High Value Loyal",
            1: "Loyal Customers",
            2: "At Risk Customers",
            3: "New Customers"
        }

        segment = segment_map.get(cluster, "Unidentified Segment")

        st.success("Customer Evaluation Complete")

        st.markdown(f"""
        ### 🎯 Customer Classification
        **Segment Identified:** {segment}
        """)

        st.markdown("## 📌 Recommended Strategic Action")

        if segment == "High Value Loyal":
            st.info("""
            ### 🌟 High Value Loyal Customers – Premium Retention Strategy

            **Profile Insight:**  
            Highest revenue contributors with strong engagement.

            **Strategic Actions:**
            - Exclusive VIP tiers
            - Early product launches
            - Personalized concierge service
            - Referral multipliers
            - Surprise loyalty bonuses

            **Business Objective:**  
            Maximize Customer Lifetime Value (CLV).
            """)

        elif segment == "Loyal Customers":
            st.info("""
            ### 🔁 Loyal Customers – Growth Acceleration Strategy

            **Profile Insight:**  
            Consistent purchasers with moderate revenue.

            **Strategic Actions:**
            - Bundle pricing
            - Cross-selling
            - Cashback campaigns
            - Subscription models

            **Business Objective:**  
            Increase average order value.
            """)

        elif segment == "At Risk Customers":
            st.warning("""
            ### ⚠️ At Risk Customers – Reactivation Strategy

            **Profile Insight:**  
            Declining engagement signals churn risk.

            **Strategic Actions:**
            - Win-back discounts
            - Personalized re-engagement emails
            - Feedback surveys
            - Remarketing ads

            **Business Objective:**  
            Reduce churn and recover revenue.
            """)

        elif segment == "New Customers":
            st.success("""
            ### 🚀 New Customers – Onboarding Strategy

            **Profile Insight:**  
            Recently acquired customers.

            **Strategic Actions:**
            - Onboarding email sequence
            - Second-purchase incentives
            - Best-seller recommendations
            - Loyalty enrollment

            **Business Objective:**  
            Convert into repeat buyers quickly.
            """)

# ============================================================
# ROUTER
# ============================================================
if not st.session_state.logged_in:
    login_page()
elif st.session_state.page == "home":
    home_page()
elif st.session_state.page == "predict":
    predict_page()
elif st.session_state.page == "simulator":
    simulator_page()
else:
    home_page()
