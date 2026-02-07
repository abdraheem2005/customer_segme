import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.io import load_customer_file
from inference.batch_predictor import BatchPredictor

# Page Configuration
st.set_page_config(
    page_title="Retail Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)

# ---------------------------------------------------------
# HELPER: Dynamic Business Labeling
# ---------------------------------------------------------
def get_segment_labels(df_results):
    """
    Dynamically assigns business labels to clusters based on their stats.
    This prevents the 'Label Switching' bug if you retrain the model.
    """
    # Calculate average metrics per cluster
    cluster_stats = df_results.groupby("Cluster").agg({
        "Monetary": "mean",
        "Recency": "mean",
        "Frequency": "mean"
    }).reset_index()

    labels = {}
    
    # 1. Loyal Customers: Highest Monetary Value
    loyal_cluster = cluster_stats.sort_values("Monetary", ascending=False).iloc[0]["Cluster"]
    labels[loyal_cluster] = "💎 High-Value Loyal"

    # 2. At-Risk Customers: Highest Recency (Longest time since last buy)
    # Exclude the one we just marked as loyal
    remaining = cluster_stats[cluster_stats["Cluster"] != loyal_cluster]
    at_risk_cluster = remaining.sort_values("Recency", ascending=False).iloc[0]["Cluster"]
    labels[at_risk_cluster] = "⚠️ At-Risk / Churning"

    # 3. Recent Low Spenders: Lowest Recency (Newest) among the rest
    remaining = remaining[remaining["Cluster"] != at_risk_cluster]
    recent_cluster = remaining.sort_values("Recency", ascending=True).iloc[0]["Cluster"]
    labels[recent_cluster] = "🌱 Recent Low Spenders"

    # 4. The Last Cluster: Usually "Bulk" or "Average"
    remaining = remaining[remaining["Cluster"] != recent_cluster]
    if not remaining.empty:
        bulk_cluster = remaining.iloc[0]["Cluster"]
        labels[bulk_cluster] = "🛒 Bulk/Average Buyers"
    
    return labels

# ---------------------------------------------------------
# SIDEBAR: Requirements & Upload
# ---------------------------------------------------------
with st.sidebar:
    st.header("📂 Data Upload")
    
    st.info("### Required Columns")
    st.markdown("""
    Your CSV/Excel **MUST** contain:
    - **`InvoiceNo`** (Text): Invoice ID
    - **`StockCode`** (Text): Product ID
    - **`Description`** (Text): Product Name
    - **`Quantity`** (Number): Units sold
    - **`InvoiceDate`** (Date): Transaction time
    - **`UnitPrice`** (Number): Cost per unit
    - **`CustomerID`** (Number): Unique Customer ID
    """)

    uploaded_file = st.file_uploader(
        "Upload Transaction History", 
        type=["csv", "xlsx", "xls"]
    )

# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
st.title("🛍️ AI Customer Segmentation System")
st.markdown("Identify your **Loyal**, **At-Risk**, and **New** customers instantly using K-Means Clustering.")

if uploaded_file:
    try:
        # 1. Load & Preview
        raw_df = load_customer_file(uploaded_file)
        
        st.write(f"**Data Loaded:** {raw_df.shape[0]:,} rows")
        with st.expander("🔍 Preview Raw Data"):
            st.dataframe(raw_df.head())

        # 2. Run Inference
        if st.button("🚀 Analyze Segments", type="primary"):
            with st.spinner("Cleaning data, engineering features, and predicting segments..."):
                predictor = BatchPredictor()
                results_df = predictor.predict(raw_df)

                # 3. Apply Business Labels
                segment_map = get_segment_labels(results_df)
                results_df["Segment Name"] = results_df["Cluster"].map(segment_map)
                
                # Success Message
                st.success("Analysis Complete!")

            # -----------------------------------------------------
            # RESULTS DASHBOARD
            # -----------------------------------------------------
            
            # --- TOP METRICS ---
            col1, col2, col3, col4 = st.columns(4)
            total_customers = len(results_df)
            avg_spend = results_df["Monetary"].mean()
            
            col1.metric("Total Customers", f"{total_customers:,}")
            col2.metric("Avg Spend per Customer", f"${avg_spend:,.2f}")
            col3.metric("Most Common Segment", results_df["Segment Name"].mode()[0])
            col4.metric("Segments Found", 4)

            st.divider()

            # --- TABBED VIEW ---
            tab1, tab2, tab3 = st.tabs(["📊 Business Insights", "📈 Visualizations", "📥 Export Data"])

            with tab1:
                st.subheader("Segment Characteristics")
                
                # Create a nice summary table
                summary = results_df.groupby("Segment Name").agg({
                    "CustomerID": "count",
                    "Monetary": "mean",
                    "Recency": "mean",
                    "Frequency": "mean"
                }).rename(columns={"CustomerID": "Count"}).sort_values("Monetary", ascending=False)

                # Formatting for display
                st.dataframe(summary.style.format({
                    "Monetary": "${:,.2f}",
                    "Recency": "{:.1f} days",
                    "Frequency": "{:.1f} orders"
                }).background_gradient(cmap="Greens", subset=["Monetary"]))
                
                # Definitions
                st.markdown("""
                ### 💡 Interpretation Guide
                - **💎 High-Value Loyal:** Your VIPs. They buy often and spend the most. **Action:** *Loyalty programs, early access.*
                - **🛒 Bulk/Average:** Regular shoppers with moderate spend. **Action:** *Upselling, bundles.*
                - **🌱 Recent Low Spenders:** New customers or small buyers. **Action:** *Welcome emails, coupons.*
                - **⚠️ At-Risk:** Haven't bought in a long time. **Action:** *Re-engagement campaigns, win-back offers.*
                """)

            with tab2:
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    st.subheader("Customer Value vs. Frequency")
                    fig = px.scatter(
                        results_df, 
                        x="Monetary", 
                        y="Frequency", 
                        color="Segment Name",
                        log_x=True, # Log scale helps view data better
                        log_y=True,
                        title="Who are your VIPs?",
                        hover_data=["CustomerID"]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col_viz2:
                    st.subheader("Recency Analysis")
                    fig2 = px.box(
                        results_df, 
                        x="Segment Name", 
                        y="Recency", 
                        color="Segment Name",
                        title="How recently did they buy?"
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            with tab3:
                st.subheader("Download Results")
                
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Full Report (CSV)",
                    data=csv,
                    file_name="segmented_customers.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.warning("Please check your file format against the requirements in the sidebar.")

else:
    # Empty State with illustration
    st.info("👋 Welcome! Please upload your transaction file to the left to begin.")