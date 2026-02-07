# 🛍️ AI-Powered Customer Segmentation System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **"Stop guessing who your best customers are. Start knowing."**

## 📖 Overview
This project is an end-to-end **Machine Learning application** designed to segment retail customers into distinct groups based on their purchasing behavior. By analyzing transactional data, the system identifies VIPs, At-Risk customers, and New buyers, enabling businesses to launch targeted marketing campaigns and improve retention.

Unlike simple scripts, this project follows an **Industry-Ready Modular Architecture**, separating data processing pipelines from inference logic, making it scalable and production-friendly.

---

## 🎯 Why This Project?
Marketing to "everyone" is marketing to *no one*. This tool solves the problem of generic marketing by implementing **RFM Analysis** (Recency, Frequency, Monetary) combined with Unsupervised Learning.

**Key Capabilities:**
* **Automated Data Cleaning:** Handles messy real-world retail datasets.
* **Feature Engineering:** transform raw invoices into meaningful customer profiles.
* **AI Segmentation:** Uses **K-Means Clustering** to find hidden patterns.
* **Interactive Dashboard:** A Streamlit app to visualize segments and export results.

---

## 📂 Project Structure
The project is organized into a modular structure to ensure separation of concerns (Training vs. Inference).

```text
customer_segmentation/
│
├── artifacts/              # 🤖 Saved Model Files (Created after training)
│   ├── scaler.pkl          # Feature Scaler
│   ├── kmeans.pkl          # Trained K-Means Model
│   └── feature_schema.json # JSON Schema for input validation
│
├── data/                   # 💾 Raw Data
│   └── customer_segmentation.csv  <-- PLACE YOUR DATASET HERE
│
├── inference/              # 🔮 Prediction Logic
│   ├── __init__.py
│   └── batch_predictor.py  # Loads model & generates predictions
│
├── pipeline/               # ⚙️ Training Logic
│   ├── __init__.py
│   ├── feature_engineering.py # Cleaning & RFM Calculation
│   └── train_model.py      # Script to train & save the model
│
├── utils/                  # 🛠️ Helper Utilities
│   ├── __init__.py
│   └── io.py               # Robust file loader (CSV/Excel)
│
├── app.py                  # 🚀 Main Streamlit Dashboard Application
├── requirements.txt        # 📦 Python Dependencies
└── README.md               # 📄 Documentation
```

## 💾 Dataset
This project requires a transactional dataset containing retail invoices.
**[🔗 Download the Dataset Here](https://www.kaggle.com/datasets/nileshbhamare/reatail-customer-sengmentation)**

### Required Columns
The input file (CSV or Excel) must contain:
* **`InvoiceNo`**: Unique ID for the transaction.
* **`StockCode`**: Product ID.
* **`Description`**: Product Name.
* **`Quantity`**: Number of units sold.
* **`InvoiceDate`**: Date and time of the transaction.
* **`UnitPrice`**: Price per unit.
* **`CustomerID`**: Unique ID for the customer.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/AshwinAshok3/customer-segmentation.git](https://github.com/AshwinAshok3/customer-segmentation.git)
cd customer-segmentation```




