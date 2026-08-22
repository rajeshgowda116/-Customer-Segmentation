import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

# -----------------------------
# Load Models
# -----------------------------
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
cluster_names = joblib.load("cluster_names.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("👥 Customer Segmentation Dashboard")
st.write(
    "Analyze customer behavior and identify meaningful customer segments using K-Means clustering."
)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Customer Details")

values = {}

for feature in features:
    values[feature] = st.sidebar.number_input(
        feature.replace("_", " ").title(),
        value=0.0
    )

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("🔍 Identify Customer Segment"):

    input_data = pd.DataFrame([values])

    input_scaled = scaler.transform(input_data)

    cluster = kmeans.predict(input_scaled)[0]

    segment = cluster_names.get(cluster, f"Cluster {cluster}")

    st.success(f"### Customer Segment: {segment}")

    st.info(f"Cluster Number: {cluster}")

# -----------------------------
# Dashboard
# -----------------------------
st.subheader("📊 Customer Segment Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Number of Segments", len(cluster_names))
col2.metric("Features Used", len(features))
col3.metric("Algorithm", "K-Means")

st.divider()

# -----------------------------
# Segment Description
# -----------------------------
st.subheader("🎯 Customer Segments")

for cluster, name in cluster_names.items():
    st.write(f"**Cluster {cluster}:** {name}")