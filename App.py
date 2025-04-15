# app.py
import streamlit as st
import pandas as pd
import sqlite3
from utils.data_processing import load_data_from_db, check_data_available

st.set_page_config(
    page_title="Pizza's Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Pizza's Predictive Purchase Order Dashboard")

DB_PATH = "pizza_data.db"
conn = sqlite3.connect(DB_PATH)

# 🔍 Check if data is already available in SQLite
if not check_data_available(conn):
    st.warning("⚠️ No data found. Please upload Sales and Ingredient CSV files to continue.")

    sales_file = st.file_uploader("Upload Sales CSV", type="csv", key="sales_upload")
    ingredient_file = st.file_uploader("Upload Ingredient CSV", type="csv", key="ingredient_upload")

    if sales_file and ingredient_file:
        sales_df = pd.read_csv(sales_file)
        ingredient_df = pd.read_csv(ingredient_file)

        sales_df.to_sql("sales", conn, if_exists="replace", index=False)
        ingredient_df.to_sql("ingredient", conn, if_exists="replace", index=False)

        st.success("✅ Data uploaded and saved successfully. Please reload the app to continue.")
        st.stop()  # Stop execution until reload

    else:
        st.stop()  # Stop here until upload is complete
else:
    # ✅ Data exists, load and show (without passing conn)
    sales_df, ingredient_df = load_data_from_db()

    st.markdown("""
    Navigate through the tabs on the left to:

    - View **Sales Overview** and identify trends.
    - Monitor **Inventory Management**.
    - Analyze **Order Behavior**.
    - Forecast future **Sales and Inventory** needs.
    """)

    st.header("📊 Sample of Sales Data")
    st.dataframe(sales_df.head())

    st.header("🍕 Sample of Ingredient Data")
    st.dataframe(ingredient_df.head())
