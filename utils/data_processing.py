# utils/data_processing.py
import sqlite3
import pandas as pd
import streamlit as st

@st.cache_data
def load_data_from_db():
    conn = sqlite3.connect("pizza_data.db")
    sales_df = pd.read_sql("SELECT * FROM sales", conn)
    ingredient_df = pd.read_sql("SELECT * FROM ingredient", conn)
    conn.close()
    return sales_df, ingredient_df

def check_data_available(conn):
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM sales")
        sales_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM ingredient")
        ingredient_count = cursor.fetchone()[0]

        return sales_count > 0 and ingredient_count > 0
    except sqlite3.OperationalError:
        return False
