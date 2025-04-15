import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_processing import load_data_from_db

st.set_page_config(page_title="Sales Overview", layout="wide")
st.title("📈 Sales Overview")

# Load data
sales_df, ing = load_data_from_db()

# Convert date column to datetime
sales_df['order_date'] = pd.to_datetime(sales_df['order_date'], format='%d-%m-%Y')

# Create reusable month filter options
sales_df['month_name'] = sales_df['order_date'].dt.strftime('%b')
month_options = ["Overall"] + sorted(sales_df['month_name'].unique(), key=lambda m: pd.to_datetime(m, format='%b').month)

# --- Overall Historical Sales Trend ---
st.subheader("📊 Daily Sales Trend")
selected_month_trend = st.selectbox("Select Month:", options=month_options, key="trend_month")

if selected_month_trend != "Overall":
    df_trend = sales_df[sales_df['month_name'] == selected_month_trend]
else:
    df_trend = sales_df

daily_sales = df_trend.groupby('order_date')['quantity'].sum().reset_index()
fig = px.line(daily_sales, x='order_date', y='quantity', title=f'Daily Sales - {selected_month_trend}')
st.plotly_chart(fig, use_container_width=True)


# --- Monthly Historical Sales Trend ---
st.subheader("📊 Monthly Sales Trend")
# No filter here; it’s always monthly
sales_df['month'] = sales_df['order_date'].dt.to_period('M')
monthly_sales = sales_df.groupby('month')['quantity'].sum().reset_index()
monthly_sales['month'] = monthly_sales['month'].dt.to_timestamp()
fig = px.line(monthly_sales, x='month', y='quantity', title='Monthly Sales')
st.plotly_chart(fig, use_container_width=True)


# --- Weekly Sales Trend ---
st.subheader("🗓️ Sales by Day of the Week")
selected_month_week = st.selectbox("Select Month:", options=month_options, key="week_month")

if selected_month_week != "Overall":
    df_week = sales_df[sales_df['month_name'] == selected_month_week]
else:
    df_week = sales_df

df_week['weekday_name'] = df_week['order_date'].dt.day_name()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_sales = df_week.groupby('weekday_name')['quantity'].sum().reset_index()
weekday_sales['weekday_name'] = pd.Categorical(weekday_sales['weekday_name'], categories=weekday_order, ordered=True)
weekday_sales = weekday_sales.sort_values('weekday_name')

fig = px.bar(weekday_sales, x='weekday_name', y='quantity', title=f'Sales by Weekday - {selected_month_week}')
st.plotly_chart(fig, use_container_width=True)


# --- Top 10 Pizza Sold ---
st.subheader("🍕 Top 10 Pizzas Sold")
selected_month_top = st.selectbox("Select Month:", options=month_options, key="top10_month")

if selected_month_top != "Overall":
    df_top = sales_df[sales_df['month_name'] == selected_month_top]
else:
    df_top = sales_df

top_items = df_top.groupby('pizza_name')['quantity'].sum().sort_values(ascending=False).head(10).reset_index()
fig = px.bar(
    top_items,
    x='pizza_name',
    y='quantity',
    title=f'Top 10 Pizzas - {selected_month_top}',
    labels={'pizza_name': 'Pizza Name', 'quantity': 'Total Sold'}
)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

st.title("🍕 Ingredient Requirements")

# Merge ingredient info with pizza metadata
ingredient_df = pd.merge(
    ing,
    sales_df[["pizza_name_id", "pizza_size", "pizza_category"]],
    on='pizza_name_id',
    how='left'
).drop_duplicates(subset=["pizza_name", "pizza_ingredients", "pizza_size"]).reset_index(drop=True)

# --- Filters (inline above chart) ---
st.markdown("### 🔍 Select Pizza and Size")

col1, col2 = st.columns([2, 1])

with col1:
    pizza_names = ingredient_df['pizza_name'].unique()
    selected_pizza = st.selectbox("Choose Pizza", pizza_names)

filtered_by_pizza = ingredient_df[ingredient_df['pizza_name'] == selected_pizza]

with col2:
    pizza_sizes = filtered_by_pizza['pizza_size'].unique()
    selected_size = st.selectbox("Choose Size", pizza_sizes)

# Final filter by pizza and size
final_df = filtered_by_pizza[filtered_by_pizza['pizza_size'] == selected_size]

# --- Chart ---
st.subheader(f"🧂 Ingredient Quantity Breakdown: **{selected_pizza}** ({selected_size})")

fig = px.pie(
    final_df,
    names='pizza_ingredients',
    values='Items_Qty_In_Grams',
    title='Ingredient Quantity (in grams)',
    hole=0.4  # donut chart style
)
st.plotly_chart(fig, use_container_width=True)

# Optional table
with st.expander("📋 View Data Table"):
    st.dataframe(final_df[['pizza_ingredients', 'Items_Qty_In_Grams']], use_container_width=True)

st.title("🍕 Last 2 Months Sales Comparison")
# Filter data for the last two months

sales_df['month'] = sales_df['order_date'].dt.month
sales_df['month_name'] = sales_df['order_date'].dt.strftime('%B')
sales_df['day'] = sales_df['order_date'].dt.day

# Filter for November and December only
sales_df = sales_df[sales_df['month'].isin([11, 12])]

# Dropdown filter for pizza category (above chart)
pizza_categories = ['All'] + sorted(sales_df['pizza_category'].dropna().unique())
selected_category = st.selectbox("🍕 Select Pizza Category", pizza_categories)

# Apply filter
if selected_category != 'All':
    sales_df = sales_df[sales_df['pizza_category'] == selected_category]

# Group by month, day
grouped = sales_df.groupby(['month_name', 'day'])['quantity'].sum().reset_index()

# Plot line chart
fig = px.line(
    grouped,
    x='day',
    y='quantity',
    color='month_name',
    markers=True,
    title=f"📈 Daily Sales Trend: November vs December ({selected_category})",
    labels={'day': 'Day of Month', 'quantity': 'Total Pizzas Sold', 'month_name': 'Month'}
)

fig.update_layout(xaxis=dict(dtick=1))

st.plotly_chart(fig, use_container_width=True)
