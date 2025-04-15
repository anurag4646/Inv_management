import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.data_processing import load_data_from_db

st.set_page_config(page_title="Order Behavior", layout="wide")
st.title("🍕 Pizza Order Behavior Analysis")

sales_df, ing = load_data_from_db()

sales_df['order_date'] = pd.to_datetime(sales_df['order_date'], format='%d-%m-%Y')
sales_df['order_time'] = pd.to_datetime(sales_df['order_time'], format='%H:%M:%S').dt.time
sales_df['hour'] = pd.to_datetime(sales_df['order_time'], format='%H:%M:%S').dt.hour

# 1. Hourly Order Distribution
st.subheader("Hourly - Grouped Order Distribution")

# Create two columns for side-by-side charts
col1, col2 = st.columns(2)

with col1:
    st.write("Hourly Breakdown")
    hourly_orders = sales_df['hour'].value_counts().sort_index()
    fig_hourly = px.bar(x=hourly_orders.index, y=hourly_orders.values, 
                        labels={'x': 'Hour of Day', 'y': 'Number of Orders'},
                        title="Orders by Hour of Day")
    st.plotly_chart(fig_hourly, use_container_width=True)

with col2:
    st.write("Time of Day Breakdown")
    # Define time periods
    def get_time_period(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    sales_df['time_period'] = sales_df['hour'].apply(get_time_period)
    time_period_orders = sales_df['time_period'].value_counts()

    # Define custom order for time periods
    custom_order = ['Morning', 'Afternoon', 'Evening', 'Night']
    time_period_orders = time_period_orders.reindex(custom_order)

    fig_time_period = px.pie(values=time_period_orders.values, names=time_period_orders.index, 
                             title='Orders by Time of Day',
                             color=time_period_orders.index,
                             color_discrete_map={'Morning':'gold', 
                                                 'Afternoon':'orangered',
                                                 'Evening':'darkblue', 
                                                 'Night':'darkslategray'})
    st.plotly_chart(fig_time_period, use_container_width=True)

# Display percentages
st.write("Percentage of Orders by Time of Day:")
percentage_orders = (time_period_orders / time_period_orders.sum() * 100).round(2)
for period, percentage in percentage_orders.items():
    st.write(f"{period}: {percentage}%")

# 2. Monthly Order Trends
st.subheader("Monthly Order Trends")
monthly_orders = sales_df.groupby(sales_df['order_date'].dt.to_period("M")).size().reset_index(name='count')
monthly_orders['order_date'] = monthly_orders['order_date'].dt.to_timestamp()
fig_monthly = px.line(monthly_orders, x='order_date', y='count', labels={'order_date': 'Month', 'count': 'Number of Orders'})
st.plotly_chart(fig_monthly, use_container_width=True)

# 3. Pizza Size Popularity
st.subheader("Pizza Size Popularity")
size_popularity = sales_df['pizza_size'].value_counts()
fig_size = px.pie(values=size_popularity.values, names=size_popularity.index, title='Pizza Size Distribution')
st.plotly_chart(fig_size, use_container_width=True)

# 4. Pizza Category Analysis
st.subheader("Pizza Category Analysis")
category_popularity = sales_df['pizza_category'].value_counts()
fig_category = px.bar(x=category_popularity.index, y=category_popularity.values, labels={'x': 'Pizza Category', 'y': 'Number of Orders'})
st.plotly_chart(fig_category, use_container_width=True)

# 5. Top Selling Pizzas
st.subheader("Top 10 Selling Pizzas")
top_pizzas = sales_df['pizza_name'].value_counts().nlargest(10)
fig_top = px.bar(x=top_pizzas.index, y=top_pizzas.values, labels={'x': 'Pizza Name', 'y': 'Number of Orders'})
st.plotly_chart(fig_top, use_container_width=True)

# 6. Average Order Value Over Time
st.subheader("Average Order Value Over Time")
sales_df['date'] = sales_df['order_date'].dt.date
avg_order_value = sales_df.groupby('date')['total_price'].mean().reset_index()
fig_avg = px.line(avg_order_value, x='date', y='total_price', labels={'date': 'Date', 'total_price': 'Average Order Value'})
st.plotly_chart(fig_avg, use_container_width=True)

# 7. Order Quantity Distribution
st.subheader("Order Quantity Distribution")
fig_qty = px.histogram(sales_df, x='quantity', nbins=20, labels={'quantity': 'Quantity per Order', 'count': 'Frequency'})
st.plotly_chart(fig_qty, use_container_width=True)

# 8. Weekday vs Weekend Order Patterns
st.subheader("Weekday vs Weekend Order Patterns")
sales_df['day_of_week'] = sales_df['order_date'].dt.day_name()
day_order_counts = sales_df['day_of_week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
fig_week = px.bar(x=day_order_counts.index, y=day_order_counts.values, labels={'x': 'Day of Week', 'y': 'Number of Orders'})
st.plotly_chart(fig_week, use_container_width=True)