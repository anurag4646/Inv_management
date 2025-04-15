import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Pizza's Dashboard", layout="wide", initial_sidebar_state="expanded")

# Application title and intro
st.title("🍕 Pizza's Sales & Inventory Forecast Dashboard (1 Month Simulation)")
st.markdown("Simulating a new shop opening and tracking orders for one month")

# Initialize session state
if 'day_counter' not in st.session_state:
    st.session_state.day_counter = 0
    st.session_state.start_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    st.session_state.sales_data = pd.DataFrame(columns=['date', 'pizza_name', 'quantity', 'size'])
    st.session_state.inventory_data = pd.DataFrame({
        'ingredient': ['Dough', 'Cheese', 'Sauce', 'Pepperoni', 'Vegetables'],
        'quantity': [10000, 5000, 3000, 2000, 1500]
    })

# Sidebar for simulation controls
with st.sidebar:
    st.header("Simulation Controls")
    if st.button("Run Next Day"):
        st.session_state.day_counter += 1
    if st.button("Run Full Month"):
        st.session_state.day_counter = 30

    st.write(f"Current Day: {st.session_state.day_counter}")

# Function to generate random orders for the day
def generate_daily_orders(day):
    pizza_types = ['Margherita', 'Pepperoni', 'Vegetarian', 'Hawaiian', 'Supreme']
    sizes = ['Small', 'Medium', 'Large']
    
    # Add some randomness and a slight upward trend as the month progresses
    base_orders = np.random.randint(20, 50)
    trend = day * 0.5  # Slight upward trend
    weekend_boost = 10 if (st.session_state.start_date + timedelta(days=day-1)).weekday() >= 5 else 0
    num_orders = int(base_orders + trend + weekend_boost)
    
    orders = []
    for _ in range(num_orders):
        orders.append({
            'date': (st.session_state.start_date + timedelta(days=day-1)).date(),
            'pizza_name': np.random.choice(pizza_types),
            'quantity': np.random.randint(1, 4),
            'size': np.random.choice(sizes)
        })
    
    return pd.DataFrame(orders)

# Function to update inventory based on orders
def update_inventory(orders):
    ingredient_usage = {
        'Margherita': {'Dough': 200, 'Cheese': 100, 'Sauce': 50},
        'Pepperoni': {'Dough': 200, 'Cheese': 100, 'Sauce': 50, 'Pepperoni': 50},
        'Vegetarian': {'Dough': 200, 'Cheese': 80, 'Sauce': 50, 'Vegetables': 100},
        'Hawaiian': {'Dough': 200, 'Cheese': 100, 'Sauce': 50, 'Vegetables': 50},
        'Supreme': {'Dough': 200, 'Cheese': 120, 'Sauce': 60, 'Pepperoni': 30, 'Vegetables': 80}
    }
    
    for _, order in orders.iterrows():
        pizza = order['pizza_name']
        quantity = order['quantity']
        for ingredient, amount in ingredient_usage[pizza].items():
            st.session_state.inventory_data.loc[st.session_state.inventory_data['ingredient'] == ingredient, 'quantity'] -= amount * quantity
    
    st.session_state.inventory_data['quantity'] = st.session_state.inventory_data['quantity'].clip(lower=0)

# Main dashboard
for day in range(st.session_state.day_counter + 1, 31):
    # Generate new orders for the day
    daily_orders = generate_daily_orders(day)
    st.session_state.sales_data = pd.concat([st.session_state.sales_data, daily_orders], ignore_index=True)
    
    # Update inventory
    update_inventory(daily_orders)
    
    st.session_state.day_counter = day
    
    if st.session_state.day_counter == 30:
        break

if st.session_state.day_counter > 0:
    # Display cumulative sales data
    st.header("Cumulative Sales Data")
    st.dataframe(st.session_state.sales_data)
    
    # Sales trend
    st.header("Sales Trend")
    daily_sales = st.session_state.sales_data.groupby('date')['quantity'].sum().reset_index()
    fig = px.line(daily_sales, x='date', y='quantity', title="Daily Sales Trend")
    st.plotly_chart(fig, use_container_width=True)
    
    # Pizza type popularity
    st.header("Pizza Type Popularity")
    pizza_popularity = st.session_state.sales_data['pizza_name'].value_counts()
    fig = px.pie(values=pizza_popularity.values, names=pizza_popularity.index, title="Pizza Type Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Simple sales forecast for next week
    st.header("Sales Forecast (Next 7 Days)")
    avg_daily_sales = daily_sales['quantity'].mean()
    last_date = daily_sales['date'].max()
    forecast_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    forecast_sales = [avg_daily_sales * (1 + i*0.01) for i in range(7)]  # Assuming slight growth
    
    fig = px.line(x=forecast_dates, y=forecast_sales, labels={'x': 'Date', 'y': 'Forecasted Sales'})
    fig.update_layout(title="7-Day Sales Forecast")
    st.plotly_chart(fig, use_container_width=True)
    
    # Inventory status
    st.header("Current Inventory Status")
    st.dataframe(st.session_state.inventory_data)
    
    # Inventory forecast
    st.header("Inventory Forecast")
    avg_daily_usage = st.session_state.inventory_data['quantity'] / st.session_state.day_counter
    days_until_reorder = st.session_state.inventory_data['quantity'] / avg_daily_usage
    
    fig = px.bar(x=st.session_state.inventory_data['ingredient'], y=days_until_reorder, 
                 labels={'x': 'Ingredient', 'y': 'Days until reorder needed'})
    fig.update_layout(title="Days Until Reorder Needed")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Welcome to your new pizza shop! Click 'Run Next Day' to simulate one day or 'Run Full Month' to simulate the entire month.")

# Footer
st.markdown("---")
st.markdown("👨‍💻 Developed for Pizza Inventory Management (1 Month Simulation) | Last updated: April 2025")