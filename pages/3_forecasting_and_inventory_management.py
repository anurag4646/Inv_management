import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Pizza's Dashboard", layout="wide", initial_sidebar_state="expanded")

# Application title and intro
st.title("🍕 Pizza's Sales & Inventory Forecast Dashboard")
st.markdown("Advanced inventory management system with AI forecasting and vendor integration")

# Connect to SQLite DB
@st.cache_resource
def get_data():
    conn = sqlite3.connect("pizza_data.db")
    sales_df = pd.read_sql("SELECT * FROM sales", conn)
    ing_df = pd.read_sql("SELECT * FROM ingredient", conn)
    
    ing_df.columns = ['pizza_name_id', 'pizza_name', 'pizza_ing', 'Items_Qty_In_Grams']
    
    # Preprocess data
    sales_df['order_date'] = pd.to_datetime(sales_df['order_date'], format='%d-%m-%Y')
    sales_df = sales_df.sort_values('order_date')
    
    return sales_df, ing_df, conn

sales_df, ing_df, conn = get_data()

# Filter train data (up to Nov)
sales_train = sales_df[
    (sales_df['order_date'] >= '2015-11-01') & 
    (sales_df['order_date'] < '2015-12-01')
]
forecast_start_date = '2015-12-01'
forecast_days = 31
forecast_dates = pd.date_range(start=forecast_start_date, periods=forecast_days)

# Add sidebar for controls (removed ARIMA parameters)
with st.sidebar:
    st.header("Dashboard Controls")
    forecast_view = st.radio("Forecast View", ["Daily", "Weekly", "Monthly"])
    
    # Ingredient shelf life settings
    st.subheader("Ingredient Settings")
    st.markdown("Set shelf life categories for ingredients")
    shelf_life_short = st.slider("Short shelf life (days)", 1, 7, 3)
    shelf_life_medium = st.slider("Medium shelf life (days)", 8, 15, 10)
    
    # Vendor notification settings
    st.subheader("Vendor Communication")
    enable_vendor_notifications = st.checkbox("Enable Vendor Notifications", value=False)
    if enable_vendor_notifications:
        email_subject = st.text_input("Email Subject", "Inventory Reorder Request")
        notification_days = st.slider("Send notifications when stock is below (days)", 1, 14, 5)

# ------ FORECASTING FUNCTIONS -------

@st.cache_data
def recommend_arima_params(series):
    """Find optimal ARIMA parameters based on ACF and PACF"""
    # Test for stationarity
    result = adfuller(series)
    p_value = result[1]
    
    # Determine d parameter based on stationarity
    d = 0
    if p_value > 0.05:  # Not stationary
        # Try first difference
        diff1 = series.diff().dropna()
        result_diff1 = adfuller(diff1)
        if result_diff1[1] <= 0.05:
            d = 1
        else:
            # Try second difference
            diff2 = diff1.diff().dropna()
            if adfuller(diff2)[1] <= 0.05:
                d = 2
            else:
                d = 1  # Default to 1 if all else fails
    
    # Calculate ACF and PACF
    diff_series = series.diff(d).dropna() if d > 0 else series
    
    try:
        acf_values = acf(diff_series, nlags=10)
        pacf_values = pacf(diff_series, nlags=10, method='ols')
        
        # Determine p based on PACF (AR component)
        # Looking for the point where PACF cuts off
        pacf_cutoff = 1.96 / np.sqrt(len(diff_series))
        p_values = np.where(np.abs(pacf_values) > pacf_cutoff)[0]
        p = max(p_values[-1] if len(p_values) > 0 else 0, 5)  # Cap at 5
        
        # Determine q based on ACF (MA component)
        # Looking for the point where ACF cuts off
        acf_cutoff = 1.96 / np.sqrt(len(diff_series))
        q_values = np.where(np.abs(acf_values) > acf_cutoff)[0]
        q = max(q_values[-1] if len(q_values) > 0 else 0, 5)  # Cap at 5
    except:
        # Fallback values if something goes wrong
        p, q = 1, 1
    
    return min(p, 5), d, min(q, 5)  # Ensure we don't exceed reasonable bounds

@st.cache_data
def evaluate_arima_models(series, p_values, d_values, q_values, max_models=5):
    """Evaluate multiple ARIMA models and find the best one"""
    best_models = []
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(series, order=(p, d, q))
                    model_fit = model.fit()
                    aic = model_fit.aic
                    best_models.append((p, d, q, aic))
                except:
                    continue
    
    # Sort by AIC (lower is better)
    best_models.sort(key=lambda x: x[3])
    
    # Return the top models
    return best_models[:max_models]

@st.cache_data
def forecast_sales(series, days):
    """Forecast sales with automatically determined optimal ARIMA parameters"""
    # Find optimal parameters
    suggested_p, suggested_d, suggested_q = recommend_arima_params(series)
    
    # Try different combinations around the suggested values
    p_values = [max(0, suggested_p-1), suggested_p, min(5, suggested_p+1)]
    d_values = [max(0, suggested_d-1), suggested_d, min(2, suggested_d+1)]
    q_values = [max(0, suggested_q-1), suggested_q, min(5, suggested_q+1)]
    
    # Evaluate models
    best_models = evaluate_arima_models(series, p_values, d_values, q_values)
    
    if best_models:
        best_p, best_d, best_q = best_models[0][:3]  # Get the best model parameters
    else:
        best_p, best_d, best_q = 1, 1, 1  # Default fallback
    
    # Display ARIMA parameters
    st.info(f"Using optimal ARIMA parameters: p={best_p}, d={best_d}, q={best_q}")
    
    model = ARIMA(series, order=(best_p, best_d, best_q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=days)
    
    return forecast, (best_p, best_d, best_q)

# ------ DATA PREPARATION -------

# Group total sales by date
daily_sales = sales_train.groupby('order_date')['quantity'].sum().reset_index()
daily_sales = daily_sales.set_index('order_date')

# Group by pizza type and size for specific forecasts
type_size_sales = sales_train.groupby(['order_date', 'pizza_name', 'pizza_size'])['quantity'].sum().reset_index()

# ------ OVERALL SALES FORECAST -------

st.header("📊 Sales Forecasts")
tab1, tab2, tab3 = st.tabs(["Total Sales", "By Pizza Type", "By Size"])

with tab1:
    # Create and display overall sales forecast
    with st.spinner("Generating total sales forecast..."):
        # Generate forecast with optimal parameters
        forecast, best_arima_params = forecast_sales(daily_sales['quantity'], forecast_days)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'order_date': forecast_dates,
            'forecast_quantity': np.round(forecast.values).astype(int)
        })
        
        # Add additional time period aggregations
        forecast_df['week'] = forecast_df['order_date'].dt.isocalendar().week
        forecast_df['month'] = forecast_df['order_date'].dt.month
        
        weekly_forecast = forecast_df.groupby('week')['forecast_quantity'].sum().reset_index()
        weekly_forecast['period'] = weekly_forecast['week'].apply(lambda x: f"Week {x}")
        
        monthly_forecast = forecast_df.groupby('month')['forecast_quantity'].sum().reset_index()
        monthly_forecast['period'] = monthly_forecast['month'].apply(lambda x: f"Month {x}")
        
        # Show forecast based on selected view
        if forecast_view == "Daily":
            fig = px.line(forecast_df, x='order_date', y='forecast_quantity', 
                         title="Daily Pizza Sales Forecast")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(forecast_df[['order_date', 'forecast_quantity']])
        
        elif forecast_view == "Weekly":
            fig = px.bar(weekly_forecast, x='period', y='forecast_quantity',
                        title="Weekly Pizza Sales Forecast")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(weekly_forecast[['period', 'forecast_quantity']])
        
        else:  # Monthly
            fig = px.bar(monthly_forecast, x='period', y='forecast_quantity',
                        title="Monthly Pizza Sales Forecast")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(monthly_forecast[['period', 'forecast_quantity']])

with tab2:
    # Forecast by pizza type
    st.subheader("Pizza Type Forecasts")
    
    # Get unique pizza types
    pizza_types = sales_train['pizza_name'].unique()
    selected_pizza = st.selectbox("Select Pizza Type", pizza_types)
    
    # Filter data for selected pizza
    pizza_data = sales_train[sales_train['pizza_name'] == selected_pizza]
    pizza_daily = pizza_data.groupby('order_date')['quantity'].sum()
    
    if len(pizza_daily) > 10:  # Ensure enough data points
        with st.spinner(f"Generating forecast for {selected_pizza}..."):
            # Generate forecast with best parameters
            pizza_forecast, pizza_params = forecast_sales(pizza_daily, forecast_days)
            
            # Create forecast dataframe
            pizza_forecast_df = pd.DataFrame({
                'order_date': forecast_dates,
                'forecast_quantity': np.round(pizza_forecast.values).astype(int)
            })
            
            fig = px.line(pizza_forecast_df, x='order_date', y='forecast_quantity',
                         title=f"{selected_pizza} Sales Forecast")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Not enough historical data for {selected_pizza} to create a reliable forecast")

with tab3:
    # Forecast by pizza size
    st.subheader("Pizza Size Forecasts")
    
    # Get unique pizza sizes
    pizza_sizes = sales_train['pizza_size'].unique()
    
    # Create side-by-side charts for each size
    size_cols = st.columns(len(pizza_sizes))
    
    for i, size in enumerate(pizza_sizes):
        with size_cols[i]:
            size_data = sales_train[sales_train['pizza_size'] == size]
            size_daily = size_data.groupby('order_date')['quantity'].sum()
            
            if len(size_daily) > 10:
                size_forecast, size_params = forecast_sales(size_daily, forecast_days)
                
                size_forecast_df = pd.DataFrame({
                    'order_date': forecast_dates,
                    'forecast_quantity': np.round(size_forecast.values).astype(int)
                })
                
                fig = px.area(size_forecast_df, x='order_date', y='forecast_quantity',
                             title=f"{size} Size Forecast")
                st.plotly_chart(fig, use_container_width=True)
                st.metric("Avg Daily", round(size_forecast_df['forecast_quantity'].mean()))
            else:
                st.warning(f"Not enough data for {size} size")

# ------ INVENTORY MANAGEMENT -------

st.header("📦 Inventory Management")

# Calculate ingredient usage from sales
inv_df = pd.merge(
    sales_train[["pizza_name_id", "order_date", "pizza_name", "pizza_size", "quantity"]],
    ing_df[["pizza_name_id", "pizza_ing", "Items_Qty_In_Grams"]],
    on="pizza_name_id", how="left"
)

# Apply size multiplier (assuming different amounts for different sizes)
size_multiplier = {'S': 0.7, 'M': 1.0, 'L': 1.3, 'XL': 1.6}
inv_df['size_factor'] = inv_df['pizza_size'].map(size_multiplier)
inv_df['adjusted_grams'] = inv_df['Items_Qty_In_Grams'] * inv_df['size_factor']

# Calculate total usage in grams
inv_df['total_grams'] = inv_df['quantity'] * inv_df['adjusted_grams']
used_inv = inv_df.groupby(["order_date", "pizza_ing"])['total_grams'].sum().reset_index()

# Calculate average daily usage by ingredient
avg_daily_usage = used_inv.groupby("pizza_ing")['total_grams'].sum().reset_index()
total_days = (sales_train['order_date'].max() - sales_train['order_date'].min()).days + 1
avg_daily_usage['daily_avg'] = (avg_daily_usage['total_grams'] / total_days).round()

# Define shelf life for ingredients (in days)
# This would typically come from a database, but we'll simulate it
np.random.seed(42)
shelf_lives = ['Short', 'Medium', 'Long']
probabilities = [0.3, 0.4, 0.3]  # 30% short, 40% medium, 30% long shelf life

ingredient_shelf_life = pd.DataFrame({
    'pizza_ing': avg_daily_usage['pizza_ing'],
    'shelf_life_category': np.random.choice(shelf_lives, size=len(avg_daily_usage), p=probabilities)
})

# Map shelf life categories to actual days
shelf_life_days = {
    'Short': shelf_life_short,
    'Medium': shelf_life_medium,
    'Long': 30  # Fixed 30 days for long shelf life items
}
ingredient_shelf_life['shelf_life_days'] = ingredient_shelf_life['shelf_life_category'].map(shelf_life_days)

# Simulate current stock levels
avg_daily_usage['min_stock_days'] = ingredient_shelf_life['shelf_life_days']
avg_daily_usage['min_stock'] = avg_daily_usage['daily_avg'] * avg_daily_usage['min_stock_days']

# Add some randomness to current stock levels
np.random.seed(100)
avg_daily_usage['stock'] = (
    avg_daily_usage['min_stock'] + 
    np.random.randint(-2, 5, size=len(avg_daily_usage)) * avg_daily_usage['daily_avg']
).clip(lower=0)  # Ensure no negative stock

# Forecast inventory requirement based on forecasted sales
# We'll use the ratio of each ingredient's usage to total pizzas sold
total_pizzas = sales_train['quantity'].sum()
ingredient_usage_ratio = {}

for ing in avg_daily_usage['pizza_ing']:
    ing_total = inv_df[inv_df['pizza_ing'] == ing]['total_grams'].sum()
    ingredient_usage_ratio[ing] = ing_total / total_pizzas

# Apply these ratios to the forecasted sales
forecast_grams = []
for i in range(forecast_days):
    day_total_pizzas = forecast_df.iloc[i]['forecast_quantity']
    for ing in avg_daily_usage['pizza_ing']:
        usage = day_total_pizzas * ingredient_usage_ratio[ing]
        forecast_grams.append({
            'order_date': forecast_dates[i],
            'pizza_ing': ing,
            'forecast_grams': round(usage)
        })
        
forecast_inv_df = pd.DataFrame(forecast_grams)

# Merge with stock and shelf life information
inv_alert = avg_daily_usage[['pizza_ing', 'daily_avg', 'stock', 'min_stock', 'min_stock_days']]
shelf_life_info = ingredient_shelf_life[['pizza_ing', 'shelf_life_category', 'shelf_life_days']]
inventory_projection = forecast_inv_df.merge(inv_alert, on='pizza_ing').merge(shelf_life_info, on='pizza_ing')

# Calculate cumulative usage and stock levels over time
inventory_projection['day_num'] = inventory_projection.groupby('pizza_ing')['order_date'].cumcount() + 1
inventory_projection['cumulative_usage'] = inventory_projection.groupby('pizza_ing')['forecast_grams'].cumsum()
inventory_projection['stock_left'] = inventory_projection['stock'] - inventory_projection['cumulative_usage']

# Detect when stock will run out
def find_stockout(df):
    if (df['stock_left'] <= 0).any():
        return df[df['stock_left'] <= 0].iloc[0]['order_date']
    else:
        return None

stockout_alerts = inventory_projection.groupby('pizza_ing').apply(find_stockout).reset_index()
stockout_alerts.columns = ['pizza_ing', 'stockout_date']

# Calculate days until stockout
stockout_alerts['days_until_stockout'] = (stockout_alerts['stockout_date'] - pd.to_datetime(forecast_start_date)).dt.days
stockout_alerts['days_until_stockout'] = stockout_alerts['days_until_stockout'].fillna(999)  # No stockout

# Create vendor mapping for ingredients
vendors = ["FreshProduce Inc.", "Cheese Masters", "Meat Supply Co.", "Bakery Solutions", "Global Ingredients"]
np.random.seed(101)
ingredient_vendors = pd.DataFrame({
    'pizza_ing': avg_daily_usage['pizza_ing'],
    'vendor': np.random.choice(vendors, size=len(avg_daily_usage)),
    'vendor_email': 'orders@example.com'  # In a real app, you'd have actual emails
})

# Merge vendor information
inventory_status = (
    avg_daily_usage
    .merge(stockout_alerts, on='pizza_ing')
    .merge(ingredient_shelf_life, on='pizza_ing')
    .merge(ingredient_vendors, on='pizza_ing')
)

# Determine status and urgency
inventory_status['status'] = 'OK'
inventory_status.loc[inventory_status['days_until_stockout'] <= inventory_status['shelf_life_days'], 'status'] = 'ORDER SOON'
inventory_status.loc[inventory_status['days_until_stockout'] <= 2, 'status'] = 'URGENT'
inventory_status.loc[inventory_status['days_until_stockout'] == 0, 'status'] = 'OUT OF STOCK'

# Display inventory status
ingredient_tab1, ingredient_tab2 = st.tabs(["Inventory Status", "Stock Projection"])

with ingredient_tab1:
    # Color code the status
    def highlight_status(val):
        if val == 'OUT OF STOCK':
            return 'background-color: red; color: white'
        elif val == 'URGENT':
            return 'background-color: orange; color: black'
        elif val == 'ORDER SOON':
            return 'background-color: yellow; color: black'
        else:
            return 'background-color: green; color: white'
    
    # Format for display
    display_inventory = inventory_status.copy()
    display_inventory['stockout_date'] = display_inventory['stockout_date'].fillna('No stockout predicted')
    display_inventory['days_until_stockout'] = display_inventory['days_until_stockout'].replace(999, 'N/A')
    
    st.subheader("Ingredient Inventory Status")
    st.dataframe(
        display_inventory[[
            'pizza_ing', 'shelf_life_category', 'stock', 'daily_avg', 
            'stockout_date', 'days_until_stockout', 'vendor', 'status'
        ]].style.applymap(highlight_status, subset=['status'])
    )

    # Vendor order recommendations
    st.subheader("📋 Vendor Order Recommendations")
    
    # Group by vendor
    vendor_orders = display_inventory[display_inventory['status'] != 'OK'].groupby('vendor')
    
    for vendor, items in vendor_orders:
        with st.expander(f"{vendor} - {len(items)} items need reordering"):
            st.dataframe(items[['pizza_ing', 'stock', 'daily_avg', 'days_until_stockout', 'status']])
            
            if enable_vendor_notifications:
                order_items = items[items['days_until_stockout'] <= notification_days]
                if not order_items.empty:
                    order_message = f"Order request for: {', '.join(order_items['pizza_ing'].tolist())}"
                    st.text_area(f"Message to {vendor}", order_message, height=100)
                    
                    if st.button(f"Send Order to {vendor}"):
                        st.success(f"Order request sent to {vendor}!")
                        
                        # In a real application, you would integrate with email API here
                        # Example code (commented out as it won't work without actual credentials):
                        '''
                        def send_email(recipient, subject, message):
                            sender = "your-email@example.com"
                            msg = MIMEMultipart()
                            msg['From'] = sender
                            msg['To'] = recipient
                            msg['Subject'] = subject
                            msg.attach(MIMEText(message, 'plain'))
                            
                            try:
                                server = smtplib.SMTP('smtp.example.com', 587)
                                server.starttls()
                                server.login("username", "password")
                                server.send_message(msg)
                                server.quit()
                                return True
                            except Exception as e:
                                print(f"Error sending email: {e}")
                                return False
                        
                        send_email(
                            items['vendor_email'].iloc[0],
                            email_subject,
                            order_message
                        )
                        '''

with ingredient_tab2:
    # Show stock projection charts
    st.subheader("Stock Level Projections")
    
    # Create categories based on urgency
    urgent_ingredients = inventory_status[inventory_status['status'] == 'URGENT']['pizza_ing'].tolist()
    order_soon_ingredients = inventory_status[inventory_status['status'] == 'ORDER SOON']['pizza_ing'].tolist()
    
    # Select ingredients to display
    display_ingredients = st.multiselect(
        "Select ingredients to display",
        options=inventory_status['pizza_ing'].tolist(),
        default=urgent_ingredients[:3] + order_soon_ingredients[:2] if urgent_ingredients or order_soon_ingredients else inventory_status['pizza_ing'].tolist()[:5]
    )
    
    if display_ingredients:
        # Filter projection data
        chart_data = inventory_projection[inventory_projection['pizza_ing'].isin(display_ingredients)]
        
        # Create chart
        fig = px.line(
            chart_data, 
            x='order_date', 
            y='stock_left', 
            color='pizza_ing',
            title="Projected Stock Levels"
        )
        
        # Add threshold line at zero
        fig.add_shape(
            type="line",
            x0=chart_data['order_date'].min(),
            y0=0,
            x1=chart_data['order_date'].max(),
            y1=0,
            line=dict(color="Red", width=2, dash="dash"),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one ingredient to display its projection.")

# ------ PACKAGING FORECAST -------

st.header("📦 Packaging Requirements")

box_sizes = sales_train['pizza_size'].unique()
box_forecast = []

for size in box_sizes:
    # Filter sales for this size
    size_sales = sales_train[sales_train['pizza_size'] == size]
    size_daily = size_sales.groupby('order_date')['quantity'].sum()
    
    if len(size_daily) > 10:
        size_forecast_raw, _ = forecast_sales(size_daily, forecast_days)
        
        # Convert forecast to numpy array for indexing safety
        size_forecast = np.array(size_forecast_raw)

        for i, date in enumerate(forecast_dates):
            if i < len(size_forecast):  # Ensure index exists
                forecast_qty = round(size_forecast[i])
                box_qty = round(forecast_qty * 1.2)

                box_forecast.append({
                    'order_date': date,
                    'box_size': size,
                    'forecast_pizzas': forecast_qty,
                    'required_boxes': box_qty,
                    'buffer': box_qty - forecast_qty
                })

# Convert to DataFrame
box_forecast_df = pd.DataFrame(box_forecast)

# Display box forecast
if not box_forecast_df.empty:
    # Group by date and size
    daily_boxes = box_forecast_df.pivot(
        index='order_date',
        columns='box_size',
        values='required_boxes'
    ).reset_index()

    # Create stacked bar chart
    fig = go.Figure()
    for size in box_sizes:
        if size in daily_boxes.columns:
            fig.add_trace(go.Bar(
                x=daily_boxes['order_date'],
                y=daily_boxes[size],
                name=f"{size} Size Boxes"
            ))

    fig.update_layout(
        title="Daily Box Requirements (Including 20% Buffer)",
        barmode='stack',
        xaxis_title="Date",
        yaxis_title="Number of Boxes"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    st.subheader("Box Requirements Summary")
    box_summary = box_forecast_df.groupby('box_size').agg({
        'required_boxes': ['sum', 'mean', 'max'],
        'buffer': 'sum'
    }).reset_index()

    box_summary.columns = ['Box Size', 'Total Boxes', 'Daily Average', 'Max Daily', 'Total Buffer']
    box_summary = box_summary.round(0).astype({
        'Total Boxes': int,
        'Daily Average': int,
        'Max Daily': int,
        'Total Buffer': int
    })

    st.dataframe(box_summary)

    # Box order recommendations
    st.subheader("Box Order Recommendations")

    # Simulated current stock
    np.random.seed(200)
    box_stock = pd.DataFrame({
        'Box Size': box_sizes,
        'Current Stock': np.random.randint(100, 500, size=len(box_sizes))
    })

    box_order = box_summary.merge(box_stock, on='Box Size')
    box_order['Days of Stock'] = (box_order['Current Stock'] / box_order['Daily Average']).round(1)
    box_order['Order Status'] = 'OK'
    box_order.loc[box_order['Days of Stock'] < 7, 'Order Status'] = 'ORDER SOON'
    box_order.loc[box_order['Days of Stock'] < 3, 'Order Status'] = 'URGENT'

    def highlight_box_status(val):
        if val == 'URGENT':
            return 'background-color: red; color: white'
        elif val == 'ORDER SOON':
            return 'background-color: yellow; color: black'
        else:
            return 'background-color: green; color: white'

    st.dataframe(box_order.style.applymap(highlight_box_status, subset=['Order Status']))
else:
    st.warning("Unable to generate box forecast. Check your data.")

# ------ DASHBOARD FOOTER -------
st.markdown("---")
st.markdown("👨‍💻 Developed for Pizza Inventory Management | Last updated: April 2025")
