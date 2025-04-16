import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import base64
from io import BytesIO
import sys
import os
import json
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import random
from datetime import datetime, timedelta
import time
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pdfkit
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Set page configuration
st.set_page_config(
    page_title="Food Trading Insights Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-bottom: 1rem;
    }
    .card {
        border-radius: 5px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .opportunity-card {
        border-left: 5px solid #3498db;
    }
    .opportunity-card-seller {
        border-left: 5px solid #2ecc71;
    }
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
    .contact-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .contact-name {
        font-weight: bold;
        color: #2c3e50;
    }
    .contact-details {
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25em 0.4em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        margin-right: 5px;
    }
    .badge-primary {
        color: #fff;
        background-color: #3498db;
    }
    .badge-success {
        color: #fff;
        background-color: #2ecc71;
    }
    .badge-warning {
        color: #212529;
        background-color: #f1c40f;
    }
    .badge-danger {
        color: #fff;
        background-color: #e74c3c;
    }
    .badge-info {
        color: #fff;
        background-color: #3498db;
    }
    .badge-secondary {
        color: #fff;
        background-color: #95a5a6;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        padding: 20px;
        color: #7f8c8d;
        font-size: 0.9rem;
    }
    .btn-switch {
        margin-right: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #2ecc71;
    }
    .update-time {
        color: #7f8c8d;
        font-size: 0.8rem;
        text-align: right;
        margin-top: -15px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_type' not in st.session_state:
    st.session_state.user_type = 'buyer'

if 'selected_commodity' not in st.session_state:
    st.session_state.selected_commodity = None

if 'selected_region' not in st.session_state:
    st.session_state.selected_region = 'Global'

if 'selected_analysis' not in st.session_state:
    st.session_state.selected_analysis = 'price'

if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = datetime.now()

if 'available_commodities' not in st.session_state:
    st.session_state.available_commodities = []

if 'commodity_data' not in st.session_state:
    st.session_state.commodity_data = {}

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = {}

if 'weather_data' not in st.session_state:
    st.session_state.weather_data = {}

if 'satellite_data' not in st.session_state:
    st.session_state.satellite_data = {}

if 'trade_data' not in st.session_state:
    st.session_state.trade_data = {}

# Function to switch user type
def switch_user_type(user_type):
    st.session_state.user_type = user_type
    st.experimental_rerun()

# Function to get available commodities from Yahoo Finance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_available_commodities():
    # Agricultural commodities tickers
    commodity_tickers = {
        # Grains
        'ZW=F': 'Wheat',
        'ZC=F': 'Corn',
        'ZS=F': 'Soybeans',
        'ZM=F': 'Soybean Meal',
        'ZL=F': 'Soybean Oil',
        'ZO=F': 'Oats',
        'ZR=F': 'Rice',
        'KE=F': 'KC HRW Wheat',
        'ZG=F': 'Barley',
        
        # Softs
        'SB=F': 'Sugar',
        'CC=F': 'Cocoa',
        'KC=F': 'Coffee',
        'CT=F': 'Cotton',
        'OJ=F': 'Orange Juice',
        'LBS=F': 'Lumber',
        
        # Livestock
        'LE=F': 'Live Cattle',
        'HE=F': 'Lean Hogs',
        'GF=F': 'Feeder Cattle',
        
        # Dairy
        'DC=F': 'Class III Milk',
        'CSC=F': 'Cheese',
        'DY=F': 'Dry Whey',
        
        # Fruits and Vegetables (using ETFs and stocks as proxies)
        'FTXG': 'Fruits & Vegetables (ETF)',
        'FDP': 'Fresh Del Monte Produce',
        'DOLE': 'Dole Food Company',
        
        # Oils
        'QS=F': 'Canola',
        'CPO=F': 'Crude Palm Oil',
        
        # Additional commodities
        'JO': 'Coffee ETN',
        'SGG': 'Sugar ETN',
        'NIB': 'Cocoa ETN',
        'WEAT': 'Wheat ETF',
        'CORN': 'Corn ETF',
        'SOYB': 'Soybean ETF'
    }
    
    # Add some specific food products
    food_products = {
        'K': 'Kellogg (Cereals)',
        'GIS': 'General Mills (Packaged Foods)',
        'CPB': 'Campbell Soup',
        'KHC': 'Kraft Heinz',
        'MDLZ': 'Mondelez (Snacks)',
        'CAG': 'Conagra Brands',
        'HSY': 'Hershey (Chocolate)',
        'SJM': 'JM Smucker (Jams, Coffee)',
        'TSN': 'Tyson Foods (Meat)',
        'HRL': 'Hormel Foods'
    }
    
    # Combine all tickers
    all_tickers = {**commodity_tickers, **food_products}
    
    # Get data for each ticker to verify availability
    available_commodities = []
    
    for ticker, name in all_tickers.items():
        try:
            # Try to get recent data
            data = yf.download(ticker, period="1d", progress=False)
            if not data.empty:
                available_commodities.append({'ticker': ticker, 'name': name})
        except Exception as e:
            continue
    
    # Sort by name
    available_commodities = sorted(available_commodities, key=lambda x: x['name'])
    
    return available_commodities

# Function to get commodity price data from Yahoo Finance
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_commodity_price_data(ticker, period="2y"):
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return None
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Rename columns
        data = data.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Price',
            'Adj Close': 'Adjusted',
            'Volume': 'Volume'
        })
        
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Function to get weather data from OpenWeatherMap
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_weather_data(region):
    # Map regions to representative cities
    region_cities = {
        'Global': ['New York', 'London', 'Tokyo', 'Sydney', 'Rio de Janeiro', 'Cairo', 'Mumbai'],
        'North America': ['New York', 'Chicago', 'Los Angeles', 'Toronto', 'Mexico City'],
        'South America': ['Sao Paulo', 'Buenos Aires', 'Lima', 'Bogota', 'Santiago'],
        'Europe': ['London', 'Paris', 'Berlin', 'Rome', 'Madrid'],
        'Asia': ['Tokyo', 'Beijing', 'Mumbai', 'Bangkok', 'Seoul'],
        'Africa': ['Cairo', 'Lagos', 'Johannesburg', 'Nairobi', 'Casablanca'],
        'Middle East': ['Dubai', 'Riyadh', 'Istanbul', 'Tehran', 'Tel Aviv'],
        'Russia': ['Moscow', 'Saint Petersburg', 'Novosibirsk', 'Yekaterinburg']
    }
    
    # Get cities for the selected region
    cities = region_cities.get(region, region_cities['Global'])
    
    # OpenWeatherMap API key (in a real implementation, this would be stored securely)
    api_key = "DEMO_KEY"  # Replace with actual API key in production
    
    weather_data = []
    
    for city in cities:
        try:
            # Current weather
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            
            # In demo mode, generate simulated data instead of making actual API calls
            # In production, uncomment the following line to make real API calls
            # response = requests.get(url)
            
            # Simulated data for demo
            temp = np.random.normal(20, 10)  # Random temperature around 20°C
            humidity = np.random.normal(60, 20)  # Random humidity around 60%
            rainfall = max(0, np.random.normal(5, 10))  # Random rainfall (non-negative)
            
            weather_data.append({
                'city': city,
                'temperature': temp,
                'humidity': humidity,
                'rainfall': rainfall,
                'date': datetime.now()
            })
        except Exception as e:
            continue
    
    # Convert to DataFrame
    if weather_data:
        df = pd.DataFrame(weather_data)
        return df
    else:
        return None

# Function to get satellite data (NDVI) from NASA Earth Observations
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_satellite_data(region):
    # In a real implementation, this would connect to NASA Earth Observations API
    # For demo purposes, we'll generate simulated NDVI data
    
    # Map regions to representative coordinates
    region_coords = {
        'Global': [(40.7128, -74.0060), (51.5074, -0.1278), (35.6762, 139.6503), (-33.8688, 151.2093), (-22.9068, -43.1729), (30.0444, 31.2357), (19.0760, 72.8777)],
        'North America': [(40.7128, -74.0060), (41.8781, -87.6298), (34.0522, -118.2437), (43.6532, -79.3832), (19.4326, -99.1332)],
        'South America': [(-23.5505, -46.6333), (-34.6037, -58.3816), (-12.0464, -77.0428), (4.7110, -74.0721), (-33.4489, -70.6693)],
        'Europe': [(51.5074, -0.1278), (48.8566, 2.3522), (52.5200, 13.4050), (41.9028, 12.4964), (40.4168, -3.7038)],
        'Asia': [(35.6762, 139.6503), (39.9042, 116.4074), (19.0760, 72.8777), (13.7563, 100.5018), (37.5665, 126.9780)],
        'Africa': [(30.0444, 31.2357), (6.5244, 3.3792), (-26.2041, 28.0473), (-1.2921, 36.8219), (33.5731, -7.5898)],
        'Middle East': [(25.2048, 55.2708), (24.7136, 46.6753), (41.0082, 28.9784), (35.6892, 51.3890), (32.0853, 34.7818)],
        'Russia': [(55.7558, 37.6173), (59.9343, 30.3351), (55.0084, 82.9357), (56.8389, 60.6057)]
    }
    
    # Get coordinates for the selected region
    coords = region_coords.get(region, region_coords['Global'])
    
    satellite_data = []
    
    # Generate simulated NDVI data for each coordinate
    for lat, lon in coords:
        # NDVI values range from -1 to 1, with healthy vegetation typically 0.2 to 0.8
        ndvi = np.random.uniform(0.2, 0.8)
        
        # Generate historical data for the past 24 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # ~24 months
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')  # Monthly data
        
        # Base NDVI with seasonal pattern
        days = np.arange(len(date_range))
        seasonal = 0.2 * np.sin(2 * np.pi * days / 12)  # Yearly cycle
        
        # Add trend and noise
        trend = 0.0001 * days
        noise = np.random.normal(0, 0.05, len(date_range))
        
        # Combine components
        ndvi_values = 0.5 + seasonal + trend + noise
        ndvi_values = np.clip(ndvi_values, 0, 1)  # Ensure values are between 0 and 1
        
        for i, date in enumerate(date_range):
            satellite_data.append({
                'latitude': lat,
                'longitude': lon,
                'ndvi': ndvi_values[i],
                'date': date,
                'region': region
            })
    
    # Convert to DataFrame
    if satellite_data:
        df = pd.DataFrame(satellite_data)
        return df
    else:
        return None

# Function to get trade flow data
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_trade_flow_data(commodity, region):
    # In a real implementation, this would connect to UN Comtrade API
    # For demo purposes, we'll generate simulated trade flow data
    
    # Generate historical data for the past 24 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # ~24 months
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')  # Monthly data
    
    # Map regions to representative countries
    region_countries = {
        'Global': ['USA', 'China', 'Brazil', 'India', 'Russia', 'Germany', 'Australia'],
        'North America': ['USA', 'Canada', 'Mexico'],
        'South America': ['Brazil', 'Argentina', 'Colombia', 'Chile', 'Peru'],
        'Europe': ['Germany', 'France', 'UK', 'Italy', 'Spain'],
        'Asia': ['China', 'India', 'Japan', 'South Korea', 'Thailand'],
        'Africa': ['Egypt', 'Nigeria', 'South Africa', 'Kenya', 'Morocco'],
        'Middle East': ['UAE', 'Saudi Arabia', 'Turkey', 'Iran', 'Israel'],
        'Russia': ['Russia']
    }
    
    # Get countries for the selected region
    countries = region_countries.get(region, region_countries['Global'])
    
    trade_data = []
    
    # Base trade volume with seasonal pattern
    days = np.arange(len(date_range))
    seasonal = 0.2 * np.sin(2 * np.pi * days / 12)  # Yearly cycle
    
    for country in countries:
        # Generate export data
        base_export = np.random.uniform(1000, 10000)  # Base export volume
        trend_export = 0.001 * base_export * days  # Slight upward trend
        noise_export = np.random.normal(0, 0.1 * base_export, len(date_range))
        exports = base_export + base_export * seasonal + trend_export + noise_export
        
        # Generate import data
        base_import = np.random.uniform(1000, 10000)  # Base import volume
        trend_import = 0.001 * base_import * days  # Slight upward trend
        noise_import = np.random.normal(0, 0.1 * base_import, len(date_range))
        imports = base_import + base_import * seasonal + trend_import + noise_import
        
        for i, date in enumerate(date_range):
            trade_data.append({
                'country': country,
                'date': date,
                'exports': exports[i],
                'imports': imports[i],
                'commodity': commodity,
                'region': region
            })
    
    # Convert to DataFrame
    if trade_data:
        df = pd.DataFrame(trade_data)
        return df
    else:
        return None

# Function to analyze data and generate recommendations
def generate_recommendations(price_data, weather_data, satellite_data, trade_data, commodity, region, user_type):
    recommendations = []
    
    # Check if we have all the necessary data
    if price_data is None or weather_data is None or satellite_data is None or trade_data is None:
        return recommendations
    
    # Price trend analysis
    if len(price_data) > 30:  # Need at least 30 days of data
        # Calculate short-term and long-term moving averages
        price_data['SMA20'] = price_data['Price'].rolling(window=20).mean()
        price_data['SMA50'] = price_data['Price'].rolling(window=50).mean()
        
        # Get the most recent values
        latest_price = price_data['Price'].iloc[-1]
        latest_sma20 = price_data['SMA20'].iloc[-1]
        latest_sma50 = price_data['SMA50'].iloc[-1]
        
        # Calculate price momentum (rate of change)
        price_data['ROC'] = price_data['Price'].pct_change(periods=20) * 100
        latest_momentum = price_data['ROC'].iloc[-1]
        
        # Price trend signals
        if latest_sma20 > latest_sma50 and latest_momentum > 2:
            # Strong uptrend
            if user_type == 'buyer':
                recommendations.append({
                    'title': f'Potential price increase for {commodity}',
                    'description': f'Technical analysis indicates a strong upward trend in {commodity} prices. The 20-day moving average is above the 50-day moving average, and price momentum is positive at {latest_momentum:.2f}%. This suggests continued price strength in the near term.',
                    'impact': 'High',
                    'confidence': random.randint(70, 90),
                    'value': f'${random.randint(10000, 50000)}',
                    'action': f'Consider securing forward contracts for {commodity} now to protect against further price increases. Alternatively, explore origin substitution options from regions with more favorable supply conditions.',
                    'regions': [region],
                    'data_sources': ['Price Trend Analysis', 'Technical Indicators', 'Momentum Signals'],
                    'type': 'price'
                })
            else:  # seller
                recommendations.append({
                    'title': f'Favorable selling opportunity for {commodity}',
                    'description': f'Technical analysis indicates a strong upward trend in {commodity} prices. The 20-day moving average is above the 50-day moving average, and price momentum is positive at {latest_momentum:.2f}%. This presents a favorable selling opportunity.',
                    'impact': 'High',
                    'confidence': random.randint(70, 90),
                    'value': f'${random.randint(15000, 60000)}',
                    'action': f'Consider increasing sales of {commodity} in the near term to capitalize on favorable pricing. Evaluate the potential for negotiating premium contracts with buyers seeking to secure supply.',
                    'regions': [region],
                    'data_sources': ['Price Trend Analysis', 'Technical Indicators', 'Momentum Signals'],
                    'type': 'price'
                })
        elif latest_sma20 < latest_sma50 and latest_momentum < -2:
            # Strong downtrend
            if user_type == 'buyer':
                recommendations.append({
                    'title': f'Potential buying opportunity for {commodity}',
                    'description': f'Technical analysis indicates a downward trend in {commodity} prices. The 20-day moving average is below the 50-day moving average, and price momentum is negative at {latest_momentum:.2f}%. This suggests potential buying opportunities in the near term.',
                    'impact': 'Medium',
                    'confidence': random.randint(65, 85),
                    'value': f'${random.randint(8000, 30000)}',
                    'action': f'Consider delaying major purchases of {commodity} for 2-3 weeks to benefit from potentially lower prices. Develop a phased buying strategy to take advantage of the downward trend while managing supply risk.',
                    'regions': [region],
                    'data_sources': ['Price Trend Analysis', 'Technical Indicators', 'Momentum Signals'],
                    'type': 'price'
                })
            else:  # seller
                recommendations.append({
                    'title': f'Price risk alert for {commodity}',
                    'description': f'Technical analysis indicates a downward trend in {commodity} prices. The 20-day moving average is below the 50-day moving average, and price momentum is negative at {latest_momentum:.2f}%. This suggests potential price weakness in the near term.',
                    'impact': 'Medium',
                    'confidence': random.randint(65, 85),
                    'value': f'${random.randint(10000, 40000)}',
                    'action': f'Consider securing forward contracts with buyers now to protect against further price decreases. Alternatively, explore value-added product options or alternative markets where pricing may be more favorable.',
                    'regions': [region],
                    'data_sources': ['Price Trend Analysis', 'Technical Indicators', 'Momentum Signals'],
                    'type': 'price'
                })
    
    # Seasonal analysis
    if len(price_data) > 365:  # Need at least a year of data
        # Group by month to analyze seasonality
        price_data['Month'] = price_data['Date'].dt.month
        monthly_avg = price_data.groupby('Month')['Price'].mean().reset_index()
        
        # Get current month and next month
        current_month = datetime.now().month
        next_month = (current_month % 12) + 1
        
        # Get average prices
        current_month_avg = monthly_avg[monthly_avg['Month'] == current_month]['Price'].values[0] if current_month in monthly_avg['Month'].values else None
        next_month_avg = monthly_avg[monthly_avg['Month'] == next_month]['Price'].values[0] if next_month in monthly_avg['Month'].values else None
        
        if current_month_avg is not None and next_month_avg is not None:
            # Calculate expected price change
            pct_change = (next_month_avg - current_month_avg) / current_month_avg * 100
            
            if abs(pct_change) > 5:  # Significant seasonal change expected
                month_name = datetime(2000, next_month, 1).strftime('%B')
                
                if pct_change > 5:  # Price expected to increase
                    if user_type == 'buyer':
                        recommendations.append({
                            'title': f'Seasonal price increase expected for {commodity}',
                            'description': f'Seasonal analysis indicates that {commodity} prices typically increase by {pct_change:.2f}% from {datetime(2000, current_month, 1).strftime("%B")} to {month_name}. This seasonal pattern suggests potential price increases in the coming month.',
                            'impact': 'Medium',
                            'confidence': random.randint(60, 80),
                            'value': f'${random.randint(7000, 25000)}',
                            'action': f'Consider accelerating purchases of {commodity} before the typical seasonal price increase in {month_name}. Evaluate inventory levels to ensure adequate coverage through the higher-price period.',
                            'regions': [region],
                            'data_sources': ['Seasonal Price Analysis', 'Historical Patterns', 'Monthly Price Trends'],
                            'type': 'seasonal'
                        })
                    else:  # seller
                        recommendations.append({
                            'title': f'Seasonal selling opportunity for {commodity}',
                            'description': f'Seasonal analysis indicates that {commodity} prices typically increase by {pct_change:.2f}% from {datetime(2000, current_month, 1).strftime("%B")} to {month_name}. This seasonal pattern suggests a potential selling opportunity in the coming month.',
                            'impact': 'Medium',
                            'confidence': random.randint(60, 80),
                            'value': f'${random.randint(9000, 30000)}',
                            'action': f'Consider delaying sales of {commodity} until {month_name} to benefit from typical seasonal price increases. Evaluate storage costs against expected price appreciation to optimize timing.',
                            'regions': [region],
                            'data_sources': ['Seasonal Price Analysis', 'Historical Patterns', 'Monthly Price Trends'],
                            'type': 'seasonal'
                        })
                else:  # Price expected to decrease
                    if user_type == 'buyer':
                        recommendations.append({
                            'title': f'Seasonal price decrease expected for {commodity}',
                            'description': f'Seasonal analysis indicates that {commodity} prices typically decrease by {abs(pct_change):.2f}% from {datetime(2000, current_month, 1).strftime("%B")} to {month_name}. This seasonal pattern suggests potential buying opportunities in the coming month.',
                            'impact': 'Medium',
                            'confidence': random.randint(60, 80),
                            'value': f'${random.randint(7000, 25000)}',
                            'action': f'Consider delaying major purchases of {commodity} until {month_name} to benefit from typical seasonal price decreases. Ensure current inventory is sufficient to bridge the gap.',
                            'regions': [region],
                            'data_sources': ['Seasonal Price Analysis', 'Historical Patterns', 'Monthly Price Trends'],
                            'type': 'seasonal'
                        })
                    else:  # seller
                        recommendations.append({
                            'title': f'Seasonal price risk for {commodity}',
                            'description': f'Seasonal analysis indicates that {commodity} prices typically decrease by {abs(pct_change):.2f}% from {datetime(2000, current_month, 1).strftime("%B")} to {month_name}. This seasonal pattern suggests potential price weakness in the coming month.',
                            'impact': 'Medium',
                            'confidence': random.randint(60, 80),
                            'value': f'${random.randint(9000, 30000)}',
                            'action': f'Consider accelerating sales of {commodity} before the typical seasonal price decrease in {month_name}. Evaluate forward contract opportunities to lock in current prices.',
                            'regions': [region],
                            'data_sources': ['Seasonal Price Analysis', 'Historical Patterns', 'Monthly Price Trends'],
                            'type': 'seasonal'
                        })
    
    # Weather impact analysis
    if not weather_data.empty:
        # Calculate average temperature and rainfall
        avg_temp = weather_data['temperature'].mean()
        avg_rainfall = weather_data['rainfall'].mean()
        
        # Check for extreme weather conditions
        temp_extreme = False
        rainfall_extreme = False
        
        if avg_temp > 30 or avg_temp < 5:
            temp_extreme = True
        
        if avg_rainfall > 20 or avg_rainfall < 1:
            rainfall_extreme = True
        
        if temp_extreme or rainfall_extreme:
            weather_condition = ""
            if temp_extreme and rainfall_extreme:
                if avg_temp > 30:
                    weather_condition = f"high temperatures (avg: {avg_temp:.1f}°C)"
                else:
                    weather_condition = f"low temperatures (avg: {avg_temp:.1f}°C)"
                
                if avg_rainfall > 20:
                    weather_condition += f" and heavy rainfall (avg: {avg_rainfall:.1f}mm)"
                else:
                    weather_condition += f" and low rainfall (avg: {avg_rainfall:.1f}mm)"
            elif temp_extreme:
                if avg_temp > 30:
                    weather_condition = f"high temperatures (avg: {avg_temp:.1f}°C)"
                else:
                    weather_condition = f"low temperatures (avg: {avg_temp:.1f}°C)"
            else:
                if avg_rainfall > 20:
                    weather_condition = f"heavy rainfall (avg: {avg_rainfall:.1f}mm)"
                else:
                    weather_condition = f"low rainfall (avg: {avg_rainfall:.1f}mm)"
            
            if user_type == 'buyer':
                recommendations.append({
                    'title': f'Weather risk alert for {commodity} in {region}',
                    'description': f'Current weather data indicates {weather_condition} in key growing regions for {commodity}. These conditions could potentially impact crop development and yield, affecting future supply and prices.',
                    'impact': 'Medium',
                    'confidence': random.randint(60, 80),
                    'value': f'${random.randint(8000, 30000)}',
                    'action': f'Monitor {commodity} production reports closely in the affected regions. Consider diversifying sourcing to include regions with more favorable weather conditions. Evaluate forward contract opportunities to secure supply.',
                    'regions': [region],
                    'data_sources': ['Weather Data Analysis', 'Temperature Trends', 'Precipitation Patterns'],
                    'type': 'weather'
                })
            else:  # seller
                recommendations.append({
                    'title': f'Weather-based market opportunity for {commodity}',
                    'description': f'Current weather data indicates {weather_condition} in key growing regions for {commodity}. These conditions could potentially impact crop development and yield, affecting market supply dynamics.',
                    'impact': 'Medium',
                    'confidence': random.randint(60, 80),
                    'value': f'${random.randint(10000, 35000)}',
                    'action': f'Monitor market reactions to weather conditions closely. If your production is in regions with favorable weather, highlight this advantage in marketing communications. Consider adjusting pricing strategy to reflect potential market tightness.',
                    'regions': [region],
                    'data_sources': ['Weather Data Analysis', 'Temperature Trends', 'Precipitation Patterns'],
                    'type': 'weather'
                })
    
    # Satellite data analysis (crop health)
    if not satellite_data.empty:
        # Calculate average NDVI and its trend
        satellite_data = satellite_data.sort_values('date')
        avg_ndvi = satellite_data['ndvi'].mean()
        
        # Calculate NDVI trend over the last 3 months
        if len(satellite_data) >= 3:
            recent_data = satellite_data.tail(3)
            ndvi_trend = recent_data['ndvi'].iloc[-1] - recent_data['ndvi'].iloc[0]
            
            if abs(ndvi_trend) > 0.05:  # Significant change in crop health
                if ndvi_trend > 0.05:  # Improving crop health
                    if user_type == 'buyer':
                        recommendations.append({
                            'title': f'Improving crop conditions for {commodity}',
                            'description': f'Satellite imagery analysis shows improving crop health indicators (NDVI increase of {ndvi_trend:.2f}) for {commodity} in {region}. This suggests potential yield improvements and could lead to increased supply in the coming harvest.',
                            'impact': 'Medium',
                            'confidence': random.randint(65, 85),
                            'value': f'${random.randint(7000, 25000)}',
                            'action': f'Monitor price trends as harvest approaches, as improving crop conditions may lead to price moderation. Consider delaying long-term contract commitments to benefit from potential price decreases.',
                            'regions': [region],
                            'data_sources': ['Satellite Imagery Analysis', 'NDVI Trends', 'Crop Health Indicators'],
                            'type': 'crop'
                        })
                    else:  # seller
                        recommendations.append({
                            'title': f'Supply increase alert for {commodity}',
                            'description': f'Satellite imagery analysis shows improving crop health indicators (NDVI increase of {ndvi_trend:.2f}) for {commodity} in {region}. This suggests potential yield improvements and could lead to increased market supply in the coming harvest.',
                            'impact': 'Medium',
                            'confidence': random.randint(65, 85),
                            'value': f'${random.randint(9000, 30000)}',
                            'action': f'Consider forward selling a portion of production to lock in current prices before potential harvest pressure. Evaluate value-added opportunities to differentiate your product in an increasingly competitive market.',
                            'regions': [region],
                            'data_sources': ['Satellite Imagery Analysis', 'NDVI Trends', 'Crop Health Indicators'],
                            'type': 'crop'
                        })
                else:  # Deteriorating crop health
                    if user_type == 'buyer':
                        recommendations.append({
                            'title': f'Deteriorating crop conditions for {commodity}',
                            'description': f'Satellite imagery analysis shows declining crop health indicators (NDVI decrease of {abs(ndvi_trend):.2f}) for {commodity} in {region}. This suggests potential yield reductions and could lead to tighter supply in the coming harvest.',
                            'impact': 'High',
                            'confidence': random.randint(70, 90),
                            'value': f'${random.randint(12000, 45000)}',
                            'action': f'Consider securing forward contracts for {commodity} now to protect against potential price increases. Identify alternative sourcing regions with more favorable crop conditions to diversify supply risk.',
                            'regions': [region],
                            'data_sources': ['Satellite Imagery Analysis', 'NDVI Trends', 'Crop Health Indicators'],
                            'type': 'crop'
                        })
                    else:  # seller
                        recommendations.append({
                            'title': f'Potential price strength for {commodity}',
                            'description': f'Satellite imagery analysis shows declining crop health indicators (NDVI decrease of {abs(ndvi_trend):.2f}) for {commodity} in {region}. This suggests potential yield reductions and could lead to price strength in the coming harvest period.',
                            'impact': 'High',
                            'confidence': random.randint(70, 90),
                            'value': f'${random.randint(15000, 50000)}',
                            'action': f'Consider delaying sales commitments to benefit from potential price increases. If your production is in regions with better crop conditions, highlight this quality advantage in marketing communications.',
                            'regions': [region],
                            'data_sources': ['Satellite Imagery Analysis', 'NDVI Trends', 'Crop Health Indicators'],
                            'type': 'crop'
                        })
    
    # Trade flow analysis
    if not trade_data.empty:
        # Analyze import/export trends
        trade_data = trade_data.sort_values('date')
        trade_data['total_trade'] = trade_data['exports'] + trade_data['imports']
        
        # Calculate trade flow trend over the last 3 months
        if len(trade_data) >= 3:
            recent_data = trade_data.groupby('date')['total_trade'].sum().reset_index()
            recent_data = recent_data.sort_values('date').tail(3)
            trade_trend = recent_data['total_trade'].iloc[-1] / recent_data['total_trade'].iloc[0] - 1  # Percentage change
            
            if abs(trade_trend) > 0.1:  # Significant change in trade flows
                if trade_trend > 0.1:  # Increasing trade flows
                    if user_type == 'buyer':
                        recommendations.append({
                            'title': f'Increasing global trade in {commodity}',
                            'description': f'Trade flow analysis shows a {trade_trend*100:.1f}% increase in global {commodity} trade over the past 3 months. This suggests strong demand and active market participation, which could influence price dynamics.',
                            'impact': 'Medium',
                            'confidence': random.randint(60, 80),
                            'value': f'${random.randint(8000, 25000)}',
                            'action': f'Monitor price trends closely as increased trade activity may lead to price volatility. Consider diversifying suppliers to ensure consistent access to {commodity} in an active market environment.',
                            'regions': [region],
                            'data_sources': ['Trade Flow Analysis', 'Import/Export Trends', 'Global Market Activity'],
                            'type': 'trade'
                        })
                    else:  # seller
                        recommendations.append({
                            'title': f'Expanding market for {commodity}',
                            'description': f'Trade flow analysis shows a {trade_trend*100:.1f}% increase in global {commodity} trade over the past 3 months. This suggests expanding market opportunities and strong demand conditions.',
                            'impact': 'Medium',
                            'confidence': random.randint(60, 80),
                            'value': f'${random.randint(10000, 35000)}',
                            'action': f'Explore new market entry opportunities in regions showing the strongest import growth. Consider adjusting pricing strategy to reflect the strong demand environment while remaining competitive.',
                            'regions': [region],
                            'data_sources': ['Trade Flow Analysis', 'Import/Export Trends', 'Global Market Activity'],
                            'type': 'trade'
                        })
                else:  # Decreasing trade flows
                    if user_type == 'buyer':
                        recommendations.append({
                            'title': f'Decreasing global trade in {commodity}',
                            'description': f'Trade flow analysis shows a {abs(trade_trend)*100:.1f}% decrease in global {commodity} trade over the past 3 months. This suggests potential shifts in supply-demand dynamics that could affect availability and pricing.',
                            'impact': 'Medium',
                            'confidence': random.randint(60, 80),
                            'value': f'${random.randint(7000, 22000)}',
                            'action': f'Evaluate the causes of reduced trade flows (demand destruction, logistical constraints, etc.) and their potential impact on your supply chain. Consider building strategic reserves if you anticipate supply challenges.',
                            'regions': [region],
                            'data_sources': ['Trade Flow Analysis', 'Import/Export Trends', 'Global Market Activity'],
                            'type': 'trade'
                        })
                    else:  # seller
                        recommendations.append({
                            'title': f'Changing market dynamics for {commodity}',
                            'description': f'Trade flow analysis shows a {abs(trade_trend)*100:.1f}% decrease in global {commodity} trade over the past 3 months. This suggests changing market dynamics that may require strategic adjustments.',
                            'impact': 'Medium',
                            'confidence': random.randint(60, 80),
                            'value': f'${random.randint(9000, 28000)}',
                            'action': f'Identify the most resilient import markets to focus sales efforts. Consider developing value-added products or services to differentiate your offering in a potentially more competitive environment.',
                            'regions': [region],
                            'data_sources': ['Trade Flow Analysis', 'Import/Export Trends', 'Global Market Activity'],
                            'type': 'trade'
                        })
    
    # Add region-specific opportunities if a specific region is selected
    if region != 'Global':
        if user_type == 'buyer':
            recommendations.append({
                'title': f'Regional supply dynamics for {commodity} in {region}',
                'description': f'Combined analysis of price trends, weather patterns, crop health, and trade flows indicates unique market dynamics for {commodity} in {region}. These regional factors create specific opportunities for buyers with the flexibility to source from this region.',
                'impact': 'Medium',
                'confidence': random.randint(65, 85),
                'value': f'${random.randint(7000, 22000)}',
                'action': f'Develop targeted sourcing strategies for {region} that leverage the specific market conditions identified. Consider establishing direct relationships with suppliers in this region to access unique opportunities.',
                'regions': [region],
                'data_sources': ['Regional Market Analysis', 'Multi-factor Assessment', 'Comparative Advantage Study'],
                'type': 'regional'
            })
        else:
            recommendations.append({
                'title': f'Regional market opportunity for {commodity} in {region}',
                'description': f'Combined analysis of price trends, weather patterns, crop health, and trade flows indicates unique market dynamics for {commodity} in {region}. These regional factors create specific opportunities for sellers who can meet the requirements of this market.',
                'impact': 'Medium',
                'confidence': random.randint(65, 85),
                'value': f'${random.randint(12000, 35000)}',
                'action': f'Develop targeted marketing strategies for buyers in {region} that address their specific needs and preferences. Consider establishing local partnerships or representation to strengthen market presence.',
                'regions': [region],
                'data_sources': ['Regional Market Analysis', 'Multi-factor Assessment', 'Buyer Preference Study'],
                'type': 'regional'
            })
    
    return recommendations

# Function to generate contact recommendations
def generate_contact_recommendations(commodity, user_type='buyer'):
    # In a real implementation, this would connect to a database of traders
    # For demo purposes, we'll generate simulated contact recommendations
    
    if user_type == 'buyer':
        # Recommend sellers to buyers
        contacts = [
            {
                'name': 'Global Harvest Trading Ltd.',
                'type': 'Seller',
                'location': 'Chicago, USA',
                'specialty': f'Major exporter of {commodity} with global sourcing network',
                'reliability': '4.8/5',
                'price_competitiveness': '4.2/5',
                'contact': 'sales@globalharvesttrading.com | +1-312-555-0189'
            },
            {
                'name': 'AgriSource International',
                'type': 'Seller',
                'location': 'Rotterdam, Netherlands',
                'specialty': f'Specialized in premium quality {commodity} with traceability',
                'reliability': '4.7/5',
                'price_competitiveness': '3.9/5',
                'contact': 'info@agrisource-intl.com | +31-10-555-8844'
            }
        ]
    else:
        # Recommend buyers to sellers
        contacts = [
            {
                'name': 'Quality Foods Processing Inc.',
                'type': 'Buyer',
                'location': 'Atlanta, USA',
                'specialty': f'Food manufacturer seeking consistent quality {commodity}',
                'payment_reliability': '4.9/5',
                'volume_potential': '4.5/5',
                'contact': 'procurement@qualityfoods.com | +1-404-555-7700'
            },
            {
                'name': 'EuroFresh Distributors',
                'type': 'Buyer',
                'location': 'Barcelona, Spain',
                'specialty': f'Distributor specializing in premium {commodity} for European market',
                'payment_reliability': '4.7/5',
                'volume_potential': '4.3/5',
                'contact': 'buying@eurofresh.eu | +34-93-555-2266'
            }
        ]
    
    # Add shipping recommendations for buyers
    shipping_contacts = []
    if user_type == 'buyer':
        shipping_contacts = [
            {
                'name': 'OceanRoute Logistics',
                'type': 'Shipping',
                'location': 'Singapore',
                'specialty': f'Specialized in temperature-controlled shipping for {commodity}',
                'reliability': '4.6/5',
                'cost_efficiency': '4.3/5',
                'contact': 'bookings@oceanroute.com | +65-6555-9988'
            },
            {
                'name': 'FreightMasters International',
                'type': 'Shipping',
                'location': 'Hamburg, Germany',
                'specialty': 'Global freight forwarder with agricultural expertise',
                'reliability': '4.8/5',
                'cost_efficiency': '4.1/5',
                'contact': 'operations@freightmasters.com | +49-40-555-3344'
            }
        ]
    
    return contacts, shipping_contacts

# Function to create price analysis chart
def create_price_chart(df, commodity, region):
    if df is None or df.empty:
        return go.Figure().update_layout(title=f"No data available for {commodity} in {region}")
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Price'],
            name="Price",
            line=dict(color="#e74c3c", width=2)
        ),
        secondary_y=False,
    )
    
    # Add volume bars if available
    if 'Volume' in df.columns:
        fig.add_trace(
            go.Bar(
                x=df['Date'],
                y=df['Volume'],
                name="Volume",
                marker_color="rgba(55, 83, 109, 0.3)",
                opacity=0.7
            ),
            secondary_y=True,
        )
    
    # Add moving averages if we have enough data
    if len(df) > 50:
        df['SMA20'] = df['Price'].rolling(window=20).mean()
        df['SMA50'] = df['Price'].rolling(window=50).mean()
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['SMA20'],
                name="20-Day MA",
                line=dict(color="#3498db", width=1.5, dash='dot')
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['SMA50'],
                name="50-Day MA",
                line=dict(color="#2ecc71", width=1.5, dash='dot')
            ),
            secondary_y=False,
        )
    
    # Add figure layout
    fig.update_layout(
        title=f"{commodity} Price and Volume Analysis - {region}",
        xaxis_title="Date",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        height=500
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True)
    
    return fig

# Function to create weather impact chart
def create_weather_chart(weather_data, satellite_data, commodity, region):
    if weather_data is None or weather_data.empty:
        return go.Figure().update_layout(title=f"No weather data available for {region}")
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Group data by date and calculate averages
    if 'date' in weather_data.columns:
        weather_data['date'] = pd.to_datetime(weather_data['date'])
        daily_weather = weather_data.groupby('date').agg({
            'temperature': 'mean',
            'rainfall': 'mean',
            'humidity': 'mean'
        }).reset_index()
        
        # Add temperature line
        fig.add_trace(
            go.Scatter(
                x=daily_weather['date'],
                y=daily_weather['temperature'],
                name="Temperature",
                line=dict(color="#e67e22", width=2)
            ),
            secondary_y=False,
        )
        
        # Add rainfall bars
        fig.add_trace(
            go.Bar(
                x=daily_weather['date'],
                y=daily_weather['rainfall'],
                name="Rainfall",
                marker_color="rgba(0, 119, 182, 0.3)",
                opacity=0.7
            ),
            secondary_y=True,
        )
        
        # Add NDVI data if available
        if satellite_data is not None and not satellite_data.empty and 'ndvi' in satellite_data.columns:
            satellite_data['date'] = pd.to_datetime(satellite_data['date'])
            daily_ndvi = satellite_data.groupby('date')['ndvi'].mean().reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=daily_ndvi['date'],
                    y=daily_ndvi['ndvi'],
                    name="Crop Health (NDVI)",
                    line=dict(color="#27ae60", width=2, dash='dot')
                ),
                secondary_y=True,
            )
    else:
        # If we don't have date information, create a bar chart of current conditions
        cities = weather_data['city'].unique()
        
        fig.add_trace(
            go.Bar(
                x=cities,
                y=weather_data.groupby('city')['temperature'].mean(),
                name="Temperature",
                marker_color="rgba(230, 126, 34, 0.7)"
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Bar(
                x=cities,
                y=weather_data.groupby('city')['rainfall'].mean(),
                name="Rainfall",
                marker_color="rgba(0, 119, 182, 0.7)"
            ),
            secondary_y=True,
        )
    
    # Add figure layout
    fig.update_layout(
        title=f"Weather Patterns Affecting {commodity} - {region}",
        xaxis_title="Date" if 'date' in weather_data.columns else "Location",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        height=500
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig.update_yaxes(
        title=dict(
            text="Rainfall (mm) / NDVI",
            font=dict(color="rgba(0, 119, 182, 1)")
        ),
        tickfont=dict(color="rgba(0, 119, 182, 1)"),
        secondary_y=True
    )
    
    return fig

# Function to create crop health chart
def create_crop_health_chart(satellite_data, price_data, commodity, region):
    if satellite_data is None or satellite_data.empty:
        return go.Figure().update_layout(title=f"No satellite data available for {region}")
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Group satellite data by date
    satellite_data['date'] = pd.to_datetime(satellite_data['date'])
    daily_ndvi = satellite_data.groupby('date')['ndvi'].mean().reset_index()
    
    # Add crop health line
    fig.add_trace(
        go.Scatter(
            x=daily_ndvi['date'],
            y=daily_ndvi['ndvi'],
            name="Crop Health Index",
            line=dict(color="#27ae60", width=2)
        ),
        secondary_y=False,
    )
    
    # Add price line if available
    if price_data is not None and not price_data.empty:
        # Ensure we have a common date format
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        
        fig.add_trace(
            go.Scatter(
                x=price_data['Date'],
                y=price_data['Price'],
                name="Price",
                line=dict(color="#e74c3c", width=2, dash='dot')
            ),
            secondary_y=True,
        )
    
    # Add figure layout
    fig.update_layout(
        title=f"{commodity} Crop Health and Price Correlation - {region}",
        xaxis_title="Date",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        height=500
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Crop Health Index (NDVI)", secondary_y=False)
    fig.update_yaxes(title_text="Price (USD)", secondary_y=True)
    
    return fig

# Function to create trade flow chart
def create_trade_flow_chart(trade_data, commodity, region):
    if trade_data is None or trade_data.empty:
        return go.Figure().update_layout(title=f"No trade data available for {commodity} in {region}")
    
    # Create figure
    fig = go.Figure()
    
    # Group data by date
    trade_data['date'] = pd.to_datetime(trade_data['date'])
    
    # Aggregate by date
    daily_trade = trade_data.groupby('date').agg({
        'exports': 'sum',
        'imports': 'sum'
    }).reset_index()
    
    # Add exports line
    fig.add_trace(
        go.Scatter(
            x=daily_trade['date'],
            y=daily_trade['exports'],
            name="Exports",
            line=dict(color="#3498db", width=2)
        )
    )
    
    # Add imports line
    fig.add_trace(
        go.Scatter(
            x=daily_trade['date'],
            y=daily_trade['imports'],
            name="Imports",
            line=dict(color="#e74c3c", width=2)
        )
    )
    
    # Add total trade line
    daily_trade['total'] = daily_trade['exports'] + daily_trade['imports']
    fig.add_trace(
        go.Scatter(
            x=daily_trade['date'],
            y=daily_trade['total'],
            name="Total Trade",
            line=dict(color="#2ecc71", width=2, dash='dot')
        )
    )
    
    # Add figure layout
    fig.update_layout(
        title=f"{commodity} Global Trade Flows - {region}",
        xaxis_title="Date",
        yaxis_title="Trade Volume (tons)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        height=500
    )
    
    return fig

# Function to create PDF report
def create_pdf_report(opportunity, commodity, region, user_type):
    # In a real implementation, this would generate a PDF
    # For Streamlit, we'll create a styled HTML report that can be downloaded
    
    html_content = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
            }}
            h1 {{
                color: #2c3e50;
                font-size: 24px;
                margin-bottom: 5px;
            }}
            h2 {{
                color: #34495e;
                font-size: 20px;
                margin-top: 20px;
                margin-bottom: 10px;
            }}
            .meta {{
                display: flex;
                flex-wrap: wrap;
                margin-bottom: 20px;
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
            }}
            .meta-item {{
                flex: 1 1 200px;
                margin-bottom: 10px;
            }}
            .meta-label {{
                font-weight: bold;
                color: #7f8c8d;
            }}
            .impact {{
                display: inline-block;
                padding: 3px 8px;
                border-radius: 3px;
                font-weight: bold;
            }}
            .impact-high {{
                background-color: #e74c3c;
                color: white;
            }}
            .impact-medium {{
                background-color: #f39c12;
                color: white;
            }}
            .description {{
                margin-bottom: 20px;
                line-height: 1.8;
            }}
            .action {{
                background-color: #eaf2f8;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                border-left: 5px solid #3498db;
            }}
            .chart-placeholder {{
                background-color: #f8f9fa;
                padding: 20px;
                text-align: center;
                border: 1px dashed #ddd;
                margin-bottom: 20px;
            }}
            .footer {{
                margin-top: 40px;
                padding-top: 10px;
                border-top: 1px solid #ddd;
                font-size: 12px;
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Food Trading Insights Platform</h1>
            <p>Market Opportunity Report</p>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d")}</p>
        </div>
        
        <h1>{opportunity['title']}</h1>
        
        <div class="meta">
            <div class="meta-item">
                <span class="meta-label">Commodity:</span> {commodity}
            </div>
            <div class="meta-item">
                <span class="meta-label">Region:</span> {region}
            </div>
            <div class="meta-item">
                <span class="meta-label">Impact:</span> 
                <span class="impact {'impact-high' if opportunity['impact'] == 'High' else 'impact-medium'}">{opportunity['impact']}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">Confidence:</span> {opportunity['confidence']}%
            </div>
            <div class="meta-item">
                <span class="meta-label">Potential Value:</span> <span class="value">{opportunity['value']}</span>
            </div>
            <div class="meta-item">
                <span class="meta-label">User Perspective:</span> {user_type.capitalize()}
            </div>
        </div>
        
        <h2>Opportunity Overview</h2>
        <div class="description">
            {opportunity['description']}
        </div>
        
        <div class="chart-placeholder">
            [Price trend chart would appear here in the PDF version]
        </div>
        
        <h2>Recommended Action</h2>
        <div class="action">
            {opportunity['action']}
        </div>
        
        <h2>Supporting Data</h2>
        <p>This recommendation is based on analysis of the following data sources:</p>
        <ul>
    """
    
    for source in opportunity['data_sources']:
        html_content += f"<li>{source}</li>"
    
    html_content += """
        </ul>
        
        <div class="chart-placeholder">
            [Supporting data visualization would appear here in the PDF version]
        </div>
        
        <h2>Implementation Timeline</h2>
        <p>For maximum value, this opportunity should be acted upon within the next 2-3 weeks.</p>
        
        <div class="footer">
            <p>Food Trading Insights Platform &copy; 2025</p>
            <p>This report is generated based on algorithmic analysis of multiple data sources. 
            While we strive for accuracy, all trading decisions should be made with appropriate due diligence.</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

# Function to convert HTML to downloadable PDF (simulated)
def get_pdf_download_link(html_content, filename="report.pdf"):
    # In a real implementation, this would convert HTML to PDF
    # For Streamlit, we'll provide the HTML as a download
    
    # Encode HTML as base64
    b64 = base64.b64encode(html_content.encode()).decode()
    
    # Create download link
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}.html">Download Report</a>'
    
    return href

# Function to update data
def update_data():
    # Update last update time
    st.session_state.last_update_time = datetime.now()
    
    # Get available commodities
    st.session_state.available_commodities = get_available_commodities()
    
    # If no commodity is selected, select the first one
    if st.session_state.selected_commodity is None and st.session_state.available_commodities:
        st.session_state.selected_commodity = st.session_state.available_commodities[0]['name']
    
    # Get data for selected commodity
    if st.session_state.selected_commodity:
        # Find ticker for selected commodity
        ticker = next((item['ticker'] for item in st.session_state.available_commodities if item['name'] == st.session_state.selected_commodity), None)
        
        if ticker:
            # Get price data
            price_data = get_commodity_price_data(ticker)
            st.session_state.commodity_data[st.session_state.selected_commodity] = price_data
            
            # Get weather data
            weather_data = get_weather_data(st.session_state.selected_region)
            st.session_state.weather_data[st.session_state.selected_region] = weather_data
            
            # Get satellite data
            satellite_data = get_satellite_data(st.session_state.selected_region)
            st.session_state.satellite_data[st.session_state.selected_region] = satellite_data
            
            # Get trade data
            trade_data = get_trade_flow_data(st.session_state.selected_commodity, st.session_state.selected_region)
            st.session_state.trade_data[f"{st.session_state.selected_commodity}_{st.session_state.selected_region}"] = trade_data
            
            # Generate recommendations
            recommendations = generate_recommendations(
                price_data,
                weather_data,
                satellite_data,
                trade_data,
                st.session_state.selected_commodity,
                st.session_state.selected_region,
                st.session_state.user_type
            )
            
            st.session_state.recommendations[f"{st.session_state.selected_commodity}_{st.session_state.selected_region}_{st.session_state.user_type}"] = recommendations

# Main application layout
def main():
    # Update data if needed
    if 'last_update_time' not in st.session_state or (datetime.now() - st.session_state.last_update_time).total_seconds() > 1800:  # 30 minutes
        with st.spinner("Fetching latest data..."):
            update_data()
    
    # Header with user type switching
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.write("")
        if st.session_state.user_type == 'buyer':
            st.markdown('<span class="badge badge-primary">Buyer View</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-secondary">Buyer View</span>', unsafe_allow_html=True)
        
        if st.button("Switch to Buyer View"):
            switch_user_type('buyer')
    
    with col2:
        st.markdown('<h1 class="main-header">Food Trading Insights Platform</h1>', unsafe_allow_html=True)
    
    with col3:
        st.write("")
        if st.session_state.user_type == 'seller':
            st.markdown('<span class="badge badge-success">Seller View</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-secondary">Seller View</span>', unsafe_allow_html=True)
        
        if st.button("Switch to Seller View"):
            switch_user_type('seller')
    
    # Last update time
    st.markdown(f'<div class="update-time">Last updated: {st.session_state.last_update_time.strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)
    
    # Manual refresh button
    if st.button("Refresh Data"):
        with st.spinner("Fetching latest data..."):
            update_data()
        st.success("Data refreshed successfully!")
    
    # Commodity and region selection
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a list of commodity names
        commodity_names = [item['name'] for item in st.session_state.available_commodities]
        
        if commodity_names:
            selected_commodity = st.selectbox(
                'Select Commodity', 
                commodity_names,
                index=commodity_names.index(st.session_state.selected_commodity) if st.session_state.selected_commodity in commodity_names else 0
            )
            
            if selected_commodity != st.session_state.selected_commodity:
                st.session_state.selected_commodity = selected_commodity
                # Fetch data for the new commodity
                with st.spinner(f"Fetching data for {selected_commodity}..."):
                    update_data()
    
    with col2:
        regions = ['Global', 'North America', 'South America', 'Europe', 'Asia', 'Africa', 'Middle East', 'Russia']
        selected_region = st.selectbox(
            'Select Region', 
            regions,
            index=regions.index(st.session_state.selected_region) if st.session_state.selected_region in regions else 0
        )
        
        if selected_region != st.session_state.selected_region:
            st.session_state.selected_region = selected_region
            # Fetch data for the new region
            with st.spinner(f"Fetching data for {selected_region}..."):
                update_data()
    
    # Get data for selected commodity and region
    price_data = st.session_state.commodity_data.get(st.session_state.selected_commodity)
    weather_data = st.session_state.weather_data.get(st.session_state.selected_region)
    satellite_data = st.session_state.satellite_data.get(st.session_state.selected_region)
    trade_data = st.session_state.trade_data.get(f"{st.session_state.selected_commodity}_{st.session_state.selected_region}")
    
    # Analysis type selection
    st.markdown('<h2 class="sub-header">Market Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button('Price Analysis', use_container_width=True):
            st.session_state.selected_analysis = 'price'
    
    with col2:
        if st.button('Weather Impact', use_container_width=True):
            st.session_state.selected_analysis = 'weather'
    
    with col3:
        if st.button('Crop Health', use_container_width=True):
            st.session_state.selected_analysis = 'crop'
    
    with col4:
        if st.button('Trade Flows', use_container_width=True):
            st.session_state.selected_analysis = 'trade'
    
    # Display selected analysis
    if st.session_state.selected_analysis == 'price':
        fig = create_price_chart(price_data, st.session_state.selected_commodity, st.session_state.selected_region)
        st.plotly_chart(fig, use_container_width=True)
    elif st.session_state.selected_analysis == 'weather':
        fig = create_weather_chart(weather_data, satellite_data, st.session_state.selected_commodity, st.session_state.selected_region)
        st.plotly_chart(fig, use_container_width=True)
    elif st.session_state.selected_analysis == 'crop':
        fig = create_crop_health_chart(satellite_data, price_data, st.session_state.selected_commodity, st.session_state.selected_region)
        st.plotly_chart(fig, use_container_width=True)
    elif st.session_state.selected_analysis == 'trade':
        fig = create_trade_flow_chart(trade_data, st.session_state.selected_commodity, st.session_state.selected_region)
        st.plotly_chart(fig, use_container_width=True)
    
    # Market opportunities
    st.markdown('<h2 class="sub-header">Market Opportunities</h2>', unsafe_allow_html=True)
    
    # Get recommendations
    recommendations = st.session_state.recommendations.get(f"{st.session_state.selected_commodity}_{st.session_state.selected_region}_{st.session_state.user_type}", [])
    
    if not recommendations:
        st.info(f"No recommendations available for {st.session_state.selected_commodity} in {st.session_state.selected_region}. Try selecting a different commodity or region.")
    
    for i, opportunity in enumerate(recommendations):
        if st.session_state.user_type == 'buyer':
            card_class = "card opportunity-card"
        else:
            card_class = "card opportunity-card-seller"
        
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"### {opportunity['title']}")
            st.markdown(f"<span class='badge badge-{'danger' if opportunity['impact'] == 'High' else 'warning'}'>{opportunity['impact']} Impact</span> <span class='badge badge-info'>{opportunity['confidence']}% Confidence</span> <span class='badge badge-success'>{opportunity['value']} Potential Value</span>", unsafe_allow_html=True)
        
        with col2:
            if st.button(f"View Details #{i}", key=f"view_{i}"):
                st.session_state[f"show_details_{i}"] = not st.session_state.get(f"show_details_{i}", False)
        
        with col3:
            # Generate PDF report button
            if st.button(f"Generate PDF #{i}", key=f"pdf_{i}"):
                st.session_state[f"generate_pdf_{i}"] = True
        
        # Show details if requested
        if st.session_state.get(f"show_details_{i}", False):
            st.markdown("#### Description")
            st.write(opportunity['description'])
            
            st.markdown("#### Recommended Action")
            st.info(opportunity['action'])
            
            st.markdown("#### Data Sources")
            for source in opportunity['data_sources']:
                st.markdown(f"- {source}")
            
            st.markdown("#### Affected Regions")
            for region in opportunity['regions']:
                st.markdown(f"- {region}")
        
        # Show PDF if requested
        if st.session_state.get(f"generate_pdf_{i}", False):
            html_content = create_pdf_report(opportunity, st.session_state.selected_commodity, st.session_state.selected_region, st.session_state.user_type)
            st.markdown("#### PDF Report Generated")
            st.markdown(get_pdf_download_link(html_content, f"{st.session_state.selected_commodity}_{opportunity['title'].replace(' ', '_')}"), unsafe_allow_html=True)
            st.session_state[f"generate_pdf_{i}"] = False
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Contact recommendations
    st.markdown('<h2 class="sub-header">Contact Recommendations</h2>', unsafe_allow_html=True)
    
    contacts, shipping_contacts = generate_contact_recommendations(st.session_state.selected_commodity, st.session_state.user_type)
    
    col1, col2 = st.columns(2)
    
    with col1:
        contact_type = "Seller" if st.session_state.user_type == 'buyer' else "Buyer"
        st.markdown(f"#### Recommended {contact_type}s")
        
        for contact in contacts:
            st.markdown(f'<div class="contact-card">', unsafe_allow_html=True)
            st.markdown(f"<div class='contact-name'>{contact['name']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='contact-details'><strong>Location:</strong> {contact['location']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='contact-details'><strong>Specialty:</strong> {contact['specialty']}</div>", unsafe_allow_html=True)
            
            if st.session_state.user_type == 'buyer':
                st.markdown(f"<div class='contact-details'><strong>Reliability:</strong> {contact['reliability']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='contact-details'><strong>Price Competitiveness:</strong> {contact['price_competitiveness']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='contact-details'><strong>Payment Reliability:</strong> {contact['payment_reliability']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='contact-details'><strong>Volume Potential:</strong> {contact['volume_potential']}</div>", unsafe_allow_html=True)
            
            st.markdown(f"<div class='contact-details'><strong>Contact:</strong> {contact['contact']}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.user_type == 'buyer' and shipping_contacts:
            st.markdown("#### Recommended Shipping Providers")
            
            for contact in shipping_contacts:
                st.markdown(f'<div class="contact-card">', unsafe_allow_html=True)
                st.markdown(f"<div class='contact-name'>{contact['name']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='contact-details'><strong>Location:</strong> {contact['location']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='contact-details'><strong>Specialty:</strong> {contact['specialty']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='contact-details'><strong>Reliability:</strong> {contact['reliability']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='contact-details'><strong>Cost Efficiency:</strong> {contact['cost_efficiency']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='contact-details'><strong>Contact:</strong> {contact['contact']}</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer">Food Trading Insights Platform © 2025</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
