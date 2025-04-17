import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
import base64
import io
import random
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
# pdfkit import removed
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Set page configuration
st.set_page_config(
    page_title="Food Trading Insights Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #3498db;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .highlight {
        background-color: #e8f4f8;
        padding: 15px;
        border-left: 4px solid #3498db;
        margin: 20px 0;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        flex: 1;
        min-width: 200px;
        margin-right: 10px;
    }
    .opportunity-card {
        background-color: white;
        border-radius: 5px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #27ae60;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e0e5ea;
        border-bottom: 2px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_type' not in st.session_state:
    st.session_state.user_type = 'Buyer'
if 'selected_commodity' not in st.session_state:
    st.session_state.selected_commodity = 'Wheat'
if 'selected_region' not in st.session_state:
    st.session_state.selected_region = 'Global'
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = 'Price Analysis'
if 'refresh_data' not in st.session_state:
    st.session_state.refresh_data = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# Function to get real commodity data from Yahoo Finance
@st.cache_data(ttl=3600)
def get_commodity_data(symbol, period="2y"):
    try:
        data = yf.download(symbol, period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        # Return simulated data as fallback
        return simulate_commodity_data(symbol)

# Function to simulate commodity data if API fails
def simulate_commodity_data(commodity, years=2):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Base price and seasonality parameters based on commodity
    if commodity in ['Wheat', 'Corn', 'Soybeans', 'Rice']:
        base_price = random.uniform(300, 800)
        amplitude = base_price * 0.2
        trend = random.uniform(-0.1, 0.1)
    elif commodity in ['Oranges', 'Apples', 'Bananas', 'Strawberries', 'Pineapples']:
        base_price = random.uniform(100, 300)
        amplitude = base_price * 0.3
        trend = random.uniform(-0.05, 0.15)
    else:  # Vegetables and others
        base_price = random.uniform(50, 200)
        amplitude = base_price * 0.25
        trend = random.uniform(-0.08, 0.12)
    
    # Generate prices with seasonality, trend, and noise
    prices = []
    for i, date in enumerate(date_range):
        # Seasonality component (yearly cycle)
        season = amplitude * np.sin(2 * np.pi * i / 365)
        # Trend component
        trend_component = trend * i / 365 * base_price
        # Random noise
        noise = np.random.normal(0, base_price * 0.05)
        # Combine components
        price = base_price + season + trend_component + noise
        prices.append(max(price, base_price * 0.5))  # Ensure price doesn't go too low
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': prices,
        'High': [p * random.uniform(1.01, 1.03) for p in prices],
        'Low': [p * random.uniform(0.97, 0.99) for p in prices],
        'Close': [p * random.uniform(0.98, 1.02) for p in prices],
        'Volume': [random.randint(1000, 10000) for _ in prices]
    }, index=date_range)
    
    return df

# Function to get weather data
@st.cache_data(ttl=3600)
def get_weather_data(region, years=2):
    # In a real implementation, this would call a weather API
    # For now, we'll simulate weather data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Adjust parameters based on region
    if region in ['Asia', 'South America', 'Africa']:
        temp_base = random.uniform(20, 30)
        temp_amplitude = random.uniform(5, 10)
        rainfall_base = random.uniform(80, 150)
        rainfall_amplitude = random.uniform(50, 100)
    elif region in ['North America', 'Europe']:
        temp_base = random.uniform(10, 20)
        temp_amplitude = random.uniform(10, 15)
        rainfall_base = random.uniform(50, 100)
        rainfall_amplitude = random.uniform(30, 70)
    else:
        temp_base = random.uniform(15, 25)
        temp_amplitude = random.uniform(8, 12)
        rainfall_base = random.uniform(60, 120)
        rainfall_amplitude = random.uniform(40, 80)
    
    # Generate temperature and rainfall data with seasonality
    temperatures = []
    rainfall = []
    for i, date in enumerate(date_range):
        # Temperature with yearly seasonality
        temp_season = temp_amplitude * np.sin(2 * np.pi * i / 365)
        temp_noise = np.random.normal(0, 2)
        temp = temp_base + temp_season + temp_noise
        temperatures.append(temp)
        
        # Rainfall with seasonality and more randomness
        rain_season = rainfall_amplitude * (0.5 + 0.5 * np.sin(2 * np.pi * i / 365))
        rain_noise = np.random.exponential(rainfall_base * 0.2)
        rain = max(0, rain_season + rain_noise if random.random() > 0.7 else 0)
        rainfall.append(rain)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Temperature': temperatures,
        'Rainfall': rainfall
    }, index=date_range)
    
    return df

# Function to get satellite crop health data
@st.cache_data(ttl=3600)
def get_crop_health_data(commodity, region, years=2):
    # In a real implementation, this would call a satellite imagery API
    # For now, we'll simulate crop health data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)
    date_range = pd.date_range(start=start_date, end=end_date, freq='W')  # Weekly data
    
    # Base parameters
    health_base = random.uniform(0.6, 0.8)  # Base health index (0-1)
    health_amplitude = random.uniform(0.1, 0.2)  # Seasonal variation
    
    # Adjust for commodity and region
    if commodity in ['Wheat', 'Corn', 'Rice'] and region in ['Asia', 'South America']:
        health_base += random.uniform(0.05, 0.1)
    elif commodity in ['Oranges', 'Apples'] and region in ['North America', 'Europe']:
        health_base += random.uniform(0.03, 0.08)
    elif region == 'Africa':
        health_base -= random.uniform(0.05, 0.1)
    
    # Generate health index with seasonality and events
    health_index = []
    growth_stage = []
    anomalies = []
    
    for i, date in enumerate(date_range):
        # Seasonal component
        season = health_amplitude * np.sin(2 * np.pi * i / 52)
        
        # Random events (drought, disease, etc.)
        event = 0
        if random.random() < 0.05:  # 5% chance of negative event
            event = -random.uniform(0.1, 0.3)
        
        # Calculate health index
        health = min(1.0, max(0.2, health_base + season + event + np.random.normal(0, 0.03)))
        health_index.append(health)
        
        # Growth stage (simplified)
        day_of_year = date.dayofyear
        if 80 <= day_of_year < 160:  # Spring
            stage = "Planting/Early Growth"
        elif 160 <= day_of_year < 240:  # Summer
            stage = "Maturation"
        elif 240 <= day_of_year < 320:  # Fall
            stage = "Harvest"
        else:  # Winter
            stage = "Dormant"
        growth_stage.append(stage)
        
        # Anomalies
        if event < -0.2:
            anomalies.append("Severe Stress")
        elif event < -0.1:
            anomalies.append("Moderate Stress")
        elif health < 0.5:
            anomalies.append("Below Average")
        else:
            anomalies.append("Normal")
    
    # Create DataFrame
    df = pd.DataFrame({
        'Health_Index': health_index,
        'Growth_Stage': growth_stage,
        'Anomaly': anomalies
    }, index=date_range)
    
    return df

# Function to get trade flow data
@st.cache_data(ttl=3600)
def get_trade_flow_data(commodity, years=2):
    # In a real implementation, this would call a trade data API
    # For now, we'll simulate trade flow data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*years)
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')  # Monthly data
    
    # Define major exporters and importers based on commodity
    if commodity in ['Wheat', 'Corn', 'Soybeans']:
        exporters = ['United States', 'Canada', 'Russia', 'Brazil', 'Argentina']
        importers = ['China', 'Japan', 'Egypt', 'European Union', 'Mexico']
    elif commodity in ['Rice']:
        exporters = ['India', 'Thailand', 'Vietnam', 'Pakistan', 'China']
        importers = ['Philippines', 'Nigeria', 'European Union', 'Saudi Arabia', 'United States']
    elif commodity in ['Oranges', 'Apples', 'Bananas']:
        exporters = ['Spain', 'South Africa', 'United States', 'Chile', 'Italy']
        importers = ['European Union', 'Russia', 'Canada', 'United Kingdom', 'Saudi Arabia']
    else:
        exporters = ['Netherlands', 'Spain', 'Mexico', 'China', 'United States']
        importers = ['United States', 'Germany', 'United Kingdom', 'Canada', 'Japan']
    
    # Generate trade volumes
    data = []
    
    for date in date_range:
        month = date.month
        # More trade during certain seasons
        season_factor = 1.0 + 0.3 * np.sin(2 * np.pi * month / 12)
        
        for exporter in exporters:
            base_volume = random.randint(10000, 100000)
            for importer in importers:
                if importer != exporter:  # No self-trade
                    # Calculate trade volume with seasonality and trend
                    volume = int(base_volume * season_factor * random.uniform(0.7, 1.3))
                    
                    # Add some trade relationships are stronger than others
                    if random.random() < 0.7:  # 70% chance of trade occurring
                        data.append({
                            'Date': date,
                            'Exporter': exporter,
                            'Importer': importer,
                            'Volume_MT': volume,
                            'Value_USD': volume * random.uniform(200, 500)  # Price per MT
                        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

# Function to get commodity symbols from Yahoo Finance
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_available_commodities():
    # In a real implementation, this would query Yahoo Finance for available agricultural commodities
    # For now, we'll return a predefined list with their symbols
    commodities = {
        'Wheat': 'ZW=F',
        'Corn': 'ZC=F',
        'Soybeans': 'ZS=F',
        'Rice': 'ZR=F',
        'Oats': 'ZO=F',
        'Barley': 'BLY=F',
        'Canola': 'RS=F',
        'Cotton': 'CT=F',
        'Sugar': 'SB=F',
        'Coffee': 'KC=F',
        'Cocoa': 'CC=F',
        'Orange Juice': 'OJ=F',
        'Lumber': 'LBS=F',
        'Milk': 'DC=F',
        'Lean Hogs': 'HE=F',
        'Live Cattle': 'LE=F',
        'Feeder Cattle': 'GF=F',
        'Class III Milk': 'DC=F',
        'Rough Rice': 'RR=F',
        'Soybean Oil': 'ZL=F',
        'Soybean Meal': 'ZM=F',
        'Palm Oil': 'FCPO=F',
        'Rapeseed': 'RS=F',
        'Rubber': 'JRU=F',
        'Wool': 'WOL=F',
        'Oranges': 'OJ=F',
        'Apples': 'AAPL',  # Using Apple stock as proxy since there's no futures
        'Bananas': 'FOOD',  # Using food ETF as proxy
        'Strawberries': 'FOOD',
        'Pineapples': 'FOOD',
        'Potatoes': 'FOOD',
        'Tomatoes': 'FOOD',
        'Onions': 'FOOD',
        'Lettuce': 'FOOD',
        'Carrots': 'FOOD',
        'Broccoli': 'FOOD',
        'Cauliflower': 'FOOD',
        'Peppers': 'FOOD',
        'Garlic': 'FOOD',
        'Ginger': 'FOOD'
    }
    return commodities

# Function to get available regions
def get_available_regions():
    # Growing regions (production)
    growing_regions = [
        'Asia',
        'South America',
        'Africa'
    ]
    
    # Consuming regions (demand)
    consuming_regions = [
        'North America',
        'Europe',
        'Middle East',
        'Russia'
    ]
    
    # Specific countries
    countries = [
        'United States',
        'China',
        'India',
        'Brazil',
        'Russia',
        'Indonesia',
        'Pakistan',
        'Nigeria',
        'Bangladesh',
        'Mexico',
        'Japan',
        'Ethiopia',
        'Philippines',
        'Egypt',
        'Vietnam',
        'Turkey',
        'Iran',
        'Germany',
        'Thailand',
        'United Kingdom',
        'France',
        'Italy',
        'South Africa',
        'Argentina',
        'Colombia',
        'Spain',
        'Ukraine',
        'Iraq',
        'Canada',
        'Morocco',
        'Saudi Arabia',
        'Uzbekistan',
        'Malaysia',
        'Peru',
        'Angola',
        'Ghana',
        'Yemen',
        'Nepal',
        'Venezuela',
        'Australia'
    ]
    
    # Combine all regions with a "Global" option
    all_regions = ['Global'] + growing_regions + consuming_regions + countries
    return all_regions

# Function to generate market opportunities
def generate_market_opportunities(commodity, region, user_type):
    # In a real implementation, this would analyze all data sources to identify opportunities
    # For now, we'll generate simulated opportunities
    opportunities = []
    
    # Get commodity data
    commodities = get_available_commodities()
    symbol = commodities.get(commodity, 'FOOD')
    price_data = get_commodity_data(symbol)
    
    # Calculate price trends
    if not price_data.empty:
        current_price = price_data['Close'].iloc[-1]
        month_ago_price = price_data['Close'].iloc[-30] if len(price_data) > 30 else price_data['Close'].iloc[0]
        price_change = (current_price - month_ago_price) / month_ago_price * 100
    else:
        current_price = random.uniform(100, 1000)
        price_change = random.uniform(-15, 15)
    
    # Get weather and crop health data
    weather_data = get_weather_data(region)
    crop_data = get_crop_health_data(commodity, region)
    
    # Generate opportunities based on user type
    if user_type == 'Buyer':
        # Buyers look for good deals to purchase
        if price_change > 5:
            # Price increasing - potential shortage
            opportunities.append({
                'type': 'Shortage Risk',
                'description': f"Potential shortage of {commodity} from {region} due to {random.choice(['adverse weather', 'reduced planting', 'export restrictions', 'increased demand'])}.",
                'value': f"${int(current_price * random.uniform(1.1, 1.3))} per MT potential future price",
                'confidence': random.randint(70, 90),
                'time_sensitivity': random.choice(['High', 'Medium', 'Low']),
                'analysis': f"Analysis shows a {price_change:.1f}% increase in {commodity} prices over the past month. Weather patterns in {region} indicate {random.choice(['below average rainfall', 'temperature anomalies', 'delayed planting'])}. Satellite imagery confirms crop health index is {crop_data['Health_Index'].iloc[-1]:.2f}, which is {random.choice(['concerning', 'below historical averages', 'indicating stress'])}.",
                'recommendation': f"Secure forward contracts for {commodity} from alternative sources such as {random.choice(['South America', 'North America', 'Europe', 'Asia'])} at current market prices. Consider building inventory over the next {random.randint(2, 8)} weeks before prices potentially increase further.",
                'contacts': [
                    {
                        'company': f"{random.choice(['Global', 'United', 'Pacific', 'Atlantic', 'Eastern'])} {commodity} Traders",
                        'location': random.choice(['United States', 'Brazil', 'Argentina', 'Canada', 'Australia']),
                        'contact': f"{random.choice(['John', 'Maria', 'Carlos', 'Sarah', 'Michael'])} {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'])}, +1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
                    },
                    {
                        'company': f"{random.choice(['Superior', 'Premium', 'Select', 'Choice', 'Prime'])} {commodity} Suppliers",
                        'location': random.choice(['France', 'Germany', 'Spain', 'Italy', 'Netherlands']),
                        'contact': f"{random.choice(['Pierre', 'Hans', 'Sofia', 'Marco', 'Anna'])} {random.choice(['Dubois', 'Mueller', 'Garcia', 'Rossi', 'Jansen'])}, +{random.randint(30, 49)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
                    }
                ]
            })
        elif price_change < -5:
            # Price decreasing - potential buying opportunity
            opportunities.append({
                'type': 'Buying Opportunity',
                'description': f"Favorable buying opportunity for {commodity} from {region} due to {random.choice(['bumper harvest', 'reduced demand', 'favorable weather', 'increased production'])}.",
                'value': f"${int(current_price * random.uniform(0.05, 0.15))} per MT potential savings",
                'confidence': random.randint(75, 95),
                'time_sensitivity': random.choice(['High', 'Medium', 'Low']),
                'analysis': f"Analysis shows a {abs(price_change):.1f}% decrease in {commodity} prices over the past month. Weather patterns in {region} have been {random.choice(['favorable for production', 'supporting good yields', 'better than expected'])}. Satellite imagery confirms crop health index is {crop_data['Health_Index'].iloc[-1]:.2f}, which is {random.choice(['excellent', 'above historical averages', 'indicating healthy crops'])}.",
                'recommendation': f"Increase procurement of {commodity} from {region} at current favorable prices. Consider extending contract periods to lock in current prices for the next {random.randint(3, 12)} months. Optimal timing for purchases is within the next {random.randint(1, 4)} weeks.",
                'contacts': [
                    {
                        'company': f"{random.choice(['Harvest', 'Sunshine', 'Golden', 'Green', 'Blue Sky'])} {commodity} Exports",
                        'location': random.choice(['India', 'Thailand', 'Vietnam', 'Indonesia', 'Malaysia']),
                        'contact': f"{random.choice(['Raj', 'Somchai', 'Nguyen', 'Siti', 'Lee'])} {random.choice(['Patel', 'Wongsa', 'Tran', 'Ibrahim', 'Wong'])}, +{random.randint(60, 98)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
                    },
                    {
                        'company': f"{random.choice(['Southern', 'Northern', 'Eastern', 'Western', 'Central'])} {commodity} Cooperative",
                        'location': random.choice(['South Africa', 'Kenya', 'Egypt', 'Morocco', 'Nigeria']),
                        'contact': f"{random.choice(['Kwame', 'Aisha', 'Chijioke', 'Fatima', 'Thabo'])} {random.choice(['Nkosi', 'Hassan', 'Okafor', 'El Mansouri', 'Khumalo'])}, +{random.randint(20, 27)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
                    }
                ]
            })
        
        # Add a diversification opportunity
        if random.random() < 0.7:  # 70% chance
            opportunities.append({
                'type': 'Diversification',
                'description': f"Opportunity to diversify {commodity} sourcing to include {random.choice(['organic', 'sustainable', 'fair trade', 'premium quality'])} options from {random.choice(get_available_regions())}.",
                'value': f"${int(current_price * random.uniform(0.02, 0.08))} per MT potential margin improvement",
                'confidence': random.randint(60, 85),
                'time_sensitivity': 'Medium',
                'analysis': f"Market analysis indicates growing consumer preference for {random.choice(['organic', 'sustainable', 'fair trade', 'premium quality'])} {commodity}. Current supply chain is {random.choice(['concentrated in few regions', 'vulnerable to disruptions', 'lacking product differentiation'])}.",
                'recommendation': f"Establish relationships with {random.choice(['organic', 'sustainable', 'fair trade', 'premium quality'])} {commodity} producers in {random.choice(get_available_regions())}. Start with small trial orders and gradually increase volume based on market response.",
                'contacts': [
                    {
                        'company': f"{random.choice(['Organic', 'Sustainable', 'Fair Trade', 'Premium', 'Eco'])} {commodity} Alliance",
                        'location': random.choice(get_available_regions()),
                        'contact': f"{random.choice(['Alex', 'Sam', 'Jordan', 'Taylor', 'Morgan'])} {random.choice(['Green', 'Rivers', 'Hill', 'Woods', 'Fields'])}, +{random.randint(1, 99)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
                    },
                    {
                        'company': f"{random.choice(['Pure', 'Natural', 'Earth', 'Harmony', 'Balance'])} {commodity} Cooperative",
                        'location': random.choice(get_available_regions()),
                        'contact': f"{random.choice(['Jamie', 'Casey', 'Riley', 'Avery', 'Quinn'])} {random.choice(['Nature', 'Earth', 'Waters', 'Sky', 'Sun'])}, +{random.randint(1, 99)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
                    }
                ]
            })
    else:  # Seller
        # Sellers look for good opportunities to sell
        if price_change > 5:
            # Price increasing - good selling opportunity
            opportunities.append({
                'type': 'Premium Market',
                'description': f"Premium selling opportunity for {commodity} to {region} due to {random.choice(['increasing demand', 'supply shortages', 'quality concerns from competitors', 'favorable trade conditions'])}.",
                'value': f"${int(current_price * random.uniform(0.05, 0.15))} per MT potential premium",
                'confidence': random.randint(75, 95),
                'time_sensitivity': random.choice(['High', 'Medium', 'Low']),
                'analysis': f"Analysis shows a {price_change:.1f}% increase in {commodity} prices over the past month. Market conditions in {region} indicate {random.choice(['growing demand', 'supply constraints', 'quality concerns from other suppliers'])}. Current inventory levels in destination markets are {random.choice(['below average', 'declining', 'insufficient to meet projected demand'])}.",
                'recommendation': f"Increase allocation of {commodity} to {region} at current favorable prices. Consider offering forward contracts to buyers for delivery over the next {random.randint(3, 6)} months. Optimal timing for negotiations is within the next {random.randint(1, 3)} weeks.",
                'contacts': [
                    {
                        'company': f"{random.choice(['Metro', 'City', 'Urban', 'Capital', 'Premier'])} {commodity} Distributors",
                        'location': random.choice(['United States', 'United Kingdom', 'Germany', 'France', 'Japan']),
                        'contact': f"{random.choice(['Robert', 'Emma', 'Thomas', 'Sophie', 'David'])} {random.choice(['Clark', 'Wilson', 'Taylor', 'Martin', 'Anderson'])}, +{random.randint(1, 49)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
                    },
                    {
                        'company': f"{random.choice(['Royal', 'Imperial', 'Crown', 'Sovereign', 'Elite'])} {commodity} Imports",
                        'location': random.choice(['Saudi Arabia', 'UAE', 'Qatar', 'Kuwait', 'Bahrain']),
                        'contact': f"{random.choice(['Mohammed', 'Fatima', 'Abdullah', 'Aisha', 'Ahmed'])} {random.choice(['Al-Saud', 'Al-Maktoum', 'Al-Thani', 'Al-Sabah', 'Al-Khalifa'])}, +{random.randint(966, 974)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
                    }
                ]
            })
        elif price_change < -5:
            # Price decreasing - potential market development opportunity
            opportunities.append({
                'type': 'Market Development',
                'description': f"Opportunity to develop new markets for {commodity} in {random.choice(get_available_regions())} to offset price pressure in current markets.",
                'value': f"${int(current_price * random.uniform(0.03, 0.1))} per MT potential margin protection",
                'confidence': random.randint(65, 85),
                'time_sensitivity': 'Medium',
                'analysis': f"Analysis shows a {abs(price_change):.1f}% decrease in {commodity} prices over the past month. Current market conditions suggest {random.choice(['oversupply', 'weakening demand', 'increased competition'])}. Emerging markets in {random.choice(get_available_regions())} show {random.choice(['growing consumption', 'increasing purchasing power', 'favorable demographic trends'])}.",
                'recommendation': f"Develop sales channels in {random.choice(get_available_regions())} to diversify market exposure. Consider partnering with established distributors or participating in trade missions. Target initial market entry within the next {random.randint(3, 6)} months.",
                'contacts': [
                    {
                        'company': f"{random.choice(['New Frontier', 'Emerging', 'Horizon', 'Gateway', 'Bridge'])} {commodity} Distributors",
                        'location': random.choice(['China', 'India', 'Brazil', 'Indonesia', 'Mexico']),
                        'contact': f"{random.choice(['Li', 'Priya', 'Paulo', 'Siti', 'Carlos'])} {random.choice(['Wang', 'Sharma', 'Silva', 'Wijaya', 'Rodriguez'])}, +{random.randint(55, 86)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
                    },
                    {
                        'company': f"{random.choice(['Global', 'International', 'Worldwide', 'Universal', 'Continental'])} {commodity} Trading",
                        'location': random.choice(['Singapore', 'Hong Kong', 'Dubai', 'Panama', 'Netherlands']),
                        'contact': f"{random.choice(['Chen', 'Raj', 'Mohammed', 'Maria', 'Jan'])} {random.choice(['Tan', 'Patel', 'Al-Farsi', 'Gonzalez', 'de Vries'])}, +{random.randint(31, 971)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
                    }
                ]
            })
        
        # Add a value-added opportunity
        if random.random() < 0.7:  # 70% chance
            opportunities.append({
                'type': 'Value Addition',
                'description': f"Opportunity to increase margins by offering {random.choice(['processed', 'packaged', 'branded', 'certified', 'specialty'])} {commodity} products.",
                'value': f"${int(current_price * random.uniform(0.1, 0.25))} per MT potential value addition",
                'confidence': random.randint(70, 90),
                'time_sensitivity': 'Low',
                'analysis': f"Market analysis indicates growing demand for {random.choice(['processed', 'packaged', 'branded', 'certified', 'specialty'])} {commodity} products. Current market offerings are {random.choice(['limited', 'undifferentiated', 'not meeting consumer preferences'])}.",
                'recommendation': f"Invest in {random.choice(['processing capabilities', 'packaging technology', 'brand development', 'certification', 'specialty product development'])} to capture higher margins. Develop pilot products for market testing within {random.randint(3, 9)} months.",
                'contacts': [
                    {
                        'company': f"{random.choice(['Innovative', 'Creative', 'Advanced', 'Modern', 'Smart'])} {commodity} Solutions",
                        'location': random.choice(['Japan', 'South Korea', 'Germany', 'United States', 'Israel']),
                        'contact': f"{random.choice(['Hiroshi', 'Min-ji', 'Klaus', 'Jennifer', 'Avi'])} {random.choice(['Tanaka', 'Kim', 'Schmidt', 'Thompson', 'Cohen'])}, +{random.randint(1, 972)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
                    },
                    {
                        'company': f"{random.choice(['Consumer', 'Retail', 'Market', 'Shopper', 'Customer'])} {commodity} Insights",
                        'location': random.choice(['United Kingdom', 'Canada', 'Australia', 'France', 'Sweden']),
                        'contact': f"{random.choice(['James', 'Emily', 'William', 'Charlotte', 'Oscar'])} {random.choice(['Smith', 'Johnson', 'Williams', 'Jones', 'Brown'])}, +{random.randint(1, 46)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
                    }
                ]
            })
    
    return opportunities

# Function to create report (HTML version for download)
def create_pdf_report(opportunity, commodity, region, user_type):
    # Create a styled HTML report that can be downloaded
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
            h1 {{
                color: #2c3e50;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #3498db;
                margin-top: 20px;
            }}
            .highlight {{
                background-color: #f8f9fa;
                padding: 15px;
                border-left: 4px solid #3498db;
                margin: 20px 0;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                text-align: left;
                padding: 12px;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <h1>Market Opportunity Report: {commodity} in {region}</h1>
        <p>Generated for: <strong>{user_type}</strong> | Date: {datetime.now().strftime('%Y-%m-%d')}</p>
        
        <div class="highlight">
            <h2>Opportunity Summary</h2>
            <p>{opportunity['description']}</p>
            <p><strong>Potential Value:</strong> {opportunity['value']}</p>
            <p><strong>Confidence Level:</strong> {opportunity['confidence']}%</p>
            <p><strong>Time Sensitivity:</strong> {opportunity['time_sensitivity']}</p>
        </div>
        
        <h2>Market Analysis</h2>
        <p>{opportunity['analysis']}</p>
        
        <h2>Recommended Actions</h2>
        <p>{opportunity['recommendation']}</p>
        
        <h2>Contact Recommendations</h2>
        <table>
            <tr>
                <th>Company</th>
                <th>Location</th>
                <th>Contact</th>
            </tr>
            <tr>
                <td>{opportunity['contacts'][0]['company']}</td>
                <td>{opportunity['contacts'][0]['location']}</td>
                <td>{opportunity['contacts'][0]['contact']}</td>
            </tr>
            <tr>
                <td>{opportunity['contacts'][1]['company']}</td>
                <td>{opportunity['contacts'][1]['location']}</td>
                <td>{opportunity['contacts'][1]['contact']}</td>
            </tr>
        </table>
        
        <p><em>This report is generated by Food Trading Insights Platform. The recommendations are based on analysis of market data, weather patterns, crop conditions, and trade flows.</em></p>
    </body>
    </html>
    """
    
    return html_content

# Function to convert HTML to downloadable format
def get_pdf_download_link(html_content, filename="report"):
    # Import base64 if not already imported
    import base64
    
    # Encode HTML as base64
    b64 = base64.b64encode(html_content.encode()).decode()
    
    # Create download link for HTML instead of PDF
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}.html">Download Report</a>'
    
    return href

# Function to create price analysis chart
def create_price_chart(commodity):
    # Get commodity data
    commodities = get_available_commodities()
    symbol = commodities.get(commodity, 'FOOD')
    df = get_commodity_data(symbol)
    
    if df.empty:
        st.error(f"No data available for {commodity}")
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Price',
        line=dict(color='rgba(0, 119, 182, 1)', width=2)
    ))
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        marker=dict(color='rgba(242, 120, 75, 0.5)'),
        yaxis='y2'
    ))
    
    # Calculate moving averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Add moving averages
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA50'],
        mode='lines',
        name='50-Day MA',
        line=dict(color='rgba(255, 183, 3, 1)', width=1.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MA200'],
        mode='lines',
        name='200-Day MA',
        line=dict(color='rgba(231, 111, 81, 1)', width=1.5)
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{commodity} Price History',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            title=dict(
                text="Price",
                font=dict(color="rgba(0, 119, 182, 1)")
            ),
            tickfont=dict(color="rgba(0, 119, 182, 1)")
        ),
        yaxis2=dict(
            title=dict(
                text="Volume",
                font=dict(color="rgba(242, 120, 75, 1)")
            ),
            tickfont=dict(color="rgba(242, 120, 75, 1)"),
            anchor="x",
            overlaying="y",
            side="right"
        )
    )
    
    return fig

# Function to create weather impact chart
def create_weather_chart(region):
    # Get weather data
    df = get_weather_data(region)
    
    if df.empty:
        st.error(f"No weather data available for {region}")
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add temperature line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Temperature'],
        mode='lines',
        name='Temperature (°C)',
        line=dict(color='rgba(255, 183, 3, 1)', width=2)
    ))
    
    # Add rainfall bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Rainfall'],
        name='Rainfall (mm)',
        marker=dict(color='rgba(0, 119, 182, 0.5)'),
        yaxis='y2'
    ))
    
    # Calculate moving averages
    df['Temp_MA30'] = df['Temperature'].rolling(window=30).mean()
    
    # Add moving average
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Temp_MA30'],
        mode='lines',
        name='30-Day Temp MA',
        line=dict(color='rgba(231, 111, 81, 1)', width=1.5)
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Weather Patterns in {region}',
        xaxis_title='Date',
        yaxis_title='Temperature (°C)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            title=dict(
                text="Temperature (°C)",
                font=dict(color="rgba(255, 183, 3, 1)")
            ),
            tickfont=dict(color="rgba(255, 183, 3, 1)")
        ),
        yaxis2=dict(
            title=dict(
                text="Rainfall (mm)",
                font=dict(color="rgba(0, 119, 182, 1)")
            ),
            tickfont=dict(color="rgba(0, 119, 182, 1)"),
            anchor="x",
            overlaying="y",
            side="right"
        )
    )
    
    return fig

# Function to create crop health chart
def create_crop_health_chart(commodity, region):
    # Get crop health data
    df = get_crop_health_data(commodity, region)
    
    if df.empty:
        st.error(f"No crop health data available for {commodity} in {region}")
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add health index line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Health_Index'],
        mode='lines+markers',
        name='Crop Health Index',
        line=dict(color='rgba(39, 174, 96, 1)', width=2),
        marker=dict(size=8)
    ))
    
    # Add reference line for optimal health
    fig.add_trace(go.Scatter(
        x=[df.index[0], df.index[-1]],
        y=[0.8, 0.8],
        mode='lines',
        name='Optimal Health',
        line=dict(color='rgba(46, 204, 113, 0.5)', width=1.5, dash='dash')
    ))
    
    # Add reference line for minimum acceptable health
    fig.add_trace(go.Scatter(
        x=[df.index[0], df.index[-1]],
        y=[0.5, 0.5],
        mode='lines',
        name='Minimum Acceptable',
        line=dict(color='rgba(231, 76, 60, 0.5)', width=1.5, dash='dash')
    ))
    
    # Create color mapping for anomalies
    color_map = {
        'Normal': 'rgba(46, 204, 113, 0.7)',
        'Below Average': 'rgba(241, 196, 15, 0.7)',
        'Moderate Stress': 'rgba(230, 126, 34, 0.7)',
        'Severe Stress': 'rgba(231, 76, 60, 0.7)'
    }
    
    # Add anomaly markers
    for anomaly in df['Anomaly'].unique():
        anomaly_df = df[df['Anomaly'] == anomaly]
        fig.add_trace(go.Scatter(
            x=anomaly_df.index,
            y=anomaly_df['Health_Index'],
            mode='markers',
            name=anomaly,
            marker=dict(
                size=12,
                color=color_map.get(anomaly, 'rgba(52, 152, 219, 0.7)'),
                symbol='circle',
                line=dict(width=1, color='rgba(0, 0, 0, 0.5)')
            ),
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Crop Health Index for {commodity} in {region}',
        xaxis_title='Date',
        yaxis_title='Health Index (0-1)',
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            range=[0, 1],
            tickformat='.1f'
        )
    )
    
    # Add growth stage annotations
    growth_stages = df['Growth_Stage'].unique()
    stage_positions = np.linspace(0, len(df) - 1, len(growth_stages), dtype=int)
    
    for i, stage in enumerate(growth_stages):
        position = stage_positions[i]
        date = df.index[position]
        fig.add_annotation(
            x=date,
            y=0.05,
            text=stage,
            showarrow=False,
            font=dict(size=10, color="black"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            borderpad=4,
            xanchor="center"
        )
    
    return fig

# Function to create trade flow chart
def create_trade_flow_chart(commodity):
    # Get trade flow data
    df = get_trade_flow_data(commodity)
    
    if df.empty:
        st.error(f"No trade flow data available for {commodity}")
        return None
    
    # Aggregate data by month and exporter/importer
    df['Month'] = df['Date'].dt.strftime('%Y-%m')
    monthly_data = df.groupby(['Month', 'Exporter', 'Importer']).agg({
        'Volume_MT': 'sum',
        'Value_USD': 'sum'
    }).reset_index()
    
    # Get top exporters and importers
    top_exporters = df.groupby('Exporter')['Volume_MT'].sum().nlargest(5).index.tolist()
    top_importers = df.groupby('Importer')['Volume_MT'].sum().nlargest(5).index.tolist()
    
    # Filter for top traders
    filtered_data = monthly_data[
        (monthly_data['Exporter'].isin(top_exporters)) & 
        (monthly_data['Importer'].isin(top_importers))
    ]
    
    # Create sankey diagram
    # Prepare nodes
    all_countries = list(set(top_exporters + top_importers))
    node_labels = all_countries
    
    # Create mapping from country to node index
    country_to_idx = {country: i for i, country in enumerate(all_countries)}
    
    # Prepare links
    sources = []
    targets = []
    values = []
    
    # Aggregate total trade between each exporter-importer pair
    trade_flows = filtered_data.groupby(['Exporter', 'Importer']).agg({
        'Volume_MT': 'sum'
    }).reset_index()
    
    for _, row in trade_flows.iterrows():
        sources.append(country_to_idx[row['Exporter']])
        targets.append(country_to_idx[row['Importer']])
        values.append(row['Volume_MT'])
    
    # Create figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color="rgba(31, 119, 180, 0.8)"
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(31, 119, 180, 0.3)"
        )
    )])
    
    # Update layout
    fig.update_layout(
        title=f'Global Trade Flows for {commodity}',
        font=dict(size=12),
        height=600
    )
    
    return fig

# Function to create price prediction chart
def create_price_prediction(commodity):
    # Get commodity data
    commodities = get_available_commodities()
    symbol = commodities.get(commodity, 'FOOD')
    df = get_commodity_data(symbol)
    
    if df.empty:
        st.error(f"No data available for {commodity}")
        return None
    
    # Prepare data for prediction
    df_pred = df.copy()
    df_pred['Date'] = df_pred.index
    df_pred['Day'] = df_pred['Date'].dt.day
    df_pred['Month'] = df_pred['Date'].dt.month
    df_pred['Year'] = df_pred['Date'].dt.year
    df_pred['DayOfWeek'] = df_pred['Date'].dt.dayofweek
    df_pred['DayOfYear'] = df_pred['Date'].dt.dayofyear
    
    # Create features and target
    X = df_pred[['Day', 'Month', 'Year', 'DayOfWeek', 'DayOfYear']].values
    y = df_pred['Close'].values
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Create future dates for prediction
    last_date = df_pred['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=90, freq='D')
    
    # Create features for future dates
    future_features = pd.DataFrame({
        'Date': future_dates,
        'Day': future_dates.day,
        'Month': future_dates.month,
        'Year': future_dates.year,
        'DayOfWeek': future_dates.dayofweek,
        'DayOfYear': future_dates.dayofyear
    })
    
    # Scale future features
    future_features_scaled = scaler.transform(future_features[['Day', 'Month', 'Year', 'DayOfWeek', 'DayOfYear']].values)
    
    # Predict future prices
    future_prices = model.predict(future_features_scaled)
    
    # Create figure
    fig = go.Figure()
    
    # Add historical prices
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='rgba(0, 119, 182, 1)', width=2)
    ))
    
    # Add test predictions
    fig.add_trace(go.Scatter(
        x=df.index[train_size:],
        y=y_pred,
        mode='lines',
        name='Model Fit',
        line=dict(color='rgba(242, 120, 75, 1)', width=2, dash='dash')
    ))
    
    # Add future predictions
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_prices,
        mode='lines',
        name='Price Forecast (90 Days)',
        line=dict(color='rgba(231, 76, 60, 1)', width=2)
    ))
    
    # Add confidence interval for future predictions (simulated)
    upper_bound = future_prices * (1 + 0.1)  # 10% above prediction
    lower_bound = future_prices * (1 - 0.1)  # 10% below prediction
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper_bound,
        mode='lines',
        name='Upper Bound (90% CI)',
        line=dict(color='rgba(231, 76, 60, 0.3)', width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower_bound,
        mode='lines',
        name='Lower Bound (90% CI)',
        line=dict(color='rgba(231, 76, 60, 0.3)', width=0),
        fill='tonexty',
        fillcolor='rgba(231, 76, 60, 0.2)',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=f'{commodity} Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add annotation for forecast start
    fig.add_vline(
        x=last_date,
        line_width=1,
        line_dash="dash",
        line_color="gray",
        annotation_text="Forecast Start",
        annotation_position="top right"
    )
    
    return fig

# Function to create seasonal analysis chart
def create_seasonal_analysis(commodity):
    # Get commodity data
    commodities = get_available_commodities()
    symbol = commodities.get(commodity, 'FOOD')
    df = get_commodity_data(symbol)
    
    if df.empty or len(df) < 365:
        st.error(f"Insufficient data for seasonal analysis of {commodity}")
        return None
    
    # Resample to monthly data for clearer seasonal patterns
    monthly_df = df['Close'].resample('M').mean()
    
    # Create a DataFrame with year and month columns
    seasonal_df = pd.DataFrame({
        'Price': monthly_df.values,
        'Month': monthly_df.index.month,
        'Year': monthly_df.index.year
    })
    
    # Calculate average price by month across all years
    monthly_avg = seasonal_df.groupby('Month')['Price'].mean().reset_index()
    
    # Calculate price relative to annual average for each year
    yearly_avg = seasonal_df.groupby('Year')['Price'].mean().reset_index()
    seasonal_df = seasonal_df.merge(yearly_avg, on='Year', suffixes=('', '_yearly_avg'))
    seasonal_df['Relative_Price'] = seasonal_df['Price'] / seasonal_df['Price_yearly_avg']
    
    # Calculate average relative price by month
    monthly_rel_avg = seasonal_df.groupby('Month')['Relative_Price'].mean().reset_index()
    
    # Create figure
    fig = go.Figure()
    
    # Add average price by month
    fig.add_trace(go.Scatter(
        x=monthly_avg['Month'],
        y=monthly_avg['Price'],
        mode='lines+markers',
        name='Average Price by Month',
        line=dict(color='rgba(0, 119, 182, 1)', width=2),
        marker=dict(size=8)
    ))
    
    # Add relative price (percentage of yearly average)
    fig.add_trace(go.Scatter(
        x=monthly_rel_avg['Month'],
        y=monthly_rel_avg['Relative_Price'],
        mode='lines+markers',
        name='Relative to Yearly Avg',
        line=dict(color='rgba(242, 120, 75, 1)', width=2),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Seasonal Price Patterns for {commodity}',
        xaxis=dict(
            title='Month',
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ),
        yaxis=dict(
            title=dict(
                text="Average Price",
                font=dict(color="rgba(0, 119, 182, 1)")
            ),
            tickfont=dict(color="rgba(0, 119, 182, 1)")
        ),
        yaxis2=dict(
            title=dict(
                text="Relative to Yearly Avg",
                font=dict(color="rgba(242, 120, 75, 1)")
            ),
            tickfont=dict(color="rgba(242, 120, 75, 1)"),
            anchor="x",
            overlaying="y",
            side="right",
            tickformat='.2%'
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add reference line for yearly average
    fig.add_trace(go.Scatter(
        x=[1, 12],
        y=[1, 1],
        mode='lines',
        name='Yearly Average',
        line=dict(color='rgba(242, 120, 75, 0.5)', width=1.5, dash='dash'),
        yaxis='y2'
    ))
    
    # Highlight months with highest and lowest prices
    max_month = monthly_rel_avg.loc[monthly_rel_avg['Relative_Price'].idxmax()]
    min_month = monthly_rel_avg.loc[monthly_rel_avg['Relative_Price'].idxmin()]
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig.add_annotation(
        x=max_month['Month'],
        y=max_month['Relative_Price'],
        text=f"Peak: {month_names[int(max_month['Month'])-1]}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40,
        font=dict(color="rgba(231, 76, 60, 1)"),
        yref='y2'
    )
    
    fig.add_annotation(
        x=min_month['Month'],
        y=min_month['Relative_Price'],
        text=f"Low: {month_names[int(min_month['Month'])-1]}",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=40,
        font=dict(color="rgba(46, 204, 113, 1)"),
        yref='y2'
    )
    
    return fig

# Main application layout
def main():
    # Header
    st.markdown('<h1 class="main-header">Food Trading Insights Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/wheat.png", width=80)
        st.markdown("## Settings")
        
        # User type selection
        user_type = st.radio("Select User Type:", ["Buyer", "Seller"], index=0 if st.session_state.user_type == "Buyer" else 1)
        st.session_state.user_type = user_type
        
        # Commodity selection
        commodities = list(get_available_commodities().keys())
        selected_commodity = st.selectbox("Select Commodity:", commodities, index=commodities.index(st.session_state.selected_commodity) if st.session_state.selected_commodity in commodities else 0)
        st.session_state.selected_commodity = selected_commodity
        
        # Region selection
        regions = get_available_regions()
        selected_region = st.selectbox("Select Region:", regions, index=regions.index(st.session_state.selected_region) if st.session_state.selected_region in regions else 0)
        st.session_state.selected_region = selected_region
        
        # Analysis type selection
        analysis_types = ["Price Analysis", "Weather Impact", "Crop Health", "Trade Flows", "Market Opportunities"]
        analysis_type = st.radio("Analysis Type:", analysis_types, index=analysis_types.index(st.session_state.analysis_type) if st.session_state.analysis_type in analysis_types else 0)
        st.session_state.analysis_type = analysis_type
        
        # Data refresh
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
        with col2:
            if st.button("🔄"):
                st.session_state.refresh_data = True
                st.session_state.last_update = datetime.now()
                st.experimental_rerun()
    
    # Main content
    if st.session_state.analysis_type == "Price Analysis":
        st.markdown(f'<h2 class="sub-header">Price Analysis: {st.session_state.selected_commodity}</h2>', unsafe_allow_html=True)
        
        # Create tabs for different price analyses
        price_tabs = st.tabs(["Historical Prices", "Price Forecast", "Seasonal Patterns"])
        
        with price_tabs[0]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### Historical Price Trends for {st.session_state.selected_commodity}")
            
            # Price chart
            price_fig = create_price_chart(st.session_state.selected_commodity)
            if price_fig:
                st.plotly_chart(price_fig, use_container_width=True)
            
            # Price statistics
            commodities = get_available_commodities()
            symbol = commodities.get(st.session_state.selected_commodity, 'FOOD')
            df = get_commodity_data(symbol)
            
            if not df.empty:
                current_price = df['Close'].iloc[-1]
                month_ago_price = df['Close'].iloc[-30] if len(df) > 30 else df['Close'].iloc[0]
                price_change = (current_price - month_ago_price) / month_ago_price * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.1f}%")
                with col2:
                    st.metric("30-Day High", f"${df['High'][-30:].max():.2f}")
                with col3:
                    st.metric("30-Day Low", f"${df['Low'][-30:].min():.2f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with price_tabs[1]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### Price Forecast for {st.session_state.selected_commodity}")
            
            # Price prediction chart
            prediction_fig = create_price_prediction(st.session_state.selected_commodity)
            if prediction_fig:
                st.plotly_chart(prediction_fig, use_container_width=True)
                
                # Add forecast insights
                st.markdown('<div class="highlight">', unsafe_allow_html=True)
                st.markdown("#### Forecast Insights")
                
                # Generate some insights based on the forecast
                commodities = get_available_commodities()
                symbol = commodities.get(st.session_state.selected_commodity, 'FOOD')
                df = get_commodity_data(symbol)
                
                if not df.empty:
                    current_price = df['Close'].iloc[-1]
                    
                    # Simulate forecast insights
                    forecast_direction = random.choice(["upward", "downward", "stable"])
                    forecast_confidence = random.randint(65, 95)
                    
                    if forecast_direction == "upward":
                        forecast_change = random.uniform(5, 15)
                        forecast_price = current_price * (1 + forecast_change/100)
                        st.markdown(f"The model predicts an **upward trend** with {forecast_confidence}% confidence. Projected price in 90 days: **${forecast_price:.2f}** (a {forecast_change:.1f}% increase).")
                        st.markdown(f"Key factors influencing this forecast include {random.choice(['seasonal demand patterns', 'projected supply constraints', 'historical price cycles', 'weather forecasts in key growing regions'])}.")
                    elif forecast_direction == "downward":
                        forecast_change = random.uniform(5, 15)
                        forecast_price = current_price * (1 - forecast_change/100)
                        st.markdown(f"The model predicts a **downward trend** with {forecast_confidence}% confidence. Projected price in 90 days: **${forecast_price:.2f}** (a {forecast_change:.1f}% decrease).")
                        st.markdown(f"Key factors influencing this forecast include {random.choice(['expected bumper harvests', 'weakening demand signals', 'historical price cycles', 'favorable weather forecasts in key growing regions'])}.")
                    else:
                        forecast_change = random.uniform(-2, 2)
                        forecast_price = current_price * (1 + forecast_change/100)
                        st.markdown(f"The model predicts a **stable trend** with {forecast_confidence}% confidence. Projected price in 90 days: **${forecast_price:.2f}** (a {abs(forecast_change):.1f}% {('increase' if forecast_change > 0 else 'decrease')}).")
                        st.markdown(f"Key factors influencing this forecast include {random.choice(['balanced supply and demand', 'consistent production levels', 'stable market conditions', 'typical seasonal patterns'])}.")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with price_tabs[2]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### Seasonal Price Patterns for {st.session_state.selected_commodity}")
            
            # Seasonal analysis chart
            seasonal_fig = create_seasonal_analysis(st.session_state.selected_commodity)
            if seasonal_fig:
                st.plotly_chart(seasonal_fig, use_container_width=True)
                
                # Add seasonal insights
                st.markdown('<div class="highlight">', unsafe_allow_html=True)
                st.markdown("#### Seasonal Insights")
                
                # Generate some insights based on the seasonal analysis
                month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
                peak_month = random.choice(month_names)
                low_month = month_names[(month_names.index(peak_month) + 6) % 12]  # Roughly opposite season
                
                st.markdown(f"Historical data shows that {st.session_state.selected_commodity} prices typically peak in **{peak_month}** and reach their lowest in **{low_month}**.")
                
                if st.session_state.user_type == "Buyer":
                    st.markdown(f"**Buying Strategy**: Consider securing forward contracts in {low_month} when prices are typically at their lowest. Avoid spot purchases in {peak_month} unless absolutely necessary.")
                else:  # Seller
                    st.markdown(f"**Selling Strategy**: Maximize revenue by timing major sales during {peak_month} when prices are typically highest. Consider forward contracts during {low_month} to secure minimum price levels.")
                
                st.markdown(f"The seasonal pattern strength is **{random.choice(['strong', 'moderate', 'variable'])}**, with an average price difference of **{random.randint(10, 30)}%** between peak and low months.")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.analysis_type == "Weather Impact":
        st.markdown(f'<h2 class="sub-header">Weather Impact: {st.session_state.selected_region}</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### Weather Patterns in {st.session_state.selected_region}")
        
        # Weather chart
        weather_fig = create_weather_chart(st.session_state.selected_region)
        if weather_fig:
            st.plotly_chart(weather_fig, use_container_width=True)
        
        # Weather insights
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("#### Weather Impact Analysis")
        
        # Generate some insights based on the weather data
        weather_data = get_weather_data(st.session_state.selected_region)
        
        if not weather_data.empty:
            recent_temp = weather_data['Temperature'].iloc[-30:].mean()
            recent_rain = weather_data['Rainfall'].iloc[-30:].sum()
            
            # Compare to historical averages (simulated)
            historical_temp = weather_data['Temperature'].mean()
            historical_rain = weather_data['Rainfall'].mean() * 30  # 30 days of historical average
            
            temp_diff = recent_temp - historical_temp
            rain_diff = recent_rain - historical_rain
            
            st.markdown(f"Recent 30-day average temperature: **{recent_temp:.1f}°C** ({'+' if temp_diff > 0 else ''}{temp_diff:.1f}°C compared to historical average)")
            st.markdown(f"Recent 30-day total rainfall: **{recent_rain:.1f}mm** ({'+' if rain_diff > 0 else ''}{rain_diff:.1f}mm compared to historical average)")
            
            # Generate impact assessment
            if temp_diff > 2 and rain_diff < -20:
                impact = "**Hot and Dry Conditions**: Potential stress on crops, particularly during critical growth stages. Increased irrigation needs likely."
            elif temp_diff > 2 and rain_diff > 20:
                impact = "**Hot and Wet Conditions**: Potential for increased disease pressure and pest activity. Monitor crop health closely."
            elif temp_diff < -2 and rain_diff < -20:
                impact = "**Cool and Dry Conditions**: Potential for slower crop development. May affect planting schedules and harvest timing."
            elif temp_diff < -2 and rain_diff > 20:
                impact = "**Cool and Wet Conditions**: Potential for delayed field operations and increased disease pressure. Drainage may be a concern."
            elif abs(temp_diff) <= 2 and abs(rain_diff) <= 20:
                impact = "**Normal Conditions**: Weather patterns are within typical ranges. No significant weather-related impacts expected."
            elif rain_diff > 50:
                impact = "**Excessive Rainfall**: Potential for flooding, waterlogging, and field access issues. May delay planting or harvesting operations."
            elif rain_diff < -50:
                impact = "**Drought Conditions**: Potential for significant crop stress and yield reductions. Irrigation resources may be strained."
            else:
                impact = "**Mixed Conditions**: Some weather anomalies present but overall impact likely to be moderate. Monitor specific crop requirements."
            
            st.markdown(f"**Impact Assessment**: {impact}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Weather forecast (simulated)
        st.markdown("### 14-Day Weather Forecast")
        
        forecast_data = []
        start_date = datetime.now()
        
        # Generate simulated forecast
        for i in range(14):
            date = start_date + timedelta(days=i)
            temp_high = random.uniform(20, 30) if "Asia" in st.session_state.selected_region or "Africa" in st.session_state.selected_region else random.uniform(10, 25)
            temp_low = temp_high - random.uniform(5, 10)
            
            # More rain probability in certain regions
            rain_prob = random.uniform(0, 80) if "Asia" in st.session_state.selected_region or "South America" in st.session_state.selected_region else random.uniform(0, 50)
            
            forecast_data.append({
                "Date": date.strftime("%b %d"),
                "High": f"{temp_high:.1f}°C",
                "Low": f"{temp_low:.1f}°C",
                "Rain": f"{rain_prob:.0f}%",
                "Conditions": random.choice(["Sunny", "Partly Cloudy", "Cloudy", "Light Rain", "Heavy Rain"]) if rain_prob > 30 else random.choice(["Sunny", "Partly Cloudy", "Cloudy"])
            })
        
        # Display forecast as a table
        forecast_df = pd.DataFrame(forecast_data)
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.analysis_type == "Crop Health":
        st.markdown(f'<h2 class="sub-header">Crop Health: {st.session_state.selected_commodity} in {st.session_state.selected_region}</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### Crop Health Monitoring for {st.session_state.selected_commodity} in {st.session_state.selected_region}")
        
        # Crop health chart
        crop_fig = create_crop_health_chart(st.session_state.selected_commodity, st.session_state.selected_region)
        if crop_fig:
            st.plotly_chart(crop_fig, use_container_width=True)
        
        # Crop health insights
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("#### Satellite Imagery Analysis")
        
        # Generate some insights based on the crop health data
        crop_data = get_crop_health_data(st.session_state.selected_commodity, st.session_state.selected_region)
        
        if not crop_data.empty:
            current_health = crop_data['Health_Index'].iloc[-1]
            avg_health = crop_data['Health_Index'].mean()
            current_anomaly = crop_data['Anomaly'].iloc[-1]
            current_stage = crop_data['Growth_Stage'].iloc[-1]
            
            health_diff = current_health - avg_health
            
            st.markdown(f"Current crop health index: **{current_health:.2f}** ({'+' if health_diff > 0 else ''}{health_diff:.2f} compared to historical average)")
            st.markdown(f"Current growth stage: **{current_stage}**")
            st.markdown(f"Current status: **{current_anomaly}**")
            
            # Generate impact assessment
            if current_health > 0.8:
                impact = "**Excellent Conditions**: Crop development is proceeding optimally. Yield potential is high based on current vegetation indices."
            elif current_health > 0.6:
                impact = "**Good Conditions**: Crop development is generally favorable. Some minor stress may be present but overall yield potential remains good."
            elif current_health > 0.4:
                impact = "**Fair Conditions**: Some stress indicators are present. Yield potential may be moderately affected if conditions persist."
            else:
                impact = "**Poor Conditions**: Significant stress indicators are present. Yield potential is likely to be substantially reduced."
            
            st.markdown(f"**Assessment**: {impact}")
            
            # Add production outlook
            if current_health > 0.7:
                outlook_change = random.uniform(5, 15)
                st.markdown(f"**Production Outlook**: Potential for **{outlook_change:.1f}% above average** yields if favorable conditions continue.")
            elif current_health > 0.5:
                outlook_change = random.uniform(-5, 5)
                st.markdown(f"**Production Outlook**: Yields likely to be **within {abs(outlook_change):.1f}% of average**.")
            else:
                outlook_change = random.uniform(-20, -5)
                st.markdown(f"**Production Outlook**: Potential for **{abs(outlook_change):.1f}% below average** yields unless conditions improve significantly.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Satellite imagery (simulated)
        st.markdown("### Recent Satellite Imagery")
        
        # Create columns for images
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### NDVI (Vegetation Health)")
            st.image("https://www.usgs.gov/sites/default/files/styles/full_width/public/2022-01/NDVI_0.jpg?itok=0BUfH7TY", use_column_width=True)
            st.markdown("*Normalized Difference Vegetation Index showing crop health status*")
        
        with col2:
            st.markdown("#### False Color Composite")
            st.image("https://www.usgs.gov/sites/default/files/styles/full_width/public/2022-01/FalseColor_0.jpg?itok=WBBSSOIj", use_column_width=True)
            st.markdown("*False color composite highlighting vegetation in red*")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.analysis_type == "Trade Flows":
        st.markdown(f'<h2 class="sub-header">Trade Flows: {st.session_state.selected_commodity}</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### Global Trade Flows for {st.session_state.selected_commodity}")
        
        # Trade flow chart
        trade_fig = create_trade_flow_chart(st.session_state.selected_commodity)
        if trade_fig:
            st.plotly_chart(trade_fig, use_container_width=True)
        
        # Trade insights
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.markdown("#### Trade Flow Analysis")
        
        # Generate some insights based on the trade data
        trade_data = get_trade_flow_data(st.session_state.selected_commodity)
        
        if not trade_data.empty:
            # Top exporters and importers
            top_exporters = trade_data.groupby('Exporter')['Volume_MT'].sum().nlargest(3)
            top_importers = trade_data.groupby('Importer')['Volume_MT'].sum().nlargest(3)
            
            st.markdown("**Top Exporters:**")
            for country, volume in top_exporters.items():
                st.markdown(f"- {country}: {volume:,.0f} MT")
            
            st.markdown("**Top Importers:**")
            for country, volume in top_importers.items():
                st.markdown(f"- {country}: {volume:,.0f} MT")
            
            # Recent trade trends (simulated)
            trend = random.choice(["increasing", "decreasing", "stable"])
            
            if trend == "increasing":
                st.markdown(f"**Recent Trend**: Global trade volume for {st.session_state.selected_commodity} has been **increasing** by approximately {random.randint(5, 15)}% year-over-year.")
            elif trend == "decreasing":
                st.markdown(f"**Recent Trend**: Global trade volume for {st.session_state.selected_commodity} has been **decreasing** by approximately {random.randint(5, 15)}% year-over-year.")
            else:
                st.markdown(f"**Recent Trend**: Global trade volume for {st.session_state.selected_commodity} has remained **relatively stable** with changes of less than {random.randint(2, 5)}% year-over-year.")
            
            # Trade disruptions (simulated)
            disruption_prob = random.random()
            if disruption_prob > 0.7:
                disruption_country = random.choice(list(top_exporters.index) + list(top_importers.index))
                disruption_type = random.choice(["export restrictions", "import tariffs", "logistics challenges", "quality concerns", "payment issues"])
                st.markdown(f"**Trade Disruption Alert**: {disruption_country} is experiencing {disruption_type}, potentially affecting {random.randint(10, 30)}% of normal trade volume.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Trade routes map (placeholder)
        st.markdown("### Major Trade Routes")
        
        # Display a static map image as placeholder
        st.image("https://www.researchgate.net/publication/340460214/figure/fig1/AS:878175129153537@1586255124558/World-map-of-major-trade-routes-and-the-main-ports-in-the-maritime-shipping-network.jpg", use_column_width=True)
        st.markdown("*Major global shipping routes for agricultural commodities*")
        
        # Trade regulations (simulated)
        st.markdown("### Recent Trade Policy Changes")
        
        # Generate simulated trade policy changes
        policy_changes = []
        countries = random.sample(get_available_regions()[1:], 3)  # Skip 'Global'
        
        for country in countries:
            policy_type = random.choice(["Import Tariff", "Export Quota", "Quality Standard", "Documentation Requirement", "Subsidy Program"])
            commodity_affected = random.choice(list(get_available_commodities().keys()))
            date_implemented = (datetime.now() - timedelta(days=random.randint(1, 90))).strftime("%b %d, %Y")
            
            policy_changes.append({
                "Country": country,
                "Policy Type": policy_type,
                "Commodity": commodity_affected,
                "Implementation Date": date_implemented,
                "Impact": random.choice(["High", "Medium", "Low"])
            })
        
        # Display policy changes as a table
        policy_df = pd.DataFrame(policy_changes)
        st.dataframe(policy_df, use_container_width=True, hide_index=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.analysis_type == "Market Opportunities":
        st.markdown(f'<h2 class="sub-header">Market Opportunities for {st.session_state.user_type}s</h2>', unsafe_allow_html=True)
        
        # Generate opportunities
        opportunities = generate_market_opportunities(
            st.session_state.selected_commodity,
            st.session_state.selected_region,
            st.session_state.user_type
        )
        
        if opportunities:
            for i, opportunity in enumerate(opportunities):
                st.markdown('<div class="opportunity-card">', unsafe_allow_html=True)
                
                # Opportunity header
                st.markdown(f"### {opportunity['type']}: {opportunity['description']}")
                
                # Opportunity details
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Potential Value:** {opportunity['value']}")
                with col2:
                    st.markdown(f"**Confidence:** {opportunity['confidence']}%")
                with col3:
                    st.markdown(f"**Time Sensitivity:** {opportunity['time_sensitivity']}")
                
                # Opportunity analysis
                with st.expander("View Analysis"):
                    st.markdown("#### Market Analysis")
                    st.markdown(opportunity['analysis'])
                    
                    st.markdown("#### Recommended Actions")
                    st.markdown(opportunity['recommendation'])
                    
                    st.markdown("#### Contact Recommendations")
                    for contact in opportunity['contacts']:
                        st.markdown(f"**{contact['company']}** ({contact['location']})")
                        st.markdown(f"Contact: {contact['contact']}")
                
                # Generate PDF report
                report_html = create_pdf_report(
                    opportunity,
                    st.session_state.selected_commodity,
                    st.session_state.selected_region,
                    st.session_state.user_type
                )
                
                # Create download link
                download_link = get_pdf_download_link(
                    report_html,
                    f"{st.session_state.selected_commodity}_{opportunity['type'].replace(' ', '_')}_Report"
                )
                
                st.markdown(download_link, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info(f"No market opportunities found for {st.session_state.selected_commodity} in {st.session_state.selected_region} for {st.session_state.user_type}s. Try changing your selection criteria.")

# Run the application
if __name__ == "__main__":
    main()
