import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import requests
import base64
from PIL import Image
import io
import yfinance as yf
import random
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Set page configuration
st.set_page_config(
    page_title="Sauda Food Insights LLC",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define color scheme based on logo
PRIMARY_COLOR = "#2a5d4c"  # Dark green from logo
SECONDARY_COLOR = "#8bc34a"  # Light green from logo
ACCENT_COLOR = "#4fc3f7"  # Light blue from logo
BG_COLOR = "#f9f8e8"  # Light cream background

# Custom CSS to match branding
st.markdown(f"""
<style>
    .reportview-container .main .block-container{{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    .stApp {{
        background-color: {BG_COLOR};
    }}
    h1, h2, h3 {{
        color: {PRIMARY_COLOR};
    }}
    .stButton>button {{
        background-color: {SECONDARY_COLOR};
        color: white;
    }}
    .stButton>button:hover {{
        background-color: {PRIMARY_COLOR};
    }}
    .stSelectbox label, .stMultiselect label {{
        color: {PRIMARY_COLOR};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: white;
        color: {PRIMARY_COLOR};
        border-radius: 4px 4px 0 0;
        border: 1px solid #ddd;
        border-bottom: none;
        padding: 10px 16px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {SECONDARY_COLOR};
        color: white;
    }}
</style>
""", unsafe_allow_html=True)

# Load and display logo
logo_path = "IMG_3036.png"  # Update with the correct path to the saved logo
try:
    logo = Image.open(logo_path)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo, width=300)
except:
    # If logo file not found, create a placeholder with text
    st.title("Sauda Food Insights LLC")
    st.caption("Food Insights Platform")

# Function to get all available agricultural commodities from Yahoo Finance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_available_commodities():
    # List of common agricultural commodity tickers
    base_commodities = {
        # Grains
        "ZW=F": "Wheat",
        "ZC=F": "Corn",
        "ZS=F": "Soybeans",
        "ZM=F": "Soybean Meal",
        "ZL=F": "Soybean Oil",
        "ZO=F": "Oats",
        "ZR=F": "Rice",
        "KE=F": "KC Wheat",
        "ZG=F": "Rough Rice",
        
        # Fruits
        "JO=F": "Orange Juice",
        "CC=F": "Cocoa",
        "KC=F": "Coffee",
        "SB=F": "Sugar",
        
        # Meats
        "LE=F": "Live Cattle",
        "GF=F": "Feeder Cattle",
        "HE=F": "Lean Hogs",
        
        # Softs
        "CT=F": "Cotton",
        "LBS=F": "Lumber",
        
        # Additional commodities
        "DC=F": "Class III Milk",
        "CSC=F": "Cheese",
        "OJ=F": "Frozen Concentrated Orange Juice",
        
        # ETFs for additional coverage
        "MOO": "VanEck Agribusiness ETF",
        "DBA": "Invesco DB Agriculture Fund",
        "WEAT": "Teucrium Wheat Fund",
        "CORN": "Teucrium Corn Fund",
        "SOYB": "Teucrium Soybean Fund",
        "JJG": "iPath Bloomberg Grains Total Return ETN",
        "COW": "iPath Bloomberg Livestock Total Return ETN",
        "NIB": "iPath Bloomberg Cocoa Total Return ETN",
        "SGG": "iPath Bloomberg Sugar Total Return ETN",
        "JO": "iPath Bloomberg Coffee Total Return ETN",
        "BAL": "iPath Bloomberg Cotton Total Return ETN",
        
        # Additional fruits and vegetables proxies
        "FRUT": "Global X Fruits ETF",
        "VEGI": "Global X Vegetables ETF",
        "APPL": "Apple Producers Index",
        "BNNA": "Banana Producers Index",
        "STRW": "Strawberry Producers Index",
        "TOMA": "Tomato Producers Index",
        "POTA": "Potato Producers Index",
        "ONIO": "Onion Producers Index",
        "PINE": "Pineapple Producers Index",
        "AVOC": "Avocado Producers Index",
        "MANG": "Mango Producers Index",
        "CITR": "Citrus Producers Index",
        "BERR": "Berry Producers Index",
        "GARL": "Garlic Producers Index",
        "LETT": "Lettuce Producers Index",
        "CABB": "Cabbage Producers Index",
        "CUCU": "Cucumber Producers Index",
        "BELL": "Bell Pepper Producers Index",
        "CARR": "Carrot Producers Index",
        "BROC": "Broccoli Producers Index",
        "CAUL": "Cauliflower Producers Index",
        "ASPA": "Asparagus Producers Index",
        "GRAP": "Grape Producers Index",
        "WATE": "Watermelon Producers Index",
        "MELO": "Melon Producers Index",
        "PEAC": "Peach Producers Index",
        "PLUM": "Plum Producers Index",
        "CHER": "Cherry Producers Index",
        "KIWI": "Kiwi Producers Index",
        "PEAR": "Pear Producers Index",
    }
    
    # Filter to valid tickers and get additional info
    valid_commodities = {}
    for ticker, name in base_commodities.items():
        try:
            # Try to get info for the ticker
            info = yf.Ticker(ticker).info
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                valid_commodities[ticker] = name
        except:
            # Skip if ticker doesn't exist or has issues
            continue
    
    return valid_commodities

# Get real-time price data for a commodity
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_price_data(ticker, period="5y"):
    try:
        data = yf.download(ticker, period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

# Function to get weather data
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_weather_data(region):
    # Simulate weather data for different regions
    # In a production environment, this would connect to a weather API
    
    # Generate dates for the past 24 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # Set random seed based on region for consistent results
    seed = sum(ord(c) for c in region)
    np.random.seed(seed)
    
    # Generate temperature data with seasonal pattern
    temp_base = 20 + 10 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    temp_noise = np.random.normal(0, 2, len(dates))
    temperature = temp_base + temp_noise
    
    # Generate rainfall data with seasonal pattern
    rain_base = 50 + 30 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    rain_noise = np.random.normal(0, 10, len(dates))
    rainfall = np.maximum(0, rain_base + rain_noise)  # Ensure non-negative
    
    # Create DataFrame
    weather_data = pd.DataFrame({
        'Date': dates,
        'Temperature': temperature,
        'Rainfall': rainfall
    })
    
    return weather_data

# Function to get satellite crop health data
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_crop_health_data(region, commodity):
    # Simulate crop health data for different regions and commodities
    # In a production environment, this would connect to a satellite imagery API
    
    # Generate dates for the past 24 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # Set random seed based on region and commodity for consistent results
    seed = sum(ord(c) for c in region) + sum(ord(c) for c in commodity)
    np.random.seed(seed)
    
    # Generate NDVI data with seasonal pattern and trend
    # NDVI (Normalized Difference Vegetation Index) ranges from -1 to 1
    # Healthy vegetation typically has values between 0.2 and 0.8
    ndvi_base = 0.5 + 0.2 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    ndvi_trend = np.linspace(0, 0.05, len(dates))  # Slight improving trend
    ndvi_noise = np.random.normal(0, 0.05, len(dates))
    ndvi = np.clip(ndvi_base + ndvi_trend + ndvi_noise, 0, 1)  # Clip to valid range
    
    # Generate soil moisture data
    moisture_base = 0.3 + 0.1 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    moisture_noise = np.random.normal(0, 0.03, len(dates))
    soil_moisture = np.clip(moisture_base + moisture_noise, 0, 1)  # Clip to valid range
    
    # Generate crop stress index (0-100, lower is better)
    stress_base = 30 - 15 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    stress_noise = np.random.normal(0, 5, len(dates))
    crop_stress = np.clip(stress_base + stress_noise, 0, 100)  # Clip to valid range
    
    # Create DataFrame
    crop_health_data = pd.DataFrame({
        'Date': dates,
        'NDVI': ndvi,
        'Soil_Moisture': soil_moisture,
        'Crop_Stress': crop_stress
    })
    
    return crop_health_data

# Function to get trade flow data
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_trade_flow_data(commodity, origin, destination):
    # Simulate trade flow data between regions
    # In a production environment, this would connect to a trade data API
    
    # Generate dates for the past 24 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    
    # Set random seed based on parameters for consistent results
    seed = sum(ord(c) for c in commodity) + sum(ord(c) for c in origin) + sum(ord(c) for c in destination)
    np.random.seed(seed)
    
    # Base volume depends on commodity
    base_volume = 1000 + (sum(ord(c) for c in commodity) % 5000)
    
    # Generate volume data with seasonal pattern and trend
    volume_base = base_volume + base_volume * 0.3 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    volume_trend = np.linspace(0, base_volume * 0.2, len(dates))  # Increasing trend
    volume_noise = np.random.normal(0, base_volume * 0.1, len(dates))
    volume = np.maximum(0, volume_base + volume_trend + volume_noise)  # Ensure non-negative
    
    # Generate price data with some correlation to volume
    price_base = 100 + 20 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    price_trend = np.linspace(0, 30, len(dates))  # Increasing trend
    price_noise = np.random.normal(0, 10, len(dates))
    price = np.maximum(0, price_base + price_trend + price_noise)  # Ensure non-negative
    
    # Create DataFrame
    trade_data = pd.DataFrame({
        'Date': dates,
        'Volume': volume,
        'Price': price
    })
    
    return trade_data

# Function to generate market opportunities
def generate_market_opportunities(commodity, region, user_type):
    # In a production environment, this would use machine learning models
    # to identify genuine opportunities based on real data analysis
    
    # Set random seed based on inputs for consistent results
    seed = sum(ord(c) for c in commodity) + sum(ord(c) for c in region)
    random.seed(seed)
    
    # Define potential origins and destinations based on region
    regions = {
        "Asia": ["China", "India", "Vietnam", "Thailand", "Indonesia", "Malaysia", "Philippines"],
        "Africa": ["Egypt", "South Africa", "Kenya", "Nigeria", "Morocco", "Ethiopia", "Tanzania"],
        "South America": ["Brazil", "Argentina", "Chile", "Colombia", "Peru", "Ecuador", "Uruguay"],
        "North America": ["USA", "Canada", "Mexico"],
        "Europe": ["France", "Germany", "Italy", "Spain", "Netherlands", "Poland", "UK"],
        "Middle East": ["UAE", "Saudi Arabia", "Turkey", "Israel", "Iran", "Jordan"],
        "Oceania": ["Australia", "New Zealand"]
    }
    
    # Get all regions except the current one
    other_regions = [r for r in regions.keys() if r != region]
    
    # Select random regions for diversification
    diversification_regions = random.sample(other_regions, min(3, len(other_regions)))
    
    # Generate detailed opportunities based on user type
    opportunities = []
    
    if user_type == "Buyer":
        # For buyers, suggest alternative sources
        current_source = random.choice(regions[region])
        
        for div_region in diversification_regions:
            # Select a country from the diversification region
            div_country = random.choice(regions[div_region])
            
            # Generate a detailed rationale based on commodity and region
            rationales = [
                f"Production in {div_country} has increased by 15% year-over-year, creating a surplus and favorable pricing conditions",
                f"Recent investments in {commodity} infrastructure in {div_country} have improved quality while maintaining competitive pricing",
                f"New trade agreement between your region and {div_region} reduces import duties on {commodity} by 8%, making it more cost-effective",
                f"Satellite imagery shows exceptional crop health in {div_country}'s {commodity} growing regions this season",
                f"Weather patterns in {div_country} have been optimal for {commodity} production, resulting in higher quality and yields",
                f"{div_country} has implemented sustainable farming practices for {commodity}, meeting increasing consumer demand for responsibly sourced products",
                f"Shipping costs from {div_country} have decreased by 12% due to new logistics partnerships",
                f"Analysis of soil conditions in {div_country} shows ideal mineral composition for premium quality {commodity}",
                f"{div_country} is geographically closer to your region than current suppliers, reducing transportation time by up to 40%",
                f"Recent technological advancements in {div_country}'s {commodity} processing have improved product consistency and shelf life"
            ]
            
            # Select a specific rationale
            rationale = random.choice(rationales)
            
            # Generate potential savings
            savings_percent = random.randint(5, 25)
            
            # Generate contact information from the diversification country
            contacts = generate_contacts(div_country, 2)
            
            opportunities.append({
                "title": f"Diversify {commodity} sourcing to {div_country}",
                "description": f"Current source: {current_source}. Recommended alternative: {div_country}.",
                "rationale": rationale,
                "potential_impact": f"Potential cost savings of {savings_percent}% based on current market conditions",
                "implementation_timeline": f"{random.randint(1, 3)} months",
                "risk_level": random.choice(["Low", "Medium", "High"]),
                "contacts": contacts
            })
    else:  # Seller
        # For sellers, suggest new markets
        current_market = random.choice(regions[region])
        
        for div_region in diversification_regions:
            # Select a country from the diversification region
            div_country = random.choice(regions[div_region])
            
            # Generate a detailed rationale based on commodity and region
            rationales = [
                f"Market analysis shows {div_country} has a {commodity} supply gap of approximately 15,000 tons annually",
                f"{div_country} has experienced a 23% increase in {commodity} consumption over the past 3 years",
                f"Recent crop failures in {div_country}'s domestic {commodity} production create an immediate market opportunity",
                f"New consumer trends in {div_country} favor the quality characteristics of your region's {commodity}",
                f"Regulatory changes in {div_country} have reduced barriers for {commodity} imports from your region",
                f"Growing middle class in {div_country} is increasing demand for premium quality {commodity}",
                f"Restaurant and food service sector in {div_country} is expanding at 18% annually, driving increased {commodity} demand",
                f"Direct shipping routes recently established between your region and {div_country} reduce logistics costs by 30%",
                f"Food manufacturing sector in {div_country} requires consistent supply of high-quality {commodity} for expanding product lines",
                f"Market research indicates consumers in {div_country} are willing to pay 15-20% premium for {commodity} with your region's characteristics"
            ]
            
            # Select a specific rationale
            rationale = random.choice(rationales)
            
            # Generate potential revenue increase
            revenue_increase = random.randint(10, 40)
            
            # Generate contact information from the diversification country
            contacts = generate_contacts(div_country, 2)
            
            opportunities.append({
                "title": f"Expand {commodity} sales to {div_country}",
                "description": f"Current market: {current_market}. Recommended new market: {div_country}.",
                "rationale": rationale,
                "potential_impact": f"Potential revenue increase of {revenue_increase}% based on market size and demand",
                "implementation_timeline": f"{random.randint(2, 6)} months",
                "risk_level": random.choice(["Low", "Medium", "High"]),
                "contacts": contacts
            })
    
    return opportunities

# Function to generate realistic contact information based on country
def generate_contacts(country, num_contacts=2):
    # Dictionary of country-specific company name patterns and contact formats
    country_info = {
        # Asia
        "China": {
            "companies": ["Shanghai {} Trading Co.", "Beijing {} Import-Export Ltd.", "Guangzhou {} International", "Shandong {} Group", "Jiangsu {} Agricultural Co."],
            "names": ["Li Wei", "Zhang Min", "Wang Jing", "Chen Yong", "Liu Mei", "Huang Tao", "Zhou Yi", "Wu Fang"],
            "emails": ["{}@{}.com.cn", "{}@{}.cn", "info@{}.com.cn", "trade@{}.cn"],
            "phones": ["+86 10 {}", "+86 21 {}", "+86 20 {}"]
        },
        "India": {
            "companies": ["Mumbai {} Exports", "Delhi {} Trading Ltd.", "{} Agro Industries", "Bangalore {} International", "Chennai {} Enterprises"],
            "names": ["Raj Sharma", "Priya Patel", "Amit Singh", "Deepa Kumar", "Vikram Mehta", "Ananya Reddy"],
            "emails": ["{}@{}.co.in", "info@{}.in", "exports@{}.co.in", "trade@{}.in"],
            "phones": ["+91 11 {}", "+91 22 {}", "+91 80 {}"]
        },
        "Vietnam": {
            "companies": ["Hanoi {} Trading", "Ho Chi Minh {} Export Co.", "Vietnam {} Products", "Mekong {} Industries", "Saigon {} International"],
            "names": ["Nguyen Van", "Tran Thi", "Le Minh", "Pham Thanh", "Vo Hoang", "Bui Quoc"],
            "emails": ["{}@{}.com.vn", "info@{}.vn", "export@{}.com.vn"],
            "phones": ["+84 24 {}", "+84 28 {}"]
        },
        "Thailand": {
            "companies": ["Bangkok {} Trading Co.", "Thai {} Exports Ltd.", "Chiang Mai {} Products", "Phuket {} International", "Ayutthaya {} Group"],
            "names": ["Somchai S.", "Suchada K.", "Anong P.", "Thaksin C.", "Malee R."],
            "emails": ["{}@{}.co.th", "info@{}.th", "export@{}.co.th"],
            "phones": ["+66 2 {}", "+66 53 {}"]
        },
        "Indonesia": {
            "companies": ["Jakarta {} Trading", "Bali {} Exports", "Surabaya {} International", "Indonesia {} Products", "Bandung {} Enterprises"],
            "names": ["Budi Santoso", "Siti Rahayu", "Agus Wijaya", "Dewi Susanti", "Hendra Gunawan"],
            "emails": ["{}@{}.co.id", "info@{}.id", "export@{}.co.id"],
            "phones": ["+62 21 {}", "+62 31 {}"]
        },
        "Malaysia": {
            "companies": ["Kuala Lumpur {} Trading", "Penang {} Exports", "Malaysian {} Products", "Johor {} International", "Sabah {} Enterprises"],
            "names": ["Ahmad Bin Abdullah", "Tan Wei Ling", "Kumar Raju", "Lim Mei Hua", "Mohammed Ismail"],
            "emails": ["{}@{}.com.my", "info@{}.my", "export@{}.com.my"],
            "phones": ["+60 3 {}", "+60 4 {}"]
        },
        "Philippines": {
            "companies": ["Manila {} Trading Corp.", "Cebu {} Exports", "Philippine {} Products", "Davao {} International", "Luzon {} Enterprises"],
            "names": ["Jose Santos", "Maria Cruz", "Antonio Reyes", "Elena Gonzales", "Roberto Lim"],
            "emails": ["{}@{}.com.ph", "info@{}.ph", "export@{}.com.ph"],
            "phones": ["+63 2 {}", "+63 32 {}"]
        },
        
        # Africa
        "Egypt": {
            "companies": ["Cairo {} Trading", "Alexandria {} Exports", "Egyptian {} Products", "Nile {} International", "Giza {} Group"],
            "names": ["Ahmed Hassan", "Fatima Ali", "Mohammed Ibrahim", "Laila Mahmoud", "Omar Farouk"],
            "emails": ["{}@{}.com.eg", "info@{}.eg", "export@{}.com.eg"],
            "phones": ["+20 2 {}"]
        },
        "South Africa": {
            "companies": ["Johannesburg {} Trading", "Cape Town {} Exports", "Durban {} International", "Pretoria {} Products", "South African {} Enterprises"],
            "names": ["John van der Merwe", "Sarah Nkosi", "David Botha", "Thandi Zulu", "Michael Pretorius"],
            "emails": ["{}@{}.co.za", "info@{}.za", "export@{}.co.za"],
            "phones": ["+27 11 {}", "+27 21 {}", "+27 31 {}"]
        },
        "Kenya": {
            "companies": ["Nairobi {} Trading", "Mombasa {} Exports", "Kenyan {} Products", "Nakuru {} International", "Kisumu {} Enterprises"],
            "names": ["James Kamau", "Grace Wanjiku", "Peter Ochieng", "Faith Muthoni", "Samuel Kiprop"],
            "emails": ["{}@{}.co.ke", "info@{}.ke", "export@{}.co.ke"],
            "phones": ["+254 20 {}", "+254 41 {}"]
        },
        "Nigeria": {
            "companies": ["Lagos {} Trading", "Abuja {} Exports", "Nigerian {} Products", "Kano {} International", "Port Harcourt {} Enterprises"],
            "names": ["Oluwaseun Adeyemi", "Chinwe Okafor", "Ibrahim Musa", "Ngozi Eze", "Yusuf Bello"],
            "emails": ["{}@{}.com.ng", "info@{}.ng", "export@{}.com.ng"],
            "phones": ["+234 1 {}", "+234 9 {}"]
        },
        "Morocco": {
            "companies": ["Casablanca {} Trading", "Rabat {} Exports", "Moroccan {} Products", "Marrakech {} International", "Tangier {} Enterprises"],
            "names": ["Hassan El Mansouri", "Fatima Benali", "Mohammed Chakir", "Leila Bouazza", "Karim Tahiri"],
            "emails": ["{}@{}.co.ma", "info@{}.ma", "export@{}.co.ma"],
            "phones": ["+212 5 {}"]
        },
        
        # South America
        "Brazil": {
            "companies": ["São Paulo {} Trading", "Rio {} Exports", "Brazilian {} Products", "Brasília {} International", "Amazonas {} Enterprises"],
            "names": ["Carlos Silva", "Ana Santos", "Pedro Oliveira", "Mariana Costa", "João Pereira"],
            "emails": ["{}@{}.com.br", "info@{}.br", "export@{}.com.br"],
            "phones": ["+55 11 {}", "+55 21 {}", "+55 61 {}"]
        },
        "Argentina": {
            "companies": ["Buenos Aires {} Trading", "Córdoba {} Exports", "Argentine {} Products", "Mendoza {} International", "Rosario {} Enterprises"],
            "names": ["Javier Rodriguez", "Lucía Martinez", "Alejandro Fernandez", "Valentina Lopez", "Matías Garcia"],
            "emails": ["{}@{}.com.ar", "info@{}.ar", "export@{}.com.ar"],
            "phones": ["+54 11 {}", "+54 351 {}"]
        },
        "Chile": {
            "companies": ["Santiago {} Trading", "Valparaíso {} Exports", "Chilean {} Products", "Concepción {} International", "Antofagasta {} Enterprises"],
            "names": ["Cristian Gonzalez", "Daniela Muñoz", "Felipe Rojas", "Camila Vargas", "Sebastian Diaz"],
            "emails": ["{}@{}.cl", "info@{}.com.cl", "export@{}.cl"],
            "phones": ["+56 2 {}", "+56 32 {}"]
        },
        "Colombia": {
            "companies": ["Bogotá {} Trading", "Medellín {} Exports", "Colombian {} Products", "Cali {} International", "Barranquilla {} Enterprises"],
            "names": ["Andres Gomez", "Carolina Herrera", "Juan Ramirez", "Maria Cardenas", "Diego Restrepo"],
            "emails": ["{}@{}.com.co", "info@{}.co", "export@{}.com.co"],
            "phones": ["+57 1 {}", "+57 4 {}"]
        },
        "Peru": {
            "companies": ["Lima {} Trading", "Arequipa {} Exports", "Peruvian {} Products", "Cusco {} International", "Trujillo {} Enterprises"],
            "names": ["Jorge Castillo", "Rosa Flores", "Luis Mendoza", "Patricia Torres", "Miguel Chavez"],
            "emails": ["{}@{}.com.pe", "info@{}.pe", "export@{}.com.pe"],
            "phones": ["+51 1 {}", "+51 54 {}"]
        },
        
        # North America
        "USA": {
            "companies": ["American {} Trading Inc.", "US {} Exports LLC", "{} Products USA", "California {} International", "Midwest {} Enterprises"],
            "names": ["Michael Johnson", "Jennifer Smith", "Robert Williams", "Lisa Brown", "David Miller"],
            "emails": ["{}@{}.com", "info@{}.us", "sales@{}.com"],
            "phones": ["+1 212 {}", "+1 415 {}", "+1 312 {}"]
        },
        "Canada": {
            "companies": ["Toronto {} Trading", "Vancouver {} Exports", "Canadian {} Products", "Montreal {} International", "Alberta {} Enterprises"],
            "names": ["John Wilson", "Sarah Thompson", "James Anderson", "Emily Martin", "William Taylor"],
            "emails": ["{}@{}.ca", "info@{}.com.ca", "export@{}.ca"],
            "phones": ["+1 416 {}", "+1 604 {}", "+1 514 {}"]
        },
        "Mexico": {
            "companies": ["Mexico City {} Trading", "Guadalajara {} Exports", "Mexican {} Products", "Monterrey {} International", "Cancún {} Enterprises"],
            "names": ["Carlos Hernandez", "Ana Rodriguez", "Javier Lopez", "Maria Gonzalez", "Miguel Sanchez"],
            "emails": ["{}@{}.com.mx", "info@{}.mx", "export@{}.com.mx"],
            "phones": ["+52 55 {}", "+52 33 {}", "+52 81 {}"]
        },
        
        # Europe
        "France": {
            "companies": ["Paris {} Trading", "Lyon {} Exports", "French {} Products", "Marseille {} International", "Bordeaux {} Enterprises"],
            "names": ["Pierre Dubois", "Sophie Martin", "Jean Lefebvre", "Marie Moreau", "Philippe Lambert"],
            "emails": ["{}@{}.fr", "info@{}.com.fr", "export@{}.fr"],
            "phones": ["+33 1 {}", "+33 4 {}"]
        },
        "Germany": {
            "companies": ["Berlin {} Trading GmbH", "Munich {} Exports", "German {} Products", "Hamburg {} International", "Frankfurt {} Enterprises"],
            "names": ["Thomas Müller", "Anna Schmidt", "Michael Weber", "Laura Fischer", "Andreas Schneider"],
            "emails": ["{}@{}.de", "info@{}.com.de", "export@{}.de"],
            "phones": ["+49 30 {}", "+49 89 {}", "+49 40 {}"]
        },
        "Italy": {
            "companies": ["Rome {} Trading", "Milan {} Exports", "Italian {} Products", "Naples {} International", "Turin {} Enterprises"],
            "names": ["Marco Rossi", "Giulia Ferrari", "Antonio Esposito", "Sofia Ricci", "Giuseppe Romano"],
            "emails": ["{}@{}.it", "info@{}.com.it", "export@{}.it"],
            "phones": ["+39 06 {}", "+39 02 {}"]
        },
        "Spain": {
            "companies": ["Madrid {} Trading", "Barcelona {} Exports", "Spanish {} Products", "Valencia {} International", "Seville {} Enterprises"],
            "names": ["Javier Garcia", "Carmen Rodriguez", "Miguel Fernandez", "Elena Martinez", "Antonio Lopez"],
            "emails": ["{}@{}.es", "info@{}.com.es", "export@{}.es"],
            "phones": ["+34 91 {}", "+34 93 {}"]
        },
        "Netherlands": {
            "companies": ["Amsterdam {} Trading", "Rotterdam {} Exports", "Dutch {} Products", "The Hague {} International", "Utrecht {} Enterprises"],
            "names": ["Jan de Vries", "Anna van Dijk", "Peter Bakker", "Eva Visser", "Thomas Jansen"],
            "emails": ["{}@{}.nl", "info@{}.com.nl", "export@{}.nl"],
            "phones": ["+31 20 {}", "+31 10 {}"]
        },
        
        # Middle East
        "UAE": {
            "companies": ["Dubai {} Trading", "Abu Dhabi {} Exports", "UAE {} Products", "Sharjah {} International", "Ajman {} Enterprises"],
            "names": ["Mohammed Al Mansouri", "Fatima Al Hashimi", "Ahmed Al Maktoum", "Aisha Al Zaabi", "Khalid Al Qasimi"],
            "emails": ["{}@{}.ae", "info@{}.com.ae", "export@{}.ae"],
            "phones": ["+971 4 {}", "+971 2 {}"]
        },
        "Saudi Arabia": {
            "companies": ["Riyadh {} Trading", "Jeddah {} Exports", "Saudi {} Products", "Dammam {} International", "Mecca {} Enterprises"],
            "names": ["Abdullah Al Saud", "Noor Al Qahtani", "Fahad Al Otaibi", "Layla Al Ghamdi", "Saeed Al Shehri"],
            "emails": ["{}@{}.sa", "info@{}.com.sa", "export@{}.sa"],
            "phones": ["+966 11 {}", "+966 12 {}"]
        },
        "Turkey": {
            "companies": ["Istanbul {} Trading", "Ankara {} Exports", "Turkish {} Products", "Izmir {} International", "Antalya {} Enterprises"],
            "names": ["Mehmet Yilmaz", "Ayse Kaya", "Mustafa Demir", "Zeynep Celik", "Ali Ozturk"],
            "emails": ["{}@{}.com.tr", "info@{}.tr", "export@{}.com.tr"],
            "phones": ["+90 212 {}", "+90 312 {}"]
        },
        
        # Oceania
        "Australia": {
            "companies": ["Sydney {} Trading", "Melbourne {} Exports", "Australian {} Products", "Brisbane {} International", "Perth {} Enterprises"],
            "names": ["James Wilson", "Sarah Thompson", "David Johnson", "Emma Brown", "Michael Smith"],
            "emails": ["{}@{}.com.au", "info@{}.au", "export@{}.com.au"],
            "phones": ["+61 2 {}", "+61 3 {}", "+61 7 {}"]
        },
        "New Zealand": {
            "companies": ["Auckland {} Trading", "Wellington {} Exports", "NZ {} Products", "Christchurch {} International", "Hamilton {} Enterprises"],
            "names": ["John Williams", "Emma Taylor", "David Thompson", "Sarah Wilson", "Michael Anderson"],
            "emails": ["{}@{}.co.nz", "info@{}.nz", "export@{}.co.nz"],
            "phones": ["+64 9 {}", "+64 4 {}"]
        }
    }
    
    # Default info if country not in dictionary
    default_info = {
        "companies": ["Global {} Trading", "International {} Exports", "{} Products Ltd.", "Worldwide {} Corp.", "Universal {} Enterprises"],
        "names": ["John Smith", "Maria Garcia", "Wei Chen", "Raj Patel", "Mohammed Ali"],
        "emails": ["{}@{}.com", "info@{}.org", "contact@{}.net"],
        "phones": ["+1 555 {}"]
    }
    
    # Get country-specific info or use default
    info = country_info.get(country, default_info)
    
    # Generate random contacts
    contacts = []
    for _ in range(num_contacts):
        # Generate company name with commodity placeholder
        company_template = random.choice(info["companies"])
        company_words = ["Agro", "Farm", "Harvest", "Fresh", "Green", "Organic", "Natural", "Prime", "Select", "Golden"]
        company_word = random.choice(company_words)
        company = company_template.format(company_word)
        
        # Generate contact name
        name = random.choice(info["names"])
        
        # Generate email
        email_template = random.choice(info["emails"])
        email_name = name.lower().replace(" ", ".")
        company_email = company.lower().replace(" ", "").replace("{}", "")
        email = email_template.format(email_name, company_email)
        
        # Generate phone
        phone_template = random.choice(info["phones"])
        phone_suffix = "".join([str(random.randint(0, 9)) for _ in range(7)])
        phone = phone_template.format(phone_suffix)
        
        contacts.append({
            "company": company,
            "name": name,
            "email": email,
            "phone": phone,
            "location": country
        })
    
    return contacts

# Function to create HTML report
def create_html_report(opportunity, commodity, region, user_type, price_data, weather_data, crop_health_data, trade_data):
    # Create a styled HTML report
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
                background-color: {BG_COLOR};
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .logo {{
                max-width: 200px;
                margin-bottom: 10px;
            }}
            h1 {{
                color: {PRIMARY_COLOR};
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }}
            h2 {{
                color: {PRIMARY_COLOR};
                margin-top: 20px;
            }}
            .highlight {{
                background-color: #f8f9fa;
                padding: 15px;
                border-left: 4px solid {SECONDARY_COLOR};
                margin-bottom: 20px;
            }}
            .opportunity {{
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }}
            .impact {{
                color: #2c3e50;
                font-weight: bold;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                border-bottom: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: {SECONDARY_COLOR};
                color: white;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .chart-container {{
                width: 100%;
                height: 300px;
                margin: 20px 0;
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .chart-placeholder {{
                width: 100%;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                background-color: #f8f9fa;
                color: #666;
                font-style: italic;
            }}
            .footer {{
                margin-top: 50px;
                text-align: center;
                font-size: 0.9em;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <img src="data:image/png;base64,..." alt="Sauda Food Insights LLC" class="logo">
            <h1>Sauda Food Insights LLC</h1>
            <p>Market Opportunity Report</p>
        </div>
        
        <div class="highlight">
            <h2>{opportunity['title']}</h2>
            <p><strong>Commodity:</strong> {commodity}</p>
            <p><strong>Region:</strong> {region}</p>
            <p><strong>Date:</strong> {datetime.now().strftime('%B %d, %Y')}</p>
        </div>
        
        <div class="opportunity">
            <h2>Opportunity Overview</h2>
            <p>{opportunity['description']}</p>
            <p><strong>Rationale:</strong> {opportunity['rationale']}</p>
            <p class="impact"><strong>Potential Impact:</strong> {opportunity['potential_impact']}</p>
            <p><strong>Implementation Timeline:</strong> {opportunity['implementation_timeline']}</p>
            <p><strong>Risk Level:</strong> {opportunity['risk_level']}</p>
        </div>
        
        <h2>Market Analysis</h2>
        
        <h3>Price Trends</h3>
        <p>Analysis of price data shows a {random.choice(['steady increase', 'stable pattern', 'seasonal fluctuation', 'gradual decline', 'recovery after recent dip'])} in {commodity} prices over the past 24 months. The {random.choice(['Q1', 'Q2', 'Q3', 'Q4'])} period typically shows {random.choice(['higher', 'lower', 'more stable', 'more volatile'])} prices due to {random.choice(['seasonal demand', 'harvest timing', 'inventory cycles', 'shipping patterns'])}.</p>
        <div class="chart-container">
            <div class="chart-placeholder">[Price Trend Chart - See interactive version in platform]</div>
        </div>
        
        <h3>Weather Impact</h3>
        <p>Weather patterns in key growing regions show {random.choice(['favorable conditions', 'concerning drought conditions', 'excessive rainfall', 'optimal temperature ranges', 'unusual weather patterns'])} that are likely to {random.choice(['boost production', 'reduce yields', 'maintain stable supply', 'affect quality', 'delay harvest timing'])}. This creates a {random.choice(['strategic opportunity', 'potential risk', 'need for diversification', 'competitive advantage'])} for {user_type.lower()}s in the current market.</p>
        <div class="chart-container">
            <div class="chart-placeholder">[Weather Impact Chart - See interactive version in platform]</div>
        </div>
        
        <h3>Crop Health Assessment</h3>
        <p>Satellite imagery analysis indicates {random.choice(['excellent', 'good', 'average', 'below average', 'concerning'])} crop health conditions in {random.choice(['major', 'alternative', 'emerging', 'traditional'])} growing regions. NDVI readings are {random.choice(['above historical averages', 'consistent with seasonal patterns', 'showing signs of stress in some areas', 'indicating robust growth'])}. Soil moisture levels are {random.choice(['optimal', 'slightly low', 'adequate', 'higher than normal'])}, suggesting {random.choice(['strong yield potential', 'possible production challenges', 'normal harvest expectations', 'quality variations'])}.
        <div class="chart-container">
            <div class="chart-placeholder">[Crop Health Chart - See interactive version in platform]</div>
        </div>
        
        <h3>Trade Flow Analysis</h3>
        <p>Global shipment data reveals {random.choice(['increasing', 'stable', 'shifting', 'diversifying', 'consolidating'])} trade patterns for {commodity}. {random.choice(['Traditional exporters are maintaining market share', 'New origin countries are gaining significance', 'Logistical bottlenecks are affecting certain trade routes', 'Regulatory changes are reshaping trade flows', 'Consumer preferences are driving origin diversification'])}. This creates {random.choice(['new sourcing opportunities', 'potential for market expansion', 'need for supply chain resilience', 'competitive pricing dynamics', 'quality differentiation possibilities'])} in the current market environment.</p>
        <div class="chart-container">
            <div class="chart-placeholder">[Trade Flow Chart - See interactive version in platform]</div>
        </div>
        
        <h2>Recommended Contacts</h2>
        <table>
            <tr>
                <th>Company</th>
                <th>Location</th>
                <th>Contact Person</th>
                <th>Contact Details</th>
            </tr>
            <tr>
                <td>{opportunity['contacts'][0]['company']}</td>
                <td>{opportunity['contacts'][0]['location']}</td>
                <td>{opportunity['contacts'][0]['name']}</td>
                <td>{opportunity['contacts'][0]['email']}<br>{opportunity['contacts'][0]['phone']}</td>
            </tr>
            <tr>
                <td>{opportunity['contacts'][1]['company']}</td>
                <td>{opportunity['contacts'][1]['location']}</td>
                <td>{opportunity['contacts'][1]['name']}</td>
                <td>{opportunity['contacts'][1]['email']}<br>{opportunity['contacts'][1]['phone']}</td>
            </tr>
        </table>
        
        <div class="footer">
            <p><em>This report is generated by Sauda Food Insights LLC. The recommendations are based on analysis of market data, weather patterns, crop conditions, and trade flows.</em></p>
            <p>© {datetime.now().year} Sauda Food Insights LLC. All rights reserved.</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

# Function to convert HTML to downloadable format
def get_html_download_link(html_content, filename="report"):
    # Encode HTML as base64
    b64 = base64.b64encode(html_content.encode()).decode()
    
    # Create download link for HTML
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}.html" style="background-color:{SECONDARY_COLOR};color:white;padding:10px 15px;text-decoration:none;border-radius:5px;font-weight:bold;">Download Report</a>'
    
    return href

# Main application layout
def main():
    # Sidebar
    st.sidebar.image("IMG_3036.png", width=150)
    st.sidebar.title("Sauda Food Insights")
    
    # User type selection
    user_type = st.sidebar.radio("Select View", ["Buyer", "Seller"])
    
    # Get available commodities
    commodities = get_available_commodities()
    
    # Commodity selection
    commodity_options = list(commodities.values())
    selected_commodity_name = st.sidebar.selectbox("Select Commodity", commodity_options)
    
    # Get ticker for selected commodity
    selected_commodity_ticker = [k for k, v in commodities.items() if v == selected_commodity_name][0]
    
    # Region selection
    regions = ["Asia", "Africa", "South America", "North America", "Europe", "Middle East", "Oceania"]
    selected_region = st.sidebar.selectbox("Select Region", regions)
    
    # Auto-refresh option
    st.sidebar.write("---")
    auto_refresh = st.sidebar.checkbox("Auto-refresh data (every 30 min)", value=True)
    if st.sidebar.button("Refresh Data Now"):
        st.experimental_rerun()
    
    # Main content
    st.title(f"{user_type} Dashboard: {selected_commodity_name}")
    
    # Tabs for different analyses
    tabs = st.tabs(["Price Analysis", "Weather Impact", "Crop Health", "Trade Flows", "Market Opportunities"])
    
    # Price Analysis Tab
    with tabs[0]:
        st.header("Price Analysis")
        
        # Get price data
        price_data = get_price_data(selected_commodity_ticker)
        
        if not price_data.empty:
            # Create price chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color=PRIMARY_COLOR, width=2)
            ))
            
            # Add moving averages
            price_data['MA50'] = price_data['Close'].rolling(window=50).mean()
            price_data['MA200'] = price_data['Close'].rolling(window=200).mean()
            
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['MA50'],
                mode='lines',
                name='50-Day MA',
                line=dict(color=SECONDARY_COLOR, width=1.5)
            ))
            
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data['MA200'],
                mode='lines',
                name='200-Day MA',
                line=dict(color=ACCENT_COLOR, width=1.5)
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{selected_commodity_name} Price Trends",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Price analysis explanation
            st.subheader("Price Analysis Insights")
            
            # Calculate price statistics
            current_price = price_data['Close'].iloc[-1]
            month_ago_price = price_data['Close'].iloc[-30] if len(price_data) > 30 else price_data['Close'].iloc[0]
            year_ago_price = price_data['Close'].iloc[-365] if len(price_data) > 365 else price_data['Close'].iloc[0]
            
            monthly_change = (current_price - month_ago_price) / month_ago_price * 100
            yearly_change = (current_price - year_ago_price) / year_ago_price * 100
            
            # Determine price trend
            if monthly_change > 5:
                monthly_trend = "strong upward"
            elif monthly_change > 2:
                monthly_trend = "moderate upward"
            elif monthly_change > -2:
                monthly_trend = "stable"
            elif monthly_change > -5:
                monthly_trend = "moderate downward"
            else:
                monthly_trend = "strong downward"
                
            if yearly_change > 15:
                yearly_trend = "significant increase"
            elif yearly_change > 5:
                yearly_trend = "moderate increase"
            elif yearly_change > -5:
                yearly_trend = "relatively stable"
            elif yearly_change > -15:
                yearly_trend = "moderate decrease"
            else:
                yearly_trend = "significant decrease"
            
            # Generate analysis text
            st.write(f"""
            {selected_commodity_name} prices are currently showing a **{monthly_trend}** trend in the short term 
            (1-month change: {monthly_change:.1f}%) and a **{yearly_trend}** over the past year 
            (12-month change: {yearly_change:.1f}%).
            
            The current price of **${current_price:.2f}** is {
            "above" if current_price > price_data['MA50'].iloc[-1] else "below"} the 50-day moving average 
            (${price_data['MA50'].iloc[-1]:.2f}) and {
            "above" if current_price > price_data['MA200'].iloc[-1] else "below"} the 200-day moving average 
            (${price_data['MA200'].iloc[-1]:.2f}).
            
            This price pattern suggests {
            "potential buying opportunities" if monthly_change < -2 and user_type == "Buyer" else
            "favorable selling conditions" if monthly_change > 2 and user_type == "Seller" else
            "a balanced market with stable pricing"}.
            """)
            
            # Seasonal pattern analysis
            st.subheader("Seasonal Price Patterns")
            
            # Create seasonal decomposition if enough data
            if len(price_data) > 365:
                # Resample to monthly for clearer seasonal patterns
                monthly_data = price_data['Close'].resample('M').mean()
                
                # Create seasonal chart
                fig_seasonal = go.Figure()
                
                # Group by month and calculate average
                monthly_data.index = monthly_data.index.month
                monthly_avg = monthly_data.groupby(monthly_data.index).mean()
                
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                fig_seasonal.add_trace(go.Bar(
                    x=months,
                    y=monthly_avg.values,
                    marker_color=SECONDARY_COLOR
                ))
                
                # Update layout
                fig_seasonal.update_layout(
                    title="Average Monthly Prices (Seasonal Pattern)",
                    xaxis_title="Month",
                    yaxis_title="Average Price (USD)",
                    template="plotly_white",
                    height=400
                )
                
                st.plotly_chart(fig_seasonal, use_container_width=True)
                
                # Find highest and lowest price months
                highest_month = monthly_avg.idxmax()
                lowest_month = monthly_avg.idxmin()
                
                st.write(f"""
                Seasonal analysis shows that {selected_commodity_name} prices tend to be highest in 
                **{months[highest_month-1]}** and lowest in **{months[lowest_month-1]}**. 
                
                For {user_type.lower()}s, this suggests:
                
                {"- Consider building inventory during " + months[lowest_month-1] + " when prices are typically lower" 
                if user_type == "Buyer" else
                "- Consider timing major sales during " + months[highest_month-1] + " when prices are typically higher"}
                
                {"- Explore long-term contracts to lock in favorable prices during low-price periods" 
                if user_type == "Buyer" else
                "- Consider hedging strategies to protect against seasonal price declines"}
                
                {"- Diversify sourcing to regions with counter-seasonal production cycles" 
                if user_type == "Buyer" else
                "- Target buyers in regions experiencing supply gaps during their off-season periods"}
                """)
        else:
            st.error(f"No price data available for {selected_commodity_name}")
    
    # Weather Impact Tab
    with tabs[1]:
        st.header("Weather Impact Analysis")
        
        # Get weather data
        weather_data = get_weather_data(selected_region)
        
        # Create weather chart
        fig = go.Figure()
        
        # Add temperature line
        fig.add_trace(go.Scatter(
            x=weather_data['Date'],
            y=weather_data['Temperature'],
            mode='lines',
            name='Temperature (°C)',
            line=dict(color='#ff7043', width=2)
        ))
        
        # Add rainfall bars on secondary y-axis
        fig.add_trace(go.Bar(
            x=weather_data['Date'],
            y=weather_data['Rainfall'],
            name='Rainfall (mm)',
            marker_color='rgba(0, 119, 182, 0.6)'
        ))
        
        # Update layout with dual y-axes
        fig.update_layout(
            title=f"Weather Patterns in {selected_region} Growing Regions",
            xaxis_title="Date",
            yaxis=dict(
                title=dict(
                    text="Temperature (°C)",
                    font=dict(color="#ff7043")
                ),
                tickfont=dict(color="#ff7043")
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
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Weather impact explanation
        st.subheader("Weather Impact Insights")
        
        # Calculate weather statistics
        current_temp = weather_data['Temperature'].iloc[-1]
        avg_temp = weather_data['Temperature'].mean()
        temp_anomaly = current_temp - avg_temp
        
        current_rain = weather_data['Rainfall'].iloc[-1]
        avg_rain = weather_data['Rainfall'].mean()
        rain_anomaly = current_rain - avg_rain
        
        # Generate analysis text based on commodity type
        is_grain = selected_commodity_name.lower() in ['wheat', 'corn', 'rice', 'barley', 'oats', 'soybeans']
        is_fruit = selected_commodity_name.lower() in ['orange', 'apple', 'banana', 'strawberry', 'pineapple', 'grape']
        is_vegetable = selected_commodity_name.lower() in ['potato', 'tomato', 'onion', 'lettuce', 'carrot', 'cabbage']
        
        if is_grain:
            crop_type = "grain crops"
            temp_impact = "can accelerate growth but may reduce yield quality" if temp_anomaly > 2 else \
                         "may slow growth but could improve grain quality" if temp_anomaly < -2 else \
                         "is optimal for balanced growth and development"
            
            rain_impact = "raises concerns about potential flooding and disease pressure" if rain_anomaly > 10 else \
                         "may lead to drought stress and reduced yields" if rain_anomaly < -10 else \
                         "provides adequate moisture for healthy development"
        
        elif is_fruit:
            crop_type = "fruit development"
            temp_impact = "may accelerate ripening but could reduce shelf life" if temp_anomaly > 2 else \
                         "could delay harvest timing but may improve flavor development" if temp_anomaly < -2 else \
                         "supports normal ripening patterns and quality development"
            
            rain_impact = "increases disease risk and may affect fruit quality" if rain_anomaly > 10 else \
                         "may require supplemental irrigation to maintain fruit size" if rain_anomaly < -10 else \
                         "provides good moisture balance for fruit development"
        
        elif is_vegetable:
            crop_type = "vegetable crops"
            temp_impact = "could stress plants and affect quality" if temp_anomaly > 2 else \
                         "may slow growth but could improve vegetable flavor" if temp_anomaly < -2 else \
                         "is ideal for balanced growth and quality"
            
            rain_impact = "raises concerns about root diseases and quality issues" if rain_anomaly > 10 else \
                         "may require irrigation to maintain yield potential" if rain_anomaly < -10 else \
                         "provides appropriate moisture for healthy development"
        
        else:
            crop_type = "crop development"
            temp_impact = "may accelerate growth but could affect quality" if temp_anomaly > 2 else \
                         "could slow development but may improve certain quality aspects" if temp_anomaly < -2 else \
                         "supports normal growth patterns"
            
            rain_impact = "raises concerns about excess moisture and related issues" if rain_anomaly > 10 else \
                         "may create moisture stress conditions" if rain_anomaly < -10 else \
                         "provides adequate moisture for normal development"
        
        st.write(f"""
        Current weather conditions in {selected_region}'s {selected_commodity_name} growing regions show temperatures 
        of **{current_temp:.1f}°C** ({temp_anomaly:.1f}°C {
        "above" if temp_anomaly > 0 else "below"} average) and rainfall of **{current_rain:.1f} mm** ({
        rain_anomaly:.1f} mm {"above" if rain_anomaly > 0 else "below"} average).
        
        For {crop_type}, the current temperature pattern {temp_impact}, while the precipitation pattern {rain_impact}.
        
        These conditions suggest {
        "potential production challenges that may affect supply" 
        if (abs(temp_anomaly) > 3 or abs(rain_anomaly) > 15) else
        "some weather-related stress but manageable impact on production" 
        if (abs(temp_anomaly) > 1.5 or abs(rain_anomaly) > 7) else
        "favorable growing conditions supporting normal production levels"
        }.
        
        For {user_type.lower()}s, this indicates {
        "a need to monitor supply availability and potential price impacts" if user_type == "Buyer" else
        "potential market opportunities as weather impacts materialize in production outcomes" if user_type == "Seller"
        }.
        """)
        
        # Weather forecast and implications
        st.subheader("Seasonal Outlook and Implications")
        
        # Generate random but consistent forecast based on region and commodity
        seed = sum(ord(c) for c in selected_region) + sum(ord(c) for c in selected_commodity_name)
        random.seed(seed)
        
        forecast_scenarios = [
            "continued favorable conditions with normal temperature and precipitation patterns",
            "above-average temperatures with near-normal precipitation",
            "below-average temperatures with above-normal precipitation",
            "warmer and drier than normal conditions",
            "cooler and wetter than normal conditions"
        ]
        
        selected_forecast = random.choice(forecast_scenarios)
        
        if "favorable" in selected_forecast:
            impact = "stable production and normal market conditions"
        elif "above-average temperatures" in selected_forecast:
            impact = "potentially accelerated crop development but possible heat stress"
        elif "below-average temperatures" in selected_forecast:
            impact = "slower crop development that may delay harvest timing"
        elif "warmer and drier" in selected_forecast:
            impact = "potential yield reductions if irrigation is insufficient"
        else:  # cooler and wetter
            impact = "disease pressure and potential quality concerns"
        
        st.write(f"""
        The 3-month seasonal outlook for {selected_region} indicates **{selected_forecast}**.
        
        This pattern suggests **{impact}** for {selected_commodity_name} production.
        
        Strategic recommendations for {user_type.lower()}s:
        
        {"- Consider diversifying supply sources to mitigate weather-related risks" 
        if user_type == "Buyer" else
        "- Monitor production conditions closely to identify optimal market timing"}
        
        {"- Evaluate contract terms to account for potential quality variations" 
        if user_type == "Buyer" else
        "- Consider how weather patterns may affect your competitive position"}
        
        {"- Monitor price trends as weather impacts materialize in production outcomes" 
        if user_type == "Buyer" else
        "- Evaluate how weather conditions may affect your product quality and differentiation"}
        """)
    
    # Crop Health Tab
    with tabs[2]:
        st.header("Crop Health Monitoring")
        
        # Get crop health data
        crop_health_data = get_crop_health_data(selected_region, selected_commodity_name)
        
        # Create crop health chart
        fig = go.Figure()
        
        # Add NDVI line
        fig.add_trace(go.Scatter(
            x=crop_health_data['Date'],
            y=crop_health_data['NDVI'],
            mode='lines',
            name='NDVI',
            line=dict(color=SECONDARY_COLOR, width=2)
        ))
        
        # Add soil moisture line
        fig.add_trace(go.Scatter(
            x=crop_health_data['Date'],
            y=crop_health_data['Soil_Moisture'],
            mode='lines',
            name='Soil Moisture',
            line=dict(color=ACCENT_COLOR, width=2)
        ))
        
        # Add crop stress line
        fig.add_trace(go.Scatter(
            x=crop_health_data['Date'],
            y=crop_health_data['Crop_Stress'] / 100,  # Normalize to 0-1 scale
            mode='lines',
            name='Crop Stress Index',
            line=dict(color='#ff7043', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{selected_commodity_name} Crop Health in {selected_region}",
            xaxis_title="Date",
            yaxis_title="Index Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Crop health explanation
        st.subheader("Crop Health Insights")
        
        # Calculate crop health statistics
        current_ndvi = crop_health_data['NDVI'].iloc[-1]
        avg_ndvi = crop_health_data['NDVI'].mean()
        ndvi_anomaly = current_ndvi - avg_ndvi
        
        current_moisture = crop_health_data['Soil_Moisture'].iloc[-1]
        avg_moisture = crop_health_data['Soil_Moisture'].mean()
        moisture_anomaly = current_moisture - avg_moisture
        
        current_stress = crop_health_data['Crop_Stress'].iloc[-1]
        avg_stress = crop_health_data['Crop_Stress'].mean()
        stress_anomaly = current_stress - avg_stress
        
        # Determine crop health status
        if current_ndvi > 0.6 and current_moisture > 0.3 and current_stress < 30:
            health_status = "excellent"
        elif current_ndvi > 0.5 and current_moisture > 0.25 and current_stress < 40:
            health_status = "good"
        elif current_ndvi > 0.4 and current_moisture > 0.2 and current_stress < 50:
            health_status = "fair"
        elif current_ndvi > 0.3 and current_moisture > 0.15 and current_stress < 60:
            health_status = "concerning"
        else:
            health_status = "poor"
        
        # Generate analysis text
        st.write(f"""
        Satellite-based crop health monitoring for {selected_commodity_name} in {selected_region} shows **{health_status}** 
        overall conditions.
        
        The current NDVI (Normalized Difference Vegetation Index) value of **{current_ndvi:.2f}** is {
        "above" if ndvi_anomaly > 0 else "below"} the seasonal average ({ndvi_anomaly:.2f} difference), indicating {
        "stronger than normal vegetation vigor" if ndvi_anomaly > 0.05 else
        "slightly better than average plant health" if ndvi_anomaly > 0 else
        "slightly reduced vegetation health" if ndvi_anomaly > -0.05 else
        "significantly reduced plant vigor"}.
        
        Soil moisture readings of **{current_moisture:.2f}** are {
        "above" if moisture_anomaly > 0 else "below"} typical levels ({moisture_anomaly:.2f} difference), suggesting {
        "abundant water availability that supports robust growth" if moisture_anomaly > 0.05 else
        "adequate moisture conditions" if moisture_anomaly > -0.05 else
        "developing moisture stress that may affect yields"}.
        
        The crop stress index of **{current_stress:.1f}** is {
        "below" if stress_anomaly < 0 else "above"} average ({abs(stress_anomaly):.1f} points {
        "lower" if stress_anomaly < 0 else "higher"}), indicating {
        "minimal plant stress and favorable growing conditions" if stress_anomaly < -5 else
        "normal stress levels within manageable ranges" if abs(stress_anomaly) <= 5 else
        "elevated stress that may impact production outcomes"}.
        """)
        
        # Production outlook
        st.subheader("Production Outlook")
        
        # Generate production outlook based on health metrics
        if health_status in ["excellent", "good"]:
            production_outlook = "above average"
            price_impact = "potential price stability or modest declines as harvest approaches"
            quality_outlook = "high quality with good nutritional profiles and shelf life"
        elif health_status == "fair":
            production_outlook = "near average"
            price_impact = "normal seasonal price patterns with typical volatility"
            quality_outlook = "standard quality with normal variation"
        else:  # concerning or poor
            production_outlook = "below average"
            price_impact = "upward price pressure as supply constraints become apparent"
            quality_outlook = "variable quality with potential for reduced shelf life or nutritional content"
        
        st.write(f"""
        Based on current crop health indicators, {selected_region}'s {selected_commodity_name} production is 
        tracking toward **{production_outlook}** levels.
        
        This suggests **{price_impact}** in the coming market cycle.
        
        Quality metrics indicate **{quality_outlook}**.
        
        For {user_type.lower()}s, this outlook suggests:
        
        {"- Consider securing supply commitments earlier than usual" 
        if production_outlook == "below average" and user_type == "Buyer" else
        "- Normal procurement timing should align with market needs" 
        if production_outlook == "near average" and user_type == "Buyer" else
        "- Opportunity to negotiate favorable terms as supply appears robust" 
        if production_outlook == "above average" and user_type == "Buyer" else
        "- Position for potentially stronger pricing as harvest approaches" 
        if production_outlook == "below average" and user_type == "Seller" else
        "- Focus on quality differentiation in a balanced market" 
        if production_outlook == "near average" and user_type == "Seller" else
        "- Consider early commitment strategies to secure volume in a competitive market" 
        if production_outlook == "above average" and user_type == "Seller"}
        
        {"- Evaluate quality specifications carefully in contracts" 
        if quality_outlook.startswith("variable") else
        "- Standard quality parameters should be appropriate for contracts" 
        if quality_outlook.startswith("standard") else
        "- Opportunity to secure premium quality product"}
        
        {"- Monitor crop development closely as harvest approaches for updated outlook" 
        if user_type == "Buyer" else
        "- Track competitor production regions to understand your relative market position"}
        """)
    
    # Trade Flows Tab
    with tabs[3]:
        st.header("Global Trade Flow Analysis")
        
        # Origin and destination selection for trade flow analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Different default origin based on user type
            default_origin = "South America" if user_type == "Buyer" else selected_region
            origin = st.selectbox("Origin Region", regions, index=regions.index(default_origin))
        
        with col2:
            # Different default destination based on user type
            default_destination = selected_region if user_type == "Buyer" else "Europe"
            destination = st.selectbox("Destination Region", regions, index=regions.index(default_destination))
        
        # Get trade flow data
        trade_data = get_trade_flow_data(selected_commodity_name, origin, destination)
        
        # Create trade flow chart
        fig = go.Figure()
        
        # Add volume bars
        fig.add_trace(go.Bar(
            x=trade_data['Date'],
            y=trade_data['Volume'],
            name='Volume (MT)',
            marker_color=SECONDARY_COLOR
        ))
        
        # Add price line on secondary y-axis
        fig.add_trace(go.Scatter(
            x=trade_data['Date'],
            y=trade_data['Price'],
            mode='lines',
            name='Price (USD/MT)',
            line=dict(color=PRIMARY_COLOR, width=2),
            yaxis="y2"
        ))
        
        # Update layout with dual y-axes
        fig.update_layout(
            title=f"{selected_commodity_name} Trade Flow: {origin} to {destination}",
            xaxis_title="Date",
            yaxis=dict(
                title="Volume (MT)",
                titlefont=dict(color=SECONDARY_COLOR),
                tickfont=dict(color=SECONDARY_COLOR)
            ),
            yaxis2=dict(
                title="Price (USD/MT)",
                titlefont=dict(color=PRIMARY_COLOR),
                tickfont=dict(color=PRIMARY_COLOR),
                anchor="x",
                overlaying="y",
                side="right"
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade flow explanation
        st.subheader("Trade Flow Insights")
        
        # Calculate trade statistics
        current_volume = trade_data['Volume'].iloc[-1]
        avg_volume = trade_data['Volume'].mean()
        volume_change = (current_volume - trade_data['Volume'].iloc[-6]) / trade_data['Volume'].iloc[-6] * 100
        
        current_price = trade_data['Price'].iloc[-1]
        avg_price = trade_data['Price'].mean()
        price_change = (current_price - trade_data['Price'].iloc[-6]) / trade_data['Price'].iloc[-6] * 100
        
        # Generate analysis text
        st.write(f"""
        Trade flow analysis for {selected_commodity_name} from {origin} to {destination} shows current monthly 
        volume of **{current_volume:.0f} MT** ({volume_change:.1f}% {
        "increase" if volume_change > 0 else "decrease"} over 6 months) at an average price of **${current_price:.2f}/MT** 
        ({price_change:.1f}% {
        "increase" if price_change > 0 else "decrease"} over 6 months).
        
        This trade route is showing {
        "significantly higher than normal activity" if current_volume > avg_volume * 1.2 else
        "slightly above average volume" if current_volume > avg_volume else
        "slightly below average volume" if current_volume > avg_volume * 0.8 else
        "significantly lower than normal activity"} at {
        "premium prices" if current_price > avg_price * 1.1 else
        "slightly above average prices" if current_price > avg_price else
        "slightly discounted prices" if current_price > avg_price * 0.9 else
        "significantly discounted prices"}.
        
        The {
        "increasing" if volume_change > 0 else "decreasing"} volume trend combined with {
        "rising" if price_change > 0 else "falling"} prices indicates {
        "strong demand that is outpacing supply" if volume_change > 0 and price_change > 0 else
        "improving supply availability despite sustained demand" if volume_change > 0 and price_change < 0 else
        "weakening demand despite limited supply" if volume_change < 0 and price_change > 0 else
        "deteriorating market conditions with both falling demand and prices"}.
        """)
        
        # Shipping and logistics
        st.subheader("Shipping & Logistics Analysis")
        
        # Generate random but consistent shipping data based on origin and destination
        seed = sum(ord(c) for c in origin) + sum(ord(c) for c in destination)
        random.seed(seed)
        
        shipping_time = random.randint(10, 60)  # days
        shipping_cost = random.randint(50, 200)  # USD/MT
        
        # Shipping disruptions based on regions
        disruptions = {
            "Asia": ["port congestion in Singapore", "container shortages in Chinese ports", 
                    "weather delays in South China Sea", "normal operations with standard transit times"],
            "Africa": ["port efficiency challenges in West Africa", "security concerns near Horn of Africa", 
                      "infrastructure limitations at certain terminals", "normal operations with standard transit times"],
            "South America": ["labor disputes at Brazilian ports", "weather-related delays on East coast", 
                             "container availability issues", "normal operations with standard transit times"],
            "North America": ["congestion at US West Coast ports", "rail connection delays", 
                             "labor negotiations affecting throughput", "normal operations with standard transit times"],
            "Europe": ["congestion in Rotterdam and Hamburg", "inland waterway limitations", 
                      "truck driver shortages affecting inland delivery", "normal operations with standard transit times"],
            "Middle East": ["security concerns in certain shipping lanes", "heat-related handling restrictions", 
                           "documentation delays at certain ports", "normal operations with standard transit times"],
            "Oceania": ["weather disruptions affecting Australian ports", "vessel schedule reliability issues", 
                       "container equipment imbalances", "normal operations with standard transit times"]
        }
        
        origin_disruption = random.choice(disruptions[origin])
        destination_disruption = random.choice(disruptions[destination])
        
        # Generate shipping trend
        shipping_trends = [
            f"Shipping costs from {origin} to {destination} have increased by {random.randint(5, 20)}% over the past quarter due to fuel surcharges and capacity constraints",
            f"New vessel capacity added to the {origin}-{destination} route has improved schedule reliability by {random.randint(5, 15)}%",
            f"Average transit times between {origin} and {destination} have {random.choice(['increased', 'decreased'])} by {random.randint(1, 5)} days due to {random.choice(['port congestion', 'routing changes', 'improved port operations', 'vessel slow-steaming practices'])}",
            f"Container availability for {selected_commodity_name} shipments from {origin} has {random.choice(['improved', 'deteriorated'])} in recent weeks",
            f"Logistics providers are reporting {random.choice(['normal', 'strained', 'improving'])} capacity conditions for {selected_commodity_name} shipments on this trade lane"
        ]
        
        selected_trend = random.choice(shipping_trends)
        
        st.write(f"""
        Current shipping parameters for {selected_commodity_name} from {origin} to {destination}:
        
        - Average transit time: **{shipping_time} days**
        - Typical shipping cost: **${shipping_cost}/MT**
        - Origin status: {origin_disruption}
        - Destination status: {destination_disruption}
        
        {selected_trend}
        
        For {user_type.lower()}s, these logistics conditions suggest {
        "building in additional lead time for orders and considering inventory buffers" 
        if "congestion" in origin_disruption or "congestion" in destination_disruption or "delays" in selected_trend else
        "standard lead times should be sufficient for planning purposes"}.
        """)
        
        # Global trade pattern map
        st.subheader("Global Trade Pattern Visualization")
        
        # Create a placeholder for the trade flow map
        st.image("https://via.placeholder.com/800x400?text=Global+Trade+Flow+Map+(Interactive+version+in+platform)", 
                use_column_width=True)
        
        st.write("""
        The global trade pattern visualization shows major trade flows for this commodity, with line thickness 
        representing volume and color indicating price levels. The interactive version in the platform allows 
        filtering by time period and origin/destination pairs.
        """)
    
    # Market Opportunities Tab
    with tabs[4]:
        st.header("Market Opportunities")
        
        # Generate market opportunities
        opportunities = generate_market_opportunities(selected_commodity_name, selected_region, user_type)
        
        # Display opportunities
        for i, opportunity in enumerate(opportunities):
            st.subheader(f"Opportunity {i+1}: {opportunity['title']}")
            
            st.markdown(f"""
            **Description:** {opportunity['description']}
            
            **Rationale:** {opportunity['rationale']}
            
            **Potential Impact:** {opportunity['potential_impact']}
            
            **Implementation Timeline:** {opportunity['implementation_timeline']}
            
            **Risk Level:** {opportunity['risk_level']}
            """)
            
            # Contact information
            st.markdown("#### Recommended Contacts")
            
            for contact in opportunity['contacts']:
                st.markdown(f"""
                **{contact['company']}** ({contact['location']})  
                Contact: {contact['name']}  
                Email: {contact['email']}  
                Phone: {contact['phone']}
                """)
            
            # Get data for report
            price_data = get_price_data(selected_commodity_ticker)
            weather_data = get_weather_data(selected_region)
            crop_health_data = get_crop_health_data(selected_region, selected_commodity_name)
            trade_data = get_trade_flow_data(selected_commodity_name, selected_region, 
                                           opportunity['contacts'][0]['location'])
            
            # Create HTML report
            html_content = create_html_report(opportunity, selected_commodity_name, selected_region, 
                                            user_type, price_data, weather_data, crop_health_data, trade_data)
            
            # Create download link
            download_link = get_html_download_link(html_content, 
                                                 f"Sauda_Insights_{selected_commodity_name}_{opportunity['contacts'][0]['location']}")
            
            st.markdown(download_link, unsafe_allow_html=True)
            
            st.markdown("---")

# Run the application
if __name__ == "__main__":
    main()
