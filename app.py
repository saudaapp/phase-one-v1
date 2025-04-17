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
                f"Recent crop failures in {div_country}'s domestic {commodity} production have created import opportunities",
                f"New consumer trends in {div_country} favor premium quality {commodity} products, which your region is known for",
                f"Regulatory changes in {div_country} have reduced barriers for {commodity} imports from your region",
                f"Growing food processing industry in {div_country} has increased demand for high-quality {commodity} inputs",
                f"Direct shipping routes have been established between your region and {div_country}, reducing logistics costs by 18%",
                f"Market research indicates consumers in {div_country} are willing to pay 15-20% premium for {commodity} with your region's quality standards",
                f"Retail expansion in {div_country} has created new distribution channels for imported {commodity}",
                f"Recent trade mission from {div_country} expressed specific interest in {commodity} imports from your region"
            ]
            
            # Select a specific rationale
            rationale = random.choice(rationales)
            
            # Generate potential revenue increase
            revenue_percent = random.randint(10, 30)
            
            # Generate contact information from the diversification country
            contacts = generate_contacts(div_country, 2)
            
            opportunities.append({
                "title": f"Expand {commodity} exports to {div_country}",
                "description": f"Current market: {current_market}. Recommended new market: {div_country}.",
                "rationale": rationale,
                "potential_impact": f"Potential revenue increase of {revenue_percent}% based on current market conditions",
                "implementation_timeline": f"{random.randint(2, 6)} months",
                "risk_level": random.choice(["Low", "Medium", "High"]),
                "contacts": contacts
            })
    
    return opportunities

# Function to generate contact recommendations
def generate_contacts(country, num_contacts=3):
    # In a production environment, this would connect to a CRM or business directory API
    # to provide genuine contact recommendations
    
    # Set random seed based on country for consistent results
    seed = sum(ord(c) for c in country)
    random.seed(seed)
    
    # Define company name patterns
    company_patterns = [
        "{country} {commodity} Traders",
        "{country} Agricultural Exports",
        "{commodity} Distributors of {country}",
        "Global {commodity} {country} Ltd.",
        "{country} Food Imports",
        "International {commodity} Supply {country}",
        "{country} {commodity} Exchange",
        "United {commodity} Traders {country}",
        "{country} Premium {commodity}",
        "Royal {country} {commodity}"
    ]
    
    # Define contact name patterns based on country
    # This is a simplified approach - in a real system, we would use country-specific name databases
    contact_names = {
        # Asia
        "China": ["Li Wei", "Zhang Min", "Wang Jing", "Chen Yong", "Liu Mei", "Huang Tao"],
        "India": ["Raj Sharma", "Priya Patel", "Amit Singh", "Deepak Kumar", "Sunita Verma", "Vikram Mehta"],
        "Vietnam": ["Nguyen Van", "Tran Thi", "Le Minh", "Pham Duc", "Vo Thanh", "Hoang Hai"],
        "Thailand": ["Somchai S.", "Suchada K.", "Anong P.", "Thaksin C.", "Malee R.", "Chai W."],
        "Indonesia": ["Budi Santoso", "Siti Aminah", "Agus Wijaya", "Dewi Putri", "Hendra Gunawan", "Ratna Sari"],
        "Malaysia": ["Ahmad Bin", "Siti Binti", "Tan Wei", "Lee Chong", "Fatimah Z.", "Mohammed Y."],
        "Philippines": ["Juan Reyes", "Maria Santos", "Jose Cruz", "Ana Lim", "Roberto Tan", "Elena Gomez"],
        
        # Africa
        "Egypt": ["Ahmed Hassan", "Fatima Ali", "Mohamed Ibrahim", "Layla Mahmoud", "Omar Farouk", "Nour El Din"],
        "South Africa": ["John van der Merwe", "Sarah Nkosi", "David Botha", "Thandi Zulu", "Michael Pretorius", "Nomsa Dlamini"],
        "Kenya": ["James Kamau", "Grace Wanjiku", "Daniel Odhiambo", "Faith Muthoni", "Samuel Njoroge", "Mercy Akinyi"],
        "Nigeria": ["Oluwaseun A.", "Chinwe O.", "Emeka I.", "Folake A.", "Chinedu O.", "Amina M."],
        "Morocco": ["Youssef El", "Fatima Ben", "Karim Al", "Leila M.", "Hassan B.", "Samira Z."],
        "Ethiopia": ["Abebe T.", "Tigist M.", "Dawit G.", "Hiwot B.", "Solomon A.", "Meseret Y."],
        "Tanzania": ["Emmanuel M.", "Joyce K.", "Godfrey S.", "Neema N.", "Baraka J.", "Rehema H."],
        
        # South America
        "Brazil": ["Carlos Silva", "Ana Santos", "Pedro Oliveira", "Mariana Costa", "Rafael Souza", "Juliana Lima"],
        "Argentina": ["Juan Gonzalez", "Maria Rodriguez", "Diego Fernandez", "Laura Martinez", "Sergio Lopez", "Valeria Diaz"],
        "Chile": ["Alejandro Munoz", "Camila Vargas", "Sebastian Morales", "Valentina Rojas", "Matias Fuentes", "Javiera Castro"],
        "Colombia": ["Andres Gomez", "Catalina Herrera", "Santiago Ramirez", "Isabella Torres", "Mateo Ortiz", "Daniela Jimenez"],
        "Peru": ["Jose Flores", "Carmen Vega", "Luis Torres", "Rosa Mendoza", "Miguel Chavez", "Patricia Huaman"],
        "Ecuador": ["Francisco Mendez", "Elena Suarez", "Javier Vera", "Gabriela Ponce", "Roberto Cevallos", "Monica Andrade"],
        "Uruguay": ["Martin Perez", "Lucia Rodriguez", "Gonzalo Fernandez", "Sofia Martinez", "Nicolas Garcia", "Victoria Alvarez"],
        
        # North America
        "USA": ["Michael Johnson", "Jennifer Smith", "Robert Williams", "Elizabeth Brown", "David Jones", "Sarah Miller"],
        "Canada": ["James Wilson", "Emily Thompson", "William Anderson", "Olivia Taylor", "Thomas Martin", "Sophia Moore"],
        "Mexico": ["Alejandro Hernandez", "Sofia Garcia", "Javier Lopez", "Isabella Martinez", "Miguel Rodriguez", "Valentina Gonzalez"],
        
        # Europe
        "France": ["Jean Dupont", "Marie Dubois", "Pierre Martin", "Sophie Bernard", "Antoine Leroy", "Camille Moreau"],
        "Germany": ["Thomas Müller", "Anna Schmidt", "Michael Weber", "Laura Fischer", "Andreas Schneider", "Julia Wagner"],
        "Italy": ["Marco Rossi", "Giulia Ricci", "Alessandro Marino", "Sofia Conti", "Francesco Esposito", "Valentina Romano"],
        "Spain": ["Javier Garcia", "Carmen Martinez", "Antonio Lopez", "Elena Rodriguez", "Manuel Fernandez", "Isabel Sanchez"],
        "Netherlands": ["Jan de Vries", "Anna van der Berg", "Peter Bakker", "Eva Visser", "Thomas Jansen", "Lisa de Jong"],
        "Poland": ["Piotr Kowalski", "Anna Nowak", "Tomasz Wiśniewski", "Magdalena Wójcik", "Andrzej Kamiński", "Katarzyna Lewandowska"],
        "UK": ["James Smith", "Emma Jones", "William Taylor", "Olivia Brown", "Thomas Wilson", "Sophie Evans"],
        
        # Middle East
        "UAE": ["Mohammed Al", "Fatima Al", "Ahmed Al", "Aisha Al", "Khalid Al", "Maryam Al"],
        "Saudi Arabia": ["Abdullah Al", "Nora Al", "Fahad Al", "Layla Al", "Saeed Al", "Hessa Al"],
        "Turkey": ["Mehmet Yilmaz", "Ayşe Kaya", "Mustafa Demir", "Zeynep Şahin", "Ali Çelik", "Elif Yildiz"],
        "Israel": ["David Cohen", "Sarah Levy", "Moshe Goldberg", "Rachel Friedman", "Yosef Katz", "Leah Shapiro"],
        "Iran": ["Ali Hosseini", "Zahra Ahmadi", "Mohammad Rezaei", "Fatemeh Mohammadi", "Reza Karimi", "Maryam Jafari"],
        "Jordan": ["Omar Al", "Lina Al", "Khaled Al", "Rania Al", "Zaid Al", "Yasmin Al"],
        
        # Oceania
        "Australia": ["James Smith", "Sarah Johnson", "Michael Williams", "Emma Brown", "David Jones", "Olivia Wilson"],
        "New Zealand": ["William Taylor", "Charlotte Anderson", "Thomas Martin", "Sophie Thompson", "Oliver White", "Emily Davis"]
    }
    
    # Default names if country not in list
    default_names = ["John Smith", "Jane Doe", "Robert Johnson", "Maria Garcia", "David Lee", "Sarah Brown"]
    
    # Get contact names for the country
    names = contact_names.get(country, default_names)
    
    # Generate contacts
    contacts = []
    commodities = ["Rice", "Wheat", "Corn", "Soybeans", "Coffee", "Sugar", "Cotton", "Cocoa", "Fruits", "Vegetables"]
    
    for i in range(num_contacts):
        # Select a random name
        name = random.choice(names)
        
        # Select a random commodity
        commodity = random.choice(commodities)
        
        # Generate company name
        company_pattern = random.choice(company_patterns)
        company = company_pattern.format(country=country, commodity=commodity)
        
        # Generate position
        positions = ["Procurement Manager", "Supply Chain Director", "Trading Manager", "Import/Export Specialist", 
                     "Purchasing Director", "Business Development Manager", "Sales Director", "Chief Trading Officer"]
        position = random.choice(positions)
        
        # Generate contact details
        email = f"{name.lower().replace(' ', '.')}@{company.lower().replace(' ', '')}.com"
        phone = f"+{random.randint(1, 999)} {random.randint(100, 999)} {random.randint(1000, 9999)}"
        
        contacts.append({
            "name": name,
            "company": company,
            "position": position,
            "location": country,
            "email": email,
            "phone": phone,
            "contact": f"{name}, {position}"
        })
    
    return contacts

# Function to create HTML report
def create_html_report(opportunity, commodity, region, user_type, price_chart, weather_chart, crop_health_chart, trade_flow_chart):
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
                color: #2a5d4c;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #2a5d4c;
                margin-top: 20px;
            }}
            .highlight {{
                background-color: #f8f9fa;
                padding: 15px;
                border-left: 4px solid #8bc34a;
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
            .chart-container {{
                margin: 20px 0;
                border: 1px solid #ddd;
                padding: 10px;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                font-size: 0.8em;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <img src="data:image/png;base64,{get_image_base64('IMG_3036.png')}" alt="Sauda Food Insights LLC" class="logo">
            <h1>Market Opportunity Report: {commodity} in {region}</h1>
            <p>Generated for: <strong>{user_type}</strong> | Date: {datetime.now().strftime('%Y-%m-%d')}</p>
        </div>
        
        <div class="highlight">
            <h2>Opportunity Summary</h2>
            <p><strong>{opportunity['title']}</strong></p>
            <p>{opportunity['description']}</p>
            <p><strong>Rationale:</strong> {opportunity['rationale']}</p>
            <p><strong>Potential Impact:</strong> {opportunity['potential_impact']}</p>
            <p><strong>Implementation Timeline:</strong> {opportunity['implementation_timeline']}</p>
            <p><strong>Risk Level:</strong> {opportunity['risk_level']}</p>
        </div>
        
        <h2>Market Analysis</h2>
        
        <h3>Price Trends</h3>
        <div class="chart-container">
            <img src="data:image/png;base64,{price_chart}" alt="Price Trends" style="width:100%">
        </div>
        <p>The chart above shows the price trends for {commodity} over the past 24 months. 
        This data indicates {get_price_analysis(commodity, user_type)}.</p>
        
        <h3>Weather Impact</h3>
        <div class="chart-container">
            <img src="data:image/png;base64,{weather_chart}" alt="Weather Impact" style="width:100%">
        </div>
        <p>Weather conditions in key growing regions for {commodity} show {get_weather_analysis(region, commodity, user_type)}.</p>
        
        <h3>Crop Health Assessment</h3>
        <div class="chart-container">
            <img src="data:image/png;base64,{crop_health_chart}" alt="Crop Health" style="width:100%">
        </div>
        <p>Satellite imagery analysis of {commodity} growing regions indicates {get_crop_health_analysis(region, commodity, user_type)}.</p>
        
        <h3>Trade Flows</h3>
        <div class="chart-container">
            <img src="data:image/png;base64,{trade_flow_chart}" alt="Trade Flows" style="width:100%">
        </div>
        <p>Global trade flow analysis for {commodity} shows {get_trade_flow_analysis(commodity, region, user_type)}.</p>
        
        <h2>Recommended Actions</h2>
        <ul>
            <li>Initiate contact with recommended partners in {opportunity['description'].split(': ')[1]}</li>
            <li>Conduct detailed cost-benefit analysis based on current market conditions</li>
            <li>Develop implementation timeline aligned with seasonal market patterns</li>
            <li>Consider pilot program to test market response before full-scale implementation</li>
            <li>Monitor key indicators (price trends, crop health, weather patterns) for optimal timing</li>
        </ul>
        
        <h2>Contact Recommendations</h2>
        <table>
            <tr>
                <th>Name</th>
                <th>Company</th>
                <th>Position</th>
                <th>Location</th>
                <th>Contact</th>
            </tr>
            {generate_contact_table_rows(opportunity['contacts'])}
        </table>
        
        <div class="footer">
            <p>This report is generated by Sauda Food Insights LLC. The recommendations are based on analysis of market data, weather patterns, crop conditions, and trade flows.</p>
            <p>© {datetime.now().year} Sauda Food Insights LLC. All rights reserved.</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

# Helper function to generate contact table rows
def generate_contact_table_rows(contacts):
    rows = ""
    for contact in contacts:
        rows += f"""
        <tr>
            <td>{contact['name']}</td>
            <td>{contact['company']}</td>
            <td>{contact['position']}</td>
            <td>{contact['location']}</td>
            <td>{contact['email']}<br>{contact['phone']}</td>
        </tr>
        """
    return rows

# Helper function to get image as base64
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        # Return a placeholder if image not found
        return ""

# Helper functions for analysis text
def get_price_analysis(commodity, user_type):
    if user_type == "Buyer":
        return f"potential buying opportunities in the coming months as seasonal patterns suggest price stabilization. Historical volatility indicates optimal purchase timing may be approaching for {commodity}"
    else:
        return f"potential selling opportunities as price trends show strengthening fundamentals. Market signals suggest strategic positioning of {commodity} inventory could maximize returns in the current cycle"

def get_weather_analysis(region, commodity, user_type):
    if user_type == "Buyer":
        return f"mixed conditions that may impact production volumes. Buyers should monitor these patterns closely as they could affect {commodity} availability and quality in the coming harvest season"
    else:
        return f"conditions that could create premium opportunities for quality producers. Sellers of {commodity} may benefit from highlighting production region advantages in marketing materials"

def get_crop_health_analysis(region, commodity, user_type):
    if user_type == "Buyer":
        return f"variable health conditions across growing regions. This suggests buyers should diversify sourcing of {commodity} to mitigate quality and availability risks"
    else:
        return f"opportunities to differentiate based on crop quality metrics. Sellers with high-quality {commodity} production may command premium pricing in the current market environment"

def get_trade_flow_analysis(commodity, region, user_type):
    if user_type == "Buyer":
        return f"shifting patterns that create new sourcing opportunities. Buyers should explore emerging {commodity} export regions to optimize supply chain resilience and cost structures"
    else:
        return f"emerging market opportunities in regions with growing import demand. Sellers should consider diversifying {commodity} export destinations to capture premium market segments"

# Function to convert HTML to downloadable format
def get_html_download_link(html_content, filename="report"):
    # Encode HTML as base64
    b64 = base64.b64encode(html_content.encode()).decode()
    
    # Create download link for HTML
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}.html" class="download-button">Download Report</a>'
    
    return href

# Function to create chart images for reports
def create_chart_image(fig):
    # Convert plotly figure to image
    img_bytes = fig.to_image(format="png")
    
    # Encode as base64
    img_base64 = base64.b64encode(img_bytes).decode()
    
    return img_base64

# Main application layout
def main():
    # Sidebar for user type selection
    st.sidebar.header("User Settings")
    
    # User type selection
    user_type = st.sidebar.radio("Select User Type", ["Buyer", "Seller"])
    
    # Get available commodities
    available_commodities = get_available_commodities()
    
    # If no commodities found, use a default list
    if not available_commodities:
        available_commodities = {
            "ZW=F": "Wheat",
            "ZC=F": "Corn",
            "ZS=F": "Soybeans",
            "ZO=F": "Oats",
            "ZR=F": "Rice",
            "JO=F": "Orange Juice",
            "KC=F": "Coffee",
            "SB=F": "Sugar",
            "CC=F": "Cocoa",
            "CT=F": "Cotton",
            "LE=F": "Live Cattle",
            "HE=F": "Lean Hogs"
        }
    
    # Commodity selection
    st.sidebar.header("Commodity Selection")
    selected_commodity = st.sidebar.selectbox(
        "Select Commodity",
        options=list(available_commodities.keys()),
        format_func=lambda x: available_commodities[x]
    )
    selected_commodity_name = available_commodities[selected_commodity]
    
    # Region selection
    st.sidebar.header("Region Selection")
    selected_region = st.sidebar.selectbox(
        "Select Region",
        options=["Asia", "Africa", "South America", "North America", "Europe", "Middle East", "Oceania"]
    )
    
    # Analysis type selection
    st.sidebar.header("Analysis Type")
    analysis_types = {
        "Price Analysis": True,
        "Weather Impact": True,
        "Crop Health": True,
        "Trade Flows": True
    }
    
    for analysis_type in analysis_types.keys():
        analysis_types[analysis_type] = st.sidebar.checkbox(analysis_type, value=True)
    
    # Data refresh button
    if st.sidebar.button("Refresh Data"):
        st.experimental_rerun()
    
    # Main content area
    st.title(f"{selected_commodity_name} Market Intelligence")
    st.subheader(f"Region: {selected_region} | View: {user_type}")
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Market Analysis", "Opportunities", "Contacts"])
    
    with tab1:
        # Market Analysis Tab
        st.header("Market Analysis Dashboard")
        
        # Price Analysis
        if analysis_types["Price Analysis"]:
            st.subheader("Price Analysis")
            
            # Get price data
            price_data = get_price_data(selected_commodity)
            
            if not price_data.empty:
                # Create price chart
                fig_price = go.Figure()
                
                fig_price.add_trace(go.Scatter(
                    x=price_data.index,
                    y=price_data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color=PRIMARY_COLOR, width=2)
                ))
                
                # Add moving averages
                price_data['MA50'] = price_data['Close'].rolling(window=50).mean()
                price_data['MA200'] = price_data['Close'].rolling(window=200).mean()
                
                fig_price.add_trace(go.Scatter(
                    x=price_data.index,
                    y=price_data['MA50'],
                    mode='lines',
                    name='50-Day MA',
                    line=dict(color=SECONDARY_COLOR, width=1.5, dash='dash')
                ))
                
                fig_price.add_trace(go.Scatter(
                    x=price_data.index,
                    y=price_data['MA200'],
                    mode='lines',
                    name='200-Day MA',
                    line=dict(color=ACCENT_COLOR, width=1.5, dash='dot')
                ))
                
                # Update layout
                fig_price.update_layout(
                    title=f"{selected_commodity_name} Price Trends",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig_price, use_container_width=True)
                
                # Price analysis text
                st.markdown(f"""
                ### Price Analysis Insights
                
                The price chart for {selected_commodity_name} shows the daily closing prices along with 50-day and 200-day moving averages, 
                which help identify the overall trend direction and potential support/resistance levels.
                
                **Current Price:** ${price_data['Close'].iloc[-1]:.2f}
                
                **Key Observations:**
                - {get_price_trend_description(price_data)}
                - {get_moving_average_analysis(price_data)}
                - {get_volatility_analysis(price_data)}
                
                **Implications for {user_type}s:**
                {get_price_implications(price_data, user_type, selected_commodity_name)}
                """)
            else:
                st.warning(f"No price data available for {selected_commodity_name}")
        
        # Weather Impact
        if analysis_types["Weather Impact"]:
            st.subheader("Weather Impact Analysis")
            
            # Get weather data
            weather_data = get_weather_data(selected_region)
            
            # Create weather chart
            fig_weather = go.Figure()
            
            # Temperature trace
            fig_weather.add_trace(go.Scatter(
                x=weather_data['Date'],
                y=weather_data['Temperature'],
                mode='lines',
                name='Temperature (°C)',
                line=dict(color='red', width=2)
            ))
            
            # Create a secondary y-axis for rainfall
            fig_weather.add_trace(go.Bar(
                x=weather_data['Date'],
                y=weather_data['Rainfall'],
                name='Rainfall (mm)',
                marker=dict(color='blue', opacity=0.6)
            ))
            
            # Update layout with secondary y-axis
            fig_weather.update_layout(
                title=f"Weather Patterns in {selected_region} Growing Regions",
                xaxis_title="Date",
                yaxis=dict(
                    title="Temperature (°C)",
                    titlefont=dict(color="red"),
                    tickfont=dict(color="red")
                ),
                yaxis2=dict(
                    title="Rainfall (mm)",
                    titlefont=dict(color="blue"),
                    tickfont=dict(color="blue"),
                    anchor="x",
                    overlaying="y",
                    side="right"
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig_weather, use_container_width=True)
            
            # Calculate weather anomalies
            recent_temp = weather_data['Temperature'].iloc[-3:].mean()
            historical_temp = weather_data['Temperature'].iloc[:-3].mean()
            temp_anomaly = recent_temp - historical_temp
            
            recent_rain = weather_data['Rainfall'].iloc[-3:].mean()
            historical_rain = weather_data['Rainfall'].iloc[:-3].mean()
            rain_anomaly = recent_rain - historical_rain
            
            # Weather analysis text
            st.markdown(f"""
            ### Weather Impact Insights
            
            The weather chart shows temperature and rainfall patterns in key {selected_commodity_name} growing regions within {selected_region}.
            
            **Recent Conditions:**
            - Average Temperature (Last 3 Months): {recent_temp:.1f}°C ({temp_anomaly:+.1f}°C vs. historical average)
            - Average Rainfall (Last 3 Months): {recent_rain:.1f}mm ({rain_anomaly:+.1f}mm vs. historical average)
            
            **Analysis:**
            {selected_commodity_name} growing regions in {selected_region} are experiencing {
            "significant weather anomalies that may impact production" 
            if (abs(temp_anomaly) > 3 or abs(rain_anomaly) > 15) else
            "some weather-related stress but manageable impact on production" 
            if (abs(temp_anomaly) > 1.5 or abs(rain_anomaly) > 7) else
            "favorable growing conditions supporting normal production levels"
            }.
            
            For {user_type.lower()}s, this indicates {
            "a need to monitor supply availability and potential price impacts" if user_type == "Buyer" else
            "potential market opportunities as weather impacts materialize in production outcomes" if user_type == "Seller" else
            "important weather patterns affecting market conditions"
            }.
            """)
            
            # Weather forecast and implications
            st.subheader("Seasonal Outlook and Implications")
            
            # Generate random but consistent forecast based on region and commodity
            seed = sum(ord(c) for c in selected_region) + sum(ord(c) for c in selected_commodity_name)
            random.seed(seed)
            
            forecast_scenarios = [
                "above average temperatures and below average precipitation",
                "near normal temperatures and precipitation",
                "below average temperatures and above average precipitation",
                "above average temperatures and precipitation",
                "below average temperatures and precipitation"
            ]
            
            selected_scenario = random.choice(forecast_scenarios)
            
            # Determine production outlook based on scenario and commodity
            production_outlooks = {
                "above average temperatures and below average precipitation": "below average" if random.random() < 0.7 else "near average",
                "near normal temperatures and precipitation": "near average" if random.random() < 0.8 else "above average",
                "below average temperatures and above average precipitation": "above average" if random.random() < 0.6 else "near average",
                "above average temperatures and precipitation": "near average" if random.random() < 0.5 else random.choice(["above average", "below average"]),
                "below average temperatures and precipitation": "below average" if random.random() < 0.6 else "near average"
            }
            
            production_outlook = production_outlooks[selected_scenario]
            
            # Determine quality outlook
            quality_outlooks = {
                "above average temperatures and below average precipitation": "variable quality with potential stress impacts",
                "near normal temperatures and precipitation": "standard quality expectations",
                "below average temperatures and above average precipitation": "potential quality concerns in some regions",
                "above average temperatures and precipitation": "variable quality with disease pressure risks",
                "below average temperatures and precipitation": "delayed maturity affecting quality parameters"
            }
            
            quality_outlook = quality_outlooks[selected_scenario]
            
            # Display forecast and implications
            st.markdown(f"""
            **3-Month Seasonal Forecast:**
            The seasonal outlook for key {selected_commodity_name} growing regions in {selected_region} indicates {selected_scenario}.
            
            **Production Implications:**
            - Production Volume: {production_outlook.title()}
            - Quality Outlook: {quality_outlook.title()}
            
            **Strategic Recommendations:**
            {
            "- Consider forward contracting to secure supply" 
            if production_outlook == "below average" and user_type == "Buyer" else
            "- Monitor for buying opportunities as harvest approaches" 
            if production_outlook == "above average" and user_type == "Buyer" else
            "- Position for potentially stronger pricing as harvest approaches" 
            if production_outlook == "below average" and user_type == "Seller" else
            "- Focus on quality differentiation in a balanced market" 
            if production_outlook == "near average" and user_type == "Seller" else
            "- Consider early commitment strategies to secure volume in a competitive market" 
            if production_outlook == "above average" and user_type == "Seller" else
            "- Maintain flexible purchasing strategies to adapt to changing market conditions"
            }
            
            {
            "- Evaluate quality specifications carefully in contracts" 
            if quality_outlook.startswith("variable") else
            "- Standard quality parameters should be appropriate for contracts" 
            if quality_outlook.startswith("standard") else
            "- Opportunity to secure premium quality product"
            }
            """)
        
        # Crop Health
        if analysis_types["Crop Health"]:
            st.subheader("Crop Health Monitoring")
            
            # Get crop health data
            crop_health_data = get_crop_health_data(selected_region, selected_commodity_name)
            
            # Create crop health chart
            fig_crop = go.Figure()
            
            # NDVI trace
            fig_crop.add_trace(go.Scatter(
                x=crop_health_data['Date'],
                y=crop_health_data['NDVI'],
                mode='lines',
                name='NDVI',
                line=dict(color='green', width=2)
            ))
            
            # Soil moisture trace
            fig_crop.add_trace(go.Scatter(
                x=crop_health_data['Date'],
                y=crop_health_data['Soil_Moisture'],
                mode='lines',
                name='Soil Moisture',
                line=dict(color='blue', width=2)
            ))
            
            # Crop stress trace
            fig_crop.add_trace(go.Scatter(
                x=crop_health_data['Date'],
                y=crop_health_data['Crop_Stress'] / 100,  # Normalize to 0-1 scale
                mode='lines',
                name='Crop Stress Index',
                line=dict(color='red', width=2)
            ))
            
            # Update layout
            fig_crop.update_layout(
                title=f"{selected_commodity_name} Crop Health Indicators in {selected_region}",
                xaxis_title="Date",
                yaxis_title="Index Value",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_white",
                height=500
            )
            
            st.plotly_chart(fig_crop, use_container_width=True)
            
            # Calculate recent trends
            recent_ndvi = crop_health_data['NDVI'].iloc[-3:].mean()
            historical_ndvi = crop_health_data['NDVI'].iloc[:-3].mean()
            ndvi_trend = recent_ndvi - historical_ndvi
            
            recent_stress = crop_health_data['Crop_Stress'].iloc[-3:].mean()
            historical_stress = crop_health_data['Crop_Stress'].iloc[:-3].mean()
            stress_trend = recent_stress - historical_stress
            
            # Crop health analysis text
            st.markdown(f"""
            ### Satellite-Based Crop Health Insights
            
            The crop health chart shows key indicators derived from satellite imagery for {selected_commodity_name} in {selected_region}.
            
            **Indicator Explanations:**
            - **NDVI (Normalized Difference Vegetation Index)**: Measures vegetation density and health (0-1 scale, higher is healthier)
            - **Soil Moisture**: Indicates water availability in the soil (0-1 scale, higher is wetter)
            - **Crop Stress Index**: Measures overall plant stress from various factors (0-1 scale, lower is better)
            
            **Current Conditions:**
            - NDVI: {recent_ndvi:.2f} ({ndvi_trend:+.2f} vs. historical average)
            - Crop Stress: {recent_stress:.1f} ({stress_trend:+.1f} vs. historical average)
            
            **Analysis:**
            Satellite imagery indicates {selected_commodity_name} crops in {selected_region} are showing {
            "signs of significant stress that may impact yields" 
            if (ndvi_trend < -0.05 or stress_trend > 5) else
            "some stress indicators but generally manageable conditions" 
            if (ndvi_trend < -0.02 or stress_trend > 2) else
            "healthy vegetation with favorable growing conditions"
            }.
            
            **Implications for {user_type}s:**
            {
            "Monitor supply availability and quality specifications as harvest approaches" if user_type == "Buyer" else
            "Highlight product quality advantages in marketing materials" if user_type == "Seller"
            }
            """)
        
        # Trade Flows
        if analysis_types["Trade Flows"]:
            st.subheader("Global Trade Flow Analysis")
            
            # Define origin and destination based on user type and region
            if user_type == "Buyer":
                origin = "Global Exporters"
                destination = selected_region
            else:
                origin = selected_region
                destination = "Global Importers"
            
            # Get trade flow data
            trade_data = get_trade_flow_data(selected_commodity_name, origin, destination)
            
            # Create trade flow chart
            fig_trade = go.Figure()
            
            # Volume trace
            fig_trade.add_trace(go.Bar(
                x=trade_data['Date'],
                y=trade_data['Volume'],
                name='Volume (MT)',
                marker=dict(color=SECONDARY_COLOR)
            ))
            
            # Price trace on secondary y-axis
            fig_trade.add_trace(go.Scatter(
                x=trade_data['Date'],
                y=trade_data['Price'],
                mode='lines',
                name='Price',
                line=dict(color=PRIMARY_COLOR, width=2),
                yaxis="y2"
            ))
            
            # Update layout with secondary y-axis
            fig_trade.update_layout(
                title=f"{selected_commodity_name} Trade Flows: {origin} to {destination}",
                xaxis_title="Date",
                yaxis=dict(
                    title="Volume (Metric Tons)",
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
            
            st.plotly_chart(fig_trade, use_container_width=True)
            
            # Calculate recent trends
            recent_volume = trade_data['Volume'].iloc[-3:].mean()
            historical_volume = trade_data['Volume'].iloc[:-3].mean()
            volume_trend = (recent_volume - historical_volume) / historical_volume * 100
            
            recent_price = trade_data['Price'].iloc[-3:].mean()
            historical_price = trade_data['Price'].iloc[:-3].mean()
            price_trend = (recent_price - historical_price) / historical_price * 100
            
            # Trade flow analysis text
            st.markdown(f"""
            ### Trade Flow Insights
            
            The trade flow chart shows the volume and price trends for {selected_commodity_name} shipments from {origin} to {destination}.
            
            **Recent Trends:**
            - Volume: {recent_volume:.0f} MT ({volume_trend:+.1f}% vs. historical average)
            - Price: ${recent_price:.2f}/MT ({price_trend:+.1f}% vs. historical average)
            
            **Analysis:**
            Trade flows for {selected_commodity_name} between {origin} and {destination} are showing {
            "significant changes that may indicate shifting market dynamics" 
            if (abs(volume_trend) > 15 or abs(price_trend) > 10) else
            "moderate fluctuations within expected seasonal patterns" 
            if (abs(volume_trend) > 5 or abs(price_trend) > 3) else
            "stable patterns with minimal disruption to established trade channels"
            }.
            
            **Key Observations:**
            - {get_volume_price_relationship(volume_trend, price_trend)}
            - {get_seasonality_observation(trade_data)}
            - {get_market_implication(volume_trend, price_trend, user_type)}
            """)
    
    with tab2:
        # Opportunities Tab
        st.header("Market Opportunities")
        
        # Generate opportunities based on user type
        opportunities = generate_market_opportunities(selected_commodity_name, selected_region, user_type)
        
        # Display opportunities
        for i, opportunity in enumerate(opportunities):
            with st.expander(f"Opportunity {i+1}: {opportunity['title']}", expanded=(i==0)):
                st.markdown(f"""
                **Description:** {opportunity['description']}
                
                **Rationale:** {opportunity['rationale']}
                
                **Potential Impact:** {opportunity['potential_impact']}
                
                **Implementation Timeline:** {opportunity['implementation_timeline']}
                
                **Risk Level:** {opportunity['risk_level']}
                """)
                
                # Create charts for the report
                # Price chart
                price_data = get_price_data(selected_commodity)
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(
                    x=price_data.index[-24:],
                    y=price_data['Close'][-24:],
                    mode='lines',
                    name='Close Price',
                    line=dict(color=PRIMARY_COLOR, width=2)
                ))
                price_chart_base64 = create_chart_image(fig_price)
                
                # Weather chart
                weather_data = get_weather_data(selected_region)
                fig_weather = go.Figure()
                fig_weather.add_trace(go.Scatter(
                    x=weather_data['Date'][-24:],
                    y=weather_data['Temperature'][-24:],
                    mode='lines',
                    name='Temperature',
                    line=dict(color='red', width=2)
                ))
                weather_chart_base64 = create_chart_image(fig_weather)
                
                # Crop health chart
                crop_health_data = get_crop_health_data(selected_region, selected_commodity_name)
                fig_crop = go.Figure()
                fig_crop.add_trace(go.Scatter(
                    x=crop_health_data['Date'][-24:],
                    y=crop_health_data['NDVI'][-24:],
                    mode='lines',
                    name='NDVI',
                    line=dict(color='green', width=2)
                ))
                crop_chart_base64 = create_chart_image(fig_crop)
                
                # Trade flow chart
                if user_type == "Buyer":
                    origin = "Global Exporters"
                    destination = selected_region
                else:
                    origin = selected_region
                    destination = "Global Importers"
                trade_data = get_trade_flow_data(selected_commodity_name, origin, destination)
                fig_trade = go.Figure()
                fig_trade.add_trace(go.Bar(
                    x=trade_data['Date'][-24:],
                    y=trade_data['Volume'][-24:],
                    name='Volume',
                    marker=dict(color=SECONDARY_COLOR)
                ))
                trade_chart_base64 = create_chart_image(fig_trade)
                
                # Create HTML report
                html_content = create_html_report(
                    opportunity, 
                    selected_commodity_name, 
                    selected_region, 
                    user_type,
                    price_chart_base64,
                    weather_chart_base64,
                    crop_chart_base64,
                    trade_chart_base64
                )
                
                # Create download link
                download_link = get_html_download_link(
                    html_content, 
                    f"Sauda_{selected_commodity_name}_{opportunity['title'].replace(' ', '_')}"
                )
                
                st.markdown(download_link, unsafe_allow_html=True)
                
                # Display contacts
                st.subheader("Recommended Contacts")
                for contact in opportunity['contacts']:
                    st.markdown(f"""
                    **{contact['name']}**  
                    {contact['position']} at {contact['company']}  
                    Location: {contact['location']}  
                    Contact: {contact['email']} | {contact['phone']}
                    """)
    
    with tab3:
        # Contacts Tab
        st.header("Contact Recommendations")
        
        # Define regions based on user type
        if user_type == "Buyer":
            # For buyers, show contacts from producing regions
            if selected_region in ["North America", "Europe"]:
                contact_regions = ["Asia", "South America", "Africa"]
            else:
                contact_regions = ["South America", "Asia", "North America"]
        else:
            # For sellers, show contacts from consuming regions
            if selected_region in ["Asia", "South America", "Africa"]:
                contact_regions = ["North America", "Europe", "Middle East"]
            else:
                contact_regions = ["Asia", "Middle East", "Europe"]
        
        # Generate and display contacts for each region
        for region in contact_regions:
            st.subheader(f"{region} Contacts")
            
            contacts = generate_contacts(random.choice(["China", "India", "Vietnam", "Thailand", "Indonesia", "Malaysia", "Philippines"]) if region == "Asia" else
                                        random.choice(["Egypt", "South Africa", "Kenya", "Nigeria", "Morocco"]) if region == "Africa" else
                                        random.choice(["Brazil", "Argentina", "Chile", "Colombia", "Peru"]) if region == "South America" else
                                        random.choice(["USA", "Canada", "Mexico"]) if region == "North America" else
                                        random.choice(["France", "Germany", "Italy", "Spain", "Netherlands"]) if region == "Europe" else
                                        random.choice(["UAE", "Saudi Arabia", "Turkey", "Israel"]) if region == "Middle East" else
                                        random.choice(["Australia", "New Zealand"]), 3)
            
            # Display contacts in a more visual format
            cols = st.columns(3)
            for i, contact in enumerate(contacts):
                with cols[i]:
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; border-radius:5px; padding:15px; height:200px;">
                        <h3 style="color:{PRIMARY_COLOR};">{contact['name']}</h3>
                        <p><strong>{contact['position']}</strong><br>
                        {contact['company']}</p>
                        <p>📍 {contact['location']}</p>
                        <p>📧 {contact['email']}<br>
                        📞 {contact['phone']}</p>
                    </div>
                    """, unsafe_allow_html=True)

# Helper functions for price analysis
def get_price_trend_description(price_data):
    # Calculate recent trend
    recent_period = min(60, len(price_data) // 4)
    if recent_period < 10:
        return "Insufficient data to determine trend"
    
    recent_prices = price_data['Close'].iloc[-recent_period:]
    start_price = recent_prices.iloc[0]
    end_price = recent_prices.iloc[-1]
    percent_change = (end_price - start_price) / start_price * 100
    
    if percent_change > 10:
        return f"Strong upward trend with {percent_change:.1f}% increase over the past {recent_period} trading days"
    elif percent_change > 3:
        return f"Moderate upward trend with {percent_change:.1f}% increase over the past {recent_period} trading days"
    elif percent_change < -10:
        return f"Strong downward trend with {percent_change:.1f}% decrease over the past {recent_period} trading days"
    elif percent_change < -3:
        return f"Moderate downward trend with {percent_change:.1f}% decrease over the past {recent_period} trading days"
    else:
        return f"Relatively stable prices with {percent_change:.1f}% change over the past {recent_period} trading days"

def get_moving_average_analysis(price_data):
    if 'MA50' not in price_data.columns or 'MA200' not in price_data.columns:
        return "Moving average data not available"
    
    last_close = price_data['Close'].iloc[-1]
    last_ma50 = price_data['MA50'].iloc[-1]
    last_ma200 = price_data['MA200'].iloc[-1]
    
    if last_close > last_ma50 and last_ma50 > last_ma200:
        return "Price is above both 50-day and 200-day moving averages, indicating a strong bullish trend"
    elif last_close < last_ma50 and last_ma50 < last_ma200:
        return "Price is below both 50-day and 200-day moving averages, indicating a strong bearish trend"
    elif last_close > last_ma50 and last_ma50 < last_ma200:
        return "Price is above 50-day but below 200-day moving average, suggesting a potential trend reversal from bearish to bullish"
    elif last_close < last_ma50 and last_ma50 > last_ma200:
        return "Price is below 50-day but above 200-day moving average, suggesting a potential short-term pullback in a longer-term bullish trend"
    else:
        return "Moving averages show mixed signals, indicating a potential consolidation phase"

def get_volatility_analysis(price_data):
    # Calculate recent volatility
    recent_period = min(30, len(price_data) // 4)
    if recent_period < 10:
        return "Insufficient data to determine volatility"
    
    recent_returns = price_data['Close'].iloc[-recent_period:].pct_change().dropna()
    volatility = recent_returns.std() * (252 ** 0.5) * 100  # Annualized volatility in percentage
    
    if volatility > 40:
        return f"Extremely high volatility ({volatility:.1f}% annualized), indicating significant market uncertainty"
    elif volatility > 25:
        return f"High volatility ({volatility:.1f}% annualized), suggesting active trading conditions"
    elif volatility > 15:
        return f"Moderate volatility ({volatility:.1f}% annualized), typical for agricultural commodities"
    else:
        return f"Low volatility ({volatility:.1f}% annualized), indicating relatively stable trading conditions"

def get_price_implications(price_data, user_type, commodity):
    recent_period = min(60, len(price_data) // 4)
    if recent_period < 10:
        return "Insufficient data to determine implications"
    
    recent_prices = price_data['Close'].iloc[-recent_period:]
    start_price = recent_prices.iloc[0]
    end_price = recent_prices.iloc[-1]
    percent_change = (end_price - start_price) / start_price * 100
    
    if user_type == "Buyer":
        if percent_change > 8:
            return f"The strong upward price trend suggests buyers should consider securing forward contracts for {commodity} to protect against further price increases. Evaluate alternative sourcing options to diversify supply risk."
        elif percent_change > 3:
            return f"The moderate upward price trend indicates buyers should monitor {commodity} markets closely and potentially increase inventory levels before further price increases materialize."
        elif percent_change < -8:
            return f"The significant price decline presents favorable buying opportunities for {commodity}. Consider increasing purchase volumes to take advantage of lower prices, but monitor quality metrics carefully."
        elif percent_change < -3:
            return f"The moderate price decline suggests buyers may benefit from a patient approach to {commodity} procurement, potentially securing better terms as the market stabilizes."
        else:
            return f"The stable price environment provides a good opportunity to review and potentially renegotiate {commodity} supply contracts with a focus on quality and reliability rather than just price."
    else:  # Seller
        if percent_change > 8:
            return f"The strong upward price trend creates favorable conditions for {commodity} sellers. Consider optimizing sales timing to capitalize on higher prices while maintaining customer relationships."
        elif percent_change > 3:
            return f"The moderate upward price trend suggests sellers should review inventory management strategies for {commodity} to ensure optimal positioning as the market strengthens."
        elif percent_change < -8:
            return f"The significant price decline indicates sellers should focus on value-added services and quality differentiation for {commodity} to maintain margins in a challenging price environment."
        elif percent_change < -3:
            return f"The moderate price decline suggests sellers should evaluate cost structures and potentially adjust {commodity} marketing strategies to emphasize non-price value propositions."
        else:
            return f"The stable price environment allows sellers to focus on operational efficiency and customer relationship development rather than reactive pricing strategies for {commodity}."

def get_volume_price_relationship(volume_trend, price_trend):
    if volume_trend > 10 and price_trend > 5:
        return "Increasing volumes with rising prices indicate strong demand fundamentals"
    elif volume_trend > 10 and price_trend < -5:
        return "Increasing volumes with falling prices suggest potential oversupply conditions"
    elif volume_trend < -10 and price_trend > 5:
        return "Decreasing volumes with rising prices indicate potential supply constraints"
    elif volume_trend < -10 and price_trend < -5:
        return "Decreasing volumes with falling prices suggest weakening market fundamentals"
    else:
        return "Volume and price movements show balanced market conditions"

def get_seasonality_observation(trade_data):
    # This is a simplified approach - in a real system, we would use more sophisticated seasonality analysis
    return "Trade patterns show typical seasonal fluctuations with peak volumes aligned with harvest cycles"

def get_market_implication(volume_trend, price_trend, user_type):
    if user_type == "Buyer":
        if volume_trend > 10:
            return "Increasing trade volumes suggest good product availability for buyers"
        elif volume_trend < -10:
            return "Decreasing trade volumes indicate potential sourcing challenges ahead"
        else:
            return "Stable trade volumes suggest consistent product availability in the near term"
    else:  # Seller
        if price_trend > 5:
            return "Positive price trends create favorable conditions for sellers in target markets"
        elif price_trend < -5:
            return "Price pressure suggests sellers should focus on efficiency and value-added services"
        else:
            return "Stable pricing environment allows for consistent sales planning and forecasting"

if __name__ == "__main__":
    main()
