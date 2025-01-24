import streamlit as st
import requests
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('wind_power_model.pkl')  # Path to your trained model
scaler = joblib.load('scaler.pkl')  # Path to your scaler object

# Constants for wind power calculation
rho = 1.225  # Air density in kg/m^3
turbine_radius = 20  # Radius of the turbine in meters (adjust based on your turbine)

# Fetch live data from Open-Meteo API
def fetch_live_data():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 9.5722,  # Latitude for your location (9°34'20"N)
        "longitude": 77.6758,  # Longitude for your location (77°40'33"E)
        "hourly": "temperature_2m,relativehumidity_2m,dewpoint_2m,windspeed_10m,windspeed_100m,winddirection_10m,winddirection_100m,windgusts_10m",
        "timezone": "Asia/Kolkata"
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data['hourly']

# Preprocess live data
def preprocess_live_data(live_data, scaler):
    df_live = pd.DataFrame(live_data)
    features = ['temperature_2m', 'relativehumidity_2m', 'dewpoint_2m', 
                'windspeed_10m', 'windspeed_100m', 'winddirection_10m', 
                'winddirection_100m', 'windgusts_10m']
    
    # Ensure the live data is in the correct format for the scaler
    scaled_data = scaler.transform(df_live[features])
    
    return scaled_data

# Calculate power production from live data
def calculate_power(windspeed):
    try:
        # Convert windspeed to float and handle errors
        windspeed = float(windspeed)
        
        if windspeed < 0:
            return 0  # Power is zero for negative wind speeds (invalid data)
        
        # Use the formula to calculate power from wind speed
        swept_area = np.pi * turbine_radius**2  # Area of the turbine blades
        power = 0.5 * rho * swept_area * (windspeed**3)  # Power in watts
        power_kw = power / 1000  # Convert to kilowatts
        return power_kw
    except (ValueError, TypeError):
        # Return None if the windspeed is invalid or cannot be converted
        return None

# Streamlit app interface
st.title("Wind Power Prediction")

# Fetch live data and process it
live_data = fetch_live_data()

# Preprocess the live data to match the model's input format
scaled_live_data = preprocess_live_data(live_data, scaler)

# Make predictions using the model
predicted_power = model.predict(scaled_live_data)

# Calculate live power production for each hour based on wind speed
windspeed_10m = pd.DataFrame(live_data)['windspeed_10m']  # Wind speed at 10m height
calculated_power = windspeed_10m.apply(calculate_power)

# Display the predictions and live power production in the app
st.write("Predicted Wind Power (kW) for the upcoming hours:")
predicted_power_df = pd.DataFrame(predicted_power, columns=["Predicted Power (kW)"])
predicted_power_df['Hour'] = live_data['time']  # Add the time of prediction for reference
st.write(predicted_power_df)

st.write("Live Power Production (kW) based on wind speed:")
calculated_power_df = pd.DataFrame(calculated_power, columns=["Live Power Production (kW)"])
calculated_power_df['Hour'] = live_data['time']  # Add the time of calculation for reference
st.write(calculated_power_df)

# If you want to display the actual live data in the same interface
st.write("Live Weather Data (Last 24 hours):")
live_data_df = pd.DataFrame(live_data)
st.write(live_data_df)
def calculate_power(windspeed):
    try:
        # Convert windspeed to numeric
        windspeed = float(windspeed)
        
        # Ensure windspeed is valid
        if windspeed < 0:
            return 0  # Return zero for invalid windspeed
        
        # Use the formula to calculate power from wind speed
        swept_area = np.pi * turbine_radius**2  # Area of the turbine blades
        power = 0.5 * rho * swept_area * (windspeed**3)  # Power in watts
        power_kw = power / 1000  # Convert to kilowatts
        return power_kw
    except ValueError:
        return None  # Return None if the windspeed is invalid

