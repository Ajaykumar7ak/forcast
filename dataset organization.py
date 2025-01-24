import pandas as pd # Data manipulation and analysis
import numpy as np # Numerical operations on large, multi-dimensional arrays and matrices
import matplotlib.pyplot as plt # Plotting and visualization of data
import seaborn as sns # Advanced data visualization based on matplotlib
import scipy.stats as stats # Scientific computing and technical computing
import sklearn # Machine learning and predictive data analysis
location1 = pd.read_csv("C:\wind_2\Location1.csv")
location2 = pd.read_csv("C:\wind_2\Location2.csv")

# Check for missing values in each DataFrame
missing_values_location1 = location1.isnull().sum()
missing_values_location2 = location2.isnull().sum()

# Filter out columns with no missing values
missing_columns_location1 = missing_values_location1[missing_values_location1 > 0]
missing_columns_location2 = missing_values_location2[missing_values_location2 > 0]

# Display the columns with missing values for each DataFrame
print("Columns with missing values in Location1:\n", missing_columns_location1)
print("\nColumns with missing values in Location2:\n", missing_columns_location2)
duplicates_location1 = location1[location1.duplicated()]
duplicates_location2 = location2[location2.duplicated()]

# Display the duplicates for each DataFrame
print("Duplicates in Location1:\n", duplicates_location1)
print("\nDuplicates in Location2:\n", duplicates_location2)

# Convert 'Time' column to datetime in each DataFrame
location1['Time'] = pd.to_datetime(location1['Time'])
location2['Time'] = pd.to_datetime(location2['Time'])

# You can check the conversion by displaying the .info() again
print("Updated Information for Location1 DataFrame:")
location1.info()
print("\nUpdated Information for Location2 DataFrame:")
location2.info()
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Creating a subplot layout
fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=("Temperature Distribution", "Power Output vs. Time", "Wind Speed vs. Power",
                    "Humidity vs. Dew Point", "Wind Gusts Distribution", "Correlation Heatmap",
                    "Power Output Comparison", "Temperature vs. Power"))

# 1. Temperature Distribution (Histogram)
fig.add_trace(go.Histogram(x=location1['temperature_2m'], marker_color='blue'), row=1, col=1)

# 2. Power Output vs. Time (Time Series)
fig.add_trace(go.Scatter(x=location1['Time'], y=location1['Power'], mode='lines', line_color='green'), row=1, col=2)

# 3. Wind Speed and Power Relationship (Scatter Plot)
fig.add_trace(go.Scatter(x=location1['windspeed_10m'], y=location1['Power'], mode='markers', marker_color='red'), row=1, col=3)

# 4. Humidity and Dew Point Analysis (Scatter Plot)
fig.add_trace(go.Scatter(x=location1['relativehumidity_2m'], y=location1['dewpoint_2m'], mode='markers', marker_color='purple'), row=2, col=1)

# 6. Wind Gusts Distribution (Histogram)
fig.add_trace(go.Histogram(x=location1['windgusts_10m'], marker_color='orange'), row=2, col=2)

# 7. Correlation Heatmap
corr = location1.corr()
fig.add_trace(go.Heatmap(z=corr, x=corr.columns, y=corr.columns, colorscale='Viridis'), row=2, col=3)

# 8. Power Output Comparison (Box Plot)
fig.add_trace(go.Box(y=location1['Power'], marker_color='cyan'), row=3, col=1)

# 9. Temperature vs. Power (Scatter Plot)
fig.add_trace(go.Scatter(x=location1['temperature_2m'], y=location1['Power'], mode='markers', marker_color='magenta'), row=3, col=2)

# Update layout
fig.update_layout(height=1200, width=1200, title_text="Data Analysis for Location 1", showlegend=False)
fig.show()

# Note: Rose plot and Time Series Decomposition are not included due to complexity and lack of direct support in Plotly.