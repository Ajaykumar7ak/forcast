import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("C:\wind_2\Location1.csv")  # Replace with your actual file path

# Ensure Time is in datetime format
df['Time'] = pd.to_datetime(df['Time'])

# Features and target
features = ['temperature_2m', 'relativehumidity_2m', 'dewpoint_2m', 
            'windspeed_10m', 'windspeed_100m', 'winddirection_10m', 
            'winddirection_100m', 'windgusts_10m']
target = 'Power'

# Scale the features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Split the data into training and testing sets (80% training, 20% testing)
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")

import joblib

# Save the trained model
joblib.dump(model, 'wind_power_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

print("Model and Scaler saved successfully!")

