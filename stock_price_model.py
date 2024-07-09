import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv('stock_prices.csv')  # Ensure your dataset is in the correct path

# Preprocessing
df.fillna(df.mean(), inplace=True)  # Handle missing values
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open('stock_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)
