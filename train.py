import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# 🔹 Load the dataset
df = pd.read_csv("C:/Users/adity/Downloads/vehicles.csv", low_memory=False)

# 🔹 Select exactly 6 features + target (MPG)
df = df[["displ", "cylinders", "trany", "drive", "fuelType1", "highway08", "comb08"]]  # Target: comb08 (MPG)

# 🔹 Handle missing values
df.dropna(inplace=True)

# 🔹 Encode categorical values
df["trany"] = df["trany"].astype("category").cat.codes  # Transmission: 0 = Manual, 1 = Automatic
df["drive"] = df["drive"].astype("category").cat.codes  # Drive Type: 0 = FWD, 1 = RWD, 2 = AWD
df["fuelType1"] = df["fuelType1"].astype("category").cat.codes  # Fuel Type Encoding

# 🔹 Define features (X) and target (y)
X = df.drop(columns=["comb08"])  # Features: 6 input features
y = df["comb08"]  # Target: Mileage (MPG)

# 🔹 Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Initialize models
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 🔹 Train both models separately on exactly 6 features
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# 🔹 Save both trained models
with open("linear_regression_model.pkl", "wb") as file:
    pickle.dump(lr, file)

with open("hybrid_mileage_model.pkl", "wb") as file:
    pickle.dump(rf, file)

print("✅ Hybrid model trained and saved successfully using Pickle!")
