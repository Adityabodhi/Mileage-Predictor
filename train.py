import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# 🔹 Load the dataset
df = pd.read_csv("vehicles.csv", low_memory=False)

# 🔹 Select only 5 features + target
df = df[["displ", "cylinders", "trany", "drive", "fuelType1", "comb08"]]  # No "highway08"

# 🔹 Handle missing values
df.dropna(inplace=True)

# 🔹 Encode categorical values
df["trany"] = df["trany"].astype("category").cat.codes  # Transmission: 0 = Manual, 1 = Automatic
df["drive"] = df["drive"].astype("category").cat.codes  # Drive Type: 0 = FWD, 1 = RWD, 2 = AWD
df["fuelType1"] = df["fuelType1"].astype("category").cat.codes  # Fuel Type Encoding

# 🔹 Define features (X) and target (y)
X = df.drop(columns=["comb08"])  # Features: Exactly 5 features now
y = df["comb08"]  # Target: Mileage (MPG)

# 🔹 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train models
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# 🔹 Save models
with open("linear_regression_model.pkl", "wb") as file:
    pickle.dump(lr, file)

with open("hybrid_mileage_model.pkl", "wb") as file:
    pickle.dump(rf, file)

print("✅ Model trained and saved with exactly 5 features!")
