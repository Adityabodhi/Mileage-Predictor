import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 🔹 Load the dataset
df = pd.read_csv("C:/Users/adity/Downloads/vehicles.csv", low_memory=False)

# 🔹 Select exactly 5 features + target
df = df[["displ", "cylinders", "trany", "drive", "fuelType1", "comb08"]]  # Removed extra features

# 🔹 Handle missing values
df.dropna(inplace=True)

# 🔹 Encode categorical values
df["trany"] = df["trany"].astype("category").cat.codes
df["drive"] = df["drive"].astype("category").cat.codes
df["fuelType1"] = df["fuelType1"].astype("category").cat.codes

# 🔹 Define features (X) and target (y)
X = df.drop(columns=["comb08"])  # Features: Now only 5 features
y = df["comb08"]  # Target: Mileage

# 🔹 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train RandomForestRegressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 🔹 Save trained model
with open("hybrid_mileage_model.pkl", "wb") as file:
    pickle.dump(rf, file)

print("✅ Model trained with exactly 5 features and saved!")
