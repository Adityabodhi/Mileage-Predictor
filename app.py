from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# 🔹 Load trained models
with open("hybrid_mileage_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("linear_regression_model.pkl", "rb") as file:
    lr_model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 🔹 Get input values from user (6 Features)
        engine_size = float(request.form["engine_size"])  # Engine displacement (L)
        cylinders = int(request.form["cylinders"])  # Number of cylinders
        transmission = int(request.form["transmission"])  # 0 = Manual, 1 = Automatic
        drive_type = int(request.form["drive_type"])  # 0 = FWD, 1 = RWD, 2 = AWD
        fuel_type = int(request.form["fuel_type"])  # Encoded Fuel Type
        highway_mpg = float(request.form["highway_mpg"])  # Highway MPG

        # 🔹 Prepare input array with exactly 6 features
        input_features = np.array([[engine_size, cylinders, transmission, drive_type, fuel_type, highway_mpg]])

        # 🔹 Predict mileage using the hybrid model (Random Forest)
        predicted_mileage = model.predict(input_features)[0]

        return render_template("index.html", prediction_text=f"Predicted Mileage: {predicted_mileage:.2f} MPG")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
