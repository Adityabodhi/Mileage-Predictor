from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# 🔹 Load trained models
with open("hybrid_mileage_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 🔹 Get input values from user (Only 5 Features)
        engine_size = float(request.form["engine_size"])
        cylinders = int(request.form["cylinders"])
        transmission = int(request.form["transmission"])
        drive_type = int(request.form["drive_type"])
        fuel_type = int(request.form["fuel_type"])

        # 🔹 Prepare input array with exactly 5 features
        input_features = np.array([[engine_size, cylinders, transmission, drive_type, fuel_type]])

        # 🔹 Predict mileage using the model
        predicted_mileage = model.predict(input_features)[0]

        return render_template("index.html", prediction_text=f"Predicted Mileage: {predicted_mileage:.2f} MPG")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
