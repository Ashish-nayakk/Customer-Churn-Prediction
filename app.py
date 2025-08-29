from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import json


# Initialize Flask app
app = Flask(__name__)

# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("Churn_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    prob = None
    gauge_json = None

    if request.method == "POST":
        # Get form inputs
        age = int(request.form["age"])
        gender = request.form["gender"]
        tenure = int(request.form["tenure"])
        monthly_charges = float(request.form["monthly_charges"])
        total_charges = float(request.form["total_charges"])
        contract = request.form["contract"]
        internet = request.form["internet"]
        techsupport = request.form["techsupport"]

        gender_value = 1 if gender == "Female" else 0

        # Build input dictionary with exact dummy names
        input_data = {
            "Age": age,
            "Gender": gender_value,
            "Tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "ContractType_One-Year": 1 if contract == "One year" else 0,
            "ContractType_Two-Year": 1 if contract == "Two year" else 0,
            "InternetService_DSL": 1 if internet == "DSL" else 0,
            "InternetService_Fiber Optic": 1 if internet == "Fiber Optic" else 0,
            "TechSupport_Yes": 1 if techsupport == "Yes" else 0,
        }

        # Convert to DataFrame
        X_df = pd.DataFrame([input_data])

        # Align with training feature order
        X_df = X_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

        # Scale & predict
        X_scaled = scaler.transform(X_df)
        prediction = model.predict(X_scaled)[0]

        # Probability handling (SVC may not support predict_proba)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_scaled)[0][1] * 100
        else:
            score = model.decision_function(X_scaled)[0]
            prob = 100 / (1 + np.exp(-score))

        # Result
        result = "Churn" if prediction == 1 else "Not Churn"

        # Gauge chart (Plotly)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': "Churn Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 50], 'color': "green"},
                    {'range': [50, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.8,
                    'value': prob
                }
            }
        ))

        gauge_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template("index.html", 
                               result=result, prob=round(prob, 2), 
                               gauge_json=gauge_json)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
