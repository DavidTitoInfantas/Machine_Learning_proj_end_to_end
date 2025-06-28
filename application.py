"""Run the Flask application."""

import os

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# Route for a home page


@app.route("/")
def index():
    """Render the index page."""
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    """Predict the data point."""
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get(
                "parental_level_of_education"
            ),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get(
                "test_preparation_course"
            ),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score")),
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template("home.html", results=results[0])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Endpoint to predict student performance from JSON input."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        custom_data = CustomData(
            gender=data.get("gender"),
            race_ethnicity=data.get("race_ethnicity"),
            parental_level_of_education=data.get(
                "parental_level_of_education"
            ),
            lunch=data.get("lunch"),
            test_preparation_course=data.get("test_preparation_course"),
            reading_score=float(data.get("reading_score")),
            writing_score=float(data.get("writing_score")),
        )
        pred_df = custom_data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return jsonify({"prediction": results[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0")  # , port=port, debug=True)
    # Port is set to http://127.0.0.1:5000/
