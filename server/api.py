from flask import Flask, request, jsonify, render_template
import json
import pickle
import os
import pandas as pd

#Environment variables
DATASET_PATH=os.getenv('DATASET_PATH')
MODEL_PATH=os.getenv('MODEL_PATH')
METRICS_PATH=os.getenv('METRICS_PATH')

app = Flask(__name__)

model = pickle.load(open(MODEL_PATH,'rb'))

@app.route("/v1/categorize", methods=["POST"])
def predict():
    prod = request.json["products"]
    if prod == json.dumps({}):
        return {"error": "Invalid content"}, 400

    return {"categories": model.predict(pd.DataFrame(prod)).tolist()}

app.run()