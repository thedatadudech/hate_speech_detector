from flask import Flask, request, jsonify
import pandas as pd
from predict import predict as model_predict

app = Flask(__name__)

# Load the model


@app.route("/predict", methods=["POST"])
def predict():
    json_input = request.json
    text = json_input["text"]
    prediction = model_predict(text)
    # For simplicity, just return the received text with a static response
    response = {"received_text": text, "prediction": str(prediction)}
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
