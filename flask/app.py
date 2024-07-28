# app.py
import mlflow
import joblib
import os
from flask import Flask, request, jsonify, redirect, url_for
from flasgger import Swagger, swag_from


bestmodel_path = os.getenv(
    "BESTMODEL_PATH", "/data/mlmodel/hate_speech_detector"
)
vectorizer_path = os.getenv(
    "VECTORIZER_PATH", "/data/cv/hate_speech_detector/cv_best_model.pkl"
)

print(bestmodel_path)

app = Flask(__name__)
swagger = Swagger(app)


# Lade das Modell als ein PyFuncModel
model = mlflow.sklearn.load_model(bestmodel_path)
cv = joblib.load(vectorizer_path)


@app.route("/")
def index():
    """Redirect to Swagger UI"""
    return redirect(url_for("flasgger.apidocs"))


@app.route("/predict", methods=["POST"])
@swag_from(
    {
        "responses": {
            200: {
                "description": "Model prediction",
                "examples": {"application/json": {"prediction": [1]}},
            }
        },
        "parameters": [
            {
                "name": "body",
                "in": "body",
                "required": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "inputs": {
                            "type": "string",
                            "example": "We should put Trump on the bullseye",
                        }
                    },
                },
            }
        ],
    }
)
def predict():
    if request.method == "POST":
        json_input = request.json
        text = json_input["inputs"]
        X = cv.transform([text]).toarray()
        prediction = model.predict(X)
        response = {"received_text": text, "prediction": str(prediction)}
        return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
