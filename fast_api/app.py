import requests
import joblib
import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# Modell- und Vektorisierer-Pfade
bestmodel_path = os.getenv(
    "BESTMODEL_PATH",
    "https://hatespeechstorage.blob.core.windows.net/"
    "hatespeech-data/mlmodel/hate_speech_detector/best_model_fittedX.pkl"
    "?sp=r&st=2024-07-29T00:16:18Z&se=2024-07-30T08:16:18Z&spr="
    "https&sv=2022-11-02&sr=b&sig="
    "n7B%2FJkuFkNVQHYZWV%2BhLJDoWrifns80UEuybp6SNBcg%3D",
)
print(bestmodel_path)
vectorizer_path = os.getenv(
    "VECTORIZER_PATH",
    "https://hatespeechstorage.blob.core.windows.net/"
    "hatespeech-data/cv/hate_speech_detector/cv_best_model.pkl"
    "?sp=r&st=2024-07-29T00:14:53Z&se=2024-07-30T08:14:53Z&spr="
    "https&sv=2022-11-02&sr=b&sig="
    "Iy8umJUwhSqkflSkuaEyFTgGBoRr5K3Phuyp3kmTTPQ%3D",
)


# function for loading models


def load_model(url_blob, local_filename):
    with requests.get(url_blob, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    model = joblib.load(local_filename)
    os.remove(local_filename)
    return model


# FastAPI-Initialisierung
app = FastAPI()

# Cross-Origin Resource Sharing (CORS) Konfiguration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update mit spezifischen Ursprüngen bei Bedarf
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lade das Modell und den Vektorisierer
model = load_model(bestmodel_path, "bestmodel.pkl")
cv = load_model(vectorizer_path, "vectorizer.pkl")


# Pydantic-Modell für die Vorhersageanforderung
class PredictionRequest(BaseModel):
    inputs: str = Field(..., example="We should put Trump on the bullseye")


@app.get("/")
def read_root():
    """Redirect to Swagger UI"""
    return RedirectResponse(url="/docs")


@app.post("/predict", response_model=dict)
def predict(request: PredictionRequest):
    text = request.inputs
    X = cv.transform([text]).toarray()
    prediction = model.predict(X)
    return {"received_text": text, "prediction": str(prediction)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80)
