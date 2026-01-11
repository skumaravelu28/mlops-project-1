from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn

# --------------------
# App initialization
# --------------------
app = FastAPI(title="Sentiment Analysis API")

# --------------------
# MLflow setup
# --------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 👉 Use the LATEST model from MLflow artifacts
MODEL_URI = "models:/sentiment-model/Production"

try:
    model = mlflow.sklearn.load_model(MODEL_URI)
except Exception:
    # fallback: load latest local model if registry is not used
	model = mlflow.pyfunc.load_model("mlartifacts/1/models/m-a92dd502847445c9ae5c80a8d582da5a/artifacts")


# --------------------
# Request schema
# --------------------
class Review(BaseModel):
    text: str

# --------------------
# Routes
# --------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict_sentiment(review: Review):
    prediction = model.predict([review.text])[0]
    return {"sentiment": prediction}
