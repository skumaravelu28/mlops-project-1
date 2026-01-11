import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

# Paths
DATA_PATH = "data/raw/reviews.csv"
MODEL_PATH = "models/model.pkl"

# Load data
df = pd.read_csv(DATA_PATH)

X = df["review"]
y = df["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Evaluation
preds = model.predict(X_test_vec)
acc = accuracy_score(y_test, preds)

# MLflow tracking
mlflow.set_experiment("sentiment-analysis")

with mlflow.start_run():
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

print(f"Training completed. Accuracy: {acc}")
