FROM python:3.12-slim

WORKDIR /app

# Install only what we need
RUN pip install --no-cache-dir fastapi uvicorn mlflow scikit-learn pandas numpy

# Copy your API code + model artifacts into the image
COPY api/ api/
COPY mlartifacts/ mlartifacts/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
