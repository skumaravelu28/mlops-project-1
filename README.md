# MLOps End-to-End Project – Reviews Classification

## Project Overview
This project demonstrates an end-to-end MLOps pipeline using AWS, DVC, MLflow, Docker, Kubernetes, and GitHub Actions.
The system trains a machine learning model on review data, tracks experiments, serves predictions via API, and provides a web UI.

---

## Architecture Diagram (Workflow)

User
 → Streamlit UI (Web App)
 → FastAPI (Inference API)
 → ML Model
 → MLflow (Experiment Tracking)

Data Flow:
S3 (reviews.csv) → DVC → Training → Model Artifact → S3  
CI/CD: GitHub Actions → DockerHub → Kubernetes (Minikube)

---

## Tech Stack

- AWS EC2 & S3
- Git & GitHub
- DVC (Data Version Control)
- MLflow (Experiment Tracking)
- FastAPI (Model API)
- Streamlit (Web UI)
- Docker & DockerHub
- Kubernetes (Minikube)
- GitHub Actions (CI/CD)

---

## Project Phases

### 1. Development
- Data versioning using DVC
- Model training & experiment tracking using MLflow
- API development using FastAPI
- UI using Streamlit

### 2. Build
- Docker image creation for API and UI
- CI pipeline using GitHub Actions
- Push images to DockerHub

### 3. Deployment
- Kubernetes deployment using Minikube
- Service exposure and testing

