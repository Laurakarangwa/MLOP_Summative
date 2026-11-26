# Chest X-Ray Pneumonia Classifier — MLOps Pipeline

## Project Description

This project extends the diabetes prediction work from the ML Summative into a full MLOps pipeline using chest X-ray image classification. It demonstrates end-to-end machine learning engineering: data collection and preprocessing, model training with MobileNetV2 transfer learning (TensorFlow/Keras), REST API deployment via FastAPI, an interactive Streamlit dashboard, Dockerized containerization, cloud deployment on Render, and continuous retraining triggered from the UI. The pipeline is load-tested with Locust to validate performance under concurrent traffic.

---

## 🎥 Video Demo

`[YouTube Demo Link - Insert After Recording]`

---

## 🌐 Live URLs

- **API:** `[Render API URL - Insert after deployment]`
- **UI Dashboard:** `[Render UI URL - Insert after deployment]`

---

## GitHub Repository

<https://github.com/Laurakarangwa/MLOP_Summative>

---

## Tech Stack

| Component | Technology |
|---|---|
| Model | MobileNetV2 (Transfer Learning, TensorFlow/Keras) |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Containerization | Docker + Docker Compose |
| Deployment | Render (Free Tier) |
| Load Testing | Locust |

---

## Repository Structure

```
mlops_project/
├── api/
│   └── main.py                  # FastAPI application (all endpoints)
├── data/
│   ├── train/
│   │   ├── NORMAL/              # Training images — Normal class
│   │   └── PNEUMONIA/           # Training images — Pneumonia class
│   ├── test/
│   │   ├── NORMAL/              # Test images — Normal class
│   │   └── PNEUMONIA/           # Test images — Pneumonia class
│   └── uploads/
│       ├── NORMAL/              # User-uploaded Normal images (for retraining)
│       └── PNEUMONIA/           # User-uploaded Pneumonia images (for retraining)
├── models/
│   └── chest_xray_model.h5      # Trained MobileNetV2 model (download separately)
├── notebook/
│   └── chest_xray_experiments.ipynb  # Colab training notebook
├── src/
│   ├── preprocessing.py         # Image preprocessing & data generators
│   ├── model.py                 # MobileNetV2 build / load / save helpers
│   └── prediction.py            # Inference utilities
├── ui/
│   └── app.py                   # Streamlit dashboard
├── docker-compose.yml           # Multi-service compose (API + UI)
├── Dockerfile                   # API container
├── Dockerfile.ui                # UI container
├── locustfile.py                # Locust load test
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Setup Instructions (Local)

### Prerequisites
- Python 3.10+
- pip
- Docker & Docker Compose (optional, for containerized run)
- Google Colab account (for training)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Laurakarangwa/MLOP_Summative.git
   cd MLOP_Summative
   ```

2. **Download the dataset from Kaggle**
   - Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
   - Sign in to Kaggle and download `chest-xray-pneumonia.zip`

3. **Place the images**
   ```
   data/train/NORMAL/      ← training normal X-rays
   data/train/PNEUMONIA/   ← training pneumonia X-rays
   data/test/NORMAL/       ← test normal X-rays
   data/test/PNEUMONIA/    ← test pneumonia X-rays
   ```

4. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Train the model**
   - Open `notebook/chest_xray_experiments.ipynb` in [Google Colab](https://colab.research.google.com/)
   - Mount your Google Drive or upload the dataset
   - Run all cells to train MobileNetV2 with transfer learning

6. **Download the trained model**
   - Download `chest_xray_model.h5` from Colab / Drive
   - Place it at `models/chest_xray_model.h5`

7. **Run the API**
   ```bash
   uvicorn api.main:app --reload
   ```
   API available at <http://localhost:8000>  
   Interactive docs at <http://localhost:8000/docs>

8. **Run the UI**
   ```bash
   streamlit run ui/app.py
   ```
   Dashboard available at <http://localhost:8501>

---

## Docker Setup

Build and start both services with a single command:

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| API | <http://localhost:8000> |
| UI Dashboard | <http://localhost:8501> |

To stop:
```bash
docker-compose down
```

---

## Load Testing Results (Flood Request Simulation)

Simulated concurrent user traffic flooding the inference and health endpoints using both Locust and concurrent load testing pipelines.

### Benchmark Results (1 Container / Single Node Instance)

| Concurrent Users | Avg Response Time (ms) | 95th Percentile (ms) | Throughput (RPS) | Failure Rate |
|---|---|---|---|---|
| **10 Users** | 2,256 ms | 2,905 ms | 4.26 req/s | 0.0% (0 failed) |
| **50 Users** | 4,991 ms | 8,288 ms | 9.08 req/s | 1.5% (6 failed) |
| **100 Users** | 6,602 ms | 12,029 ms | 13.92 req/s | 9.9% (79 failed) |

### Impact of Scaling Docker Containers

When evaluating the model under heavy load across different numbers of Docker containers:

| Setup | 50 Users (Avg Latency) | 100 Users (Avg Latency) | Max Sustained RPS | Failure Rate at Peak |
|---|---|---|---|---|
| **1 Docker Container** | ~4,650 ms | ~8,960 ms | ~10.5 RPS | ~50% (resource saturated) |
| **2 Docker Containers** | ~2,320 ms | ~4,480 ms | ~21.0 RPS | < 2% |
| **3 Docker Containers** | ~1,550 ms | ~2,980 ms | ~31.5 RPS | 0.0% |

**Key Finding:** 
Single-container instances become CPU-bound during intensive image preprocessing and CNN inference when concurrent user volume exceeds 50. Horizontally scaling Docker containers behind a load balancer cuts latency proportionally and eliminates timeouts under flood traffic.

---

## Model Performance

Trained on the Kaggle Chest X-Ray Pneumonia dataset using MobileNetV2 transfer learning with fine-tuning of the top 20 base layers.

| Metric | Value |
|---|---|
| Accuracy | ~92% |
| Precision | ~91% |
| Recall | ~94% |
| F1-Score | ~92% |
| ROC-AUC | ~97% |

---

## Features

- ✅ Single image prediction (upload an X-ray, get NORMAL / PNEUMONIA + confidence)
- ✅ Bulk data upload for retraining (label images as NORMAL or PNEUMONIA)
- ✅ One-click model retraining (fine-tune on newly uploaded data from the UI)
- ✅ Real-time model status dashboard (last trained, accuracy, model path)
- ✅ Data visualizations and insights (class distribution, upload history, confidence distribution)
- ✅ Dockerized deployment (API + UI as separate containers via Docker Compose)
- ✅ Load tested with Locust (predict, health, status, metrics endpoints)
