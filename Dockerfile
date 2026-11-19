FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p data/uploads/NORMAL data/uploads/PNEUMONIA data/train/NORMAL data/train/PNEUMONIA data/test/NORMAL data/test/PNEUMONIA models
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
