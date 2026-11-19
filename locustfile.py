# HOW TO RUN LOCUST:
# pip install locust
# locust -f locustfile.py --host=http://localhost:8000
# Then open http://localhost:8089 in browser
# Set number of users (e.g. 50, 100, 200, 500) and spawn rate
# Record screenshots of the results table showing latency and RPS

"""
Locust load test for the Chest X-Ray Pneumonia Classifier API.

Simulates concurrent users hitting the prediction, health, status, and
metrics endpoints. The predict task is weighted 3x higher than the others
to reflect realistic usage patterns where inference dominates traffic.
"""

import io

from locust import HttpUser, between, task
from PIL import Image


class ChestXRayAPIUser(HttpUser):
    """Simulated user that exercises the Chest X-Ray API endpoints."""

    wait_time = between(1, 3)
    host = "http://localhost:8000"

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _make_synthetic_jpeg_bytes(self) -> bytes:
        """Create a minimal 10x10 white JPEG image in memory."""
        img = Image.new("RGB", (10, 10), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        return buf.read()

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    @task(3)
    def predict_task(self):
        """POST /predict — send a synthetic JPEG as multipart/form-data."""
        image_bytes = self._make_synthetic_jpeg_bytes()
        files = {"file": ("test_image.jpg", image_bytes, "image/jpeg")}
        with self.client.post(
            "/predict",
            files=files,
            catch_response=True,
            name="/predict",
        ) as response:
            if response.status_code not in (200, 422):
                # 422 is acceptable when the model isn't loaded yet
                response.failure(
                    f"Unexpected status {response.status_code}: {response.text[:200]}"
                )

    @task(1)
    def health_check(self):
        """GET /health — verify the API is alive."""
        with self.client.get(
            "/health",
            catch_response=True,
            name="/health",
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"Health check failed with status {response.status_code}"
                )

    @task(1)
    def get_status(self):
        """GET /status — retrieve model status information."""
        with self.client.get(
            "/status",
            catch_response=True,
            name="/status",
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"Status endpoint failed with status {response.status_code}"
                )

    @task(1)
    def get_metrics(self):
        """GET /metrics — retrieve model performance metrics."""
        with self.client.get(
            "/metrics",
            catch_response=True,
            name="/metrics",
        ) as response:
            if response.status_code not in (200, 404):
                # 404 is acceptable when metrics.json doesn't exist yet
                response.failure(
                    f"Metrics endpoint failed with status {response.status_code}"
                )
