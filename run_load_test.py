"""
run_load_test.py
----------------
Concurrent load test simulator for the Chest X-Ray Pneumonia Classifier API.
Simulates a flood of concurrent user requests across endpoints (/predict, /health, /status, /metrics),
measuring latency (min, max, avg, p50, p95), requests per second (RPS), and failure rate.
Works cleanly across all OS environments without requiring gevent C-extensions.
"""

import time
import io
import concurrent.futures
from typing import List, Dict, Any
import requests
from PIL import Image
import numpy as np

API_BASE_URL = "http://localhost:8000"

def generate_synthetic_image_bytes() -> bytes:
    """Generate a valid JPEG byte payload."""
    img = Image.new("RGB", (224, 224), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()

TEST_IMAGE_BYTES = generate_synthetic_image_bytes()

def send_predict_request() -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        files = {"file": ("test.jpg", TEST_IMAGE_BYTES, "image/jpeg")}
        resp = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=10)
        duration_ms = (time.perf_counter() - start) * 1000.0
        return {
            "endpoint": "/predict",
            "status_code": resp.status_code,
            "success": resp.status_code == 200,
            "duration_ms": duration_ms
        }
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000.0
        return {
            "endpoint": "/predict",
            "status_code": 0,
            "success": False,
            "error": str(exc),
            "duration_ms": duration_ms
        }

def send_health_request() -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        duration_ms = (time.perf_counter() - start) * 1000.0
        return {
            "endpoint": "/health",
            "status_code": resp.status_code,
            "success": resp.status_code == 200,
            "duration_ms": duration_ms
        }
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000.0
        return {
            "endpoint": "/health",
            "status_code": 0,
            "success": False,
            "error": str(exc),
            "duration_ms": duration_ms
        }

def simulate_user_cycle(user_id: int, requests_per_user: int = 5) -> List[Dict[str, Any]]:
    results = []
    for _ in range(requests_per_user):
        # 3 predict requests for every 1 health request (matching locustfile weight)
        results.append(send_predict_request())
        results.append(send_predict_request())
        results.append(send_predict_request())
        results.append(send_health_request())
    return results

def run_flood_test(num_users: int = 50, requests_per_user: int = 4):
    total_requests_planned = num_users * requests_per_user * 4
    print("=" * 65)
    print(f"🚀 FLOOD TEST SIMULATION: {num_users} Concurrent Users")
    print(f"Target: {API_BASE_URL} | Total Requests: {total_requests_planned}")
    print("=" * 65)

    all_results: List[Dict[str, Any]] = []
    overall_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
        futures = [
            executor.submit(simulate_user_cycle, u, requests_per_user)
            for u in range(num_users)
        ]
        for f in concurrent.futures.as_completed(futures):
            all_results.extend(f.result())

    overall_duration = time.perf_counter() - overall_start
    total_requests = len(all_results)
    successes = sum(1 for r in all_results if r["success"])
    failures = total_requests - successes
    failure_rate = (failures / total_requests) * 100 if total_requests > 0 else 0
    rps = total_requests / overall_duration if overall_duration > 0 else 0

    durations = [r["duration_ms"] for r in all_results]
    avg_latency = float(np.mean(durations)) if durations else 0
    p50_latency = float(np.percentile(durations, 50)) if durations else 0
    p95_latency = float(np.percentile(durations, 95)) if durations else 0
    min_latency = float(np.min(durations)) if durations else 0
    max_latency = float(np.max(durations)) if durations else 0

    # Per endpoint stats
    predict_durations = [r["duration_ms"] for r in all_results if r["endpoint"] == "/predict"]
    health_durations = [r["duration_ms"] for r in all_results if r["endpoint"] == "/health"]

    print("\n📊 RESULTS SUMMARY:")
    print(f"  Total Requests   : {total_requests}")
    print(f"  Duration         : {overall_duration:.2f} seconds")
    print(f"  Throughput (RPS) : {rps:.2f} requests/sec")
    print(f"  Failure Rate     : {failure_rate:.1f}% ({failures} failed)")
    print(f"  Avg Latency      : {avg_latency:.2f} ms")
    print(f"  50th Percentile  : {p50_latency:.2f} ms")
    print(f"  95th Percentile  : {p95_latency:.2f} ms")
    print(f"  Min / Max Latency: {min_latency:.2f} ms / {max_latency:.2f} ms")
    if predict_durations:
        print(f"  /predict Avg     : {np.mean(predict_durations):.2f} ms")
    if health_durations:
        print(f"  /health  Avg     : {np.mean(health_durations):.2f} ms")
    print("=" * 65)

    return {
        "users": num_users,
        "rps": round(rps, 2),
        "avg_ms": round(avg_latency, 2),
        "p95_ms": round(p95_latency, 2),
        "failure_rate": f"{failure_rate:.1f}%"
    }

if __name__ == "__main__":
    for users in [10, 50, 100]:
        run_flood_test(num_users=users, requests_per_user=2)
        time.sleep(1)
