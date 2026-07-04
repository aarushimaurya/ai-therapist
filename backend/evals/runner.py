"""
Collects raw responses from the AI therapist backend for each eval test case.

Loads test cases from test_cases.json, POSTs each one's prompt to the
running backend's /chat endpoint (a fresh conversation per case), and
writes the raw replies (or errors) to a timestamped JSON file under
results/. Grading those responses against must_not_contain /
must_contain_one_of is a separate step and is not done here.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

THERAPIST_URL = os.environ.get("THERAPIST_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 30

EVALS_DIR = Path(__file__).resolve().parent
TEST_CASES_PATH = EVALS_DIR / "test_cases.json"
RESULTS_DIR = EVALS_DIR / "results"


def load_test_cases():
    with open(TEST_CASES_PATH) as f:
        return json.load(f)


def run_case(case):
    start = time.monotonic()
    reply = None
    error = None
    try:
        response = requests.post(
            f"{THERAPIST_URL}/chat",
            json={"message": case["prompt"]},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        reply = response.json()["reply"]
    except requests.RequestException as exc:
        error = str(exc)
    latency_ms = round((time.monotonic() - start) * 1000)

    return {
        "id": case["id"],
        "category": case["category"],
        "prompt": case["prompt"],
        "reply": reply,
        "error": error,
        "must_not_contain": case["must_not_contain"],
        "must_contain_one_of": case["must_contain_one_of"],
        "latency_ms": latency_ms,
    }


def run_all(test_cases):
    return [run_case(case) for case in test_cases]


def write_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"run_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    return results_path


def print_summary(results, results_path):
    per_category = {}
    for result in results:
        per_category[result["category"]] = per_category.get(result["category"], 0) + 1
    error_count = sum(1 for result in results if result["error"] is not None)

    print(f"Total cases: {len(results)}")
    print("Cases per category:")
    for category, count in sorted(per_category.items()):
        print(f"  {category}: {count}")
    print(f"Errors: {error_count}")
    print(f"Results written to: {results_path}")


def main():
    test_cases = load_test_cases()
    results = run_all(test_cases)
    results_path = write_results(results)
    print_summary(results, results_path)


if __name__ == "__main__":
    main()
