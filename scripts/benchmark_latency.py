"""Compare per-message response time of the initial pipeline and the deployed one.

baseline   word + char TF-IDF (min_df=1), CalibratedClassifierCV ensemble of 5 LinearSVCs
deployed   models/disease_model.joblib as produced by train.py: a single calibrated
           LinearSVC (ensemble=False), min_df=2, char n-grams only if they helped on val

Both are timed on every test message one at a time, which is how the chatbot
calls the model. Results go to reports/latency.json.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from symptom_chatbot.model import build_pipeline, calibrated_svc, canonical_symptoms  # noqa: E402


def n_features(pipe):
    feats = pipe.named_steps["features"]
    if hasattr(feats, "vocabulary_"):
        return len(feats.vocabulary_)
    return sum(len(t.vocabulary_) for _, t in feats.transformer_list)


def time_messages(pipe, texts, repeats=5):
    for t in texts[:20]:  # warm-up
        pipe.predict_proba([t])
    samples = []
    for _ in range(repeats):
        for t in texts:
            t0 = time.perf_counter()
            pipe.predict_proba([t])
            samples.append((time.perf_counter() - t0) * 1000)
    return np.array(samples)


def main(C=3.0):
    df = pd.read_csv(ROOT / "data/processed/dataset.csv")
    trva, te = df[df.split != "test"], df[df.split == "test"]
    symptoms = canonical_symptoms()
    baseline = build_pipeline(symptoms, calibrated_svc(C, ensemble=True, cv=5), char_ngrams=True, min_df=1)
    baseline.fit(trva.text, trva.disease)
    deployed = joblib.load(ROOT / "models/disease_model.joblib")
    results = {}
    for name, pipe in (("baseline", baseline), ("deployed", deployed)):
        ms = time_messages(pipe, te.text.tolist())
        results[name] = {
            "test_accuracy": round(accuracy_score(te.disease, pipe.predict(te.text)), 4),
            "features": n_features(pipe),
            "latency_ms_mean": round(float(ms.mean()), 3),
            "latency_ms_p50": round(float(np.percentile(ms, 50)), 3),
            "latency_ms_p95": round(float(np.percentile(ms, 95)), 3),
            "timed_calls": int(ms.size),
        }
        print(name, results[name])
    b, o = results["baseline"], results["deployed"]
    results["mean_latency_reduction_pct"] = round(100 * (1 - o["latency_ms_mean"] / b["latency_ms_mean"]), 1)
    results["p95_latency_reduction_pct"] = round(100 * (1 - o["latency_ms_p95"] / b["latency_ms_p95"]), 1)
    (ROOT / "reports/latency.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
