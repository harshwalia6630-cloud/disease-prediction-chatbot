"""Download the raw datasets (if missing) and build data/processed/dataset.csv."""

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from symptom_chatbot.data import RAW, write_processed  # noqa: E402

SOURCES = {
    # Symptom2Disease (Apache-2.0), mirrored on Hugging Face
    "symptom2disease.csv":
        "https://huggingface.co/datasets/NeuronZero/Symptom2Disease/resolve/main/Symptom2Disease.csv",
    # Kaggle "Disease Symptom Prediction" with precautions merged in
    "disease_symptom_precaution.csv":
        "https://huggingface.co/datasets/shanover/disease_symptoms_prec_full/resolve/main/disease_sympts_prec_full.csv",
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        dest = RAW / name
        if not dest.exists():
            print(f"Downloading {name}")
            urllib.request.urlretrieve(url, dest)
    data, stats = write_processed()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
