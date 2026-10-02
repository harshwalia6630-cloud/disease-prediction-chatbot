"""Compare classifiers, run ablations, train the final model and evaluate it on the test split.

Model selection and ablations use the validation split only. The selected
configuration is then retrained on train+val and scored once on test.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from symptom_chatbot.model import build_pipeline, calibrated_svc, canonical_symptoms  # noqa: E402

REPORTS = ROOT / "reports"


def scores(y_true, y_pred, source):
    out = {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro")}
    for s in ("natural", "structured", "lay"):
        m = source == s
        out[f"{s}_accuracy"] = accuracy_score(y_true[m], y_pred[m])
    return {k: round(float(v), 4) for k, v in out.items()}


def main():
    df = pd.read_csv(ROOT / "data/processed/dataset.csv")
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    symptoms = canonical_symptoms()
    REPORTS.mkdir(exist_ok=True)

    # ---- 1. classifier comparison (fit on train, score on val)
    candidates = {
        "Complement Naive Bayes": ComplementNB(alpha=0.3),
        "k-NN (k=5, cosine)": KNeighborsClassifier(5, metric="cosine"),
        "Random Forest (300 trees)": RandomForestClassifier(300, n_jobs=-1, random_state=0),
        "Logistic Regression": LogisticRegression(C=20, max_iter=3000),
        **{f"LinearSVC (C={c})": LinearSVC(C=c) for c in (0.3, 1.0, 3.0)},
    }
    rows = []
    for name, clf in candidates.items():
        pipe = build_pipeline(symptoms, clf)
        t0 = time.perf_counter()
        pipe.fit(tr.text, tr.disease)
        fit_s = time.perf_counter() - t0
        rows.append({"model": name, **scores(va.disease.values, pipe.predict(va.text), va.source.values),
                     "fit_seconds": round(fit_s, 2)})
        print(rows[-1])
    comparison = pd.DataFrame(rows).sort_values(["accuracy", "macro_f1"], ascending=False)
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)
    best_c = max((r for r in rows if r["model"].startswith("LinearSVC")), key=lambda r: r["accuracy"])["model"]
    C = float(best_c.split("C=")[1].rstrip(")"))

    # ---- 2. ablations on the preprocessing / features (LinearSVC, val split)
    ablations = {}
    for name, kw in {
        "full pipeline": {},
        "without symptom normalisation": {"normalize": False},
        "without char n-grams": {"char_ngrams": False},
    }.items():
        pipe = build_pipeline(symptoms, LinearSVC(C=C), **kw)
        pipe.fit(tr.text, tr.disease)
        ablations[name] = scores(va.disease.values, pipe.predict(va.text), va.source.values)
        print(name, ablations[name])

    # ---- 3. stability: 5-fold CV over all natural-language descriptions in train+val
    nat = pd.concat([tr, va])
    nat = nat[nat.source == "natural"]
    cv = cross_val_score(build_pipeline(symptoms, LinearSVC(C=C)), nat.text, nat.disease,
                         cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=-1)

    # ---- 4. final model: calibrated LinearSVC on train+val, scored once on test
    # char n-grams roughly double feature extraction cost; keep them only if they earn it on val
    use_char = ablations["full pipeline"]["accuracy"] > ablations["without char n-grams"]["accuracy"]
    trva = pd.concat([tr, va])
    final = build_pipeline(symptoms, calibrated_svc(C, ensemble=False), char_ngrams=use_char, min_df=2)
    final.fit(trva.text, trva.disease)
    pred = final.predict(te.text)
    test = scores(te.disease.values, pred, te.source.values)
    nat_mask = te.source.values == "natural"
    report = {
        "selected_model": f"TF-IDF (word 1-2{' + char 3-5' if use_char else ''}) + calibrated LinearSVC (C={C})",
        "n_classes": int(df.disease.nunique()),
        "n_classes_natural_test": int(te[nat_mask].disease.nunique()),
        "test": test,
        "test_rows": te.source.value_counts().to_dict(),
        "cv_natural_5fold": {"mean": round(float(cv.mean()), 4), "std": round(float(cv.std()), 4),
                             "folds": [round(float(x), 4) for x in cv]},
        "ablations_val": ablations,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2))
    (REPORTS / "classification_report.txt").write_text(classification_report(te.disease, pred, digits=3))

    labels = sorted(df.disease.unique())
    cm = confusion_matrix(te.disease, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    for i, j in zip(*np.nonzero(cm)):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Test confusion matrix (accuracy {test['accuracy']:.1%})")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=110)

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(final, ROOT / "models/disease_model.joblib", compress=3)
    print(comparison.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
