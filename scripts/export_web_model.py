"""Export the trained pipeline so the website can run it entirely in the browser.

Writes
    web/model/model.json   vocabularies, idf weights, calibration, lexicon, knowledge
    web/model/coef.bin     LinearSVC weights, float32 little-endian, shape (classes, features)
    tests/fixtures/web_parity.json
        reference outputs from Python (stems, preprocessed text, probabilities,
        whole conversations) that tests/web_parity.mjs compares the JS port against
"""

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from symptom_chatbot import preprocess  # noqa: E402
from symptom_chatbot.chatbot import DISCLAIMER, URGENT, DiseaseChatbot, Session  # noqa: E402
from symptom_chatbot.lexicon import NEGATIONS  # noqa: E402

WEB = ROOT / "web" / "model"

DIALOGUES = [
    ["hi", "I have been throwing up and I have a really bad tummy ache, also loose motions"],
    ["I've been feeling feverish with chills and body aches, and I keep feeling nauseous", "yes", "yes"],
    ["I've been feeling feverish with chills and body aches, and I keep feeling nauseous", "no", "no", "no"],
    ["my head hurts", "yes", "no", "no"],
    ["my head hurts", "no", "no", "no"],
    ["I have a skin rash", "yes"],
    ["I have a skin rash", "no", "no", "no"],
    ["I have a skin rash", "yes and also I have been vomiting a lot"],
    ["severe chest pain, sweating a lot, breathless and vomiting"],
    ["I have a cough but no fever, runny nose and sneezing"],
    ["help"],
    ["I have a fever and a headache", "reset", "burning when I pee and smelly urine"],
    ["", "hello", "I don't have a fever, but my joints hurt and I have a rash"],
]


def main():
    bot = DiseaseChatbot()
    pipe = bot.model
    prep = pipe.named_steps["prep"]._prep()
    word, char = (t for _, t in pipe.named_steps["features"].transformer_list)
    cal = pipe.named_steps["clf"].calibrated_classifiers_[0]
    svc = cal.estimator
    assert list(svc.classes_) == list(pipe.classes_)

    WEB.mkdir(parents=True, exist_ok=True)
    svc.coef_.astype("<f4").tofile(WEB / "coef.bin")

    def vocab(vec):
        terms = [None] * len(vec.vocabulary_)
        for t, i in vec.vocabulary_.items():
            terms[i] = t
        return terms

    knowledge = json.loads((ROOT / "models/knowledge.json").read_text())
    model = {
        "version": 1,
        "classes": list(svc.classes_),
        "n_features": int(svc.coef_.shape[1]),
        "word": {"terms": vocab(word), "idf": np.round(word.idf_, 7).tolist(), "ngram": list(word.ngram_range)},
        "char": {"terms": vocab(char), "idf": np.round(char.idf_, 7).tolist(), "ngram": list(char.ngram_range)},
        "intercept": svc.intercept_.tolist(),
        "calibration": [[c.a_, c.b_] for c in cal.calibrators],
        "phrases": prep.normalizer.phrase_to_symptom,
        "stopwords": sorted(preprocess._STOP),
        "negations": sorted(NEGATIONS),
        "profiles": knowledge["profiles"],
        "precautions": knowledge["precautions"],
        "chat": {"confident": bot.confident, "max_questions": bot.max_questions, "top_k": bot.top_k,
                 "urgent": sorted(URGENT), "disclaimer": DISCLAIMER},
        "metrics": json.loads((ROOT / "reports/metrics.json").read_text())["test"],
    }
    (WEB / "model.json").write_text(json.dumps(model, separators=(",", ":")))

    # ---- parity fixtures
    df = pd.read_csv(ROOT / "data/processed/dataset.csv")
    texts = df[df.split == "test"].text.tolist() + [
        "No fever. Feeling nauseous", "I DON'T have a cough, but I’m sneezing!!", "itchy   rash\n\non my arms",
        "x", "", "pain pain pain pain", "I've been having sky-high fevers; can't sleep",
    ]
    words = sorted({w for t in df.text.str.lower() for w in re.findall(r"[a-z]+", t)} | set(prep.normalizer.phrase_to_symptom)
                   | {"skies", "dying", "lying", "news", "proceed", "agreed", "hopping", "hoping", "falling", "happily",
                      "relational", "generalization", "electrical", "adjustment", "controlling", "rolling", "cries", "ties"})
    words = [w for w in words if " " not in w]
    dialogues = []
    for msgs in DIALOGUES:
        s = Session()
        dialogues.append({"messages": msgs, "replies": [bot.respond(s, m) for m in msgs]})
    fixtures = {
        "stems": {w: preprocess._stem(w) for w in words},
        "texts": texts,
        "preprocessed": [prep(t) for t in texts],
        "probabilities": np.round(pipe.predict_proba(texts), 7).tolist(),
        "dialogues": dialogues,
    }
    (ROOT / "tests/fixtures").mkdir(parents=True, exist_ok=True)
    (ROOT / "tests/fixtures/web_parity.json").write_text(json.dumps(fixtures))
    sizes = {p.name: f"{p.stat().st_size / 1e6:.2f} MB" for p in WEB.iterdir()}
    print(f"exported {model['n_features']} features x {len(model['classes'])} classes", sizes)
    print(f"fixtures: {len(words)} stems, {len(texts)} texts, {len(dialogues)} dialogues")


if __name__ == "__main__":
    main()
