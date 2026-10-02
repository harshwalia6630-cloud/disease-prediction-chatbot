"""Load, clean, merge and split the two source datasets.

Sources
-------
* Symptom2Disease: 1,200 patient-style descriptions covering 24 diseases.
* Kaggle Disease-Symptom-Precaution: 4,920 structured rows covering 41 diseases.
  Only 304 of them are unique, so duplicates are removed *before* splitting
  so that no test record can also appear in training.

The structured rows are turned into sentences with templates, using lay
synonyms for some symptoms so the model learns to read everyday language.
Training and test sentences come from disjoint template sets.
"""

import json
import random
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .lexicon import SYNONYMS, canonical

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

DISEASE_NAMES = {
    "(vertigo) paroymsal positional vertigo": "Paroxysmal Positional Vertigo",
    "dimorphic hemmorhoids(piles)": "Hemorrhoids (Piles)",
    "dimorphic hemorrhoids": "Hemorrhoids (Piles)",
    "osteoarthristis": "Osteoarthritis",
    "peptic ulcer diseae": "Peptic Ulcer Disease",
    "peptic ulcer disease": "Peptic Ulcer Disease",
    "gerd": "GERD",
    "gastroesophageal reflux disease": "GERD",
    "aids": "AIDS",
    "chicken pox": "Chickenpox",
    "paralysis (brain hemorrhage)": "Paralysis (Brain Hemorrhage)",
}


def disease_name(raw):
    key = " ".join(raw.lower().split())
    return DISEASE_NAMES.get(key, key.title())


TRAIN_TEMPLATES = [
    "I have {s}.",
    "I've been experiencing {s} for the past few days.",
    "My symptoms are {s}.",
    "Lately I am suffering from {s}.",
    "For about a week now I have had {s}.",
    "I keep getting {s} and it is getting worse.",
    "Doctor, I have {s}. What could it be?",
    "Since yesterday I've noticed {s}.",
]
TEST_TEMPLATES = [
    "I am dealing with {s} right now.",
    "Recently I started having {s}, should I be worried?",
    "What illness causes {s}?",
]


def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _verbalise(symptoms, template, rng, lay_terms, p_lay=0.5):
    words = []
    for s in rng.sample(symptoms, len(symptoms)):
        options = lay_terms.get(s)
        words.append(rng.choice(options) if options and rng.random() < p_lay else s)
    return template.format(s=_join(words))


def load_structured():
    df = pd.read_csv(RAW / "disease_symptom_precaution.csv")
    df["disease"] = df["disease"].map(disease_name)
    df["symptoms"] = df["symptoms"].map(lambda r: tuple(sorted({canonical(x) for x in r.split(",") if x.strip()})))
    df["precautions"] = df["precautions"].fillna("")
    return df


def load_symptom2disease():
    df = pd.read_csv(RAW / "symptom2disease.csv")
    df = df.rename(columns={"label": "disease"})[["text", "disease"]]
    df["disease"] = df["disease"].map(disease_name)
    df["text"] = df["text"].str.strip()
    return df


PRECAUTION_FIXES = {
    "antiboitic": "antibiotic", "movment": "movement", "alovera": "aloe vera", "fiberous": "fibrous",
    "vegitables": "vegetables", "cloothing": "clothing", "reliver": "reliever", "detol": "Dettol",
    "oinments": "ointments", "poloroid": "polarised", "mosquitos": "mosquitoes", "dont": "don't",
    "otc": "OTC", "ppe": "PPE", "wash hands through": "wash hands thoroughly",
    "use clean cloths": "use clean clothes", "stop eating solid food for while": "stop eating solid food for a while",
}


def _clean_precaution(text):
    text = text.strip().lower()
    for wrong, right in PRECAUTION_FIXES.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text)
    return text[0].upper() + text[1:]


def build_knowledge(structured):
    """Per-disease symptom frequencies and precautions, used by the chatbot."""
    symptom_counts = defaultdict(Counter)
    precautions = {}
    for row in structured.itertuples():
        symptom_counts[row.disease].update(row.symptoms)
        if row.precautions and row.disease not in precautions:
            precautions[row.disease] = [_clean_precaution(p) for p in row.precautions.split(",") if p.strip()]
    rows_per_disease = structured["disease"].value_counts().to_dict()
    profiles = {
        d: {s: round(c / rows_per_disease[d], 3) for s, c in counts.most_common()}
        for d, counts in symptom_counts.items()
    }
    return {"profiles": profiles, "precautions": precautions}


def build_splits(seed=42, train_variants=8, test_variants=2):
    rng = random.Random(seed)
    structured = load_structured()
    s2d = load_symptom2disease()
    stats = {
        "structured_rows_raw": len(structured),
        "symptom2disease_rows_raw": len(s2d),
    }

    # every third lay phrase (by stable hash) is held out of training text entirely, so
    # the "lay" test rows contain phrasings the classifier has never seen
    train_lay, all_lay, unseen_lay = defaultdict(list), defaultdict(list), defaultdict(list)
    for phrase, target in SYNONYMS.items():
        all_lay[target].append(phrase)
        (unseen_lay if zlib.crc32(phrase.encode()) % 3 == 0 else train_lay)[target].append(phrase)
    stats["lay_phrases_unseen_in_training"] = sum(map(len, unseen_lay.values()))

    # --- structured: dedupe, then split unique symptom combinations per disease
    combos = structured.drop_duplicates(["disease", "symptoms"])[["disease", "symptoms"]].reset_index(drop=True)
    stats["structured_rows_unique"] = len(combos)
    # some diseases have only 5 unique combinations, so hold out exactly one for val and one for test
    split_of = {}
    for _, idx in combos.groupby("disease").groups.items():
        idx = list(idx)
        rng.shuffle(idx)
        split_of.update({idx[0]: "test", idx[1]: "val", **{i: "train" for i in idx[2:]}})
    combos["split"] = combos.index.map(split_of)
    tr_c, va_c, te_c = (combos[combos["split"] == s] for s in ("train", "val", "test"))

    def expand(frame, templates, n, split, lay_terms, source="structured", p_lay=0.5):
        rows = []
        for r in frame.itertuples():
            for _ in range(n):
                text = _verbalise(list(r.symptoms), rng.choice(templates), rng, lay_terms, p_lay)
                rows.append((text, r.disease, source, split))
        return rows

    # --- Symptom2Disease: drop exact duplicate texts, stratified 70/15/15
    s2d = s2d.drop_duplicates("text").reset_index(drop=True)
    stats["symptom2disease_rows_unique"] = len(s2d)
    tr_t, rest_t = train_test_split(s2d, test_size=0.3, stratify=s2d["disease"], random_state=seed)
    va_t, te_t = train_test_split(rest_t, test_size=0.5, stratify=rest_t["disease"], random_state=seed)

    def natural(frame, split):
        return [(t, d, "natural", split) for t, d in zip(frame["text"], frame["disease"])]

    rows = expand(tr_c, TRAIN_TEMPLATES, train_variants, "train", train_lay) + natural(tr_t, "train")
    for split, frame, texts in (("val", va_c, va_t), ("test", te_c, te_t)):
        rows += expand(frame, TEST_TEMPLATES, test_variants, split, all_lay)
        rows += natural(texts, split)
        # lay-language stress test: every symptom with an unseen synonym is written that way
        rows += expand(frame, TEST_TEMPLATES, test_variants, split, unseen_lay, source="lay", p_lay=1.0)
    data = pd.DataFrame(rows, columns=["text", "disease", "source", "split"])
    stats["diseases"] = int(data["disease"].nunique())
    stats["rows_per_split"] = data.groupby(["split", "source"]).size().unstack(fill_value=0).to_dict("index")
    return data, build_knowledge(structured), stats, combos


def write_processed(out_dir=ROOT / "data" / "processed", seed=42):
    data, knowledge, stats, combos = build_splits(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(out_dir / "dataset.csv", index=False)
    combos.assign(symptoms=combos["symptoms"].map("|".join)).to_csv(out_dir / "symptom_combos.csv", index=False)
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / "knowledge.json").write_text(json.dumps(knowledge, indent=1))
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return data, stats
