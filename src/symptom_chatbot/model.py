"""TF-IDF + linear classifier pipeline."""

import json
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .preprocess import SymptomNormalizer, TextPreprocessor

ROOT = Path(__file__).resolve().parents[2]


def canonical_symptoms(knowledge_path=ROOT / "models" / "knowledge.json"):
    profiles = json.loads(Path(knowledge_path).read_text())["profiles"]
    return sorted({s for p in profiles.values() for s in p})


class Preprocess(BaseEstimator, TransformerMixin):
    """Pipeline step that runs symptom normalisation + tokenisation once per document."""

    def __init__(self, symptoms=None, normalize=True):
        self.symptoms = symptoms
        self.normalize = normalize

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        prep = self._prep()
        return [prep(t) for t in X]

    def _prep(self):
        # built lazily so clones and unpickled pipelines rebuild the compiled regex themselves
        if getattr(self, "_cached", None) is None:
            normalizer = SymptomNormalizer(self.symptoms) if self.normalize else _NoSymptoms()
            self._cached = TextPreprocessor(normalizer)
        return self._cached

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_cached", None)
        return state


class _NoSymptoms:
    """Normaliser stand-in for the ablation without symptom normalisation."""

    def find(self, text):
        return []


def build_features(char_ngrams=True, max_features=None, min_df=1):
    word = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, ngram_range=(1, 2),
                           sublinear_tf=True, min_df=min_df, max_features=max_features)
    if not char_ngrams:
        return word
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=min_df,
                           max_features=max_features)
    return FeatureUnion([("word", word), ("char", char)])


def build_pipeline(symptoms, clf=None, normalize=True, char_ngrams=True, max_features=None, min_df=1):
    return Pipeline([
        ("prep", Preprocess(symptoms, normalize)),
        ("features", build_features(char_ngrams, max_features, min_df)),
        ("clf", clf if clf is not None else LinearSVC(C=1.0)),
    ])


def calibrated_svc(C=1.0, ensemble=True, cv=5):
    """LinearSVC with Platt-style calibration so the chatbot can report confidences."""
    return CalibratedClassifierCV(LinearSVC(C=C), method="sigmoid", cv=cv, ensemble=ensemble)
