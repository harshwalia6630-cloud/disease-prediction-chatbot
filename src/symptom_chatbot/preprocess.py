"""Text preprocessing: symptom normalisation, negation marking, tokenisation and stemming.

    "I've been throwing up and I don't have a fever"
      -> symptoms found: vomiting (+), high fever (negated)
      -> tokens: ['vomit', 'sym_vomiting', 'not_sym_high_fever', ...]

Symptom phrases are replaced by their canonical name and also emitted as a
single `sym_*` token, so "throwing up", "puking" and "vomiting" all produce the
same feature. Negated symptoms get a `not_` prefix so that "no fever" and
"fever" do not look alike to the classifier.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .lexicon import NEGATIONS, SYNONYMS

_WORD = re.compile(r"[a-z][a-z_']*")
_CLAUSE_BREAK = re.compile(r"[.,;:!?]|\bbut\b|\bhowever\b|\balthough\b")
_STOP = frozenset(ENGLISH_STOP_WORDS) - NEGATIONS - {"back", "side", "eye", "eyes", "full"}
_stemmer = PorterStemmer()


@lru_cache(maxsize=50_000)
def _stem(word):
    return _stemmer.stem(word)


@dataclass(frozen=True)
class Mention:
    symptom: str
    negated: bool
    span: tuple


class SymptomNormalizer:
    """Finds symptom phrases (canonical or lay synonyms) in free text."""

    def __init__(self, canonical_symptoms, synonyms=SYNONYMS):
        self.phrase_to_symptom = {s: s for s in canonical_symptoms}
        self.phrase_to_symptom.update({k: v for k, v in synonyms.items() if v in self.phrase_to_symptom})
        # longest phrases first so "high fever" wins over "fever" at the same position
        phrases = sorted(self.phrase_to_symptom, key=len, reverse=True)
        self._pattern = re.compile(r"\b(" + "|".join(re.escape(p) for p in phrases) + r")\b")
        self.symptoms = sorted(set(self.phrase_to_symptom.values()))

    def find(self, text):
        text = text.lower()
        mentions = []
        for m in self._pattern.finditer(text):
            mentions.append(Mention(self.phrase_to_symptom[m.group(1)], self._is_negated(text, m.start()), m.span()))
        return mentions

    @staticmethod
    def _is_negated(text, start, window=4):
        clause = _CLAUSE_BREAK.split(text[:start])[-1]
        return any(w in NEGATIONS for w in _WORD.findall(clause)[-window:])


class TextPreprocessor:
    """Callable used as the TF-IDF `analyzer` input: returns a cleaned string."""

    def __init__(self, normalizer):
        self.normalizer = normalizer

    def __call__(self, text):
        text = text.lower().replace("’", "'")
        mentions = self.normalizer.find(text)
        # rebuild the text with every symptom phrase replaced by its canonical form
        parts, last = [], 0
        for m in mentions:
            parts.append(text[last:m.span[0]])
            parts.append(m.symptom)
            last = m.span[1]
        parts.append(text[last:])
        words = [w.replace("'", "") for w in _WORD.findall("".join(parts))]
        tokens = [_stem(w) for w in words if w not in _STOP and len(w) > 1]
        tokens += [("not_" if m.negated else "") + "sym_" + m.symptom.replace(" ", "_") for m in mentions]
        return " ".join(tokens)
