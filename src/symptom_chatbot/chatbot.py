"""Conversational layer on top of the classifier.

The bot keeps per-session state: what the user has described, which symptoms
they confirmed or denied, and which follow-up question is pending. When the
model is not confident it asks about the symptom that best separates the
current top candidates, using per-disease symptom frequencies from the
structured dataset.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from .model import ROOT, canonical_symptoms
from .preprocess import SymptomNormalizer

DISCLAIMER = ("This is an educational tool, not a medical diagnosis. "
              "Please consult a qualified doctor for advice about your health.")
URGENT = {"Heart Attack", "Paralysis (Brain Hemorrhage)"}
YES = re.compile(r"^\s*(y|yes|yeah|yep|yup|sure|correct|i do|haan|ha|han)\b", re.I)
NO = re.compile(r"^\s*(n|no|nope|nah|not really|i don'?t|i do not|nahi|na)\b", re.I)
GREETING = re.compile(r"^\s*(hi|hello|hey|hii+|good (morning|afternoon|evening)|namaste)\b[\s!.]*$", re.I)
RESET = re.compile(r"^\s*(reset|restart|start over|new chat|clear)\s*$", re.I)


@dataclass
class Session:
    texts: list = field(default_factory=list)
    confirmed: set = field(default_factory=set)
    denied: set = field(default_factory=set)
    asked: list = field(default_factory=list)
    answers: dict = field(default_factory=dict)
    pending: str = None


class DiseaseChatbot:
    def __init__(self, model_path=ROOT / "models/disease_model.joblib",
                 knowledge_path=ROOT / "models/knowledge.json",
                 confident=0.9, max_questions=3, top_k=3):
        self.model = joblib.load(model_path)
        knowledge = json.loads(Path(knowledge_path).read_text())
        self.profiles = knowledge["profiles"]
        self.precautions = knowledge["precautions"]
        self.normalizer = SymptomNormalizer(canonical_symptoms(knowledge_path))
        self.classes = list(self.model.classes_)
        self.confident, self.max_questions, self.top_k = confident, max_questions, top_k

    # ---------- model access
    def predict(self, text, top_k=None, answers=None):
        """Top-k diseases for `text`.

        `answers` ({symptom: True/False} from follow-up questions) are folded in with a
        naive-Bayes update, P(d | text, answers) ∝ P(d | text) · Π P(answer | d), using
        per-disease symptom frequencies. A single extra token in a long text barely moves
        a linear model, so this is what makes a yes/no answer actually count.
        """
        probs = self.model.predict_proba([text])[0]
        for symptom, present in (answers or {}).items():
            freq = np.array([self.profiles.get(c, {}).get(symptom, 0.0) for c in self.classes])
            likelihood = np.clip(freq, 0.1, 0.9)  # cap any single answer at a 9x likelihood ratio
            probs = probs * (likelihood if present else 1 - likelihood)
        probs = probs / probs.sum()
        order = np.argsort(probs)[::-1][: top_k or self.top_k]
        return [{"disease": self.classes[i], "probability": round(float(probs[i]), 4)} for i in order]

    def detect_symptoms(self, text):
        found = {}
        for m in self.normalizer.find(text):
            found[m.symptom] = not m.negated
        return found

    # ---------- conversation
    def respond(self, session, message):
        message = message.strip()
        if not message:
            return self._reply("Please describe how you are feeling.")
        if RESET.match(message):
            session.__init__()
            return self._reply("Okay, let's start over. What symptoms are you experiencing?")
        if GREETING.match(message) and not session.texts:
            return self._reply("Hello! I can suggest possible conditions based on your symptoms. "
                               "Describe what you're feeling, e.g. \"I have a fever, headache and body aches\".")

        # a short yes/no answers the pending question; anything longer is treated as a new description
        short_answer = len(message.split()) <= 3
        if session.pending and short_answer and (YES.match(message) or NO.match(message)):
            present = bool(YES.match(message))
            (session.confirmed if present else session.denied).add(session.pending)
            session.answers[session.pending] = present
            session.pending = None
        else:
            session.pending = None
            session.texts.append(message)
            for symptom, present in self.detect_symptoms(message).items():
                (session.confirmed if present else session.denied).add(symptom)
                (session.denied if present else session.confirmed).discard(symptom)

        if not session.confirmed and len(" ".join(session.texts).split()) < 4:
            return self._reply("Could you tell me a bit more about your symptoms? For example: where it hurts, "
                               "whether you have a fever, and how long it has been going on.")

        preds = self.predict(self._session_text(session), answers=session.answers)
        if preds[0]["probability"] < self.confident and len(session.asked) < self.max_questions:
            question = self._next_question(session, preds)
            if question:
                session.pending = question
                session.asked.append(question)
                return self._reply(f"Do you also have {question}? (yes/no)", preds, session, follow_up=question)
        return self._final(preds, session)

    @staticmethod
    def _session_text(session):
        # follow-up answers are applied in predict(); adding them here too would count them twice
        return " ".join(session.texts)

    def _next_question(self, session, preds):
        """Pick the unasked symptom whose presence differs most across the top candidates."""
        known = session.confirmed | session.denied | set(session.asked)
        cands = [(p["disease"], p["probability"]) for p in preds if p["disease"] in self.profiles]
        if len(cands) < 2:
            return None
        weights = np.array([p for _, p in cands])
        weights = weights / weights.sum()
        best, best_score = None, 0.0
        # sorted so ties are broken the same way on every run (set order is hash-randomised)
        for symptom in sorted({s for d, _ in cands for s in self.profiles[d]} - known):
            freq = np.array([self.profiles[d].get(symptom, 0.0) for d, _ in cands])
            mean = (weights * freq).sum()
            score = (weights * (freq - mean) ** 2).sum()  # weighted variance: high = discriminative
            if score > best_score:
                best, best_score = symptom, score
        return best

    def _final(self, preds, session):
        top = preds[0]
        if top["probability"] < 0.4:
            listed = ", ".join(f"{p['disease']} ({p['probability']:.0%})" for p in preds)
            lines = [f"I couldn't narrow this down with confidence. Conditions that fit your symptoms include: "
                     f"{listed}. A doctor can examine you and run tests to find the actual cause."]
        else:
            lines = [f"Based on what you've described, the most likely condition is **{top['disease']}** "
                     f"({top['probability']:.0%} confidence)."]
            others = [f"{p['disease']} ({p['probability']:.0%})" for p in preds[1:] if p["probability"] >= 0.05]
            if others:
                lines.append("Other possibilities: " + ", ".join(others) + ".")
            if top["disease"] in self.precautions:
                lines.append("General precautions: " + "; ".join(self.precautions[top["disease"]]) + ".")
        if any(p["disease"] in URGENT and p["probability"] >= 0.2 for p in preds):
            lines.insert(0, "⚠️ Some of your symptoms can indicate a medical emergency. "
                            "If they are severe or sudden, call emergency services or go to the nearest hospital now.")
        lines.append(DISCLAIMER)
        return self._reply("\n\n".join(lines), preds, session, final=True)

    @staticmethod
    def _reply(text, preds=None, session=None, follow_up=None, final=False):
        return {
            "reply": text,
            "predictions": preds or [],
            "symptoms_confirmed": sorted(session.confirmed) if session else [],
            "symptoms_denied": sorted(session.denied) if session else [],
            "follow_up": follow_up,
            "final": final,
        }
