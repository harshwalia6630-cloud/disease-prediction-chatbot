import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from symptom_chatbot.chatbot import DiseaseChatbot, Session  # noqa: E402
from symptom_chatbot.model import canonical_symptoms  # noqa: E402
from symptom_chatbot.preprocess import SymptomNormalizer, TextPreprocessor  # noqa: E402


@pytest.fixture(scope="module")
def normalizer():
    return SymptomNormalizer(canonical_symptoms())


@pytest.fixture(scope="module")
def bot():
    return DiseaseChatbot()


def found(normalizer, text):
    return {(m.symptom, m.negated) for m in normalizer.find(text)}


# ---------- preprocessing

def test_lay_phrases_map_to_canonical(normalizer):
    assert found(normalizer, "I keep throwing up and have a tummy ache") == {
        ("vomiting", False), ("stomach pain", False)}


def test_longest_match_wins(normalizer):
    # "high fever" must not be split into "high" + the "fever" synonym
    assert found(normalizer, "I have a high fever") == {("high fever", False)}


def test_negation_is_scoped_to_clause(normalizer):
    assert found(normalizer, "I don't have a fever, but I have a cough") == {
        ("high fever", True), ("cough", False)}


def test_preprocessor_emits_symptom_tokens(normalizer):
    tokens = TextPreprocessor(normalizer)("No fever. Feeling nauseous").split()
    assert "not_sym_high_fever" in tokens
    assert "sym_nausea" in tokens


# ---------- model + conversation

@pytest.mark.parametrize("text, disease", [
    ("burning when I pee, smelly urine and bladder pain", "Urinary Tract Infection"),
    ("high fever with chills and sweating every other day, headache and nausea", "Malaria"),
    ("pus filled pimples and blackheads on my face, some scarring", "Acne"),
])
def test_clear_cases(bot, text, disease):
    assert bot.predict(text)[0]["disease"] == disease


def test_probabilities_are_sorted_and_bounded(bot):
    preds = bot.predict("I have a cough and a runny nose", top_k=5)
    probs = [p["probability"] for p in preds]
    assert probs == sorted(probs, reverse=True)
    assert all(0 <= p <= 1 for p in probs)


def test_vague_input_asks_for_detail(bot):
    r = bot.respond(Session(), "help")
    assert not r["predictions"] and not r["final"]


def test_follow_up_then_answer(bot):
    s = Session()
    r = bot.respond(s, "I have a skin rash")
    assert r["follow_up"] and s.pending == r["follow_up"]
    asked = r["follow_up"]
    r = bot.respond(s, "yes")
    assert asked in r["symptoms_confirmed"]


def test_long_reply_is_not_treated_as_yes(bot):
    s = Session()
    bot.respond(s, "I have a skin rash")
    r = bot.respond(s, "yes and also I have been vomiting a lot")
    assert "vomiting" in r["symptoms_confirmed"]


def test_emergency_warning(bot):
    r = bot.respond(Session(), "severe chest pain, sweating a lot, breathless and vomiting")
    assert r["final"] and "emergency" in r["reply"].lower()


def test_reset_clears_state(bot):
    s = Session()
    bot.respond(s, "I have a fever and a headache")
    bot.respond(s, "reset")
    assert not s.texts and not s.confirmed


# ---------- API

def test_api_roundtrip():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/api/predict", json={"text": "itching, skin rash and discoloured patches"}).json()
    assert r["predictions"][0]["disease"] == "Fungal Infection"
    first = client.post("/api/chat", json={"message": "I have a skin rash"}).json()
    second = client.post("/api/chat", json={"message": "no", "session_id": first["session_id"]}).json()
    assert second["session_id"] == first["session_id"]
    assert client.delete(f"/api/chat/{first['session_id']}").status_code == 200
