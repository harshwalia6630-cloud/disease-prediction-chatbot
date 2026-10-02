"""Measure how much follow-up questions help when a user only mentions part of their symptoms.

For every held-out test symptom combination (never seen in training), a
simulated patient first mentions 2 of their symptoms and then answers the
bot's yes/no questions truthfully from their full symptom list. Top-1 and
top-3 accuracy are compared with the bot's answer before any questions.
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from symptom_chatbot.chatbot import DiseaseChatbot, Session  # noqa: E402


def combos_for(split):
    df = pd.read_csv(ROOT / "data/processed/symptom_combos.csv")
    df = df[df.split == split]
    return [(d, s.split("|")) for d, s in zip(df.disease, df.symptoms)]


def simulate(bot, combos, initial=2, trials=5, seed=0):
    rng = random.Random(seed)
    totals = {"before": [0, 0], "after": [0, 0]}
    n, questions = 0, 0
    for disease, symptoms in combos:
        for _ in range(trials):
            said = rng.sample(symptoms, min(initial, len(symptoms)))
            session = Session()
            r = bot.respond(session, "I have " + " and ".join(said) + ".")
            first = [p["disease"] for p in bot.predict(" ".join(session.texts), top_k=3)]
            while r["follow_up"]:
                questions += 1
                r = bot.respond(session, "yes" if r["follow_up"] in symptoms else "no")
            final = [p["disease"] for p in r["predictions"]]
            for key, preds in (("before", first), ("after", final)):
                totals[key][0] += preds[0] == disease
                totals[key][1] += disease in preds[:3]
            n += 1
    return {
        "dialogues": n,
        "avg_follow_up_questions": round(questions / n, 2),
        "top1_before_questions": round(totals["before"][0] / n, 4),
        "top1_after_questions": round(totals["after"][0] / n, 4),
        "top3_before_questions": round(totals["before"][1] / n, 4),
        "top3_after_questions": round(totals["after"][1] / n, 4),
    }


def main(initial=2):
    bot = DiseaseChatbot()
    # choose the "stop asking" confidence on validation dialogues only
    tuning = {}
    for conf in (0.5, 0.6, 0.7, 0.8, 0.9):
        bot.confident = conf
        tuning[conf] = simulate(bot, combos_for("val"), initial)
        print(conf, tuning[conf])
    best = max(tuning, key=lambda c: (tuning[c]["top1_after_questions"], -tuning[c]["avg_follow_up_questions"]))
    bot.confident = best
    report = {"initial_symptoms_mentioned": initial, "selected_confidence_threshold": best,
              "validation_sweep": {str(k): v for k, v in tuning.items()},
              "test": simulate(bot, combos_for("test"), initial)}
    (ROOT / "reports/dialogue_simulation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["test"], indent=2), "threshold", best)


if __name__ == "__main__":
    main()
