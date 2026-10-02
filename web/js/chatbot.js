// Dialogue manager, mirroring src/symptom_chatbot/chatbot.py.

const YES = /^\s*(y|yes|yeah|yep|yup|sure|correct|i do|haan|ha|han)\b/i;
const NO = /^\s*(n|no|nope|nah|not really|i don'?t|i do not|nahi|na)\b/i;
const GREETING = /^\s*(hi|hello|hey|hii+|good (morning|afternoon|evening)|namaste)\b[\s!.]*$/i;
const RESET = /^\s*(reset|restart|start over|new chat|clear)\s*$/i;
const EMERGENCY = "⚠️ Some of your symptoms can indicate a medical emergency. "
  + "If they are severe or sudden, call emergency services or go to the nearest hospital now.";

// Python's round()/format() round on the exact binary value and break exact ties to even;
// toFixed() also uses the exact value, so only exact .5 ties need special handling.
function pyRound(x, digits) {
  const scaled = x * 10 ** digits;
  if (Number.isInteger(scaled - 0.5)) {
    const lo = Math.floor(scaled);
    return (lo % 2 === 0 ? lo : lo + 1) / 10 ** digits;
  }
  return Number(x.toFixed(digits));
}
const pct = p => `${pyRound(p * 100, 0)}%`;
const round4 = x => pyRound(x, 4);

export class Session {
  constructor() {
    this.texts = [];
    this.confirmed = new Set();
    this.denied = new Set();
    this.asked = [];
    this.answers = new Map();
    this.pending = null;
  }
}

export class DiseaseChatbot {
  constructor(model) {
    this.model = model;
    const cfg = model.meta.chat;
    this.confident = cfg.confident;
    this.maxQuestions = cfg.max_questions;
    this.topK = cfg.top_k;
    this.urgent = new Set(cfg.urgent);
    this.disclaimer = cfg.disclaimer;
    this.profiles = model.meta.profiles;
    this.precautions = model.meta.precautions;
    this.classes = model.classes;
  }

  predict(text, topK = null, answers = null) {
    let probs = Array.from(this.model.predictProba(text));
    for (const [symptom, present] of answers || []) {
      probs = probs.map((p, i) => {
        const freq = (this.profiles[this.classes[i]] || {})[symptom] || 0;
        const lik = Math.min(0.9, Math.max(0.1, freq));
        return p * (present ? lik : 1 - lik);
      });
    }
    const total = probs.reduce((a, b) => a + b, 0);
    probs = probs.map(p => p / total);
    return probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, topK || this.topK)
      .map(([p, i]) => ({ disease: this.classes[i], probability: round4(p) }));
  }

  detectSymptoms(text) {
    const found = new Map();
    for (const m of this.model.findSymptoms(text)) found.set(m.symptom, !m.negated);
    return found;
  }

  respond(session, message) {
    message = message.trim();
    if (!message) return this._reply("Please describe how you are feeling.");
    if (RESET.test(message)) {
      Object.assign(session, new Session());
      return this._reply("Okay, let's start over. What symptoms are you experiencing?");
    }
    if (GREETING.test(message) && !session.texts.length) {
      return this._reply("Hello! I can suggest possible conditions based on your symptoms. "
        + 'Describe what you\'re feeling, e.g. "I have a fever, headache and body aches".');
    }

    const shortAnswer = message.split(/\s+/).length <= 3;
    if (session.pending && shortAnswer && (YES.test(message) || NO.test(message))) {
      const present = YES.test(message);
      (present ? session.confirmed : session.denied).add(session.pending);
      session.answers.set(session.pending, present);
      session.pending = null;
    } else {
      session.pending = null;
      session.texts.push(message);
      for (const [symptom, present] of this.detectSymptoms(message)) {
        (present ? session.confirmed : session.denied).add(symptom);
        (present ? session.denied : session.confirmed).delete(symptom);
      }
    }

    if (!session.confirmed.size && session.texts.join(" ").split(/\s+/).filter(Boolean).length < 4) {
      return this._reply("Could you tell me a bit more about your symptoms? For example: where it hurts, "
        + "whether you have a fever, and how long it has been going on.");
    }

    const preds = this.predict(session.texts.join(" "), null, session.answers);
    if (preds[0].probability < this.confident && session.asked.length < this.maxQuestions) {
      const question = this._nextQuestion(session, preds);
      if (question) {
        session.pending = question;
        session.asked.push(question);
        return this._reply(`Do you also have ${question}? (yes/no)`, preds, session, question);
      }
    }
    return this._final(preds, session);
  }

  _nextQuestion(session, preds) {
    const known = new Set([...session.confirmed, ...session.denied, ...session.asked]);
    const cands = preds.filter(p => p.disease in this.profiles).map(p => [p.disease, p.probability]);
    if (cands.length < 2) return null;
    const sum = cands.reduce((a, [, p]) => a + p, 0);
    const weights = cands.map(([, p]) => p / sum);
    const pool = new Set();
    for (const [d] of cands) for (const s of Object.keys(this.profiles[d])) if (!known.has(s)) pool.add(s);
    let best = null, bestScore = 0;
    // JS default sort compares UTF-16 code units, which matches Python's sorted() for these ASCII names
    for (const symptom of [...pool].sort()) {
      const freq = cands.map(([d]) => this.profiles[d][symptom] || 0);
      const mean = weights.reduce((a, w, i) => a + w * freq[i], 0);
      const score = weights.reduce((a, w, i) => a + w * (freq[i] - mean) ** 2, 0);
      if (score > bestScore) { best = symptom; bestScore = score; }
    }
    return best;
  }

  _final(preds, session) {
    const top = preds[0];
    let lines;
    if (top.probability < 0.4) {
      const listed = preds.map(p => `${p.disease} (${pct(p.probability)})`).join(", ");
      lines = [`I couldn't narrow this down with confidence. Conditions that fit your symptoms include: `
        + `${listed}. A doctor can examine you and run tests to find the actual cause.`];
    } else {
      lines = [`Based on what you've described, the most likely condition is **${top.disease}** `
        + `(${pct(top.probability)} confidence).`];
      const others = preds.slice(1).filter(p => p.probability >= 0.05).map(p => `${p.disease} (${pct(p.probability)})`);
      if (others.length) lines.push("Other possibilities: " + others.join(", ") + ".");
      if (this.precautions[top.disease]) lines.push("General precautions: " + this.precautions[top.disease].join("; ") + ".");
    }
    if (preds.some(p => this.urgent.has(p.disease) && p.probability >= 0.2)) lines.unshift(EMERGENCY);
    lines.push(this.disclaimer);
    return this._reply(lines.join("\n\n"), preds, session, null, true);
  }

  _reply(text, preds = null, session = null, followUp = null, final = false) {
    return {
      reply: text,
      predictions: preds || [],
      symptoms_confirmed: session ? [...session.confirmed].sort() : [],
      symptoms_denied: session ? [...session.denied].sort() : [],
      follow_up: followUp,
      final,
    };
  }
}
