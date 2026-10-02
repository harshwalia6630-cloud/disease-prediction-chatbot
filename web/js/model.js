// Browser implementation of the trained scikit-learn pipeline:
//   symptom normalisation -> tokenise/stem -> TF-IDF (word 1-2 + char_wb 3-5) -> LinearSVC -> sigmoid calibration
// Every step mirrors the Python code in src/symptom_chatbot so results match to ~1e-6.

import { stem } from "./porter.js";

const WORD_RE = /[a-z][a-z_']*/g;
const CLAUSE_BREAK = /[.,;:!?]|\bbut\b|\bhowever\b|\balthough\b/;
const escapeRe = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export class SymptomModel {
  static async load(base = "model/") {
    const [meta, coef] = await Promise.all([
      fetch(base + "model.json").then(r => r.json()),
      fetch(base + "coef.bin").then(r => r.arrayBuffer()),
    ]);
    return new SymptomModel(meta, new Float32Array(coef));
  }

  constructor(meta, coef) {
    this.meta = meta;
    this.classes = meta.classes;
    this.coef = coef;
    this.nFeatures = meta.n_features;
    this.stop = new Set(meta.stopwords);
    this.negations = new Set(meta.negations);
    this.phraseToSymptom = meta.phrases;
    const phrases = Object.keys(meta.phrases).sort((a, b) => b.length - a.length);
    this.pattern = new RegExp("\\b(" + phrases.map(escapeRe).join("|") + ")\\b", "g");
    this.symptoms = [...new Set(Object.values(meta.phrases))].sort();
    this.word = this._vectorizer(meta.word, 0);
    this.char = this._vectorizer(meta.char, meta.word.terms.length);
  }

  _vectorizer(spec, offset) {
    const index = new Map(spec.terms.map((t, i) => [t, i]));
    return { index, idf: spec.idf, offset, ngram: spec.ngram };
  }

  // ---------- preprocessing (preprocess.py)
  findSymptoms(text) {
    text = text.toLowerCase();
    const mentions = [];
    for (const m of text.matchAll(this.pattern)) {
      mentions.push({ symptom: this.phraseToSymptom[m[1]], negated: this._isNegated(text, m.index),
                      start: m.index, end: m.index + m[1].length });
    }
    return mentions;
  }

  _isNegated(text, start, window = 4) {
    const clauses = text.slice(0, start).split(CLAUSE_BREAK);
    const words = clauses[clauses.length - 1].match(WORD_RE) || [];
    return words.slice(-window).some(w => this.negations.has(w));
  }

  preprocess(text) {
    text = text.toLowerCase().replaceAll("’", "'");
    const mentions = this.findSymptoms(text);
    let rebuilt = "", last = 0;
    for (const m of mentions) {
      rebuilt += text.slice(last, m.start) + m.symptom;
      last = m.end;
    }
    rebuilt += text.slice(last);
    const words = (rebuilt.match(WORD_RE) || []).map(w => w.replaceAll("'", ""));
    const tokens = words.filter(w => !this.stop.has(w) && w.length > 1).map(stem);
    for (const m of mentions) tokens.push((m.negated ? "not_" : "") + "sym_" + m.symptom.replaceAll(" ", "_"));
    return tokens.join(" ");
  }

  // ---------- TF-IDF (sklearn TfidfVectorizer, sublinear_tf, l2 norm)
  static wordNgrams(doc, [minN, maxN]) {
    const toks = doc.split(/\s+/).filter(Boolean);
    const out = minN === 1 ? [...toks] : [];
    for (let n = Math.max(minN, 2); n <= Math.min(maxN, toks.length); n++)
      for (let i = 0; i + n <= toks.length; i++) out.push(toks.slice(i, i + n).join(" "));
    return out;
  }

  static charWbNgrams(doc, [minN, maxN]) {
    const out = [];
    for (let w of doc.toLowerCase().replace(/\s\s+/g, " ").split(/\s+/).filter(Boolean)) {
      w = " " + w + " ";
      for (let n = minN; n <= maxN; n++) {
        let offset = 0;
        out.push(w.slice(offset, offset + n));
        while (offset + n < w.length) { offset += 1; out.push(w.slice(offset, offset + n)); }
        if (offset === 0) break; // word shorter than n: counted once, like sklearn
      }
    }
    return out;
  }

  _tfidf(grams, vec, features) {
    const counts = new Map();
    for (const g of grams) {
      const i = vec.index.get(g);
      if (i !== undefined) counts.set(i, (counts.get(i) || 0) + 1);
    }
    let norm = 0;
    const entries = [];
    for (const [i, c] of counts) {
      const v = (Math.log(c) + 1) * vec.idf[i];
      entries.push([i, v]);
      norm += v * v;
    }
    norm = Math.sqrt(norm);
    for (const [i, v] of entries) features.push([vec.offset + i, v / norm]);
  }

  features(text) {
    const doc = this.preprocess(text);
    const features = [];
    this._tfidf(SymptomModel.wordNgrams(doc, this.word.ngram), this.word, features);
    this._tfidf(SymptomModel.charWbNgrams(doc, this.char.ngram), this.char, features);
    return features;
  }

  // ---------- LinearSVC + CalibratedClassifierCV(sigmoid)
  predictProba(text) {
    const x = this.features(text);
    const K = this.classes.length, F = this.nFeatures;
    const proba = new Float64Array(K);
    let total = 0;
    for (let k = 0; k < K; k++) {
      let d = this.meta.intercept[k];
      const row = k * F;
      for (const [j, v] of x) d += this.coef[row + j] * v;
      const [a, b] = this.meta.calibration[k];
      proba[k] = 1 / (1 + Math.exp(a * d + b));
      total += proba[k];
    }
    for (let k = 0; k < K; k++) proba[k] = total ? proba[k] / total : 1 / K;
    return proba;
  }
}
