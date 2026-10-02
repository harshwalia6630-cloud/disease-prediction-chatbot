// Port of NLTK's PorterStemmer (default NLTK_EXTENSIONS mode), so the browser
// produces exactly the same tokens the Python model was trained on.
// Parity is checked against NLTK in tests/web_parity.mjs.

const VOWELS = new Set(["a", "e", "i", "o", "u"]);
const POOL = {
  sky: "sky", skies: "sky", dying: "die", lying: "lie", tying: "tie", news: "news",
  innings: "inning", inning: "inning", outings: "outing", outing: "outing",
  cannings: "canning", canning: "canning", howe: "howe",
  proceed: "proceed", exceed: "exceed", succeed: "succeed",
};

function isConsonant(word, i) {
  if (VOWELS.has(word[i])) return false;
  if (word[i] === "y") {
    let negate = false;
    while (i > 0 && word[i] === "y") { negate = !negate; i -= 1; }
    return (!VOWELS.has(word[i])) !== negate;
  }
  return true;
}

function consonantFlags(word) {
  const flags = [];
  for (let i = 0; i < word.length; i++) {
    const ch = word[i];
    if (VOWELS.has(ch)) flags.push(false);
    else if (ch === "y") flags.push(i === 0 ? true : !flags[i - 1]);
    else flags.push(true);
  }
  return flags;
}

function measure(stem) {
  const seq = consonantFlags(stem).map(c => (c ? "c" : "v")).join("");
  return (seq.match(/vc/g) || []).length;
}
const positiveMeasure = stem => measure(stem) > 0;
const containsVowel = stem => !consonantFlags(stem).every(Boolean);

function endsDoubleConsonant(word) {
  return word.length >= 2 && word[word.length - 1] === word[word.length - 2] && isConsonant(word, word.length - 1);
}

function endsCVC(word) {
  const n = word.length;
  return (n >= 3 && isConsonant(word, n - 3) && !isConsonant(word, n - 2) && isConsonant(word, n - 1)
          && !["w", "x", "y"].includes(word[n - 1]))
      || (n === 2 && !isConsonant(word, 0) && isConsonant(word, 1));
}

const strip = (word, suffix) => (suffix === "" ? word : word.slice(0, -suffix.length));

function applyRules(word, rules) {
  for (const [suffix, replacement, condition] of rules) {
    if (suffix === "*d") {
      if (endsDoubleConsonant(word)) {
        const stem = word.slice(0, -2);
        return !condition || condition(stem) ? stem + replacement : word;
      }
      continue;
    }
    if (word.endsWith(suffix)) {
      const stem = strip(word, suffix);
      return !condition || condition(stem) ? stem + replacement : word;
    }
  }
  return word;
}

function step1a(word) {
  if (word.endsWith("ies") && word.length === 4) return strip(word, "ies") + "ie";
  return applyRules(word, [["sses", "ss"], ["ies", "i"], ["ss", "ss"], ["s", ""]]);
}

function step1b(word) {
  if (word.endsWith("ied")) return strip(word, "ied") + (word.length === 4 ? "ie" : "i");
  if (word.endsWith("eed")) {
    const stem = strip(word, "eed");
    return measure(stem) > 0 ? stem + "ee" : word;
  }
  let inter = null;
  for (const suffix of ["ed", "ing"]) {
    if (word.endsWith(suffix)) {
      const s = strip(word, suffix);
      if (containsVowel(s)) { inter = s; break; }
    }
  }
  if (inter === null) return word;
  const last = inter[inter.length - 1];
  return applyRules(inter, [
    ["at", "ate"], ["bl", "ble"], ["iz", "ize"],
    ["*d", last, () => !["l", "s", "z"].includes(last)],
    ["", "e", stem => measure(stem) === 1 && endsCVC(stem)],
  ]);
}

function step1c(word) {
  return applyRules(word, [["y", "i", stem => stem.length > 1 && isConsonant(stem, stem.length - 1)]]);
}

function step2(word) {
  if (word.endsWith("alli") && positiveMeasure(strip(word, "alli"))) return step2(strip(word, "alli") + "al");
  const p = positiveMeasure;
  return applyRules(word, [
    ["ational", "ate", p], ["tional", "tion", p], ["enci", "ence", p], ["anci", "ance", p],
    ["izer", "ize", p], ["bli", "ble", p], ["alli", "al", p], ["entli", "ent", p], ["eli", "e", p],
    ["ousli", "ous", p], ["ization", "ize", p], ["ation", "ate", p], ["ator", "ate", p],
    ["alism", "al", p], ["iveness", "ive", p], ["fulness", "ful", p], ["ousness", "ous", p],
    ["aliti", "al", p], ["iviti", "ive", p], ["biliti", "ble", p], ["fulli", "ful", p],
    ["logi", "log", () => positiveMeasure(word.slice(0, -3))],
  ]);
}

function step3(word) {
  const p = positiveMeasure;
  return applyRules(word, [["icate", "ic", p], ["ative", "", p], ["alize", "al", p], ["iciti", "ic", p],
                           ["ical", "ic", p], ["ful", "", p], ["ness", "", p]]);
}

function step4(word) {
  const m1 = stem => measure(stem) > 1;
  return applyRules(word, [
    ["al", "", m1], ["ance", "", m1], ["ence", "", m1], ["er", "", m1], ["ic", "", m1], ["able", "", m1],
    ["ible", "", m1], ["ant", "", m1], ["ement", "", m1], ["ment", "", m1], ["ent", "", m1],
    ["ion", "", stem => measure(stem) > 1 && ["s", "t"].includes(stem[stem.length - 1])],
    ["ou", "", m1], ["ism", "", m1], ["ate", "", m1], ["iti", "", m1], ["ous", "", m1], ["ive", "", m1], ["ize", "", m1],
  ]);
}

function step5a(word) {
  if (word.endsWith("e")) {
    const stem = strip(word, "e");
    const m = measure(stem);
    if (m > 1) return stem;
    if (m === 1 && !endsCVC(stem)) return stem;
  }
  return word;
}

function step5b(word) {
  return applyRules(word, [["ll", "l", () => measure(word.slice(0, -1)) > 1]]);
}

const cache = new Map();

export function stem(word) {
  const cached = cache.get(word);
  if (cached !== undefined) return cached;
  let s = word.toLowerCase();
  if (Object.prototype.hasOwnProperty.call(POOL, s)) s = POOL[s];
  else if (word.length > 2) s = step5b(step5a(step4(step3(step2(step1c(step1b(step1a(s))))))));
  cache.set(word, s);
  return s;
}
