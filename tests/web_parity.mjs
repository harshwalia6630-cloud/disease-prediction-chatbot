// Checks that the browser port (web/js) reproduces the Python pipeline exactly.
// Fixtures come from scripts/export_web_model.py.   Run: node tests/web_parity.mjs

import { readFileSync } from "node:fs";
import { DiseaseChatbot, Session } from "../web/js/chatbot.js";
import { SymptomModel } from "../web/js/model.js";
import { stem } from "../web/js/porter.js";

const root = new URL("../", import.meta.url);
const meta = JSON.parse(readFileSync(new URL("web/model/model.json", root)));
const buf = readFileSync(new URL("web/model/coef.bin", root));
const model = new SymptomModel(meta, new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4));
const bot = new DiseaseChatbot(model);
const fx = JSON.parse(readFileSync(new URL("tests/fixtures/web_parity.json", root)));

let failures = 0;
const fail = (what, detail) => { failures++; if (failures <= 15) console.log("FAIL", what, detail); };

// 1. stemmer
let n = 0;
for (const [w, expected] of Object.entries(fx.stems)) { n++; if (stem(w) !== expected) fail("stem", { w, got: stem(w), expected }); }
console.log(`stems: ${n} checked`);

// 2. preprocessing + probabilities
let maxDiff = 0, top1Mismatch = 0;
fx.texts.forEach((t, i) => {
  const pre = model.preprocess(t);
  if (pre !== fx.preprocessed[i]) fail("preprocess", { t, got: pre, expected: fx.preprocessed[i] });
  const p = model.predictProba(t), q = fx.probabilities[i];
  for (let k = 0; k < p.length; k++) maxDiff = Math.max(maxDiff, Math.abs(p[k] - q[k]));
  const argmax = a => a.indexOf(Math.max(...a));
  if (argmax(Array.from(p)) !== argmax(q)) top1Mismatch++;
});
console.log(`texts: ${fx.texts.length} checked, max |p_js - p_py| = ${maxDiff.toExponential(2)}, top-1 mismatches = ${top1Mismatch}`);
if (maxDiff > 1e-5) fail("probabilities", maxDiff);
if (top1Mismatch) fail("top1", top1Mismatch);

// 3. whole conversations
for (const d of fx.dialogues) {
  const s = new Session();
  d.messages.forEach((m, i) => {
    const got = bot.respond(s, m), exp = d.replies[i];
    if (JSON.stringify(got) !== JSON.stringify(exp)) fail("dialogue", { message: m, got, expected: exp });
  });
}
console.log(`dialogues: ${fx.dialogues.length} checked`);

console.log(failures ? `\n${failures} FAILURE(S)` : "\nALL PARITY CHECKS PASSED");
process.exit(failures ? 1 : 0);
