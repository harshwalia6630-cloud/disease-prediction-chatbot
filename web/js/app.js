import { DiseaseChatbot, Session } from "./chatbot.js";
import { SymptomModel } from "./model.js";

const $ = id => document.getElementById(id);
const log = $("log"), form = $("form"), input = $("input"), send = $("send"), chips = $("chips");
const STARTERS = ["I have a skin rash and itching", "High fever with chills and sweating",
                  "Burning when I pee and I always need to pee", "My head hurts and I feel dizzy"];

let bot, session = new Session();

const esc = s => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const md = s => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (role === "user") div.textContent = text;
  else div.innerHTML = text.split("\n\n").map(p => (p.startsWith("⚠️") ? `<span class="warn">${md(p)}</span>` : md(p))).join("\n\n");
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

function setChips(list) {
  chips.innerHTML = "";
  for (const label of list) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "chip"; b.textContent = label;
    b.addEventListener("click", () => submit(label));
    chips.appendChild(b);
  }
}

function renderPanel(r, ms) {
  const preds = $("preds"), tags = $("tags");
  if (r.predictions.length) {
    preds.innerHTML = r.predictions.map(p => {
      const pc = Math.round(p.probability * 100);
      return `<div class="pred-row"><div class="pred-top"><span>${esc(p.disease)}</span><span>${pc}%</span></div>
              <div class="track"><div class="fill" style="width:${pc}%"></div></div></div>`;
    }).join("");
  }
  const t = r.symptoms_confirmed.map(s => `<span class="tag">${esc(s)}</span>`)
    .concat(r.symptoms_denied.map(s => `<span class="tag no" title="you said you don't have this">${esc(s)}</span>`));
  tags.innerHTML = t.length ? t.join("") : '<p class="muted small">Nothing detected yet.</p>';
  if (ms !== null) $("timing").textContent = `${ms.toFixed(2)} ms in your browser`;
}

function resetPanel() {
  $("preds").innerHTML = '<p class="muted small">No prediction yet.</p>';
  $("tags").innerHTML = '<p class="muted small">Nothing detected yet.</p>';
  $("timing").textContent = "–";
}

async function submit(text) {
  text = text.trim();
  if (!text || !bot) return;
  addMessage("user", text);
  input.value = "";
  setChips([]);
  const typing = addMessage("bot", "");
  typing.innerHTML = '<span class="typing"><i></i><i></i><i></i></span>';
  const t0 = performance.now();
  const r = bot.respond(session, text);
  const ms = performance.now() - t0;
  await new Promise(res => setTimeout(res, 350)); // brief pause so the reply doesn't feel abrupt
  typing.remove();
  addMessage("bot", r.reply);
  renderPanel(r, r.predictions.length ? ms : null);
  setChips(r.follow_up ? ["Yes", "No"] : []);
  input.focus();
}

function restart() {
  session = new Session();
  log.innerHTML = "";
  resetPanel();
  addMessage("bot", "Hi! Tell me what symptoms you're experiencing and I'll suggest what they might indicate.");
  setChips(STARTERS);
}

form.addEventListener("submit", e => { e.preventDefault(); submit(input.value); });
$("reset").addEventListener("click", restart);

(async () => {
  try {
    const model = await SymptomModel.load("model/");
    bot = new DiseaseChatbot(model);
    // measure real in-browser latency for the hero stat
    const t0 = performance.now();
    for (let i = 0; i < 20; i++) model.predictProba("I have a high fever, chills and body aches");
    $("stat-latency").textContent = `${((performance.now() - t0) / 20).toFixed(1)} ms`;
    $("status").textContent = `Model loaded · ${model.classes.length} diseases · ${model.nFeatures.toLocaleString()} features`;
    for (const el of [input, send, $("reset")]) el.disabled = false;
    restart();
  } catch (err) {
    $("status").textContent = "Could not load the model. Please refresh the page.";
    console.error(err);
  }
})();
