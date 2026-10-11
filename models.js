// Settings › Image Studio (owner 2026-10-10: the models Image Studio downloaded, «with size and a Remove button»): a section of the
// board's settings panel (Hyimg ui/settings-win.js, section «imagestudio»), drawn by this plugin when the panel opens. A row a model:
// its name, what uses it, its licence, its size and Remove (asked first: the model's folder goes, a later use asks for the download again,
// README «Models»); a model not installed says so. The runtime they share (~/Library/Caches/Hyimg/venvs/seg) shows its size.
// The data is the plugin's route POST /api/plugin/frames/models (inpaint/seg.py status), Remove is .../models_remove.
let t = k => k, timer = 0;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const fmt = b => (window.hyStore && window.hyStore.fmt ? window.hyStore.fmt(b) : `${Math.round(b / 1e6)} MB`);
const host = () => (window.hySetPanel && window.hySetPanel.body()) || document.getElementById("sets");
const UP = () => !!(window.customElements && customElements.get("hy-button"));
const btn = (id, words) => `<hy-button size="s" data-imr="${esc(id)}">${UP() ? esc(words) : `<button type="button">${esc(words)}</button>`}</hy-button>`;
let S = null, busy = "";

async function load() {
  try { const r = await fetch("/api/plugin/frames/models", { method: "POST", body: "{}" }); S = r.ok ? await r.json() : null; } catch (e) { S = null; }
  paint();
}
function section() {
  let el = document.getElementById("hyImgModels");
  const sets = host();
  if (!el && sets) {
    el = document.createElement("section"); el.id = "hyImgModels"; el.className = "hs"; el.dataset.sec = "imagestudio";
    el.setAttribute("aria-label", "Image Studio"); sets.appendChild(el);
    el.style.gridTemplateColumns = "minmax(0, 1fr)";   // a long licence or path ends in … instead of widening the panel
    el.addEventListener("click", e => { const b = e.target.closest("[data-imr]"); if (b && !b.hasAttribute("disabled")) ask(b.dataset.imr); });
  }
  return el;
}
function paint() {
  const el = section(); if (!el) return;
  let h = `<div class="sh"><span class="hs-t">Image Studio</span><span>${esc(t("Models on this Mac"))}</span></div>`;
  if (!S) h += `<div class="hs-row"><span>${esc(t("Measuring…"))}</span></div>`;
  else {
    const sub = words => `<div class="hs-row hs-sub"><span title="${esc(words)}">${esc(words)}</span></div>`;
    for (const m of S.models) {
      h += `<div class="hs-row"><span>${esc(m.name)}</span><b>${esc(m.installed ? fmt(m.bytes) : t("Not installed"))}</b>`
        + (m.installed ? btn(m.id, busy === m.id ? t("Removing…") : t("Remove")) : "") + "</div>" + sub(`${t(m.for)} · ${m.licence}`);
    }
    const r = S.runtime;
    h += `<div class="hs-row"><span>${esc(t("Runtime for SAM 3 and BiRefNet"))}</span><b>${esc(r.installed ? fmt(r.bytes) : t("Not installed"))}</b></div>` + sub(r.path);
    h += `<div class="hy-hint">${esc(t("A model that is not installed is downloaded only after you agree: README «Models»"))}</div>`;
  }
  if (S) el.dataset.sum = t("{n} of {m} installed", { n: S.models.filter(m => m.installed).length, m: S.models.length });   // the header's words
  if (el._h !== h) { el.innerHTML = h; el._h = h; }
}
function ask(id) {
  const m = S && S.models.find(x => x.id === id); if (!m) return;
  const go = async () => {
    busy = id; paint();
    try {
      const r = await fetch("/api/plugin/frames/models_remove", { method: "POST", body: JSON.stringify({ id }) }), j = await r.json();
      if (window.hyToast) window.hyToast(j.error ? t("Not removed: {why}", { why: j.error }) : t("{name} removed, {size} freed", { name: m.name, size: fmt(j.freed) }), j.error ? "error" : "success");
    } finally { busy = ""; load(); }
  };
  if (window.hyConfirm) window.hyConfirm({ title: t("Remove {name}?", { name: m.name }), note: t("{size} on disk. Its tools ask to download it again", { size: fmt(m.bytes) }),
    ok: t("Remove"), icon: "trash", onOk: go });
  else if (confirm(t("Remove {name}?", { name: m.name }))) go();
}

export function register(hy, tr) {
  if (tr) t = tr;
  const watch = () => {
    const sets = document.getElementById("sets"); if (!sets) return;
    new MutationObserver(() => {
      const on = sets.classList.contains("open");
      if (on && !timer) { timer = 1; paint(); load(); } else if (!on) timer = 0;
    }).observe(sets, { attributes: true, attributeFilter: ["class"] });
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", watch); else watch();
}
