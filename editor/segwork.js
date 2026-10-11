// Objects by a word or a click, and edges that hold (owner 2026-10-10: «давай качай конечно»): SAM 3 and BiRefNet lite, run by the
// plugin's server (inpaint/seg.py, segworker.py, POST /api/plugin/frames/segment|matte), on window.hyEdK (index.html):
//   Smart Select (⇧W, with the wand)  a click selects the object under it; ⇧-click adds more of the same object, ⌥-click takes some away
//                                      (SAM's points, label 1 and 0); ↵ or Esc ends the object, the next click starts another. The options
//                                      bar: the mode (new, add, subtract, intersect), «Select by text…» (↵: the best match, or all of them)
//   Select › Select by Text…, Refine Edge   the field above; the selection's edge onto the picture's edges (BiRefNet, else the guided
//                                      filter with no model)
//   Remove Background                   BiRefNet lite when it is installed, macOS Vision as before when not (index.html removeBackground)
// A model at work: the LED on the bar blinks (Hyimg's hy-led), a note says which; a missing model says so, with where the README tells
// how to get it. The agent's commands call the same functions (agentops.js: select.prompt, select.click, mask.refine, matte.export):
//   K.seg.prompt({text, pick, mode, box})  K.seg.click({points, mode})  K.seg.refine({method})  K.seg.exportMatte({as})
//   K.bgMatte(url)  the background mask of a layer for Remove Background, or null (then Vision)
(() => {
  const K = window.hyEdK; if (!K || !K.S || !K.MB) return;
  const { S, T } = K, E = () => window.__ed;
  const W = window.hyEdTr({
    smart: 'Smart Select Tool', byText: 'Select by Text…', field: 'Select by text…', fieldTip: 'A word for what to select: phone, hand, sky. ↵ selects',
    best: 'Best match', all: 'All matches', refine: 'Refine Edge', refineTip: 'The selection\'s edge onto the picture\'s edges (BiRefNet)',
    hint: 'Click an object · ⇧ more of it · ⌥ less · ↵ done', points: n => `${n} points`, working: m => `${m} is working…`,
    loading: m => `${m} is loading, the first time takes up to a minute…`, found: n => `Selected: ${n}`, none: w => `Nothing found for «${w}»`,
    refined: how => `Edge refined (${how})`, noSel: 'No selection', cutout: 'Cut-out', failed: 'Failed: ',
  });
  T.tool.smart = W.smart;
  const PLUGIN = (location.pathname.match(/^\/plugins\/([^/]+)\//) || [])[1] || 'frames';
  const NAME = { segment: 'SAM 3', matte: 'BiRefNet', segwarm: 'SAM 3' };
  const warmed = {};
  let busy = 0;

  // the tool: with the wand (W, ⇧W the next), in the rail's selection group and the board dock's (dockwork.js reads K.TOOLS)
  if (!K.TOOLS.some(t => t.id === 'smart')) K.TOOLS.push({ id: 'smart', name: W.smart, key: 'W', code: 'KeyW' });
  const g = (E().GROUPS || []).find(x => x.tools.includes('wand')); if (g && !g.tools.includes('smart')) g.tools.push('smart');
  const css = document.createElement('style');
  css.textContent = `:root.seg-smart #view[style*="cursor: none"]{cursor:crosshair!important}
    .segbar{display:contents} .segq{width:150px;height:24px;border-radius:7px;border:0;padding:0 8px;background:var(--raise);color:var(--ink);font:inherit;font-size:12px}
    .segq:focus{outline:none;background:var(--raise2, var(--raise))} .segbar hy-led{margin:0 2px}`;
  document.head.append(css);

  // the server: one function for every route, the model's LED on while it works -------------------------------------------------------
  const leds = () => document.querySelectorAll('hy-led.segled');
  function setBusy(d) { busy = Math.max(0, busy + d); S.segBusy = busy > 0; leds().forEach(l => l.setAttribute('state', busy ? 'busy' : 'off')); }
  async function post(route, body) {
    const first = !warmed[route]; K.toast(first ? W.loading(NAME[route]) : W.working(NAME[route]), route === 'matte' ? 'feather' : 'subject');
    setBusy(1); const ac = new AbortController(), to = setTimeout(() => ac.abort(), 300000);
    try {
      const r = await fetch(`/api/plugin/${PLUGIN}/${route}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: ac.signal });
      const j = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
      if (r.ok) warmed[route] = true;
      return { status: r.status, ...j };
    } catch (e) { return { status: 0, error: e.name === 'AbortError' ? 'no answer in 5 min' : e.message }; }
    finally { clearTimeout(to); setBusy(-1); }
  }
  const fail = m => { throw new Error(m); };
  // the document as the person sees it, at most `side` px on its long side, and the factor from document px to its px
  function flat(side) { const [w, h] = K.dims(), s = Math.min(1, side / Math.max(w, h)), c = K.flatAt(s); return { url: c.toDataURL('image/png'), s, w: c.width, h: c.height }; }
  // the selection as a grey picture (white selected) at a size
  function selPNG(w, h) {
    const s = K.sel(), c = K.mk(w, h), x = c.getContext('2d'); x.fillStyle = '#000'; x.fillRect(0, 0, w, h);
    const t = K.mk(w, h), tx = t.getContext('2d'); tx.drawImage(s.c, 0, 0, w, h); tx.globalCompositeOperation = 'source-in'; tx.fillStyle = '#fff'; tx.fillRect(0, 0, w, h);
    x.drawImage(t, 0, 0); return c.toDataURL('image/png');
  }
  // masks (data URLs, white in) onto the selection by a mode, as a marquee is combined (index.html commitSelDraw)
  async function apply(urls, mode, base) {
    K.ensureSel(); const s = K.sel(), cs = [];
    for (const u of urls) cs.push(await K.maskFromURL(u, s.c.width, s.c.height));
    if (base !== undefined) { if (base) K.commitSelDraw(x => x.drawImage(base, 0, 0), 'new'); else { S.hasSel = false; S.selV++; } }
    K.commitSelDraw(x => cs.forEach(c => x.drawImage(c, 0, 0)), mode);
  }
  const modeOf = m => (['new', 'add', 'sub', 'int'].includes(m) ? m : S.selo.mode || 'new');

  // by a word: every instance SAM 3 finds, best first; pick 0 the best, a number that one, "all" their union -----------------------------
  async function prompt(o = {}) {
    const text = String(o.text || '').trim(); if (!text) fail('select.prompt: text, the word for what to select');
    const f = flat(1536), sc = f.s, b = o.box && { x: o.box.x * sc, y: o.box.y * sc, w: o.box.w * sc, h: o.box.h * sc };
    const j = await post('segment', { image: f.url, text, box: b });
    if (j.error) fail(j.error);
    if (!j.masks.length) { K.toast(W.none(text), 'subject'); return { count: 0, ms: j.ms, empty: true }; }
    const pick = o.pick == null ? 0 : o.pick, idx = pick === 'all' ? j.masks.map((_, i) => i) : [Math.max(0, Math.min(j.masks.length - 1, +pick || 0))];
    await apply(idx.map(i => j.masks[i]), modeOf(o.mode));
    K.toast(W.found(idx.length > 1 ? idx.length : text), 'subject');
    const doc = bx => bx.map(v => Math.round(v / sc));
    return { count: j.masks.length, picked: idx, scores: j.scores, boxes: j.boxes.map(doc), ms: j.ms };
  }

  // by clicks: one object, its points (label 1 in, -1 or 0 out); the object is applied over the selection it started from -------------
  const SM = { pts: [], base: null, mode: 'new', v: -1 };
  const endObj = () => { SM.pts = []; SM.v = -1; hint(); };
  async function click(o = {}) {
    const pts = (o.points || []).map(p => Array.isArray(p) ? { x: +p[0], y: +p[1], label: p[2] == null ? 1 : +p[2] } : { x: +p.x, y: +p.y, label: p.label == null ? 1 : +p.label });
    if (!pts.length || pts.some(p => !isFinite(p.x) || !isFinite(p.y))) fail('select.click: points [{x, y, label: 1|-1}]');
    const f = flat(1536), sc = f.s, mode = modeOf(o.mode), base = mode === 'new' ? undefined : (o.base !== undefined ? o.base : S.hasSel ? K.copyC(K.sel().c) : null);
    const j = await post('segment', { image: f.url, points: pts.map(p => ({ x: p.x * sc, y: p.y * sc, label: p.label > 0 ? 1 : -1 })) });
    if (j.error) fail(j.error);
    if (!j.masks.length) return { count: 0, empty: true };
    await apply([j.masks[0]], mode, base);
    return { count: 1, score: j.scores[0], box: j.boxes[0].map(v => Math.round(v / sc)), ms: j.ms };
  }
  async function toolClick(p, e) {
    if (busy) return;
    if (SM.v !== S.selV) SM.pts = [];   // the selection changed by another tool: a new object
    const more = SM.pts.length && (e.shiftKey || e.altKey);
    if (!more) { SM.mode = e.shiftKey ? 'add' : e.altKey ? 'sub' : modeOf(); SM.base = S.hasSel ? K.copyC(K.sel().c) : null; SM.pts = []; }
    SM.pts.push({ x: p.x, y: p.y, label: more && e.altKey ? -1 : 1 });
    try { await click({ points: SM.pts, mode: SM.mode, base: SM.mode === 'new' ? undefined : SM.base }); SM.v = S.selV; }
    catch (er) { SM.pts.pop(); K.toast(W.failed + er.message, 'x'); }
    hint();
  }
  addEventListener('pointerdown', e => {
    if (S.tool !== 'smart' || S.space || S.xf || e.button !== 0 || e.target !== K.view || !S.ready) return;
    e.preventDefault(); e.stopImmediatePropagation(); toolClick(K.toDoc(e.clientX, e.clientY), e);
  }, true);
  addEventListener('keydown', e => {
    if (S.tool !== 'smart' || !SM.pts.length || (e.target.closest && e.target.closest('input, textarea'))) return;
    if (e.key === 'Enter' || e.key === 'Escape') { e.preventDefault(); e.stopImmediatePropagation(); endObj(); }
  }, true);

  // the selection's edge: BiRefNet (the picture's edges, its background away) or the guided filter alone --------------------------------
  async function refine(o = {}) {
    if (!S.hasSel) fail(W.noSel);
    const method = o.method === 'guided' ? 'guided' : 'birefnet', f = flat(2560);
    const j = await post('matte', { image: f.url, mask: selPNG(f.w, f.h), method });
    if (j.error) fail(j.error);
    await apply([j.alpha], 'new'); K.toast(W.refined(j.how === 'birefnet' ? 'BiRefNet' : 'guided'), 'feather');
    return { how: j.how, ms: j.ms, note: j.note || undefined };
  }
  // the selection out: a cut-out layer (⌘J), the selected layer's mask, or PNGs (the alpha and the cut-out) for an agent
  async function exportMatte(o = {}) {
    if (!S.hasSel) fail(W.noSel);
    const as = o.as || 'layer';
    if (as === 'layer') { const n0 = S.undo.length; K.copyToLayer(); if (S.undo.length === n0) fail('matte.export: no layer was made (select a picture layer)');
      const n = K.one(); K.edit(T.h.rename, () => { n.name = String(o.name || W.cutout); }); return { layer: n.id }; }
    if (as === 'mask') { K.addMask(true); return { layer: K.one() && K.one().id }; }
    if (as === 'png') { const [w, h] = K.dims(), alpha = selPNG(w, h), c = E().renderDoc(1), x = c.getContext('2d');
      x.globalCompositeOperation = 'destination-in'; x.drawImage(K.sel().c, 0, 0, w, h); return { alpha, cutout: c.toDataURL('image/png'), size: [w, h] }; }
    fail('matte.export: as layer, mask or png');
  }
  // Remove Background (index.html): BiRefNet on the layer, or null so the studio asks macOS Vision as before
  K.bgMatte = async url => {
    const j = await post('matte', { image: url });
    if (j.status === 503 || j.error) { if (j.status !== 503) K.toast(W.failed + j.error, 'x'); return null; }
    return { mask: j.alpha, ms: j.ms, how: 'birefnet' };
  };
  const guard = fn => async (...a) => { try { return await fn(...a); } catch (e) { K.toast(W.failed + e.message, 'x'); } };
  K.seg = { prompt, click, refine, exportMatte, get busy() { return busy > 0; }, end: endObj };

  // the options bar of Smart Select (index.html updModeBar asks K.barHint for a tool's extras) ---------------------------------------
  const before = K.barHint;
  let hintEl = null;
  function hint() { if (hintEl) hintEl.textContent = SM.pts.length > 1 ? `${W.points(SM.pts.length)} · ${W.hint}` : W.hint; }
  function bar() {
    const wrap = document.createElement('span'); wrap.className = 'segbar';
    const sg = K.Seg([['new', K.svg('sNew', 16), T.selNew], ['add', K.svg('sAdd', 16), T.selAdd, '⇧'], ['sub', K.svg('sSub', 16), T.selSub, '⌥'],
      ['int', K.svg('sInt', 16), T.selInt, '⇧ ⌥']], () => S.selo.mode, v => { S.selo.mode = v; });
    sg.el.querySelectorAll('button').forEach(b => { b.dataset.side = 'bottom'; });
    const q = document.createElement('input'); q.type = 'search'; q.className = 'segq'; q.placeholder = W.field; q.dataset.tip = W.fieldTip; q.dataset.side = 'bottom';
    q.spellcheck = false; q.value = S.segText || '';
    const pk = K.osel([['0', W.best], ['all', W.all]], () => S.segPick || '0', v => { S.segPick = v; }, W.byText);
    q.addEventListener('input', () => { S.segText = q.value; });
    q.addEventListener('keydown', e => { e.stopPropagation(); if (e.key === 'Enter') { e.preventDefault(); guard(prompt)({ text: q.value, pick: S.segPick === 'all' ? 'all' : 0 }); }
      if (e.key === 'Escape') { e.preventDefault(); q.blur(); } });
    const led = document.createElement('hy-led'); led.className = 'segled'; led.setAttribute('state', busy ? 'busy' : 'off'); led.title = 'SAM 3 · BiRefNet';
    const rf = K.otext('feather', W.refine, () => guard(refine)({}), W.refineTip);
    hintEl = document.createElement('span'); hintEl.className = 'hint2'; hint();
    wrap.append(sg.el, q, pk, led, rf, hintEl); return wrap;
  }
  K.barHint = t => (t === 'smart' ? bar() : before ? before(t) : null);
  // the models load while the person aims: picking the tool warms SAM 3 (once per studio)
  const onTool0 = K.onTool;
  K.onTool = id => {
    if (onTool0) onTool0(id);
    document.documentElement.classList.toggle('seg-smart', id === 'smart');
    if (id !== 'smart') endObj(); else if (!warmed.segwarm) { warmed.segwarm = true; fetch(`/api/plugin/${PLUGIN}/segwarm`, { method: 'POST', body: '{"models":["sam"]}' }).catch(() => {}); }
  };

  // the Select menu: Select by Text… and Smart Select after Subject, Refine Edge at the end ------------------------------------------
  const sel = K.MB.find(m => m[0] === 'select');
  if (sel) {
    const items0 = sel[2];
    sel[2] = () => {
      const items = items0(), at = items.findIndex(it => it && it.icon === 'subject');
      const focus = () => { K.setTool('smart'); requestAnimationFrame(() => { const q = document.querySelector('.segq'); if (q) q.focus(); }); };
      items.splice(at + 1, 0, { icon: 'search', label: W.byText, fn: focus }, { icon: 'smart', label: W.smart, keys: ['⇧', 'W'], fn: () => K.setTool('smart') });
      items.push(null, { icon: 'feather', label: W.refine, fn: () => guard(refine)({}), dis: !S.hasSel });
      return items;
    };
  }
})();
