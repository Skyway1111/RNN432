// Chapters 4 and 5 share one figure: the inputs over time, and core neurons drawn against the quantity each computes,
// with the scale and fit measured live on 2,000 random inputs. n2 holds the record; n4 the jump past it; n7, n6, n1,
// n8 how far the new input falls below a record.
(function () {
  const rec = (x, b) => { let m = 0; for (let i = 0; i <= b; i++) m = Math.max(m, x[i]); return m; };
  const second = (x, b) => { let a = 0, c = 0; for (let i = 0; i <= b; i++) { if (x[i] > a) { c = a; a = x[i]; } else if (x[i] > c) c = x[i]; } return c; };
  const R = v => Math.max(0, v);
  // Each panel: the neuron, the quantity at step t (after reading x_t), and what to call it.
  const PANELS = {
    2: { q: (x, t) => rec(x, t - 1), name: 'the record before this input' },
    4: { q: (x, t) => R(x[t] - rec(x, t - 1)), name: 'how far this input beats the record' },
    7: { q: (x, t) => R(rec(x, t - 1) - x[t]), name: 'how far this input falls below the record' },
    6: { q: (x, t) => R(rec(x, t - 2) - x[t]), name: 'below the record, leaving out the previous input' },
    1: { q: (x, t) => R(Math.max(rec(x, t - 3), t >= 1 ? x[t - 1] : 0) - x[t]), name: 'below the record, leaving out the input two back' },
    8: { q: (x, t) => R(second(x, t - 1) - x[t]), name: 'below the runner-up so far' },
  };
  const fits = {};
  function fit(cell) {
    if (fits[cell]) return fits[cell];
    const X = M.bank(2000, 11), ys = [], qs = [];
    for (let s = 0; s < 2000; s++) {
      const x = X.subarray(s * 10, s * 10 + 10), r = M.run(M.base, x);
      for (let t = 1; t < 10; t++) { ys.push(r.h[t + 1][cell]); qs.push(PANELS[cell].q(x, t)); }
    }
    let a = 0, b = 0; qs.forEach((q, i) => { a += q * ys[i]; b += q * q; });
    const k = a / b, ym = ys.reduce((s, v) => s + v, 0) / ys.length;
    let ss = 0, st = 0; ys.forEach((y, i) => { ss += (y - k * qs[i]) ** 2; st += (y - ym) ** 2; });
    return (fits[cell] = { k, r2: 1 - ss / st });
  }
  const W = DATA.weights;
  const term = (v, what, first) => `${v >= 0 ? (first ? '' : '+ ') : '− '}${Math.abs(v).toFixed(2)} × ${what}`;

  // A timeline figure: inputs on top, then one panel per neuron in `cells()`, inputs draggable.
  function timeline(fig, bar, cells, formula) {
    let x = Float64Array.from([0.31, -1.2, 1.54, 0.1, -0.4, 1.95, -0.9, 0.66, 1.2, 0.4]);
    let r = M.run(M.base, x), drag = -1, hover = -1, geo = null;
    const g = M.rng(21);
    const cv = V.canvas(fig, (ctx, w, h) => {
      const list = cells(), left = 84, right = 24, top = 56, slot = (w - left - right) / 10;
      const aH = Math.min(150, h * 0.26), pH = (h - top - aH - 48 - (formula() ? 58 : 0)) / Math.max(1, list.length);
      const cx = t => left + slot * (t + 0.5);
      geo = { left, slot, mid: top + aH / 2, unit: aH / 5.6 };
      V.title(ctx, 'The ten inputs, in the order they arrive', left, top - 26);
      ctx.strokeStyle = V.C.axis; ctx.lineWidth = 1; V.hline(ctx, left, w - right, geo.mid);
      for (let t = 0; t < 10; t++) {
        const v = x[t] * geo.unit, on = t === hover || t === drag;
        ctx.fillStyle = V.alpha(V.C.other, on ? 1 : 0.8); V.bar(ctx, cx(t) - 12, geo.mid, 24, v, 3);
        V.handle(ctx, cx(t), geo.mid - v, on ? V.C.ink : V.alpha(V.C.ink, 0.5));
        V.text(ctx, V.xi(t), cx(t), top + aH + 16, { size: 12, align: 'center', color: V.C.ink2 });
      }
      ctx.strokeStyle = V.C.ink; ctx.lineWidth = 2; ctx.beginPath();
      for (let t = 0; t < 10; t++) { const y = geo.mid - rec(x, t) * geo.unit; t ? ctx.lineTo(cx(t) - slot / 2, y) : ctx.moveTo(cx(t) - slot / 2, y); ctx.lineTo(cx(t) + slot / 2, y); }
      ctx.stroke();
      if (w >= 560) V.text(ctx, 'black line: the record so far', w - right, top - 26, { size: 13, color: V.C.ink, align: 'right', face: 'serif', style: 'italic' });  // a phone's figure has no room beside the title
      list.forEach((cell, i) => {
        const y0 = top + aH + 40 + i * pH, base = y0 + pH - 22, f = fit(cell), P = PANELS[cell];
        const vals = Array.from({ length: 10 }, (_, t) => r.h[t + 1][cell]), qv = Array.from({ length: 10 }, (_, t) => f.k * P.q(x, t));
        const vmax = Math.max(1, ...vals, ...qv), sc = (pH - 44) / vmax;
        V.text(ctx, `n${cell}`, left - 52, y0 + 13, { size: 15, color: V.C.ink, weight: 700, face: 'serif' });
        ctx.fillStyle = V.C.core; ctx.beginPath(); ctx.arc(left - 60, y0 + 8, 4.5, 0, 2 * Math.PI); ctx.fill();
        V.text(ctx, `dashed: ${f.k.toFixed(1)} × ${P.name}`, left, y0 + 13, { size: 13, color: V.C.ink2, face: 'serif', style: 'italic' });
        V.text(ctx, `R² ${f.r2.toFixed(2)}`, w - right, y0 + 13, { size: 12.5, align: 'right', color: V.C.ink, weight: 600 });
        ctx.strokeStyle = V.C.axis; ctx.lineWidth = 1; V.hline(ctx, left, w - right, base);
        for (let t = 0; t < 10; t++) { ctx.fillStyle = V.C.core; V.bar(ctx, cx(t) - 10, base, 20, vals[t] * sc, 3); }
        ctx.strokeStyle = V.C.ink; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
        for (let t = 0; t < 10; t++) { ctx.beginPath(); ctx.moveTo(cx(t) - 16, base - qv[t] * sc); ctx.lineTo(cx(t) + 16, base - qv[t] * sc); ctx.stroke(); }
        ctx.setLineDash([]);
      });
      const fm = formula();
      if (fm) fm.forEach((line, i) => V.text(ctx, line, left, h - 42 + i * 21, { size: 13, color: V.C.ink, base: 'middle' }));
    });
    const at = e => { const p = cv.pt(e); return { t: Math.floor((p.x - geo.left) / geo.slot), v: Math.max(-3, Math.min(3, (geo.mid - p.y) / geo.unit)), y: p.y }; };
    const inBand = p => p.t >= 0 && p.t < 10 && Math.abs(p.y - geo.mid) < geo.unit * 3.4;
    const set = () => { r = M.run(M.base, x); cv.redraw(); };
    cv.cv.addEventListener('pointerdown', e => { const p = at(e); if (inBand(p)) { drag = p.t; x[drag] = p.v; cv.cv.setPointerCapture(e.pointerId); set(); } });
    cv.cv.addEventListener('pointermove', e => {
      if (drag >= 0) { x[drag] = at(e).v; set(); return; }
      const p = at(e), nh = inBand(p) ? p.t : -1;
      if (nh !== hover) { hover = nh; cv.cv.style.cursor = hover >= 0 ? 'ns-resize' : 'default'; cv.redraw(); }
    });
    cv.cv.addEventListener('pointerup', () => { drag = -1; cv.redraw(); });
    cv.cv.addEventListener('pointerleave', () => { if (hover >= 0) { hover = -1; cv.redraw(); } });
    V.button(bar, 'Deal new numbers', () => { x = Float64Array.from({ length: 10 }, g); set(); });
    V.note(bar, 'or drag an input bar');
    return cv;
  }

  Story.chapter('record', {
    title: 'The record keeper',
    hint: { text: 'Drag an input bar', icon: 'drag' },
    steps: [
      `<p>Start with <span class="core">n2</span>. The gray bars on top are the ten inputs, x<sub>0</sub> to x<sub>9</sub>, and the black line over them is the <b>record so far</b>: the largest input seen yet, or zero if all were negative.</p>
       <p>Below, the gold bars are n2's value after each input. It climbs when a new record arrives and otherwise holds still. It <b>is</b> the record, scaled up and running one step behind. The dashed lines show the scale, measured on random inputs, and the R² at right says how well they match: 1 is a perfect match, 0 no better than a flat line.</p>`,
      `<p>How do the weights make it hold? Look at its formula under the figure. It has almost no input weight, so new numbers don't reach it directly. It keeps almost all of its own value each step. And it adds <span class="core">n4</span> times a weight.</p>
       <p>So n2 only grows when n4 fires.</p>`,
      `<p><span class="core">n4</span> is the <b>jump</b>: how far the new number beats the record, and zero if it doesn't. Its formula adds the input with a big positive weight and subtracts n2, the record.</p>
       <p>A new record makes n4 fire. One step later, n2 adds that jump. Records only go up, so n2 climbs a staircase.</p>`,
      `<p>Drag the gray bars or deal new numbers: the gold bars follow the dashed lines on every input, not just this one.</p>
       <details><summary>Why n4 subtracts itself</summary><p>n2 runs one step behind, so on the step after a record it still holds the old one. n4's weight on itself, close to −1, subtracts the jump it just reported, so it compares against the new record anyway.</p></details>`,
    ],
    mount(fig, bar) {
      let stage = 0;
      const cv = timeline(fig, bar, () => (stage >= 2 ? [2, 4] : [2]), () => {
        if (stage < 1) return null;
        const lines = [`n2 = max(0, ${term(W.Wh[2][2], 'n2', true)} ${term(W.Wh[2][4], 'n4')} ${term(W.Wi[2], 'input')} + small terms)`];
        if (stage >= 2) lines.push(`n4 = max(0, ${term(W.Wi[4], 'input', true)} ${term(W.Wh[4][2], 'n2')} ${term(W.Wh[4][4], 'n4')} + small terms)`);
        return lines;
      });
      return { step(k) { stage = k; cv.redraw(); } };
    },
  });

  Story.chapter('compare', {
    title: 'How far below the record?',
    hint: { text: 'Drag an input bar', icon: 'drag' },
    steps: [
      `<p>Four more core neurons answer one question: <b>how far below a record did the new number land?</b></p>
       <p><span class="core">n7</span> measures it against the record so far. It adds the input with a big <i>negative</i> weight and adds n2, the record, so it switches on once the new number lands somewhat below the record (<a data-ch="break"></a> works out where), and grows the lower it lands.</p>`,
      `<p>The other three ask the same question against a different yardstick. <span class="core">n6</span> uses the record with the previous input left out, as if it had never arrived, and <span class="core">n1</span> the record with the input two back left out. <span class="core">n8</span> compares against the runner-up so far.</p>
       <p>The descriptions are good, not perfect: the figure gives each one's R², and n8's is the loosest.</p>`,
      `<p>Why would a network care how far below the record a number lands? Because the runner-up is <b>the number that lands just below the record</b>. The smaller n7's reading, the more a number looks like the runner-up.</p>
       <p>Together the six core neurons keep the books. And measuring against records with one input left out is what lets four of them keep time: with n2 they make the clock of the next chapter, which remembers <b>when</b> things happened.</p>`,
    ],
    mount(fig, bar) {
      let stage = 0;
      const cv = timeline(fig, bar, () => (stage >= 1 ? [7, 6, 1, 8] : [7]), () =>
        [`n7 = max(0, ${term(W.Wi[7], 'input', true)} ${term(W.Wh[7][2], 'n2')} ${term(W.Wh[7][4], 'n4')} + small terms)`]);
      return { step(k) { stage = k; cv.redraw(); } };
    },
  });
})();
