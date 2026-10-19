// Chapter 3: the one rule. Memory before and after one step, the weighted sum into a chosen neuron, the clip at zero,
// the scores read out after ten steps, and what the missing constant implies: scaling every input scales every score.
Story.chapter('rule', {
  title: 'One rule, ten times',
  hint: { text: 'Hover a neuron to see what feeds it', icon: 'hover', from: 1 },
  steps: [
    `<p>Two words to keep apart. The ten <b>inputs</b> are the numbers that arrive: x<sub>0</sub> first, x<sub>9</sub> last. The sixteen <b>neurons</b> are the network's memory, and they are all it carries from one input to the next.</p>
     <p>Before the first input arrives, every neuron is zero.</p>`,
    `<p>When an input arrives, every neuron computes its new value the same way: multiply each of the 16 old values by its <b>memory weight</b>, add them all up, and add the new input times the neuron's <b>input weight</b>.</p>
     <p>The lines show this for one neuron: blue lines push it up, red lines push it down, and thicker lines push harder. Hover any neuron on the right to see its own lines.</p>`,
    `<p>Then comes the one twist. <b>If the sum is negative, the neuron is set to zero.</b> This clip is called a ReLU, and it is the only bend in the whole machine. Everything else is multiplying and adding.</p>
     <details><summary>The rule in symbols</summary><p>new n<sub>i</sub> = max(0, Σ<sub>j</sub> W<sub>mem</sub>[i, j] · old n<sub>j</sub> + W<sub>in</sub>[i] · x). Ten inputs, ten applications of this rule.</p></details>`,
    `<p>Do it ten times, once per input. Then each of the ten answer positions gets a <b>score</b>: a weighted sum of the 16 neurons, using the <b>output weights</b>. The highest score is the answer.</p>
     <p>Press play to watch the memory change as the numbers stream in. That is the entire machine.</p>`,
    `<p>One consequence is worth keeping. Nothing in the rule adds a constant, so if every input doubled, every neuron would double, every score would double, and the answer could not change.</p>
     <p>Try the slider: only the <b>pattern</b> of the ten numbers matters, never their overall size. The puzzle is what the 432 weights make this rule compute.</p>`,
  ],
  mount(fig, bar) {
    const x0 = [0.31, -1.2, 1.54, 0.1, -0.4, 1.05, -0.9, 0.66, -0.1, 0.4];
    let x = x0, r = M.run(M.base, x), k = 1;
    const W = M.base;
    let stage = 0, t = 3, target = 7, clip = 1;
    const tok = {};
    const scale = Math.max(...r.pre.flatMap(z => Array.from(z, Math.abs)));
    const lmax1 = Math.max(...Array.from(r.logits, Math.abs));
    const len = (v, room) => Math.sign(v) * Math.sqrt(Math.abs(v) / scale) * room;

    let rows = [];
    const cv = V.canvas(fig, (ctx, w, h) => {
      const top = 128, rh = Math.min(30, (h - top - 56) / 16), colA = w * 0.11, colB = w * 0.52, room = w * 0.15;
      const showOut = stage >= 3 && t === 10;
      // the input tape
      const slot = Math.min(58, (w - 80) / 10);
      for (let s = 0; s < 10; s++) {
        const cx = 40 + slot * (s + 0.5), on = s === t, past = s < t;
        if (on) { ctx.fillStyle = V.C.surface2; V.roundRect(ctx, cx - slot * 0.42, 14, slot * 0.84, 46, 6); ctx.fill(); }
        V.text(ctx, V.fmt(x[s]), cx, 36, { color: on ? V.C.ink : past ? V.C.ink2 : V.C.muted, size: 13, align: 'center', weight: on ? 700 : 400 });
        V.text(ctx, V.xi(s), cx, 52, { size: 11.5, align: 'center', color: on ? V.C.ink2 : V.C.muted });
      }
      V.text(ctx, t < 10 ? `Step ${t + 1} of 10: input ${V.xi(t)} = ${V.fmt(x[t])} arrives` : 'All ten inputs read', 40, 88, { color: V.C.ink, size: 15, weight: 600, face: 'serif' });
      const hb = r.h[Math.min(t, 10)], ha = t < 10 ? r.h[t + 1] : r.h[10], za = t < 10 ? r.pre[t] : r.pre[9];
      rows = [];
      for (let i = 0; i < 16; i++) {
        const y = top + i * rh + rh / 2, col = stage >= 3 ? V.cellColor(i) : V.C.ink2, picked = i === target && stage >= 1 && !showOut;
        V.text(ctx, `n${i}`, colA - 8, y, { size: 11.5, align: 'right', base: 'middle', color: picked ? V.C.ink : V.C.muted, weight: picked ? 700 : 400 });
        ctx.fillStyle = V.alpha(col, 0.85); ctx.fillRect(colA, y - rh * 0.32, Math.max(1, len(hb[i], room)), rh * 0.64);
        if (!showOut) {
          const zl = len(za[i], room), keep = za[i] > 0;
          if (stage >= 2 && !keep) {
            ctx.fillStyle = V.alpha(V.C.neg, 0.4 * (1 - clip) + 0.1); ctx.fillRect(colB, y - rh * 0.32, zl, rh * 0.64);
            ctx.strokeStyle = V.C.neg; ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.strokeRect(colB + zl, y - rh * 0.32, -zl, rh * 0.64); ctx.setLineDash([]);
          } else {
            ctx.fillStyle = V.alpha(col, 0.85); ctx.fillRect(colB, y - rh * 0.32, Math.max(1, stage >= 2 ? len(ha[i], room) : len(Math.max(0, za[i]), room)), rh * 0.64);
          }
          rows.push({ y, i, x0: colB - 20, x1: colB + room + 10 });
        }
      }
      ctx.strokeStyle = V.C.axis; ctx.lineWidth = 1; V.vline(ctx, colA, top - 6, top + 16 * rh); if (!showOut) V.vline(ctx, colB, top - 6, top + 16 * rh);
      V.text(ctx, t < 10 ? 'Memory before' : 'Memory after ten steps', colA, top - 14, { size: 13, color: V.C.ink2, face: 'serif', style: 'italic' });
      if (!showOut) V.text(ctx, 'Memory after', colB, top - 14, { size: 13, color: V.C.ink2, face: 'serif', style: 'italic' });
      if (stage >= 1 && !showOut && t < 10) {
        const yT = top + target * rh + rh / 2;
        let wmax = 0; for (let j = 0; j < 16; j++) wmax = Math.max(wmax, Math.abs(W.Wh[target * 16 + j] * hb[j]));
        for (let j = 0; j < 16; j++) {
          const c = W.Wh[target * 16 + j] * hb[j]; if (!c) continue;
          ctx.strokeStyle = c > 0 ? V.C.pos : V.C.neg; ctx.globalAlpha = 0.25 + 0.75 * Math.abs(c) / (wmax || 1); ctx.lineWidth = 0.5 + 5 * Math.abs(c) / (wmax || 1);
          const y0 = top + j * rh + rh / 2;
          ctx.beginPath(); ctx.moveTo(colA + Math.max(2, len(hb[j], room)) + 4, y0); ctx.bezierCurveTo(colB - 80, y0, colB - 80, yT, colB - 4, yT); ctx.stroke();
        }
        ctx.globalAlpha = 1;
        const zin = W.Wi[target] * x[t];
        const verdict = stage >= 2 ? (za[target] > 0 ? ', kept' : ', clipped to 0') : '';
        V.text(ctx, `n${target}: the lines add to ${V.sfmt(za[target] - zin)}, plus input ${V.sfmt(zin)}, makes ${V.sfmt(za[target])}${verdict}`, colA, top + 16 * rh + 26, { size: 13, color: V.C.ink });
      }
      if (showOut) {
        const colC = w * 0.5, L = r.logits, lmax = stage >= 4 ? 3 * lmax1 : lmax1, ans = M.argmax(L), oh = Math.min(30, (h - top - 56) / 16);
        const arm = w * 0.17, zero = colC + 86 + arm;
        V.text(ctx, stage >= 4 ? `Scores, with every input × ${k.toFixed(2)}` : 'Score for each position', colC, top - 14, { size: 13, color: V.C.ink2, face: 'serif', style: 'italic' });
        ctx.strokeStyle = V.C.axis; V.vline(ctx, zero, top - 4, top + 16 * oh);
        for (let q = 0; q < 10; q++) {
          const y = top + q * oh * 1.6 + oh / 2, lw = (L[q] / lmax) * arm;
          V.text(ctx, `Position ${q}`, colC, y, { size: 11.5, base: 'middle', color: q === ans ? V.C.ink : V.C.muted, weight: q === ans ? 700 : 400 });
          ctx.fillStyle = q === ans ? V.C.second : L[q] >= 0 ? V.alpha(V.C.pos, 0.7) : V.alpha(V.C.neg, 0.7);
          ctx.fillRect(zero, y - oh * 0.35, lw, oh * 0.7);
          V.text(ctx, V.fmt(L[q], 0), zero + lw + (lw >= 0 ? 6 : -6), y, { size: 11.5, align: lw >= 0 ? 'left' : 'right', base: 'middle', color: V.C.ink2 });
        }
        V.chip(ctx, `Answer: position ${ans}   (the runner-up is at ${M.label(x)}) ${ans === M.label(x) ? '✓' : '✗'}`, colC, top + 16 * oh + 24, { color: ans === M.label(x) ? V.C.right : V.C.wrong });
      }
    });

    cv.cv.addEventListener('mousemove', e => {
      const { x: px, y: py } = cv.pt(e);
      const hit = rows.find(q => Math.abs(py - q.y) < 12 && px > q.x0 && px < q.x1);
      if (hit && stage >= 1 && hit.i !== target) { target = hit.i; cv.redraw(); }
    });
    const play = V.button(bar, 'Play', () => {
      V.tween(tok, 5000, u => { const nt = Math.min(10, Math.floor(u * 11)); if (nt !== t) { t = nt; slider.value = t; cv.redraw(); } });
    });
    const slider = V.slider(bar, 'Step', { min: 0, max: 10, step: 1, value: t }, v => { t = v; cv.redraw(); });
    const scaler = V.slider(bar, 'Scale every input', { min: 0.25, max: 3, step: 0.01, value: 1, width: '10rem' }, v => {
      k = v; x = x0.map(a => a * k); r = M.run(M.base, x); cv.redraw();
    });
    const show = (el, on) => { el.style.display = on ? '' : 'none'; };

    return {
      step(s) {
        stage = s;
        if (s === 2) { clip = 0; V.tween(tok, 900, u => { clip = u; cv.redraw(); }); }
        if (s >= 3) { t = 10; slider.value = 10; }
        else if (t === 10) { t = 3; slider.value = 3; }
        if (s < 4 && k !== 1) { k = 1; scaler.value = 1; x = x0; r = M.run(M.base, x); }
        show(scaler.parentElement, s >= 4); show(play, s < 4); show(slider.parentElement, s < 4);
        cv.redraw();
      },
    };
  },
});
