// Chapter 12: breaking it. Drag one weight and re-measure accuracy live; why W_hh[7,2] matters, from the weights;
// all 432 weights by fragility (results/experiments/weight_whatif.npy via DATA.whatif); shaking every weight.
Story.chapter('break', {
  title: 'Breaking it',
  hint: { text: 'Drag the slider to nudge the weight', icon: 'controls' },
  steps: [
    `<p>Time to break things. The weight on the slider connects <span class="core">n2</span>, the record, to <span class="core">n7</span>, the comparator. It's <b data-fact="w72"></b>.</p>
     <p>Drag the slider to nudge it. Every move re-tests the network on 20,000 random inputs. A nudge of just 0.02 either way costs <b data-fact="w72_drop"></b> of accuracy.</p>`,
    `<p>Why so touchy? n7 fires when a number lands below a certain point, and this weight sets that point. n2 holds the record scaled up, and n7 takes the input with a big negative weight, so the switch point is this weight times n2's scale, divided by n7's input weight. The figure works it out from the weights.</p>
     <p>Nudge the weight by 0.02 and the switch point moves by <b data-live="switch-move"></b> of the record. That's the size of the near-ties where the hard decisions live, so thousands of them flip at once.</p>`,
    `<p>Here are all 432 weights, colored by how much accuracy a nudge of 0.1 costs. Hover to read one, and click to put it on the slider.</p>
     <p>The fragile ones are <b>memory weights</b> into the comparators and the record keeper. The input and output weights barely notice a nudge this size.</p>`,
    `<p>Real weights are never exact, so shake them all at once by a tiny random amount, about 0.002 each. Press <b>Shake</b> a few times. On average, accuracy falls from <b data-fact="acc"></b> to <b data-fact="noise_acc"></b>.</p>
     <p>Almost all of the damage comes through the memory weights. A shake of 0.003 on the memory weights alone leaves <b data-fact="noise_wh"></b>; the same shake on the input weights alone leaves <b data-fact="noise_wi"></b>.</p>`,
  ],
  mount(fig, bar) {
    const B = M.base, N = 16, drop = DATA.whatif, dmax = Math.max(...drop);
    let sel = { kind: 'Wh', i: 7, j: 2 }, stage = 0, curve = [], shakes = [], boxes = [], hot = null, sweep = 0;
    const base = M.accuracy(B);
    // how far a 0.02 nudge moves n7's switch point, as a share of the record: 0.02 × n2's scale / n7's input weight
    const move = 0.02 * B.Wh[2 * N + 4] * B.Wi[4] / -B.Wi[7];
    fig.closest('.chapter').querySelectorAll('[data-live="switch-move"]').forEach(e => (e.textContent = V.pct(move)));
    const idx = s => (s.kind === 'Wi' ? s.i : s.kind === 'Wh' ? 16 + s.i * 16 + s.j : 272 + s.i * 16 + s.j);
    const arr = s => (s.kind === 'Wi' ? 'Wi' : s.kind === 'Wh' ? 'Wh' : 'Wo'), off = s => (s.kind === 'Wi' ? s.i : s.i * N + s.j);
    const label = s => (s.kind === 'Wi' ? `W_in[${s.i}]` : s.kind === 'Wh' ? `W_mem[${s.i}, ${s.j}]` : `W_out[${s.i}, ${s.j}]`);
    const what = s => (s.kind === 'Wi' ? `input → n${s.i}` : s.kind === 'Wh' ? `n${s.j} → n${s.i}` : `n${s.j} → position ${s.i}`);
    let delta = 0;
    const acc = d => { const W = M.clone(B); W[arr(sel)][off(sel)] += d; return M.accuracy(W); };
    function resweep() {
      curve = []; sweep++;
      const id = sweep, steps = Array.from({ length: 41 }, (_, k) => -0.1 + k * 0.005);
      const next = () => { if (id !== sweep || !steps.length) return; const d = steps.shift(); curve.push([d, acc(d)]); cv.redraw(); requestAnimationFrame(next); };
      requestAnimationFrame(next);
    }
    const cv = V.canvas(fig, (ctx, w, h) => {
      if (stage < 2) {
        const now = acc(delta);
        const left = 74, top = 60, pw = w - left - 40, ph = h * 0.52;
        const X = d => left + (d + 0.1) / 0.2 * pw, Y = a => top + (1 - a) * ph;
        for (const a of [0.2, 0.4, 0.6, 0.8, 1]) { ctx.strokeStyle = V.C.grid; ctx.lineWidth = 1; V.hline(ctx, left, left + pw, Y(a)); V.text(ctx, `${a * 100}%`, left - 8, Y(a), { size: 11, align: 'right', base: 'middle' }); }
        for (const d of [-0.1, -0.05, 0, 0.05, 0.1]) V.text(ctx, V.sfmt(d, 2), X(d), top + ph + 18, { size: 11, align: 'center' });
        V.title(ctx, `Accuracy when ${label(sel)} = ${V.fmt(B[arr(sel)][off(sel)], 3)} (${what(sel)}) is nudged by`, left, top - 30);
        ctx.strokeStyle = V.C.ink; ctx.lineWidth = 2; ctx.beginPath(); curve.forEach(([d, a], k) => (k ? ctx.lineTo(X(d), Y(a)) : ctx.moveTo(X(d), Y(a)))); ctx.stroke();
        ctx.fillStyle = V.C.core; ctx.strokeStyle = V.C.surface; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(X(delta), Y(now), 7, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
        V.text(ctx, `${(100 * now).toFixed(1)}%`, X(delta) + 10, Y(now) - 10, { size: 14, color: V.C.ink, weight: 600 });
        if (stage >= 1 && sel.kind === 'Wh' && sel.i === 7 && sel.j === 2) {
          const k2 = B.Wh[2 * N + 4] * B.Wi[4], k7 = -B.Wi[7], w72 = B.Wh[7 * N + 2] + delta;
          const lines = [`n2 ≈ ${k2.toFixed(1)} × record   (${V.fmt(B.Wh[2 * N + 4], 2)} × n4, and n4 ≈ ${B.Wi[4].toFixed(1)} × each jump)`,
            `n7 = max(0, −${k7.toFixed(1)} × input + ${w72.toFixed(3)} × n2 + …)`,
            `switch point ≈ ${w72.toFixed(3)} × ${k2.toFixed(1)} / ${k7.toFixed(1)} = ${(w72 * k2 / k7).toFixed(3)} × record;  a nudge of 0.02 moves it ${(0.02 * k2 / k7 * 100).toFixed(1)}% of the record`];
          lines.forEach((s, i) => V.text(ctx, s, left, top + ph + 60 + i * 24, { size: 13, color: i === 2 ? V.C.ink : V.C.ink2 }));
        }
      } else {
        const gap = 26, left = 70, s = Math.min((h - 150) / 16, (w - left - 30 - 2 * gap) / 27), top = 70;
        let x = left; boxes = [];
        [['Input', 1, (i) => ({ kind: 'Wi', i })], ['Memory', 16, (i, j) => ({ kind: 'Wh', i, j })], ['Output', 10, (i, j) => ({ kind: 'Wo', i: j, j: i })]].forEach(([title, cols, mk]) => {
          V.text(ctx, title, x + cols * s / 2, top - 14, { size: 14, align: 'center', color: V.C.ink, weight: 600, face: 'serif' });
          for (let i = 0; i < 16; i++) for (let j = 0; j < cols; j++) {
            const sw = mk(i, j), d = drop[idx(sw)];
            ctx.fillStyle = V.mix(V.C.surface2, V.C.wrong, Math.sqrt(Math.max(0, d) / dmax));
            V.rect(ctx, x + j * s + 1, top + i * s + 1, s - 2, s - 2);
            boxes.push({ x: x + j * s, y: top + i * s, s, sw, d });
            if (sw.kind === sel.kind && sw.i === sel.i && sw.j === sel.j) { ctx.strokeStyle = V.C.ink; ctx.lineWidth = 2; ctx.strokeRect(x + j * s, top + i * s, s, s); }
          }
          x += cols * s + gap;
        });
        for (let i = 0; i < 16; i++) {
          V.text(ctx, `n${i}`, left - 10, top + i * s + s / 2, { size: 11.5, align: 'right', base: 'middle', color: V.C.ink });
          ctx.fillStyle = V.cellColor(i); ctx.beginPath(); ctx.arc(left - 40, top + i * s + s / 2, 4, 0, 2 * Math.PI); ctx.fill();
        }
        V.text(ctx, `Darker red: a nudge of 0.1 costs more accuracy, up to ${(100 * dmax).toFixed(0)} points. Outlined: the weight on the slider.`, left, top + 16 * s + 26, { size: 12 });
        if (stage >= 3) {
          const y0 = top + 16 * s + 54;
          V.text(ctx, `Shakes of 0.002 on every weight (unshaken: ${(100 * base).toFixed(1)}%):`, left, y0, { size: 12.5, color: V.C.ink2 });
          shakes.forEach((a, k) => V.text(ctx, `${(100 * a).toFixed(1)}%`, left + 300 + k * 58, y0, { size: 13, color: V.C.ink, weight: 600 }));
        }
      }
    });
    cv.cv.addEventListener('mousemove', e => {
      if (stage < 2) return;
      const { x: px, y: py } = cv.pt(e);
      hot = boxes.find(b => px >= b.x && px < b.x + b.s && py >= b.y && py < b.y + b.s);
      cv.cv.style.cursor = hot ? 'pointer' : 'default';
      V.tip(hot ? `<b>${label(hot.sw)} = ${V.fmt(B[arr(hot.sw)][off(hot.sw)], 3)}</b><br>${what(hot.sw)}<br>a nudge of 0.1 costs ${(100 * hot.d).toFixed(1)} points` : null, e.clientX, e.clientY);
    });
    cv.cv.addEventListener('mouseleave', () => V.tip(null));
    cv.cv.addEventListener('click', () => { if (hot) { sel = hot.sw; delta = 0; slider.value = 0; resweep(); cv.redraw(); } });
    const slider = V.slider(bar, 'Nudge', { min: -0.1, max: 0.1, step: 0.002, value: 0, width: '13rem' }, v => { delta = v; cv.redraw(); });
    const g = M.rng(41);
    const shake = () => {
      const W = M.clone(B); for (const k of ['Wi', 'Wh', 'Wo']) for (let i = 0; i < W[k].length; i++) W[k][i] += 0.002 * g();
      shakes = [...shakes.slice(-7), M.accuracy(W)];
    };
    V.button(bar, 'Shake', () => { shake(); cv.redraw(); });
    V.button(bar, 'Reset', () => { delta = 0; slider.value = 0; shakes = []; cv.redraw(); });
    resweep();
    // the last step arrives with a few shakes already done, so its numbers are on the page before any press
    return { step(k) { stage = k; if (k === 3 && !shakes.length) for (let i = 0; i < 4; i++) shake(); cv.redraw(); } };
  },
});
