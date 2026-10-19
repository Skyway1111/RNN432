// Chapter 12: confident mistakes. How rare they are, one of them, the shortcut behind it read from the weights
// (Wo relu(-Wi): a last input far below the record vetoes position 9), and what knowing it buys a search.
Story.chapter('tail', {
  title: 'Confidently wrong',
  steps: [
    `<p>Most mistakes happen near ties, where the margin is small and the network is unsure. A few are different: the network's favorite beats the right answer by a mile. It is <b>confidently wrong</b>.</p>
     <p>The chart counts how confidently wrong the network is on many random inputs, by how far its favorite beats the right answer. Since only the pattern of an input matters (<a data-ch="rule"></a>), the gap is measured per unit of the input's size. Right of zero is wrong, and each grid line up is ten times more inputs.</p>
     <p>The most confident mistakes we measure are rare. In copies of this network with slightly jiggled weights, typically about <b data-fact="rare_c8"></b> random inputs lands there.</p>`,
    `<p>Here is one. The <span class="second">runner-up</span> is the last number, at position 9. It's negative, but every number except the largest is even lower.</p>
     <p>The network answers a different position, and it isn't close. Position 9, the right answer, gets the <b>worst</b> score of all.</p>`,
    `<p>Why? Look at what the last number alone does. A negative last number lands far below the record, so the comparator <span class="core">n7</span> fires hard. Its output weights vote heavily <b>against position 9</b> and gently for everything else.</p>
     <p>That is all in the weights. Per unit of the last number, they put position 9 <b data-fact="tail_D"></b> points behind the rest. On the real mistakes of this kind we measure <b data-fact="tail_slope"></b>.</p>`,
    `<p>It's a <b>shortcut</b>. On ordinary inputs, a last number far below the record is almost never the runner-up, so voting against it is a great bet. It only fails in a corner where everything else is lower still, a corner training almost never visited. At the rarest level, <b data-fact="tail_share_c8"></b> of the confident mistakes come from this one corner.</p>
     <p>Knowing the corner pays. Testing inputs drawn from such corners finds these mistakes about <b data-fact="modes_c8"></b> times faster than random testing. A smart search that ignores the structure manages <b data-fact="ce_c8"></b> times.</p>`,
  ],
  mount(fig) {
    const x = [0.61, -0.86, -0.86, -0.96, -0.84, -1.13, -0.75, -1.56, -0.93, -0.74];
    const W = M.base, N = 16;
    const alone = Array.from({ length: 10 }, (_, k) => { let s = 0; for (let j = 0; j < N; j++) s += W.Wo[k * N + j] * Math.max(0, -W.Wi[j]); return s * Math.abs(x[9]); });
    const cell7 = W.Wo[9 * N + 7] * Math.max(0, -W.Wi[7]);
    let stage = 0, hist = null;
    function histogram() {
      const n = 200000, lo = -30, hi = 15, nb = 90, cnt = new Float64Array(nb);
      for (const c of M.margins(W, n, 29)) { const b = Math.floor((c - lo) / (hi - lo) * nb); if (b >= 0 && b < nb) cnt[b]++; }
      return { lo, hi, nb, cnt, n };
    }
    const scores = (ctx, L, x0, y0, w, h, title, color) => {
      const lmax = 45, zero = y0 + h * 0.55, bw = w / 10;
      V.title(ctx, title, x0, y0 - 8);
      ctx.strokeStyle = V.C.axis; ctx.lineWidth = 1; V.hline(ctx, x0, x0 + w, zero);
      L.forEach((v, k) => {
        const y = zero - (v / lmax) * h * 0.5, cx = x0 + bw * (k + 0.5);
        ctx.fillStyle = k === 9 ? V.C.second : color; V.bar(ctx, cx - bw * 0.3, zero, bw * 0.6, (v / lmax) * h * 0.5, 3);
        V.text(ctx, V.fmt(v, 0), cx, v >= 0 ? y - 5 : y + 14, { size: 10.5, align: 'center', color: V.C.ink2 });
        V.text(ctx, String(k), cx, y0 + h + 14, { size: 12, align: 'center', color: k === 9 ? V.C.ink : V.C.muted, weight: k === 9 ? 700 : 400 });
      });
    };
    const cv = V.canvas(fig, (ctx, w, h) => {
      if (stage === 0) {
        hist = hist || histogram();
        const left = 80, right = 30, top = 56, bottom = 70, pw = w - left - right, ph = h - top - bottom;
        const ymax = Math.log10(Math.max(...hist.cnt)), X = c => left + (c - hist.lo) / (hist.hi - hist.lo) * pw, Y = v => top + ph - (v / (ymax + 0.3)) * ph;
        V.title(ctx, `${hist.n.toLocaleString()} random inputs, counted by how far the favorite beats the right answer`, left, top - 30);
        V.text(ctx, 'inputs', left - 8, top - 10, { size: 11, align: 'right' });
        for (let d = 0; d <= ymax; d++) { ctx.strokeStyle = V.C.grid; ctx.lineWidth = 1; V.hline(ctx, left, left + pw, Y(d)); V.text(ctx, (10 ** d).toLocaleString(), left - 8, Y(d), { size: 11, align: 'right', base: 'middle' }); }
        ctx.fillStyle = V.alpha(V.C.wrong, 0.10); ctx.fillRect(X(0), top, X(hist.hi) - X(0), ph);
        const bw = pw / hist.nb;
        hist.cnt.forEach((c, b) => { if (!c) return; const v = Math.log10(c) + 0.3, c0 = hist.lo + (b + 0.5) * (hist.hi - hist.lo) / hist.nb; ctx.fillStyle = c0 > 0 ? V.C.wrong : V.C.ink2; ctx.fillRect(left + b * bw + 1, Y(v), bw - 2, top + ph - Y(v)); });
        for (const c of [0, 8]) { ctx.strokeStyle = V.C.ink; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(X(c), top); ctx.lineTo(X(c), top + ph); ctx.stroke(); ctx.setLineDash([]); }
        V.text(ctx, 'right ✓', X(-4), top + 16, { size: 13, align: 'center', color: V.C.ink, weight: 600 }); V.text(ctx, 'wrong ✗', X(4), top + 16, { size: 13, align: 'center', color: V.C.ink, weight: 600 });
        V.text(ctx, 'the most confident level', X(8) + 6, top + 40, { size: 12, color: V.C.ink2, face: 'serif', style: 'italic' });
        V.text(ctx, 'how far the favorite beats the right answer, per unit of input size', left + pw / 2, h - 22, { size: 13, align: 'center', color: V.C.ink2, face: 'serif', style: 'italic' });
      } else {
        const r = M.run(W, x), o = M.order(x), pad = 40, colW = (w - 3 * pad) / 2;
        // the input
        const slot = Math.min(48, (w - 80) / 10), tx = (w - slot * 10) / 2, mid = 100, unit = 30;
        ctx.strokeStyle = V.C.axis; ctx.lineWidth = 1; V.hline(ctx, tx, tx + slot * 10, mid);
        x.forEach((v, t) => { ctx.fillStyle = V.rankColor(o.indexOf(t)); const cx = tx + slot * (t + 0.5); V.bar(ctx, cx - slot * 0.3, mid, slot * 0.6, v * unit, 3); V.text(ctx, V.fmt(v), cx, v >= 0 ? mid - v * unit - 6 : mid - v * unit + 14, { size: 10.5, align: 'center', color: V.C.ink2 }); });
        V.title(ctx, 'The input', tx, 30);
        const ans = M.argmax(r.logits);
        V.chip(ctx, `Network says position ${ans} ✗   right answer 9`, w / 2, 178, { color: V.C.wrong, align: 'center' });
        const sy = 240, sh = h - sy - 80;
        scores(ctx, Array.from(r.logits), pad, sy, stage >= 2 ? colW : w - 2 * pad, sh, 'The network\'s score for each position', V.alpha(V.C.ink2, 0.6));
        if (stage >= 2) {
          scores(ctx, alone, 2 * pad + colW, sy, colW, sh, 'What the last input alone contributes', V.alpha(V.C.core, 0.7));
          V.text(ctx, `|${V.xi(9)}| × output weights · max(0, −input weights)`, 2 * pad + colW, h - 40, { size: 11.5 });
          V.text(ctx, `n7's share of the veto on position 9: ${V.fmt(cell7 * Math.abs(x[9]), 0)} of ${V.fmt(alone[9], 0)}`, 2 * pad + colW, h - 20, { size: 13, color: V.C.ink, weight: 600 });
        }
      }
    });
    return { step(k) { stage = k; cv.redraw(); } };
  },
});
