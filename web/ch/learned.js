// Chapter 9: how training built it, told through ARC's five sibling networks, saved at 12 checkpoints each
// (0, 2^20 ... 2^30 training sequences): accuracy, and how well all 16 neurons encode the record, at each one
// (results/experiments/core_emergence.txt via DATA.training); then how many core quantities each network keeps in a
// single neuron (results/experiments/role_probe.txt). research/findings/training.md holds the facts.
Story.chapter('learned', {
  title: 'How it learned',
  steps: [
    `<p>This network was published finished, without its training history. But ARC trained five more networks of the same shape on the same game, and saved each one twelve times along the way, from the first game to the billionth.</p>
     <p>Training is one move, repeated: show a network a random input, nudge all 432 numbers a little toward the right answer, and do it again. Each line is one sibling's accuracy as it trains.</p>`,
    `<p>All five learn in the same three phases. In the first, a few million games, accuracy climbs to <b data-fact="train_acc22"></b>, well above the 10% of guessing, while the record keeper and the jump are still forming. Even read from all 16 neurons at once, the record is only partly there: R² <b data-fact="train_rec22"></b>, where 1 would be a perfect reading.</p>`,
    `<p>Then, within the next twelve million games, the record and the jump become readable. Read from all 16 neurons together, the record now comes out almost perfectly, at R² <b data-fact="train_rec24"></b>; in <b data-fact="train_jump24"></b> of the five siblings a single neuron tracks the jump; and accuracy has jumped to <b data-fact="train_acc24"></b>.</p>`,
    `<p>The rest of training, 98% of it, buys the last points, up to <b data-fact="train_acc30"></b>. The memory and output weights are rewritten together, and what improves is the fine print: orderings it used to get wrong, and the near ties of <a data-ch="ties"></a>.</p>
     <p>One sibling switched strategy partway, from comparators to tracking the runner-up directly, and finished best.</p>`,
    `<p>One more surprise. All five learn much the same quantities, but they spread them across many neurons: each keeps only <b data-fact="clean_seeds"></b> of them in a single neuron, against <b data-fact="clean_example"></b> in the network on this page.</p>
     <p>So this network is readable partly by luck. The game does not force a network to be this tidy, which is part of why bigger networks are so much harder to read.</p>`,
  ],
  mount(fig) {
    const S = DATA.training.seeds, LABELS = ['0', '1M', '2M', '4M', '8M', '17M', '34M', '67M', '134M', '268M', '537M', '1.07B'];
    const PHASES = [[0, 3, 'phase 1'], [3, 5, 'phase 2'], [5, 11, 'phase 3']];
    let stage = 0, grow = 1;
    const tok = {};
    function panel(ctx, x0, y0, w, h, key, title, note, ymin) {
      const X = i => x0 + (i / 11) * w, Y = v => y0 + h - ((v - ymin) / (1 - ymin)) * h;
      V.title(ctx, title, x0, y0 - 30);
      V.text(ctx, note, x0, y0 - 11, { size: 12.5, face: 'serif', style: 'italic' });
      PHASES.forEach(([a, b, name], k) => {
        const on = stage === k + 1;
        ctx.fillStyle = on ? V.alpha(V.C.washGold, 0.22) : V.alpha(V.C.ink, k % 2 ? 0.035 : 0);
        ctx.fillRect(X(a), y0, X(b) - X(a), h);
        if (key === 'acc') V.text(ctx, name, (X(a) + X(b)) / 2, y0 + 16, { size: 12.5, align: 'center', color: on ? V.C.ink : V.C.muted, face: 'serif', style: 'italic', weight: on ? 700 : 400 });
      });
      for (const v of key === 'acc' ? [0, 0.25, 0.5, 0.75, 1] : [0, 0.5, 1]) {
        if (v < ymin) continue;
        ctx.strokeStyle = V.C.grid; ctx.lineWidth = 1; V.hline(ctx, x0, x0 + w, Y(v));
        V.text(ctx, key === 'acc' ? `${100 * v}%` : v.toFixed(1), x0 - 8, Y(v), { size: 11.5, align: 'right', base: 'middle' });
      }
      const upto = grow * 11;
      S.forEach((s, k) => {
        const ys = s[key].map(v => Math.max(ymin, v)), best = k === 2;
        ctx.strokeStyle = best && stage >= 3 ? V.C.second : V.alpha(V.C.ink, 0.7); ctx.lineWidth = best && stage >= 3 ? 2.4 : 1.5;
        ctx.beginPath(); ctx.moveTo(X(0), Y(ys[0]));
        for (let i = 1; i <= 11 && i - 1 < upto; i++) {
          const f = Math.min(1, upto - (i - 1));
          ctx.lineTo(X(i - 1) + (X(i) - X(i - 1)) * f, Y(ys[i - 1] + (ys[i] - ys[i - 1]) * f));
        }
        ctx.stroke();
      });
      if (key === 'acc') {
        ctx.setLineDash([2, 3]); ctx.strokeStyle = V.C.muted; V.hline(ctx, x0, x0 + w, Y(0.1)); ctx.setLineDash([]);
        V.text(ctx, 'guessing', x0 + w - 4, Y(0.1) - 6, { size: 12, align: 'right', face: 'serif', style: 'italic' });
        if (stage >= 3 && grow >= 1) {
          ctx.fillStyle = V.C.second; ctx.fillRect(X(11) - 236, Y(0.3) - 6, 18, 3);
          V.text(ctx, 'the sibling that switched strategy', X(11) - 4, Y(0.3), { size: 12.5, align: 'right', color: V.C.ink, base: 'middle' });
        }
      }
      return X;
    }
    const cv = V.canvas(fig, (ctx, w, h) => {
      const left = 70, right = 34, pw = w - left - right;
      if (stage === 4) {
        // single-neuron roles: this network against its five siblings
        const C = DATA.training.clean, names = ['example', 's0', 's1', 's2', 's3', 's4'], bh = Math.min(56, (h - 210) / 8), X = v => left + 90 + (v / 6) * (pw - 140);
        const top = Math.max(130, (h - 6 * bh * 1.3) / 2 + 20);
        V.title(ctx, 'Core quantities each network keeps in a single neuron', left, top - 70);
        V.text(ctx, 'a neuron counts when one scaled quantity matches it at R² above 0.9', left, top - 48, { size: 12.5, face: 'serif', style: 'italic' });
        names.forEach((m, k) => {
          const y = top + k * bh * 1.3, ex = m === 'example';
          V.text(ctx, ex ? 'this page' : `sibling ${k}`, left + 80, y + bh / 2, { size: 13, align: 'right', base: 'middle', color: V.C.ink, weight: ex ? 700 : 400 });
          for (let i = 0; i < C[m]; i++) { ctx.fillStyle = ex ? V.C.core : V.C.timer; ctx.beginPath(); ctx.arc(X(i + 0.5), y + bh / 2, Math.min(14, bh * 0.32), 0, 2 * Math.PI); ctx.fill(); }
          V.text(ctx, String(C[m]), X(C[m]) + 8, y + bh / 2, { size: 13, base: 'middle', color: V.C.ink2 });
        });
        return;
      }
      const avail = h - 92 - 64, ph = stage >= 1 ? (avail - 86) * 0.56 : avail;
      const X = panel(ctx, left, 92, pw, ph, 'acc', 'Accuracy of the five siblings as they train', 'one line per network; the shading marks the three phases', 0);
      if (stage >= 1) panel(ctx, left, 92 + ph + 86, pw, avail - 86 - ph, 'record', 'How well all 16 neurons encode the record', 'R² of reading the record from the neurons; 1 is perfect', 0);
      const yl = h - 34;
      LABELS.forEach((s, i) => { if (i % 2 === 0 || i === 11) V.text(ctx, s, X(i), yl, { size: 11.5, align: 'center' }); });
      V.text(ctx, 'training inputs seen', left + pw / 2, h - 12, { size: 12.5, align: 'center', face: 'serif', style: 'italic' });
    });
    return {
      step(k) {
        const first = stage === 0 && k === 0;
        stage = k;
        if (first) { grow = 0; V.tween(tok, 2400, u => { grow = u; cv.redraw(); }); } else cv.redraw();
      },
    };
  },
});
