// Chapter 1: the 432 weights as ARC posted them, in their three grids, one row per neuron, the groups marked last.
Story.chapter('numbers', {
  title: '432 numbers',
  hint: { text: 'Hover a square to read its number', icon: 'hover' },
  steps: [
    `<p>This is a complete neural network. Every number it uses is in this picture: <b>432</b> of them. Blue squares are positive numbers, red squares are negative, and the deeper the color, the bigger the number.</p>
     <p>Hover over any square to read it.</p>`,
    `<p>Researchers at the Alignment Research Center trained this network, published it, and asked a simple question: <b>can anyone explain what these numbers do?</b></p>
     <p>Months later they reported no real progress. One of them pointed out that finishing by the end of the year works out to four hours per number. The networks people rely on are millions of times bigger, so a network this small that still resists explanation is worth taking seriously.</p>`,
    `<p>The numbers come in three grids, and each row of all three belongs to one of the network's 16 <b>neurons</b>. Together the neurons are its memory: sixteen numbers it carries from one step to the next.</p>
     <p><b>Input weights</b>: one per neuron, saying how a new number enters it. <b>Memory weights</b>: 256 numbers saying how the neurons push each other around. <b>Output weights</b>: 160 numbers turning the neurons into an answer. There is nothing else.</p>`,
    `<p>By the end of this page you will know what every row does. Six neurons, marked <span class="core">gold</span>, do the core bookkeeping. The other ten, marked <span class="timer">slate</span>, are timers, and they cast the deciding votes.</p>
     <p>We'll call neuron 7 <b>n7</b> for short. Let's start with the job it was trained to do.</p>`,
  ],
  mount(fig) {
    const W = DATA.weights;
    const maxabs = a => Math.max(...a.flat().map(Math.abs));
    const mats = [
      { title: 'Input weights', sub: ['how a new', 'number enters'], cols: 1, get: (i) => W.Wi[i], scale: maxabs(W.Wi), name: (i) => `W<sub>in</sub>[${i}]`, what: (i) => `input → n${i}` },
      { title: 'Memory weights', sub: ['how the neurons push', 'each other, step to step'], cols: 16, get: (i, j) => W.Wh[i][j], scale: maxabs(W.Wh), name: (i, j) => `W<sub>mem</sub>[${i}, ${j}]`, what: (i, j) => `n${j} → n${i}` },
      { title: 'Output weights', sub: ['how the neurons', 'score each position'], cols: 10, get: (i, j) => W.Wo[j][i], scale: maxabs(W.Wo), name: (i, j) => `W<sub>out</sub>[${j}, ${i}]`, what: (i, j) => `n${i} → position ${j}` },
    ];
    let stage = 0, show = 0, hot = null, boxes = [];
    const tok = {};

    const cv = V.canvas(fig, (ctx, w, h) => {
      const gap = 34, left = 64, bottom = 72, head = 112;
      const s = Math.max(8, Math.min((h - head - bottom) / 16, (w - left - 24 - 2 * gap) / 27));
      const top = head + Math.max(0, (h - head - bottom - 16 * s) / 2);
      const total = 27 * s + 2 * gap, x0 = left + Math.max(0, (w - left - 24 - total) / 2);
      boxes = [];
      V.legend(ctx, x0, 28, [['positive', V.C.pos], ['negative', V.C.neg]]);
      V.text(ctx, 'deeper color, bigger number', x0 + 196, 28, { size: 12.5, base: 'middle', face: 'serif', style: 'italic' });
      let x = x0;
      mats.forEach((m, mi) => {
        const a = Math.max(0, Math.min(1, show * 3 - mi));
        for (let i = 0; i < 16; i++) for (let j = 0; j < m.cols; j++) {
          const v = m.get(i, j);
          ctx.globalAlpha = a;
          ctx.fillStyle = V.diverge(v, m.scale);
          V.rect(ctx, x + j * s + 1, top + i * s + 1, s - 2, s - 2);
          if (s >= 30) V.text(ctx, V.fmt(v, Math.abs(v) >= 10 ? 0 : 1), x + j * s + s / 2, top + i * s + s / 2 + 1, { color: Math.abs(v) / m.scale > 0.4 ? V.C.surface : V.C.ink2, size: 9, align: 'center', base: 'middle' });
          boxes.push({ x: x + j * s, y: top + i * s, s, m, i, j, v });
        }
        ctx.globalAlpha = 1;
        if (stage >= 2) {
          V.text(ctx, m.title, x + m.cols * s / 2, top - 52, { color: V.C.ink, size: 15, align: 'center', weight: 600, face: 'serif' });
          m.sub.forEach((line, k) => V.text(ctx, line, x + m.cols * s / 2, top - 34 + k * 14, { size: 12, align: 'center', face: 'serif', style: 'italic' }));
          V.text(ctx, `${16 * m.cols} numbers`, x + m.cols * s / 2, top + 16 * s + 22, { color: V.C.ink2, size: 12.5, align: 'center' });
          if (m.cols > 1 && s >= 13) for (let j = 0; j < m.cols; j++) V.text(ctx, String(j), x + j * s + s / 2, top - 4, { size: 9.5, align: 'center', color: V.C.muted });
          if (m.cols > 1) V.text(ctx, m.cols === 16 ? 'from n0 to n15' : 'position 0 to 9', x + m.cols * s / 2, top + 16 * s + 40, { size: 11.5, align: 'center' });
        }
        m.x = x; x += m.cols * s + gap;
      });
      for (let i = 0; i < 16; i++) {
        const core = V.CORE.includes(i), y = top + i * s + s / 2;
        V.text(ctx, `n${i}`, x0 - 12, y, { color: stage >= 3 && core ? V.C.ink : V.C.ink2, size: 12, align: 'right', base: 'middle', weight: stage >= 3 && core ? 700 : 400 });
        if (stage >= 3) { ctx.fillStyle = V.cellColor(i); ctx.beginPath(); ctx.arc(x0 - 46, y, 4.5, 0, 2 * Math.PI); ctx.fill(); }
      }
      if (stage >= 3) V.legend(ctx, x0 + 380, 28, [['core', V.C.core], ['timer', V.C.timer]]);
      if (hot) { ctx.strokeStyle = V.C.ink; ctx.lineWidth = 2; ctx.strokeRect(hot.x + 0.5, hot.y + 0.5, hot.s - 1, hot.s - 1); }
    });

    cv.cv.addEventListener('mousemove', e => {
      const { x: px, y: py } = cv.pt(e);
      hot = boxes.find(b => px >= b.x && px < b.x + b.s && py >= b.y && py < b.y + b.s) || null;
      V.tip(hot ? `<b>${hot.m.name(hot.i, hot.j)} = ${V.fmt(hot.v, 3)}</b><br>${hot.m.what(hot.i, hot.j)}` : null, e.clientX, e.clientY);
      cv.redraw();
    });
    cv.cv.addEventListener('mouseleave', () => { hot = null; V.tip(null); cv.redraw(); });

    return {
      step(k) {
        stage = k;
        if (show < 1) V.tween(tok, 1400, t => { show = t; cv.redraw(); }); else cv.redraw();
      },
    };
  },
});
