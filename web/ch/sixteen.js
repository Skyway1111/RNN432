// Chapter 8: all sixteen neurons. A map of the costliest links (accuracy lost when each is cut alone, from
// results/experiments/neuron_atlas.npy), who points to the runner-up (each neuron's vote for the runner-up's own
// position, computed from the weights), what silencing each costs (computed live), and below the steps a field guide:
// one plate per neuron with its words (web/neurons.js), its costliest links in and out, what it holds after a lone
// input at each position, and its accuracy when silenced. The facts: research/findings/neurons.md.
(function () {
  const W = M.base, N = 16, TIMERS = [...Array(N).keys()].filter(n => !V.CORE.includes(n));
  const memo = {}, once = (k, f) => (k in memo ? memo[k] : (memo[k] = f()));
  const base = () => once('base', () => M.accuracy(W));
  const lift = () => once('lift', () => M.runnerUpParts(W).map((row, n) => row.map((v, q) => W.Wo[q * N + n] * v)));
  const impulse = () => once('imp', () => Array.from({ length: 10 }, (_, p) => { const x = new Array(10).fill(0); x[p] = 1; return M.run(W, x).h[10]; }));
  // cost[i][s]: points of accuracy lost when the link from source s (0 the input, 1 + j neuron j) into neuron i is cut
  const cost = DATA.links.cut.map(row => row.map(a => 100 * (DATA.links.base - a)));
  const weightOf = (i, s) => (s === 0 ? W.Wi[i] : W.Wh[i * N + s - 1]);
  const reads = n => cost[n].map((c, s) => ({ s, c, w: weightOf(n, s) })).sort((a, b) => b.c - a.c).slice(0, 3);
  const sends = n => [...Array(N).keys()].filter(i => i !== n).map(i => ({ i, c: cost[i][n + 1], w: W.Wh[i * N + n] })).sort((a, b) => b.c - a.c).slice(0, 3);
  const name = s => (s === 0 ? 'input' : 'n' + (s - 1));
  // the positions a neuron points the runner-up to: its largest vote, and a second within 40% of it
  function pointsTo(n) {
    const l = lift()[n], q = l.map((v, k) => [v, k]).sort((a, b) => b[0] - a[0]);
    if (q[0][0] < 15) return [];
    return q[1][0] >= 0.4 * q[0][0] ? [q[0][1], q[1][1]].sort((a, b) => a - b) : [q[0][1]];
  }

  // Silencing: a neuron with every incoming weight zeroed never fires. One neuron a frame, then everyone redraws.
  const silenced = new Array(N).fill(null), listeners = [];
  function silence() {
    let n = 0;
    const next = () => {
      if (n >= N) return;
      const S = M.clone(W); S.Wi[n] = 0; for (let j = 0; j < N; j++) S.Wh[n * N + j] = 0;
      silenced[n++] = M.accuracy(S);
      listeners.forEach(f => f());
      requestAnimationFrame(next);
    };
    requestAnimationFrame(next);
  }

  // the map: core around the record keeper, timers around the dial at the runner-up position each points to (V.PLACE)
  function map(ctx, w, h, hot) {
    const cx = w * 0.5, cy = h * 0.54, R = Math.min(w * 0.29, h * 0.29), pos = V.place(cx, cy, R);
    V.title(ctx, 'The sixteen neurons and their costliest links', 26, 38);
    V.text(ctx, 'thicker: more accuracy lost when the link alone is cut', 26, 60, { size: 13, face: 'serif', style: 'italic' });
    ctx.strokeStyle = V.alpha(V.C.ink, 0.18); ctx.lineWidth = 0.8; ctx.beginPath(); ctx.arc(cx, cy, 1.12 * R, 0, 2 * Math.PI); ctx.stroke();
    for (let q = 0; q < 10; q++) {
      const a = (-90 + q * 36) * Math.PI / 180;
      V.text(ctx, String(q), cx + 1.45 * R * Math.cos(a), cy + 1.45 * R * Math.sin(a), { size: 14, align: 'center', base: 'middle', color: V.C.muted, face: 'serif', style: 'italic' });
    }
    V.text(ctx, 'around the dial: the runner-up position each timer points to', 26, h - 18, { size: 12.5, face: 'serif', style: 'italic' });
    const links = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (i !== j) links.push({ i, j, c: cost[i][j + 1], w: W.Wh[i * N + j] });
    links.sort((a, b) => b.c - a.c).splice(30);
    for (const l of links) {
      const a = pos[l.j], b = pos[l.i], on = hot === null || hot === l.i || hot === l.j;
      ctx.strokeStyle = l.w >= 0 ? V.C.pos : V.C.neg; ctx.globalAlpha = on ? 0.3 + 0.6 * (l.c / 85) : 0.07;
      ctx.lineWidth = 0.6 + 3.2 * (l.c / 85) ** 1.5;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.quadraticCurveTo((a.x + b.x) / 2 + (b.y - a.y) * 0.12, (a.y + b.y) / 2 - (b.x - a.x) * 0.12, b.x, b.y); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    const mean = once('mean', () => M.meanState(W));
    for (let n = 0; n < N; n++) {
      const p = pos[n], r = 7 + 2.1 * Math.sqrt(mean[n]), on = hot === null || hot === n;
      ctx.globalAlpha = on ? 1 : 0.35;
      ctx.fillStyle = V.cellColor(n); ctx.strokeStyle = V.C.surface; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      V.text(ctx, 'n' + n, p.x + r + 4, p.y + r * 0.6 + 6, { size: 13, color: V.C.ink, face: 'serif', style: 'italic', weight: hot === n ? 700 : 400 });
      p.r = r;
    }
    ctx.globalAlpha = 1;
    V.legend(ctx, w - 300, 38, [['core', V.C.core], ['timer', V.C.timer]]);
    V.legend(ctx, w - 300, 60, [['pushes up', V.C.pos], ['pushes down', V.C.neg]]);
    return pos;
  }

  // who points to the runner-up: rows are neurons (timers in the order of the positions they cover, then the core)
  function grid(ctx, w, h, sums) {
    const L = lift(), order = [...TIMERS].sort((a, b) => (pointsTo(a)[0] ?? 99) - (pointsTo(b)[0] ?? 99) || b - a)
      .concat([7, 2, 4, 6, 1, 8]);
    const left = 92, top = 116, extra = sums ? 3.6 : 0, s = Math.min((w - left - 40) / 10, (h - top - 70) / (16 + extra));
    V.title(ctx, 'Each neuron\'s vote for the runner-up\'s own position', 26, 38);
    V.text(ctx, 'per unit of the runner-up\'s value, by where the runner-up lands; 3,000 random inputs', 26, 60, { size: 13, face: 'serif', style: 'italic' });
    for (let q = 0; q < 10; q++) V.text(ctx, String(q), left + q * s + s / 2, top - 10, { size: 12.5, align: 'center', color: V.C.ink, weight: 600 });
    V.text(ctx, 'runner-up at position', left, top - 30, { size: 12, color: V.C.ink2 });
    const cell = (v, x, y, sc) => {
      ctx.fillStyle = V.diverge(v, sc); V.rect(ctx, x + 1, y + 1, s - 2, s - 2);
      if (Math.abs(v) >= 15) V.text(ctx, V.sfmt(v, 0), x + s / 2, y + s / 2 + 1, { size: Math.min(12, s * 0.36), align: 'center', base: 'middle', color: Math.abs(v) > 0.45 * sc ? V.C.surface : V.C.ink, weight: 600 });
    };
    order.forEach((n, row) => {
      const y = top + row * s + (row >= TIMERS.length ? 8 : 0);
      ctx.fillStyle = V.cellColor(n); ctx.beginPath(); ctx.arc(left - 58, y + s / 2, 4.5, 0, 2 * Math.PI); ctx.fill();
      V.text(ctx, 'n' + n, left - 10, y + s / 2, { size: 12.5, align: 'right', base: 'middle', color: V.C.ink });
      for (let q = 0; q < 10; q++) cell(L[n][q], left + q * s, y, 50);
    });
    if (sums) {
      const y0 = top + 16 * s + 22, rows = [['ten timers', TIMERS], ['six core', V.CORE], ['all sixteen', [...Array(N).keys()]]];
      rows.forEach(([label, set], k) => {
        const y = y0 + k * s * 1.1;
        V.text(ctx, label, left - 10, y + s / 2, { size: 12.5, align: 'right', base: 'middle', color: V.C.ink, weight: 600 });
        for (let q = 0; q < 10; q++) cell(set.reduce((a, n) => a + L[n][q], 0), left + q * s, y, 60);
      });
    }
    V.legend(ctx, left + 10 * s + 18, top + 8, [['for', V.C.pos]]);
    V.legend(ctx, left + 10 * s + 18, top + 30, [['against', V.C.neg]]);
  }

  // what is left when each neuron is silenced
  function bars(ctx, w, h) {
    const left = 268, top = 96, right = 70, bh = Math.min(30, (h - top - 70) / N), X = a => left + a * (w - left - right);
    const done = silenced.filter(a => a !== null).length, order = [...Array(N).keys()].sort((a, b) => (silenced[a] ?? 2) - (silenced[b] ?? 2));
    V.title(ctx, 'Accuracy left when one neuron is silenced', 26, 38);
    V.text(ctx, done < N ? `testing on 20,000 random inputs: ${done} of 16` : 'each tested on 20,000 random inputs', 26, 60, { size: 13, face: 'serif', style: 'italic' });
    for (const a of [0, 0.25, 0.5, 0.75, 1]) {
      ctx.strokeStyle = V.C.grid; ctx.lineWidth = 1; V.vline(ctx, X(a), top - 8, top + N * bh);
      V.text(ctx, `${100 * a}%`, X(a), top + N * bh + 18, { size: 11.5, align: 'center' });
    }
    order.forEach((n, row) => {
      const y = top + row * bh, a = silenced[n];
      ctx.fillStyle = V.cellColor(n); ctx.beginPath(); ctx.arc(24, y + bh / 2, 4.5, 0, 2 * Math.PI); ctx.fill();
      V.text(ctx, `n${n}  ${NEURONS[n].name}`, 36, y + bh / 2, { size: 12, base: 'middle', color: V.C.ink });
      if (a === null) return;
      ctx.fillStyle = V.cellColor(n); V.roundRect(ctx, X(0), y + bh * 0.18, Math.max(1, X(a) - X(0)), bh * 0.64, [0, 3, 3, 0]); ctx.fill();
      V.text(ctx, V.pct(a), X(a) + 6, y + bh / 2, { size: 12, base: 'middle', color: V.C.ink });
    });
    const full = base(), chance = 0.1;
    ctx.setLineDash([5, 4]); ctx.strokeStyle = V.C.ink; ctx.lineWidth = 1.5; V.vline(ctx, X(full), top - 14, top + N * bh);
    ctx.setLineDash([2, 3]); ctx.strokeStyle = V.C.muted; V.vline(ctx, X(chance), top - 14, top + N * bh); ctx.setLineDash([]);
    V.text(ctx, `the full network, ${V.pct(full)}`, X(full), top - 20, { size: 12, align: 'right', color: V.C.ink, weight: 600 });
    V.text(ctx, 'guessing, 10%', X(chance) + 4, top - 20, { size: 12, color: V.C.muted });
  }

  // one plate: the neuron's disc, words, costliest links in and out, impulse strip, and silenced accuracy
  function plate(n, el) {
    const core = V.CORE.includes(n), art = document.createElement('article');
    art.className = 'plate'; art.id = 'plate-' + n;
    art.innerHTML = `<header><div class="disc"></div><div><div class="nid">n${n}</div><h4>${NEURONS[n].name}</h4></div></header>
      <p>${NEURONS[n].text}</p><div class="wires"></div>
      <footer><span class="sil">Silenced: testing…</span></footer>`;
    el.appendChild(art);
    V.canvas(art.querySelector('.disc'), (ctx, w, h) => {
      const wash = V.wash(w / 2 - 2, h / 2 + 2, w * 0.4, h * 0.38, core ? V.C.washGold : V.C.washSlate, 300 + n, { layers: 26, alpha: 0.07, rough: 0.16, per: 2, spots: 3 });
      wash.grow(V.K); wash.draw(ctx);
      ctx.strokeStyle = V.alpha(V.C.ink, 0.85); ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(w / 2, h / 2, w * 0.34, 0, 2 * Math.PI); ctx.stroke();
      V.text(ctx, String(n), w / 2, h / 2 + 1, { size: 16, align: 'center', base: 'middle', color: V.C.ink, weight: 600, face: 'serif' });
    });
    V.canvas(art.querySelector('.wires'), (ctx, w, h) => {
      const cy = 68, cx = w / 2, rows = [cy - 32, cy, cy + 32], inR = reads(n), outR = sends(n);
      V.text(ctx, 'costliest links in', 0, 11, { size: 12, face: 'serif', style: 'italic' });
      V.text(ctx, 'and out', w, 11, { size: 12, align: 'right', face: 'serif', style: 'italic' });
      const wire = (x0, y0, x1, y1, l) => {
        ctx.strokeStyle = l.w >= 0 ? V.C.pos : V.C.neg; ctx.globalAlpha = 0.35 + 0.6 * Math.min(1, l.c / 85); ctx.lineWidth = 0.7 + 3 * Math.min(1, l.c / 85) ** 1.4;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.bezierCurveTo((x0 + x1) / 2, y0, (x0 + x1) / 2, y1, x1, y1); ctx.stroke(); ctx.globalAlpha = 1;
      };
      inR.forEach((l, k) => {
        wire(70, rows[k], cx - 14, cy, l);
        const lab = l.s === n + 1 ? 'itself' : name(l.s);
        V.text(ctx, lab, 0, rows[k] - 2, { size: 12.5, color: V.C.ink, weight: 600 });
        V.text(ctx, V.sfmt(l.w, 2), V.width(ctx, lab, 12.5, 600) + 5, rows[k] - 2, { size: 11.5, color: V.C.ink2 });
        V.text(ctx, `cut: −${l.c.toFixed(0)} pts`, 0, rows[k] + 12, { size: 10.5, color: V.C.muted });
      });
      outR.forEach((l, k) => {
        wire(cx + 14, cy, w - 70, rows[k], l);
        V.text(ctx, 'n' + l.i, w, rows[k] - 2, { size: 12.5, color: V.C.ink, weight: 600, align: 'right' });
        V.text(ctx, V.sfmt(l.w, 2), w - V.width(ctx, 'n' + l.i, 12.5, 600) - 5, rows[k] - 2, { size: 11.5, color: V.C.ink2, align: 'right' });
        V.text(ctx, `cut: −${l.c.toFixed(0)} pts`, w, rows[k] + 12, { size: 10.5, color: V.C.muted, align: 'right' });
      });
      ctx.fillStyle = V.cellColor(n); ctx.beginPath(); ctx.arc(cx, cy, 11, 0, 2 * Math.PI); ctx.fill();
      V.text(ctx, String(n), cx, cy + 1, { size: 11, align: 'center', base: 'middle', color: V.C.surface, weight: 700 });
      // two strips over the ten positions: what it holds after a lone 1 there, and its vote for a runner-up landing there
      const I = impulse().map(hv => hv[n]), mx = Math.max(...I), Lq = lift()[n], cw = w / 10, y1 = 140, y2 = 180;
      V.text(ctx, 'what it holds after a lone 1 at each position', 0, y1 - 6, { size: 12, face: 'serif', style: 'italic' });
      V.text(ctx, 'its vote for the runner-up, wherever it lands', 0, y2 - 6, { size: 12, face: 'serif', style: 'italic' });
      for (let q = 0; q < 10; q++) {
        ctx.fillStyle = mx > 0 ? V.alpha(V.cellColor(n), 0.1 + 0.9 * Math.max(0, I[q]) / mx) : V.C.surface2;
        V.rect(ctx, q * cw + 1, y1, cw - 2, 14);
        ctx.fillStyle = V.diverge(Lq[q], 50); V.rect(ctx, q * cw + 1, y2, cw - 2, 14);
        V.text(ctx, String(q), q * cw + cw / 2, y2 + 28, { size: 10.5, align: 'center' });
      }
    });
    const sil = art.querySelector('.sil');
    listeners.push(() => { if (silenced[n] !== null) sil.innerHTML = `Silenced, accuracy falls to <b>${V.pct(silenced[n])}</b>.`; });
  }

  Story.chapter('sixteen', {
    title: 'Meet the sixteen',
    hint: { text: 'Hover a neuron; click to open its plate', icon: 'hover' },
    steps: [
      `<p>You have met six neurons up close. Here are all sixteen at once, joined by the links that matter most: the thicker the line, the more accuracy the network loses when that one link is cut.</p>
       <p>The <span class="core">gold</span> six keep the books. The <span class="timer">slate</span> ten are timers: each holds more or less of the record depending on when it arrived. They sit around a ring of answer positions, each at the position it watches. Hover a neuron for its name; click it to open its plate in the field guide below.</p>`,
      `<p>So who points to the runner-up? Seven of the timers each watch one or two positions. Five of them use a <b>veto</b>: they vote against their positions, and a runner-up arriving there switches them down, lifting the veto. Lifting a veto raises that position's score, so the grid counts it as a vote for. Two more, n15 and n5, add a vote outright.</p>
       <p>The grid shows each neuron's vote for the runner-up's own position, wherever the runner-up lands. Read it as a staircase: n14 covers position 0, n15 positions 1 and 2, and so on down to n5 at position 8. Position 9 falls to a comparator, n7.</p>`,
      `<p>Now add up each kind. Wherever the runner-up lands, up to position 8, the ten timers push its position up, and at most of those positions the six core neurons pull it down.</p>
       <p>The core's votes still matter: cutting the vote of any core neuron but n4 costs most of the accuracy. But the vote that points at the runner-up comes from the timers.</p>`,
      `<p>And every one of the sixteen matters. Silence a neuron, so it never fires, and accuracy collapses; the bars show what is left.</p>
       <p>Even the faintest, n9, whose own vote changes nothing, costs a large share through what it feeds the others. There are no spare parts in 432 numbers. The field guide below has a plate for each neuron.</p>`,
    ],
    mount(fig) {
      let stage = 0, hot = null, pos = null;
      const cv = V.canvas(fig, (ctx, w, h) => {
        if (stage === 0) pos = map(ctx, w, h, hot);
        else if (stage <= 2) grid(ctx, w, h, stage === 2);
        else bars(ctx, w, h);
      });
      listeners.push(() => { if (stage === 3) cv.redraw(); });
      const near = e => { if (!pos) return null; const p = cv.pt(e); for (let n = 0; n < N; n++) if (Math.hypot(p.x - pos[n].x, p.y - pos[n].y) <= pos[n].r + 4) return n; return null; };
      cv.cv.addEventListener('mousemove', e => {
        if (stage !== 0) return V.tip(null);
        const n = near(e);
        if (n !== hot) { hot = n; cv.cv.style.cursor = n === null ? 'default' : 'pointer'; cv.redraw(); }
        V.tip(n === null ? null : `<b>n${n}</b> ${NEURONS[n].name}`, e.clientX, e.clientY);
      });
      cv.cv.addEventListener('mouseleave', () => { hot = null; V.tip(null); cv.redraw(); });
      cv.cv.addEventListener('click', e => { const n = near(e); if (n !== null && stage === 0) document.getElementById('plate-' + n)?.scrollIntoView({ behavior: V.reduced ? 'auto' : 'smooth', block: 'center' }); });
      silence();
      return { step(k) { stage = k; hot = null; cv.redraw(); } };
    },
    after(el) {
      el.innerHTML = `<div class="guide"><h3>A field guide to the sixteen</h3>
        <p>One plate per neuron: what it does in a sentence or two; the links into and out of it that cost the most accuracy when cut, with their weights; what it holds at the end after a lone 1 at each position; and what is left when it is silenced.</p>
        <h4 class="group"><span class="core">The core six</span></h4><div class="plates" data-g="core"></div>
        <h4 class="group"><span class="timer">The ten timers</span></h4><div class="plates" data-g="timers"></div></div>`;
      const [coreEl, timerEl] = el.querySelectorAll('.plates');
      [2, 4, 7, 6, 1, 8].forEach(n => plate(n, coreEl));
      [...TIMERS].sort((a, b) => (pointsTo(a)[0] ?? 99) - (pointsTo(b)[0] ?? 99) || b - a).forEach(n => plate(n, timerEl));
      listeners.forEach(f => f());
    },
  });
})();
