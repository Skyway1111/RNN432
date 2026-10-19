// Chapter 2: the task. Ten numbers drop in; ranks light up; the network answers; drag bars to try to fool it.
Story.chapter('game', {
  title: 'The game',
  hint: { text: 'Drag a bar up or down', icon: 'drag', from: 2 },
  steps: [
    `<p>Here is the whole job. Ten numbers arrive, one at a time, at positions 0 to 9. They are drawn at random from a bell curve centered on zero, so most land between −2 and 2, and some are negative.</p>
     <p>When the last one lands, the network must point to <b>the position of the second-largest number</b>. We'll call that number the <b>runner-up</b>.</p>`,
    `<p>The <span class="max">largest</span> is orange, the <span class="second">runner-up</span> green, and the <span class="third">third-largest</span> violet. The right answer is the green bar's position.</p>
     <p>For you this is easy: you can see all ten at once. The network can't. It reads each number once, in order, and never looks back, so it has to <b>remember</b> what it has seen.</p>`,
    `<p>The marker shows the network's answer. It is right on <b data-fact="acc"></b> of random inputs.</p>
     <p><b>Drag any bar</b> to change a number, or deal new ones, and watch the answer follow.</p>`,
    `<p>Now try to make it fail. A hint: make the runner-up and the third-largest <b>almost equal</b>.</p>
     <p>Hold on to whatever you find. By <a data-ch="ties"></a> you'll know exactly why it happens.</p>`,
  ],
  mount(fig, bar) {
    let x = Float64Array.from([0.31, -1.2, 1.54, 0.1, -0.4, 1.05, -0.9, 0.66, -0.1, 0.4]);
    let stage = 0, landed = 10, drag = -1, hover = -1;
    const tally = { right: 0, n: 0 }, tok = {}, g = M.rng(7);
    const read = V.readout(bar);

    const geo = (w, h) => {
      const left = 56, right = 30, slot = (w - left - right) / 10;
      return { left, right, slot, mid: h * 0.54, unit: Math.min(h * 0.13, 110) };
    };
    const cv = V.canvas(fig, (ctx, w, h) => {
      const { left, right, slot, mid, unit } = geo(w, h);
      const o = M.order(x), rank = new Array(10); o.forEach((p, r) => (rank[p] = r));
      ctx.strokeStyle = V.C.axis; ctx.lineWidth = 1;
      V.hline(ctx, left - 10, w - right + 10, mid);
      for (const v of [-2, -1, 1, 2]) V.text(ctx, String(v).replace('-', '−'), left - 18, mid - v * unit, { size: 11.5, align: 'right', base: 'middle' });
      if (stage >= 1) V.legend(ctx, left, 30, [['largest', V.C.max], ['runner-up', V.C.second], ['third-largest', V.C.third]]);
      const r = M.run(M.base, x), ans = M.argmax(r.logits), q = o[1];
      for (let t = 0; t < 10; t++) {
        const cx = left + slot * (t + 0.5), bw = Math.min(46, slot * 0.6);
        const fall = Math.max(0, Math.min(1, landed - t));
        V.text(ctx, String(t), cx, h - 40, { size: 13, align: 'center', color: V.C.ink2, weight: 600 });
        if (fall <= 0) continue;
        const v = x[t] * V.ease(fall), y = mid - v * unit;
        ctx.globalAlpha = fall;
        const col = stage >= 1 ? V.rankColor(rank[t]) : V.C.ink2;
        ctx.fillStyle = (t === hover || t === drag) && stage >= 2 ? V.mix(col, '#ffffff', 0.2) : col;
        V.bar(ctx, cx - bw / 2, mid, bw, v * unit, 4);
        ctx.globalAlpha = 1;
        V.text(ctx, V.fmt(x[t]), cx, v >= 0 ? y - 16 : y + 24, { color: V.C.ink, size: 13, align: 'center', weight: 600 });
        if (stage >= 2 && fall >= 1) V.handle(ctx, cx, y, t === hover || t === drag ? V.C.ink : V.alpha(V.C.ink, 0.65));
      }
      V.text(ctx, 'position: the order the numbers arrived in', left + slot * 5, h - 16, { size: 12.5, align: 'center', face: 'serif', style: 'italic' });
      if (stage >= 2 && landed >= 10) {
        const ax = left + slot * (ans + 0.5), good = ans === q;
        ctx.fillStyle = good ? V.C.right : V.C.wrong;
        ctx.beginPath(); ctx.moveTo(ax, h - 56); ctx.lineTo(ax - 8, h - 70); ctx.lineTo(ax + 8, h - 70); ctx.fill();
        V.chip(ctx, `Network says position ${ans} ${good ? '✓' : '✗'}`, ax, h - 88, { color: good ? V.C.right : V.C.wrong, align: 'center' });
      }
      if (stage >= 2) read.textContent = tally.n ? `You dealt ${tally.n}; the network was right on ${tally.right}.` : '';
    });

    function deal() {
      x = Float64Array.from({ length: 10 }, g);
      tally.n++; if (M.argmax(M.run(M.base, x).logits) === M.label(x)) tally.right++;
      drop();
    }
    function drop() { V.tween(tok, 1600, t => { landed = t * 10.5; cv.redraw(); }); }
    const at = e => { const p = cv.pt(e), { left, slot, mid, unit } = geo(cv.w, cv.h); return { t: Math.floor((p.x - left) / slot), v: Math.max(-3, Math.min(3, (mid - p.y) / unit)) }; };
    cv.cv.addEventListener('pointerdown', e => { const p = at(e); if (p.t >= 0 && p.t < 10 && stage >= 2) { drag = p.t; x[drag] = p.v; cv.cv.setPointerCapture(e.pointerId); cv.redraw(); } });
    cv.cv.addEventListener('pointermove', e => {
      if (drag >= 0) { x[drag] = at(e).v; cv.redraw(); return; }
      const t = at(e).t, nh = stage >= 2 && t >= 0 && t < 10 ? t : -1;
      if (nh !== hover) { hover = nh; cv.cv.style.cursor = hover >= 0 ? 'ns-resize' : 'default'; cv.redraw(); }
    });
    cv.cv.addEventListener('pointerup', () => { drag = -1; cv.redraw(); });
    cv.cv.addEventListener('pointerleave', () => { if (hover >= 0) { hover = -1; cv.redraw(); } });
    V.button(bar, 'Deal new numbers', deal);
    V.button(bar, 'Replay the deal', drop);

    return {
      step(k) {
        const first = stage === 0 && k === 0;
        stage = k;
        if (first) { landed = 0; drop(); } else cv.redraw();
      },
    };
  },
});
