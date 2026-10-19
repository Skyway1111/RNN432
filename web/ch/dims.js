// Chapter 10: higher dimensions. Three inputs vary over every direction (a ball), seven stay fixed: ties are great
// circles and the errors hug them. Then the memory's paths through 16 dimensions, projected on their top 3 directions.
Story.chapter('dims', {
  title: 'Seeing in ten dimensions',
  hint: { text: 'Drag to spin the ball', icon: 'spin' },
  steps: [
    `<p>An input is ten numbers, so you can think of it as a point in ten-dimensional space. Nobody can picture that, but we can take a slice.</p>
     <p>Let three of the inputs, x<sub>2</sub>, x<sub>5</sub> and x<sub>7</sub>, move around a ball while the other seven stay fixed and lower, so the moving three are always the top three. Each dot on the ball is one input, one way of ordering and spacing the three. Drag to spin it.</p>`,
    `<p>The white circles are <b>ties</b>: inputs where two of the three moving numbers are equal. They cut the ball into six pieces, one for each way of ordering the three. Inside a piece the right answer never changes; crossing a circle can change it.</p>`,
    `<p>Now the red dots: inputs the network gets wrong. Every one of them hugs a circle. This is the sliver from <a data-ch="ties"></a>, seen in 3-D.</p>
     <p>The full input space has 45 tie planes, one for every pair of positions, and the errors hug all of them. Everywhere else, the network is right.</p>`,
    `<p>One more shadow. The network's memory is a point in 16-dimensional space, and each input moves it along a path, one step per number. Here are 80 paths, flattened onto the three directions the <b>output weights</b> read most.</p>
     <p>The answer is decided by which of ten wedge-shaped regions the memory ends in, one per answer. Labels mark each answer's average direction. In 16 dimensions the wedges are cleanly apart. Squeezed into any 3-D shadow they overlap, and the counter shows how badly. No picture of this network shows the whole decision, which is part of why it is hard to explain.</p>`,
  ],
  mount(fig, bar) {
    // the moving three are MU + RHO u, always above the fixed seven, so the top three never change hands with them
    const FREE = [2, 5, 7], FIX = { 0: -0.6, 1: 0.25, 3: -1.1, 4: 0.1, 6: -0.4, 8: 0.3, 9: -0.8 }, MU = 1.2, RHO = 0.8, NP = 30000;
    let stage = 0, yaw = 0.6, pitch = 0.35, spinning = !V.reduced, drag = null, visible = false;
    // the ball: Fibonacci directions, each run through the network once
    const pts = [];
    const x = new Array(10).fill(0); for (const k in FIX) x[k] = FIX[k];
    const ga = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < NP; i++) {
      const y = 1 - 2 * (i + 0.5) / NP, rr = Math.sqrt(1 - y * y), th = ga * i, u = [rr * Math.cos(th), y, rr * Math.sin(th)];
      FREE.forEach((p, k) => (x[p] = MU + RHO * u[k]));
      pts.push({ u, wrong: M.argmax(M.run(M.base, x).logits) !== M.label(x) });
    }
    const circles = [[0, 1], [0, 2], [1, 2]].map(([i, j]) => {
      const n = [0, 0, 0]; n[i] = Math.SQRT1_2; n[j] = -Math.SQRT1_2;
      const a = [0, 0, 0]; a[i] = Math.SQRT1_2; a[j] = Math.SQRT1_2;
      const b = [n[1] * a[2] - n[2] * a[1], n[2] * a[0] - n[0] * a[2], n[0] * a[1] - n[1] * a[0]];
      return Array.from({ length: 241 }, (_, s) => { const t = 2 * Math.PI * s / 240; return a.map((v, d) => Math.cos(t) * v + Math.sin(t) * b[d]); });
    });
    // the paths: memories after each step, projected on the readout's top three directions (eigenvectors of Wo'Wo);
    // each answer's average end direction, and the share of 2,000 ends nearest their own answer's direction
    let paths = null;
    function buildPaths() {
      const A = Array.from({ length: 16 }, (_, i) => Array.from({ length: 16 }, (_, j) => { let s = 0; for (let k = 0; k < 10; k++) s += M.base.Wo[k * 16 + i] * M.base.Wo[k * 16 + j]; return s; }));
      const E = M.eig(A, 3), proj = hv => E.map(e => e.reduce((s, v, i) => s + v * hv[i], 0)), unit = v => { const n = Math.hypot(...v) || 1; return v.map(c => c / n); };
      const X = M.bank(2000, 23), cen = Array.from({ length: 10 }, () => [0, 0, 0]), ends = [];
      for (let s = 0; s < 2000; s++) { const xi = X.subarray(s * 10, s * 10 + 10), e = unit(proj(M.run(M.base, xi).h[10])), q = M.label(xi); ends.push([e, q]); e.forEach((c, d) => (cen[q][d] += c)); }
      const dirs = cen.map(unit);
      const own = ends.filter(([e, q]) => { let b = 0, bd = -2; dirs.forEach((d, k) => { const c = d[0] * e[0] + d[1] * e[1] + d[2] * e[2]; if (c > bd) { bd = c; b = k; } }); return b === q; }).length / ends.length;
      const out = [];
      for (let s = 0; s < 80; s++) { const xi = X.subarray(s * 10, s * 10 + 10), r = M.run(M.base, xi); out.push({ lab: M.label(xi), pts: r.h.map(proj) }); }
      const scale = Math.max(...out.flatMap(p => p.pts.flat().map(Math.abs)));
      out.forEach(p => (p.pts = p.pts.map(v => v.map(c => c / scale))));
      return { out, dirs, own };
    }

    const cv = V.canvas(fig, (ctx, w, h) => {
      const cx = w / 2, cy = h / 2 + 6, R = Math.min(w, h) * 0.37;
      const P = v => { const q = V.project(v, yaw, pitch); return [cx + q.x * R, cy - q.y * R, q.z]; };
      if (stage < 3) {
        ctx.strokeStyle = V.C.axis; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
        // back to front, errors last so they sit on top
        for (const pass of [0, 1, 2, 3]) for (const p of pts) {
          const err = stage >= 2 && p.wrong;
          if (err !== pass >= 2) continue;
          const [px, py, z] = P(p.u), front = z >= 0;
          if (front !== (pass % 2 === 1)) continue;
          ctx.fillStyle = err ? (front ? V.C.wrong : V.alpha(V.C.wrong, 0.3)) : (front ? V.alpha(V.C.ink2, 0.16 + 0.24 * z) : V.alpha(V.C.ink2, 0.06));
          const s = err ? (front ? 3.4 : 2.2) : 1.8; ctx.fillRect(px - s / 2, py - s / 2, s, s);
        }
        if (stage >= 1) for (const c of circles) {
          for (let s = 0; s < c.length - 1; s++) {
            const [x0, y0, z0] = P(c[s]), [x1, y1] = P(c[s + 1]);
            ctx.strokeStyle = z0 >= 0 ? V.alpha(V.C.ink, 0.85) : V.alpha(V.C.ink, 0.18); ctx.lineWidth = z0 >= 0 ? 1.5 : 1;
            ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
          }
        }
        const axes = FREE.map((p, k) => { const v = [0, 0, 0]; v[k] = 1.18; return [p, v]; });
        for (const [p, v] of axes) { const [px, py] = P(v); V.text(ctx, `${V.xi(p)} up`, px, py, { size: 13, align: 'center', base: 'middle', color: V.C.ink, face: 'serif', style: 'italic' }); }
        const nw = pts.filter(p => p.wrong).length;
        V.text(ctx, stage >= 2 ? `${nw} of ${NP.toLocaleString()} inputs wrong (${(100 * nw / NP).toFixed(1)}%)` : `${NP.toLocaleString()} inputs, one dot each`, 16, h - 14, { size: 12 });
        if (stage >= 1) V.legend(ctx, 16, 24, stage >= 2 ? [['tie between two of the three', V.C.ink], ['the network is wrong', V.C.wrong]] : [['tie between two of the three', V.C.ink]]);
      } else {
        paths = paths || buildPaths();
        for (const p of paths.out) {
          ctx.strokeStyle = V.alpha(V.C.ink2, 0.25); ctx.lineWidth = 1; ctx.beginPath();
          p.pts.forEach((v, t) => { const [px, py] = P(v); t ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }); ctx.stroke();
          const [ex, ey] = P(p.pts[10]); ctx.fillStyle = V.mix('#c4d3ea', V.C.pos, p.lab / 9); ctx.beginPath(); ctx.arc(ex, ey, 4, 0, 2 * Math.PI); ctx.fill();
        }
        paths.dirs.forEach((d, lab) => {
          const [lx, ly, z] = P(d.map(c => c * 1.1)), [ox, oy] = P([0, 0, 0]);
          ctx.strokeStyle = V.alpha(V.C.ink, z >= 0 ? 0.6 : 0.2); ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(lx, ly); ctx.stroke(); ctx.setLineDash([]);
          V.text(ctx, `Position ${lab}`, lx, ly, { size: 12, align: 'center', base: 'middle', color: z >= 0 ? V.C.ink : V.C.muted, weight: 600 });
        });
        V.text(ctx, `In this 3-D shadow, only ${(100 * paths.own).toFixed(0)}% of 2,000 memories end nearest their own answer's direction.`, 16, h - 14, { size: 12, color: V.C.ink2 });
        V.text(ctx, 'Dots: where each path ends, lighter for low positions, darker for high', 16, 24, { size: 12 });
      }
    });
    new IntersectionObserver(es => { visible = es[0].isIntersecting; }).observe(fig);
    const loop = () => { if (spinning && visible && !drag) { yaw += 0.003; cv.redraw(); } requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    cv.cv.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, yaw, pitch }; cv.cv.setPointerCapture(e.pointerId); cv.cv.style.cursor = 'grabbing'; });
    cv.cv.addEventListener('pointermove', e => { if (!drag) return; yaw = drag.yaw + (e.clientX - drag.x) * 0.008; pitch = Math.max(-1.4, Math.min(1.4, drag.pitch + (e.clientY - drag.y) * 0.008)); cv.redraw(); });
    cv.cv.addEventListener('pointerup', () => { drag = null; cv.cv.style.cursor = 'grab'; });
    cv.cv.style.cursor = 'grab';
    V.button(bar, 'Pause spin', function () { spinning = !spinning; this.textContent = spinning ? 'Pause spin' : 'Spin'; });
    V.note(bar, 'or drag to rotate');
    return { step(k) { stage = k; cv.redraw(); } };
  },
});
