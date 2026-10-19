// The three parts' paintings, each from the weights, in the part's pigment. Overview: the 432 numbers as blots of
// ultramarine and vermilion, in their three grids. How it works: the clock's dial, its ten ticks at the readout's
// directions, the hand's path on a clean input and the lamp lit at the runner-up's tick. Where it fails: the ball of
// inputs from the dimensions chapter, its tie circles, and the errors hugging them. A part registers one as
// paint(host); story.js draws it once, when the part nears the screen.
(function () {
  const W = M.base;

  // one pigment blot per weight, deeper for a bigger number, on the same sqrt scale as the chapter's grid
  function overview(host) {
    const mats = [[1, i => W.Wi[i]], [16, (i, j) => W.Wh[i * 16 + j]], [10, (i, j) => W.Wo[j * 16 + i]]];
    const scale = mats.map(([cols, get]) => { let m = 0; for (let i = 0; i < 16; i++) for (let j = 0; j < cols; j++) m = Math.max(m, Math.abs(get(i, j))); return m; });
    V.canvas(host, (ctx, w, h) => {
      const gap = 1.3, units = 27 + 2 * gap, s = Math.min(w / (units + 1), h / 17.5), x0 = (w - units * s) / 2, y0 = (h - 16 * s) / 2;
      let x = x0, k = 0;
      mats.forEach(([cols, get], m) => {
        for (let i = 0; i < 16; i++) for (let j = 0; j < cols; j++) {
          const v = get(i, j), t = Math.sqrt(Math.abs(v) / scale[m]), r = s * (0.3 + 0.17 * t);
          const b = V.wash(x + (j + 0.5) * s, y0 + (i + 0.5) * s, r, r, v >= 0 ? V.C.washIndigo : V.C.washVermilion, 700 + k++,
            { shape: 'rect', layers: 5, alpha: 0.03 + 0.17 * t, rough: 0.16, per: 1, jitter: 0.1, spots: 1, res: 0.5 });
          b.grow(V.K); b.draw(ctx);
        }
        x += (cols + gap) * s;
      });
      V.spatter(ctx, w / 2, h / 2, w * 0.42, h * 0.42, V.C.washIndigo, 710, 40, 0.98, 0.9);
    });
  }

  // the dial: a slate face bare at the center, the ten ticks where each position reads, the hand's path on the clean
  // input of the clock chapter (2 at position 2, 1 at position 5), and the runner-up's tick lit in gold
  function works(host) {
    const P = M.plane(W), x = new Array(10).fill(0); x[2] = 2; x[5] = 1;
    const r = M.run(W, x), lab = M.label(x), ans = M.argmax(r.logits);
    const ang = r.h.slice(1).map(v => (Math.hypot(...v) > 1e-9 ? P.angle(v) : null)), ticks = Array.from({ length: 10 }, (_, k) => P.angle(P.row(k)));
    V.canvas(host, (ctx, w, h) => {
      const cx = w / 2, cy = h / 2, R = Math.min(w, h) * 0.34, at = (a, rr) => [cx + rr * Math.cos(a), cy - rr * Math.sin(a)];
      const face = V.wash(cx, cy, 1.3 * R, 1.3 * R, V.C.washSlate, 31, { alpha: 0.018, layers: 40, rough: 0.24, hole: 0.58, spots: 6 });
      face.grow(V.K); face.draw(ctx);
      const [gx, gy] = at(ticks[ans], R);
      const lamp = V.wash(gx, gy, 0.34 * R, 0.34 * R, V.C.washGold, 32, { alpha: 0.05, layers: 30, rough: 0.2 });
      lamp.grow(V.K); lamp.draw(ctx);
      V.spatter(ctx, cx, cy, 1.15 * R, 1.15 * R, V.C.washSlate, 33, 60, 1.05, 1);
      ctx.strokeStyle = V.alpha(V.C.ink, 0.35); ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
      ticks.forEach((a, k) => {
        const win = k === ans, [x1, y1] = at(a, R - 9), [x2, y2] = at(a, R + 9), [lx, ly] = at(a, R + 30);
        ctx.strokeStyle = V.alpha(V.C.ink, win ? 0.95 : 0.6); ctx.lineWidth = win ? 3.2 : 1.6; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        V.text(ctx, String(k), lx, ly, { size: win ? 19 : 15, align: 'center', base: 'middle', color: V.alpha(V.C.ink, win ? 0.95 : 0.7), weight: win ? 600 : 400, face: 'serif', style: 'italic', halo: V.alpha(V.C.page, 0.85) });
      });
      // the path of the hand, one dot per input, on an inner ring; then the hand
      let prev = null, last = null;
      ang.forEach(a => {
        if (a === null) return;
        const p = at(a, 0.58 * R);
        if (prev) { ctx.strokeStyle = V.alpha(V.C.ink, 0.4); ctx.lineWidth = 1.3; ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(p[0], p[1]); ctx.stroke(); }
        ctx.fillStyle = V.alpha(V.C.ink, 0.7); ctx.beginPath(); ctx.arc(p[0], p[1], 3, 0, 2 * Math.PI); ctx.fill();
        prev = p; last = a;
      });
      if (last !== null) { const [hx, hy] = at(last, 0.86 * R); ctx.strokeStyle = V.C.ink; ctx.fillStyle = V.C.ink; ctx.lineWidth = 3.2; V.arrow(ctx, cx, cy, hx, hy, 12); }
      ctx.fillStyle = V.C.ink; ctx.beginPath(); ctx.arc(cx, cy, 4.5, 0, 2 * Math.PI); ctx.fill();
      if (ans === lab) return;
      console.error('parts.works: the clean input is not answered right');
    });
  }

  // the ball of inputs: three inputs over every direction, the rest fixed and lower, as in the dimensions chapter
  function fails(host) {
    const FREE = [2, 5, 7], FIX = { 0: -0.6, 1: 0.25, 3: -1.1, 4: 0.1, 6: -0.4, 8: 0.3, 9: -0.8 }, MU = 1.2, RHO = 0.8, NP = 16000;
    const x = new Array(10).fill(0); for (const k in FIX) x[k] = FIX[k];
    const ga = Math.PI * (3 - Math.sqrt(5)), pts = [];
    for (let i = 0; i < NP; i++) {
      const y = 1 - 2 * (i + 0.5) / NP, rr = Math.sqrt(1 - y * y), th = ga * i, u = [rr * Math.cos(th), y, rr * Math.sin(th)];
      FREE.forEach((p, k) => (x[p] = MU + RHO * u[k]));
      pts.push({ u, wrong: M.argmax(M.run(W, x).logits) !== M.label(x) });
    }
    const circles = [[0, 1], [0, 2], [1, 2]].map(([i, j]) => {
      const n = [0, 0, 0]; n[i] = Math.SQRT1_2; n[j] = -Math.SQRT1_2;
      const a = [0, 0, 0]; a[i] = Math.SQRT1_2; a[j] = Math.SQRT1_2;
      const b = [n[1] * a[2] - n[2] * a[1], n[2] * a[0] - n[0] * a[2], n[0] * a[1] - n[1] * a[0]];
      return Array.from({ length: 181 }, (_, s) => { const t = 2 * Math.PI * s / 180; return a.map((v, d) => Math.cos(t) * v + Math.sin(t) * b[d]); });
    });
    V.canvas(host, (ctx, w, h) => {
      const cx = w / 2, cy = h / 2, R = Math.min(w, h) * 0.37, yaw = 0.6, pitch = 0.35;
      const P = v => { const q = V.project(v, yaw, pitch); return [cx + q.x * R, cy - q.y * R, q.z]; };
      // the ball, shaded toward its lower right, and the spatter it threw
      const ball = V.wash(cx, cy, 1.03 * R, 1.03 * R, V.C.washIndigo, 41, { alpha: 0.012, layers: 44, rough: 0.07, per: 2, jitter: 0.015, spots: 6 });
      ball.grow(V.K); ball.draw(ctx);
      ctx.save(); ctx.beginPath(); ctx.arc(cx, cy, 1.03 * R, 0, 2 * Math.PI); ctx.clip();
      const shade = V.wash(cx + 0.38 * R, cy + 0.34 * R, 0.8 * R, 0.8 * R, V.C.washIndigo, 42, { alpha: 0.014, layers: 30, rough: 0.2 });
      shade.grow(V.K); shade.draw(ctx);
      ctx.restore();
      V.spatter(ctx, cx, cy, 1.15 * R, 1.15 * R, V.C.washIndigo, 43, 50, 1.02, 1);
      // the ties, scraped back to paper in front and faint behind; the errors in vermilion, hugging them
      for (const pass of [0, 1]) for (const c of circles) for (let s = 0; s < c.length - 1; s++) {
        const [x0, y0, z0] = P(c[s]), [x1, y1] = P(c[s + 1]);
        if ((z0 >= 0) !== (pass === 1)) continue;
        ctx.strokeStyle = pass ? V.alpha(V.C.page, 0.95) : V.alpha(V.C.page, 0.3); ctx.lineWidth = pass ? 2 : 1.2;
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      }
      for (const pass of [0, 1]) for (const p of pts) {
        if (!p.wrong) continue;
        const [px, py, z] = P(p.u);
        if ((z >= 0) !== (pass === 1)) continue;
        ctx.fillStyle = pass ? V.C.washVermilion : V.alpha(V.C.washVermilion, 0.25);
        ctx.beginPath(); ctx.arc(px, py, pass ? 2.4 : 1.6, 0, 2 * Math.PI); ctx.fill();
      }
      ctx.strokeStyle = V.alpha(V.C.ink, 0.35); ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
    });
  }

  window.Parts = { overview, works, fails };
})();
