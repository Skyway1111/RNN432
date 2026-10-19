// The hero: the network painted from its weights, on the map V.place lays out. Each neuron is an ink circle sized by
// its mean activity over a loose wash, gold for the core six and slate for the ten timers; a blue ground leaves the
// paper bare around the core and under each timer; ink joins the 30 links whose cut costs the most accuracy (results/experiments/
// neuron_atlas.npy), tinted by sign. The paint blooms once on load; reduced motion shows it finished.
(function () {
  const host = document.getElementById('painting');
  if (!host) return;
  const W = M.base, mean = M.meanState(W), INK_POS = V.mix(V.C.ink, V.C.pos, 0.65), INK_NEG = V.mix(V.C.ink, V.C.neg, 0.7);
  const links = [];
  DATA.links.cut.forEach((row, i) => row.forEach((a, s) => { if (s > 0 && s - 1 !== i) links.push({ i, j: s - 1, c: DATA.links.base - a, w: W.Wh[i * 16 + s - 1] }); }));
  links.sort((a, b) => b.c - a.c).splice(30);
  const cmax = links[0].c;

  let g = null, t0 = null, done = V.reduced;
  const progress = () => (done ? 1 : t0 === null ? 0 : Math.min(1, (performance.now() - t0) / 2800));
  const ramp = (u, a, b) => Math.min(1, Math.max(0, (u - a) / (b - a)));
  function layout(w, h) {
    const vh = Math.min(h, innerHeight / V.S), wide = w > 900;
    const cx = wide ? w * 0.715 : w * 0.5, cy = wide ? vh * 0.46 : vh * 0.25, R = wide ? Math.min(w * 0.165, vh * 0.29) : Math.min(w * 0.3, vh * 0.17);
    const pos = V.place(cx, cy, R), rad = n => R * (0.05 + 0.017 * Math.sqrt(mean[n]));
    const washes = [
      { at: 0, w: V.wash(cx + 0.06 * R, cy - 0.04 * R, 1.55 * R, 1.42 * R, V.C.washIndigo, 11, { alpha: 0.016, layers: 44, rough: 0.3, hole: 0.5 }) },
      { at: 0.08, w: V.wash(cx - 0.02 * R, cy + 0.02 * R, 0.5 * R, 0.46 * R, V.C.washGold, 12, { alpha: 0.012, layers: 30, rough: 0.28 }) },
      // the paper left bare under each timer, as if masked before the ground went down
      ...V.PLACE.filter(([n]) => !V.CORE.includes(n)).map(([n], k) => {
        const r = rad(n) * 1.5, p = pos[n];
        return { at: 0.12 + 0.02 * k, w: V.wash(p.x, p.y, r, r, V.C.ink, 200 + n, { alpha: 0.3, layers: 8, rough: 0.14, per: 2, spots: 0, glaze: 'destination-out' }) };
      }),
      ...V.PLACE.map(([n], k) => {
        const core = V.CORE.includes(n), r = rad(n) * (core ? 1.45 : 1.2), p = pos[n];
        return { at: 0.18 + 0.025 * k, w: V.wash(p.x - 0.12 * r, p.y + 0.1 * r, r, r * 0.95, core ? V.C.washGold : V.C.washSlate, 100 + n, { alpha: core ? 0.06 : 0.04, layers: 28, rough: 0.1, per: 2, spots: 3 }) };
      }),
    ];
    return { w, h, cx, cy, R, pos, rad, washes };
  }

  const cv = V.canvas(host, (ctx, w, h) => {
    if (!g || g.w !== w || g.h !== h) g = layout(w, h);
    const u = progress();
    for (const s of g.washes) { s.w.grow(V.K, Math.round(ramp(u, s.at, s.at + 0.5) * s.w.size)); s.w.draw(ctx); }
    ink(ctx, u);
  });

  function ink(ctx, u) {
    const { pos, R, cx, cy, rad } = g, rnd = V.seeded(7);
    // the dial of answer positions, in pencil
    ctx.globalAlpha = ramp(u, 0.55, 0.9);
    ctx.strokeStyle = V.alpha(V.C.ink, 0.22); ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.arc(cx, cy, 1.12 * R, 0, 2 * Math.PI); ctx.stroke();
    for (let q = 0; q < 10; q++) {
      const a = (-90 + q * 36) * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
      ctx.beginPath(); ctx.moveTo(cx + 1.09 * R * c, cy + 1.09 * R * s); ctx.lineTo(cx + 1.15 * R * c, cy + 1.15 * R * s); ctx.stroke();
      V.text(ctx, String(q), cx + 1.35 * R * c, cy + 1.35 * R * s, { size: 16, align: 'center', base: 'middle', color: V.alpha(V.C.ink, 0.55), face: 'serif', style: 'italic' });
    }
    // ink between the neurons, drawn from sender to receiver, ending at their circles
    ctx.lineCap = 'round';
    links.forEach((l, k) => {
      const f = ramp(u, 0.3 + 0.01 * k, 0.58 + 0.01 * k), a = pos[l.j], b = pos[l.i], bend = (rnd() - 0.5) * 0.3;
      if (f <= 0) return;
      const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy), mx = (a.x + b.x) / 2 - dy * bend, my = (a.y + b.y) / 2 + dx * bend;
      const t1 = rad(l.j) / d, t2 = 1 - rad(l.i) / d, s = l.c / cmax;
      ctx.strokeStyle = l.w >= 0 ? INK_POS : INK_NEG; ctx.globalAlpha = 0.3 + 0.55 * s; ctx.lineWidth = 0.6 + 2.2 * s ** 1.5;
      ctx.beginPath();
      const end = t1 + (t2 - t1) * f;
      for (let i = 0; i <= 24; i++) { const t = t1 + (end - t1) * i / 24, x = (1 - t) ** 2 * a.x + 2 * (1 - t) * t * mx + t * t * b.x, y = (1 - t) ** 2 * a.y + 2 * (1 - t) * t * my + t * t * b.y; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.stroke();
    });
    // the neurons: an ink circle each, and its name like a specimen's label
    ctx.globalAlpha = ramp(u, 0.45, 0.8);
    for (const [n] of V.PLACE) {
      const p = pos[n], r = rad(n);
      ctx.strokeStyle = V.alpha(V.C.ink, 0.85); ctx.lineWidth = V.CORE.includes(n) ? 1.4 : 1.1;
      ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 2 * Math.PI); ctx.stroke();
      V.text(ctx, 'n' + n, p.x + r * 0.75 + 4, p.y + r * 0.75 + 11, { size: 13.5, color: V.alpha(V.C.ink, 0.75), face: 'serif', style: 'italic' });
    }
    ctx.globalAlpha = 1;
  }

  if (!done) {
    const frame = now => {
      if (t0 === null) t0 = now;
      cv.redraw();
      if (progress() < 1) requestAnimationFrame(frame); else done = true;
    };
    requestAnimationFrame(frame);
  }
})();
