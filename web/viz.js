// Drawing helpers every chapter shares: palette, canvases in design pixels (scaled with the page's root size and
// snapped to device pixels, so a 4K screen gets the same picture, larger and crisp), colors for signed values,
// tweens that respect reduced motion, a 3-D projection, toolbar controls, one tooltip, and the watercolor: washes,
// spatter, the panel every figure sits on, and the brush swipe under a part's title.
(function () {
  const root = document.documentElement;
  const cssVar = k => getComputedStyle(root).getPropertyValue('--' + k).trim();
  const C = {};
  for (const k of ['page', 'surface', 'surface-2', 'ink', 'ink-2', 'muted', 'grid', 'axis', 'pos', 'neg', 'mid', 'max', 'second', 'third', 'other',
    'core', 'timer', 'right', 'wrong', 'wash-gold', 'wash-indigo', 'wash-slate', 'wash-vermilion'])
    C[k.replace(/-(\w)/, (_, c) => (c === '2' ? '2' : c.toUpperCase()))] = cssVar(k);
  const FONT = cssVar('sans') || 'system-ui, sans-serif', SERIF = cssVar('serif') || 'Georgia, serif';
  const CORE = [2, 4, 7, 6, 1, 8];
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const SUB = '₀₁₂₃₄₅₆₇₈₉';
  const xi = t => 'x' + SUB[t];

  // One design pixel is S css pixels, S = root font size / 16; K device pixels. Both follow the latest resize.
  let S = 1, K = 1;
  const measure = () => { S = parseFloat(getComputedStyle(root).fontSize) / 16; K = (devicePixelRatio || 1) * S; };
  measure();
  const snap = v => Math.round(v * K) / K;           // a device-pixel edge
  const crisp = v => (Math.round(v * K) + 0.5) / K;  // the center of a device pixel, for 1-px lines

  // A canvas filling `parent`, redrawn by draw(ctx, w, h) in design pixels on every resize, and once more when the
  // page's fonts finish loading, since a canvas drawn before that set its text in the fallback face.
  const canvases = new Set();
  function canvas(parent, draw) {
    const cv = document.createElement('canvas');
    parent.appendChild(cv);
    const ctx = cv.getContext('2d');
    const api = { cv, ctx, w: 0, h: 0, draw,
      redraw() {
        if (api.w < 50 || api.h < 50) return;  // not laid out yet; the ResizeObserver redraws once it is
        ctx.setTransform(K, 0, 0, K, 0, 0); ctx.clearRect(0, 0, api.w, api.h); api.draw(ctx, api.w, api.h);
      },
      // a pointer event's position in design pixels
      pt(e) { const r = cv.getBoundingClientRect(); return { x: (e.clientX - r.left) / S, y: (e.clientY - r.top) / S }; },
    };
    new ResizeObserver(() => {
      measure();
      const r = parent.getBoundingClientRect(), d = devicePixelRatio || 1;
      api.w = r.width / S; api.h = r.height / S; cv.width = Math.round(r.width * d); cv.height = Math.round(r.height * d);
      api.redraw();
    }).observe(parent);
    canvases.add(api);
    return api;
  }
  if (document.fonts) document.fonts.addEventListener('loadingdone', () => canvases.forEach(c => c.redraw()));

  function hex(c) { const n = parseInt(c.slice(1), 16); return [n >> 16, (n >> 8) & 255, n & 255]; }
  function mix(a, b, t) { const p = hex(a), q = hex(b); return `rgb(${p.map((v, i) => Math.round(v + (q[i] - v) * t)).join(',')})`; }
  // Signed value on the ultramarine-vermilion diverging pair, paper at zero; |v| >= max is a full pole.
  function diverge(v, max) { const t = Math.min(1, Math.abs(v) / max); return mix(C.mid, v >= 0 ? C.pos : C.neg, Math.sqrt(t)); }
  function alpha(c, a) { const p = hex(c); return `rgba(${p.join(',')},${a})`; }
  function rankColor(rank) { return rank === 0 ? C.max : rank === 1 ? C.second : rank === 2 ? C.third : C.other; }
  function cellColor(n) { return CORE.includes(n) ? C.core : C.timer; }

  const ease = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const lerp = (a, b, t) => a + (b - a) * t;
  // Run fn(t) for t in [0, 1] over ms; a newer tween on the same token cancels the older one.
  function tween(token, ms, fn) {
    token.id = (token.id || 0) + 1;
    const id = token.id;
    if (reduced || ms <= 0) { fn(1); return Promise.resolve(); }
    return new Promise(done => {
      const t0 = performance.now();
      const frame = now => {
        if (token.id !== id) return done();
        const t = Math.min(1, (now - t0) / ms);
        fn(ease(t));
        if (t < 1) requestAnimationFrame(frame); else done();
      };
      requestAnimationFrame(frame);
    });
  }

  // Rotate by yaw about the vertical axis, then pitch, and project orthographically; z is depth for sorting.
  function project([x, y, z], yaw, pitch) {
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const x1 = cy * x + sy * z, z1 = -sy * x + cy * z;
    return { x: x1, y: cp * y - sp * z1, z: sp * y + cp * z1 };
  }

  const fmt = (v, d = 2) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
  const sfmt = (v, d = 1) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
  const pct = (v, d = 1) => (100 * v).toFixed(d) + '%';

  let tipEl = null;
  function tip(html, x, y) {
    if (!tipEl) { tipEl = document.createElement('div'); tipEl.id = 'tip'; document.body.appendChild(tipEl); }
    if (html == null) { tipEl.style.display = 'none'; return; }
    tipEl.innerHTML = html; tipEl.style.display = 'block';
    const r = tipEl.getBoundingClientRect();
    tipEl.style.left = Math.min(innerWidth - r.width - 8, x + 14) + 'px';
    tipEl.style.top = Math.min(innerHeight - r.height - 8, y + 14) + 'px';
  }

  // Toolbar controls under a figure.
  function button(bar, label, onclick) { const b = document.createElement('button'); b.textContent = label; b.onclick = onclick; bar.appendChild(b); return b; }
  function slider(bar, label, { min, max, step, value, width }, oninput) {
    const l = document.createElement('label'), s = document.createElement('input');
    s.type = 'range'; s.min = min; s.max = max; s.step = step; s.value = value; if (width) s.style.width = width;
    s.oninput = () => oninput(+s.value);
    l.append(label, s); bar.appendChild(l); return s;
  }
  function select(bar, label, options, value, onchange) {
    const l = document.createElement('label'), s = document.createElement('select');
    for (const [v, name] of options) s.add(new Option(name, v));
    s.value = value; s.onchange = () => onchange(s.value);
    l.append(label, s); bar.appendChild(l); return s;
  }
  function note(bar, s) { const n = document.createElement('span'); n.className = 'note'; n.textContent = s; bar.appendChild(n); return n; }
  function readout(bar) { const n = document.createElement('span'); n.className = 'readout'; bar.appendChild(n); return n; }

  // Text on a canvas: sans for numbers and ticks; serif italic (face: 'serif') for a figure's notes and titles.
  function font(size, weight, face, style) { return `${style || ''} ${weight} ${size}px ${face === 'serif' ? SERIF : FONT}`; }
  // Text that would leave the canvas shrinks just enough to stay on it, so a phone's narrow figure keeps its titles.
  function text(ctx, s, x, y, { color = C.muted, size = 13, align = 'left', base = 'alphabetic', weight = 400, face, style } = {}) {
    ctx.fillStyle = color; ctx.font = font(size, weight, face, style); ctx.textAlign = align; ctx.textBaseline = base;
    const W = ctx.canvas.width / K - 6, wd = ctx.measureText(s).width;
    const room = align === 'left' ? W - x : align === 'right' ? x - 6 : 2 * Math.min(x - 6, W - x);
    if (wd > room && room > 0) ctx.font = font(Math.max(8, size * room / wd), weight, face, style);
    ctx.fillText(s, x, y);
  }
  // A figure's title: serif, in ink, at the top left of a panel.
  function title(ctx, s, x, y, size = 15) { text(ctx, s, x, y, { color: C.ink, size, face: 'serif', weight: 600 }); }
  function width(ctx, s, size = 13, weight = 400, face) { ctx.font = font(size, weight, face); return ctx.measureText(s).width; }
  function roundRect(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.roundRect(x, y, w, h, r); }
  // A filled rectangle on device-pixel edges: heatmap cells and bars with no soft seams.
  function rect(ctx, x, y, w, h) { const x0 = snap(x), y0 = snap(y); ctx.fillRect(x0, y0, snap(x + w) - x0, snap(y + h) - y0); }
  // A rounded bar from a baseline: positive v goes up from `base` by len; the free end is rounded (the dataviz mark).
  function bar(ctx, x, base, w, len, r = 3) {
    const top = len >= 0 ? base - len : base, h = Math.abs(len);
    if (h < 0.5) { ctx.fillRect(x, base - 0.5, w, 1); return; }
    const rr = Math.min(r, w / 2, h);
    ctx.beginPath(); ctx.roundRect(x, top, w, h, len >= 0 ? [rr, rr, 0, 0] : [0, 0, rr, rr]); ctx.fill();
  }
  // Crisp 1-px rules.
  function hline(ctx, x0, x1, y) { const yy = crisp(y); ctx.beginPath(); ctx.moveTo(x0, yy); ctx.lineTo(x1, yy); ctx.stroke(); }
  function vline(ctx, x, y0, y1) { const xx = crisp(x); ctx.beginPath(); ctx.moveTo(xx, y0); ctx.lineTo(xx, y1); ctx.stroke(); }
  // A drag handle on a bar's free end: a pill with two grip lines.
  function handle(ctx, cx, y, color = C.ink2) {
    ctx.fillStyle = color; roundRect(ctx, cx - 9, y - 4, 18, 8, 4); ctx.fill();
    ctx.strokeStyle = C.surface; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx - 4, y - 1.5); ctx.lineTo(cx + 4, y - 1.5); ctx.moveTo(cx - 4, y + 1.5); ctx.lineTo(cx + 4, y + 1.5); ctx.stroke();
  }
  // A short verdict in a tinted chip: the text stays ink, the tint and a leading mark carry the color.
  function chip(ctx, s, x, y, { color = C.ink, size = 13, align = 'left', weight = 600 } = {}) {
    const w = width(ctx, s, size, weight) + 22, h = size + 13, x0 = align === 'center' ? x - w / 2 : align === 'right' ? x - w : x;
    ctx.fillStyle = alpha(color, 0.13); roundRect(ctx, x0, y - h / 2, w, h, h / 2); ctx.fill();
    ctx.strokeStyle = alpha(color, 0.45); ctx.lineWidth = 1; roundRect(ctx, x0 + 0.5, y - h / 2 + 0.5, w - 1, h - 1, h / 2); ctx.stroke();
    text(ctx, s, x0 + w / 2, y + 0.5, { color: C.ink, size, align: 'center', base: 'middle', weight });
    return w;
  }
  // A row of swatches with names.
  function legend(ctx, x, y, items, size = 12) {
    for (const [name, col] of items) {
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x + 5, y, 5, 0, 2 * Math.PI); ctx.fill();
      text(ctx, name, x + 15, y, { size, color: C.ink2, base: 'middle' });
      x += width(ctx, name, size) + 34;
    }
    return x;
  }
  function arrow(ctx, x0, y0, x1, y1, head = 7) {
    const a = Math.atan2(y1 - y0, x1 - x0);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - head * Math.cos(a - 0.45), y1 - head * Math.sin(a - 0.45)); ctx.lineTo(x1 - head * Math.cos(a + 0.45), y1 - head * Math.sin(a + 0.45)); ctx.closePath(); ctx.fill();
  }

  // Watercolor (Tyler Hobbs' layered-polygon method). A wash is many translucent layers of one torn polygon, each torn
  // again and scaled a little, so the middle pools and the edge feathers; each layer loses a few soft spots, the
  // pigment's texture. A wash paints on its own sheet, so its hue holds, and the sheet is glazed onto the paper. With
  // `hole`, a torn inner polygon leaves the paper blank inside. Seeded, so the page paints the same picture each load.
  function seeded(seed) {
    let a = seed >>> 0;
    const u = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    u.gauss = () => Math.sqrt(-2 * Math.log(u() || 1e-12)) * Math.cos(2 * Math.PI * u());
    return u;
  }
  function tear(pts, depth, rnd) {
    for (let d = 0; d < depth; d++) {
      const out = [];
      for (let i = 0; i < pts.length; i++) {
        const a = pts[i], b = pts[(i + 1) % pts.length], len = Math.hypot(b.x - a.x, b.y - a.y), v = (a.v + b.v) / 2;
        out.push(a, { x: (a.x + b.x) / 2 + rnd.gauss() * len * v, y: (a.y + b.y) / 2 + rnd.gauss() * len * v, v: v * (0.7 + 0.5 * rnd()) });
      }
      pts = out;
    }
    return pts;
  }
  const ring = (cx, cy, rx, ry, rough, rnd) => Array.from({ length: 10 }, (_, i) => { const a = 2 * Math.PI * i / 10; return { x: cx + rx * Math.cos(a), y: cy + ry * Math.sin(a), v: rough * (0.5 + rnd()) }; });
  // a rectangle with five points a side, so its torn edge wanders a little but stays an edge
  function box(cx, cy, rx, ry, rough, rnd) {
    const c = [[cx - rx, cy - ry], [cx + rx, cy - ry], [cx + rx, cy + ry], [cx - rx, cy + ry]], pts = [];
    c.forEach(([x0, y0], i) => { const [x1, y1] = c[(i + 1) % 4]; for (let k = 0; k < 5; k++) pts.push({ x: x0 + (x1 - x0) * k / 5, y: y0 + (y1 - y0) * k / 5, v: rough * (0.5 + rnd()) }); });
    return pts;
  }
  function trace(ctx, pts, ox, oy) { ctx.moveTo(pts[0].x - ox, pts[0].y - oy); for (const p of pts) ctx.lineTo(p.x - ox, p.y - oy); ctx.closePath(); }
  // shape 'ellipse' (the default) or 'rect'; hole is a fraction of the radius for an ellipse and an inset in design
  // pixels for a rect, where each layer re-tears its own inner edge, jittered by holeJitter, so the pigment pools at
  // the rim and fades inward; rim strokes the outer edge once more, as the drying line; res scales the sheet below
  // device resolution for washes too soft to need it.
  function wash(cx, cy, rx, ry, color, seed, { layers = 40, alpha: a = 0.03, rough = 0.26, per = 3, jitter = 0.06, holeJitter = jitter, spots = 5, hole = 0, glaze = 'multiply', shape = 'ellipse', rim = 0, rimAlpha = 0.18, res = 1 } = {}) {
    const rnd = seeded(seed), rect = shape === 'rect', [r, g, b] = hex(color);
    const padx = rect ? rx * 1.1 + 36 : 1.7 * Math.max(rx, ry), pady = rect ? ry * 1.1 + 36 : padx;
    const base = rect ? box : ring;
    const outer = tear(base(cx, cy, rx, ry, rough, rnd), 4, rnd), inner = hole && !rect ? tear(ring(cx, cy, rx * hole, ry * hole, rough, rnd), 4, rnd) : null;
    const scaled = (pts, s) => pts.map(p => ({ x: cx + (p.x - cx) * s, y: cy + (p.y - cy) * s, v: p.v }));
    const polys = Array.from({ length: layers }, () => {
      // a rect's inner edge sits a little high, so the pigment settles thicker along the bottom, as on a tilted board
      const s = 1 + jitter * rnd.gauss(), hk = hole * Math.min(2.4, Math.max(0.25, 1 + holeJitter * rnd.gauss())), lift = hole * (0.2 + 0.3 * rnd());
      return { out: tear(scaled(outer, s), per, rnd),
        in: inner ? tear(scaled(inner, 1 + holeJitter * rnd.gauss()), per, rnd) : hole && rect ? tear(box(cx, cy - lift, rx - hk, ry - hk, rough * 1.6, rnd), 3, rnd) : null,
        spots: Array.from({ length: spots }, () => [(rnd() - 0.5) * 2 * rx, (rnd() - 0.5) * 2 * ry, (0.08 + 0.3 * rnd()) * Math.min(rx, ry)]) };
    });
    let sheet = null, tmp = null, k = 0, drawn = 0, rimmed = false;
    return {
      size: polys.length,
      // lay down layers up to `to` (all by default) at device scale K
      grow(K, to = polys.length) {
        const Ks = K * res;
        if (!sheet || k !== Ks) {
          k = Ks; drawn = 0; rimmed = false;
          sheet = document.createElement('canvas'); tmp = document.createElement('canvas');
          sheet.width = tmp.width = Math.ceil(2 * padx * Ks); sheet.height = tmp.height = Math.ceil(2 * pady * Ks);
        }
        if (drawn && drawn >= Math.min(to, polys.length)) return;  // nothing new to lay down
        const sc = sheet.getContext('2d'), tc = tmp.getContext('2d'), x0 = cx - padx, y0 = cy - pady;
        sc.setTransform(Ks, 0, 0, Ks, 0, 0); tc.setTransform(Ks, 0, 0, Ks, 0, 0);
        // Layers lay down coverage only, then the sheet takes the pigment's exact color: at a few percent per layer an
        // 8-bit color quantizes to a different hue (#24507d at 1.6% is (1, 1, 2), violet), but coverage keeps.
        for (; drawn < Math.min(to, polys.length); drawn++) {
          const P = polys[drawn];
          tc.clearRect(0, 0, 2 * padx, 2 * pady);
          tc.globalCompositeOperation = 'source-over'; tc.fillStyle = `rgba(0,0,0,${a})`;
          tc.beginPath(); trace(tc, P.out, x0, y0); if (P.in) trace(tc, P.in, x0, y0); tc.fill('evenodd');
          tc.globalCompositeOperation = 'destination-out';
          for (const [dx, dy, sr] of P.spots) {
            const gr = tc.createRadialGradient(padx + dx, pady + dy, 0, padx + dx, pady + dy, sr);
            gr.addColorStop(0, 'rgba(0,0,0,0.9)'); gr.addColorStop(1, 'rgba(0,0,0,0)');
            tc.fillStyle = gr; tc.beginPath(); tc.arc(padx + dx, pady + dy, sr, 0, 2 * Math.PI); tc.fill();
          }
          sc.setTransform(1, 0, 0, 1, 0, 0); sc.drawImage(tmp, 0, 0); sc.setTransform(Ks, 0, 0, Ks, 0, 0);
        }
        if (rim && drawn === polys.length && !rimmed) {
          rimmed = true;
          sc.strokeStyle = `rgba(0,0,0,${rimAlpha})`; sc.lineWidth = rim; sc.lineJoin = 'round'; sc.filter = 'blur(0.5px)';
          sc.beginPath(); trace(sc, outer, x0, y0); sc.stroke(); sc.filter = 'none';
        }
        sc.setTransform(1, 0, 0, 1, 0, 0); sc.globalCompositeOperation = 'source-in';
        sc.fillStyle = `rgb(${r},${g},${b})`; sc.fillRect(0, 0, sheet.width, sheet.height);
        sc.globalCompositeOperation = 'source-over';
        if (drawn === polys.length) tmp = null;
      },
      // glaze the sheet onto ctx (in design pixels)
      draw(ctx) {
        if (!sheet) return;
        ctx.save(); ctx.globalCompositeOperation = glaze; ctx.drawImage(sheet, cx - padx, cy - pady, 2 * padx, 2 * pady); ctx.restore();
      },
    };
  }
  // Flicked pigment: n dots of `color` around (cx, cy), spread `sx` by `sy`, kept off the disc of radius `clear`.
  function spatter(ctx, cx, cy, sx, sy, color, seed, n = 60, clear = 0, size = 1) {
    const rnd = seeded(seed), base = ctx.globalAlpha;
    ctx.save(); ctx.globalCompositeOperation = 'multiply'; ctx.fillStyle = color;
    for (let i = 0; i < n; i++) {
      const x = cx + rnd.gauss() * sx, y = cy + rnd.gauss() * sy;
      if (Math.hypot((x - cx) / sx, (y - cy) / sy) < clear) continue;
      ctx.globalAlpha = base * (0.2 + 0.5 * rnd()); ctx.beginPath(); ctx.arc(x, y, size * (0.4 + 1.8 * rnd() ** 2), 0, 2 * Math.PI); ctx.fill();
    }
    ctx.restore();
  }
  // The panel a figure sits on: a pale wash over its box, bleeding a little past it, the pigment pooled at a torn
  // rim and faded inward, so the chart's own paper stays nearly bare. `host` is the bleed wrapper around the box.
  function panel(host, seed, color = C.washIndigo) {
    return canvas(host, (ctx, w, h) => {
      const fig = host.parentElement.getBoundingClientRect(), P = (w - fig.width / S) / 2, Q = (h - fig.height / S) / 2;
      const cx = w / 2, cy = h / 2, rx = w / 2 - P - 3, ry = h / 2 - Q - 3, opts = { shape: 'rect', rough: 0.045, per: 2, jitter: 0.006, res: 0.5 };
      const tint = wash(cx, cy, rx, ry, color, seed, { ...opts, layers: 9, alpha: 0.0045, spots: 9 });
      const edge = wash(cx, cy, rx, ry, color, seed + 1, { ...opts, layers: 24, alpha: 0.0105, hole: 16, holeJitter: 0.7, spots: 2, rim: 3, rimAlpha: 0.32 });
      tint.grow(K); tint.draw(ctx); edge.grow(K); edge.draw(ctx);
      spatter(ctx, cx, cy, rx * 0.62, ry * 0.62, color, seed + 2, 16, 1.52, 0.8);  // only where the band is, never on a chart's text
    });
  }
  // One stroke of a wide brush, for a title to sit on: `host` is a box behind the text. A darker pigment goes on
  // thinner, so the three parts' strokes weigh the same.
  function swipe(host, color, seed) {
    const [r, g, b] = hex(color), lum = (0.3 * r + 0.59 * g + 0.11 * b) / 255;
    return canvas(host, (ctx, w, h) => {
      const s = wash(w * 0.5, h * 0.56, w * 0.46, h * 0.27, color, seed, { shape: 'rect', layers: 24, alpha: 0.017 + 0.017 * lum, rough: 0.13, per: 2, jitter: 0.045, spots: 4, res: 0.5 });
      s.grow(K); s.draw(ctx);
      spatter(ctx, w * 0.5, h * 0.56, w * 0.5, h * 0.3, color, seed + 1, 18, 1.1, 0.9);
    });
  }

  // The network's map, shared by the hero and the field guide: n2 at the center, the comparators around it in the
  // order of their phase on the clock, and the timers around a dial of answer positions (0 at the top, clockwise), each
  // at the runner-up position it points to (research/findings/neurons.md); the faint ones sit outside the ring.
  // [neuron, dial position, radius in units of the dial]
  const PLACE = [[2, null, 0], [4, 2.8, 0.2], [8, 1.5, 0.46], [1, 4, 0.46], [6, 6.5, 0.46], [7, 9, 0.46],
    [14, 0, 1], [15, 1.55, 1], [3, 0.85, 1.24], [10, 2.6, 1], [11, 4, 1], [12, 5.45, 1], [13, 4.9, 1.24], [0, 7, 1], [5, 8.1, 1], [9, 8.35, 1.25]];
  function place(cx, cy, R) {
    const pos = {};
    for (const [n, q, r] of PLACE) {
      const a = (-90 + (q ?? 0) * 36) * Math.PI / 180;
      pos[n] = q === null ? { x: cx - 0.03 * R, y: cy + 0.02 * R } : { x: cx + r * R * Math.cos(a), y: cy + r * R * Math.sin(a) };
    }
    return pos;
  }

  window.V = { C, CORE, reduced, FONT, SERIF, xi, get S() { return S; }, snap, crisp, canvas, mix, diverge, alpha, rankColor, cellColor, ease, lerp, tween, project,
    fmt, sfmt, pct, tip, button, slider, select, note, readout, text, title, width, roundRect, rect, bar, hline, vline, handle, chip, legend, arrow,
    get K() { return K; }, seeded, wash, spatter, panel, swipe, PLACE, place };
})();
