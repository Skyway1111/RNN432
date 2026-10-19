// The 432-parameter model in the browser: h_{t+1} = ReLU(Wh h_t + Wi x_t), logits = Wo h_10 (az/model.py).
// Weights are flat Float64Arrays (Wh row-major) so a forward pass allocates nothing; an input bank of seeded Gaussian
// inputs is drawn once, so accuracy under a dragged weight is re-measured on the same inputs and moves smoothly.
(function () {
  const N = 16, T = 10;

  function weights(src) {
    return { Wi: Float64Array.from(src.Wi), Wh: Float64Array.from(src.Wh.flat()), Wo: Float64Array.from(src.Wo.flat()) };
  }
  function clone(W) { return { Wi: W.Wi.slice(), Wh: W.Wh.slice(), Wo: W.Wo.slice() }; }

  // Everything one input does: the state after each step (h[0] = 0), pre-activations, and the logits.
  function run(W, x) {
    const h = [new Float64Array(N)], pre = [];
    for (let t = 0; t < T; t++) {
      const z = new Float64Array(N), hp = h[t], s = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        let a = W.Wi[i] * x[t];
        for (let j = 0; j < N; j++) a += W.Wh[i * N + j] * hp[j];
        z[i] = a; s[i] = a > 0 ? a : 0;
      }
      pre.push(z); h.push(s);
    }
    return { h, pre, logits: readout(W, h[T]) };
  }
  function readout(W, h) {
    const L = new Float64Array(T);
    for (let k = 0; k < T; k++) { let a = 0; for (let j = 0; j < N; j++) a += W.Wo[k * N + j] * h[j]; L[k] = a; }
    return L;
  }

  // Positions sorted largest first; the label is the 2nd.
  function order(x) { return Array.from(x.keys()).sort((a, b) => x[b] - x[a]); }
  function label(x) { return order(x)[1]; }
  function argmax(a) { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }
  // The label's logit minus the best other: positive means right, and its size is the confidence.
  function margin(L, q) { let m = -Infinity; for (let k = 0; k < T; k++) if (k !== q && L[k] > m) m = L[k]; return L[q] - m; }

  // Each input's exact part of h_10 under the input's own on/off pattern, and its vote for each output position.
  // h_10 = sum_t P_t Wi x_t, P_t the gated product of the steps after t; the parts sum to h_10 exactly.
  function votes(W, x, r = run(W, x)) {
    const parts = [];
    for (let t = 0; t < T; t++) {
      let v = new Float64Array(N);
      for (let i = 0; i < N; i++) v[i] = r.pre[t][i] > 0 ? W.Wi[i] * x[t] : 0;
      for (let s = t + 1; s < T; s++) {
        const u = new Float64Array(N);
        for (let i = 0; i < N; i++) {
          if (r.pre[s][i] <= 0) continue;
          let a = 0; for (let j = 0; j < N; j++) a += W.Wh[i * N + j] * v[j];
          u[i] = a;
        }
        v = u;
      }
      parts.push(v);
    }
    return { parts, votes: parts.map(p => readout(W, p)) };
  }

  // Seeded normals (mulberry32 and Box-Muller), so a page shows the same inputs on every load.
  function rng(seed) {
    let a = seed >>> 0, spare = null;
    const u = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    return () => {
      if (spare !== null) { const s = spare; spare = null; return s; }
      const r = Math.sqrt(-2 * Math.log(u() || 1e-12)), th = 2 * Math.PI * u();
      spare = r * Math.sin(th); return r * Math.cos(th);
    };
  }
  const banks = new Map();
  function bank(n, seed = 1) {
    const key = n + ':' + seed;
    if (!banks.has(key)) { const g = rng(seed), X = new Float64Array(n * T); for (let i = 0; i < X.length; i++) X[i] = g(); banks.set(key, X); }
    return banks.get(key);
  }

  // The hot loop of every slider and histogram: logits of input X[o..o+10) into L, with preallocated state, and the
  // label (2nd largest's position) returned.
  const hA = new Float64Array(N), hB = new Float64Array(N);
  function forwardInto(W, X, o, L) {
    let h = hA, g = hB; h.fill(0);
    for (let t = 0; t < T; t++) {
      const xt = X[o + t];
      for (let i = 0; i < N; i++) {
        let a = W.Wi[i] * xt; const r = i * N;
        for (let j = 0; j < N; j++) a += W.Wh[r + j] * h[j];
        g[i] = a > 0 ? a : 0;
      }
      const tmp = h; h = g; g = tmp;
    }
    let m1 = -Infinity, m2 = -Infinity, q1 = 0, q2 = 0;
    for (let k = 0; k < T; k++) {
      let a = 0; const r = k * N;
      for (let j = 0; j < N; j++) a += W.Wo[r + j] * h[j];
      L[k] = a;
      const v = X[o + k];
      if (v > m1) { m2 = m1; q2 = q1; m1 = v; q1 = k; } else if (v > m2) { m2 = v; q2 = k; }
    }
    return q2;
  }
  const Lbuf = new Float64Array(T);
  // Share of bank inputs answered right.
  function accuracy(W, n = 20000, seed = 1) {
    const X = bank(n, seed);
    let right = 0;
    for (let s = 0; s < n; s++) { const q = forwardInto(W, X, s * T, Lbuf); if (argmax(Lbuf) === q) right++; }
    return right / n;
  }
  // How confidently wrong on each bank input: (best other score - the label's score) / |x|, the rare-event level c.
  function margins(W, n, seed = 1) {
    const X = bank(n, seed), out = new Float64Array(n);
    for (let s = 0; s < n; s++) {
      const o = s * T, q = forwardInto(W, X, o, Lbuf);
      let bo = -Infinity, nx = 0;
      for (let k = 0; k < T; k++) { if (k !== q && Lbuf[k] > bo) bo = Lbuf[k]; nx += X[o + k] * X[o + k]; }
      out[s] = (bo - Lbuf[q]) / Math.sqrt(nx);
    }
    return out;
  }

  // Each neuron's mean activity in the states the steps read (h_0 … h_9) over bank inputs: a link j -> i pushes
  // |Wh[i, j]| times neuron j's mean on average, since activity is never negative.
  function meanState(W, n = 2000, seed = 1) {
    const X = bank(n, seed), m = new Float64Array(N);
    for (let s = 0; s < n; s++) {
      let h = new Float64Array(N);
      for (let t = 0; t < T; t++) {
        for (let j = 0; j < N; j++) m[j] += h[j];
        const g = new Float64Array(N), xt = X[s * T + t];
        for (let i = 0; i < N; i++) { let a = W.Wi[i] * xt; for (let j = 0; j < N; j++) a += W.Wh[i * N + j] * h[j]; g[i] = a > 0 ? a : 0; }
        h = g;
      }
    }
    return m.map(v => v / (n * T));
  }
  // push[i][j]: the mean size of what neuron j sends neuron i each step; push[i][16]: what the input sends it.
  function pushes(W, mean = meanState(W)) {
    return Array.from({ length: N }, (_, i) => [...Array.from({ length: N }, (_, j) => Math.abs(W.Wh[i * N + j]) * mean[j]), Math.abs(W.Wi[i]) * Math.sqrt(2 / Math.PI)]);
  }

  // The runner-up's exact part of each neuron at the end, per unit of its value, averaged over bank inputs by the
  // runner-up's position: part[n][q]. Each input's part is carried through the gates of the steps after it, as in
  // votes(); with the readout, W.Wo[q][n] * part[n][q] is neuron n's vote for the runner-up's own position.
  function runnerUpParts(W, n = 3000, seed = 3) {
    const X = bank(n, seed), sum = Array.from({ length: N }, () => new Float64Array(T)), cnt = new Float64Array(T);
    const C = new Float64Array(T * N), D = new Float64Array(T * N), h = new Float64Array(N), z = new Float64Array(N);
    for (let s = 0; s < n; s++) {
      const o = s * T;
      h.fill(0);
      for (let t = 0; t < T; t++) {
        for (let i = 0; i < N; i++) { let a = W.Wi[i] * X[o + t]; for (let j = 0; j < N; j++) a += W.Wh[i * N + j] * h[j]; z[i] = a; }
        for (let u = 0; u < t; u++) for (let i = 0; i < N; i++) {
          let a = 0;
          if (z[i] > 0) for (let j = 0; j < N; j++) a += W.Wh[i * N + j] * C[u * N + j];
          D[u * N + i] = a;
        }
        for (let i = 0; i < N; i++) { D[t * N + i] = z[i] > 0 ? W.Wi[i] * X[o + t] : 0; h[i] = z[i] > 0 ? z[i] : 0; }
        C.set(D.subarray(0, (t + 1) * N));
      }
      const q = label(X.subarray(o, o + T));
      for (let i = 0; i < N; i++) sum[i][q] += C[q * N + i] / X[o + q];
      cnt[q]++;
    }
    return sum.map(row => Array.from(row, (v, q) => v / cnt[q]));
  }

  // Eigenvalues of a small real matrix (array of rows), as [re, im] pairs: the characteristic polynomial by
  // Faddeev-LeVerrier, its roots by Durand-Kerner. Exact enough for the 5×5 blocks the clock chapter reads.
  function eigvals(A) {
    const n = A.length, c = [1];
    let Mk = A.map(r => r.map(() => 0));
    for (let k = 1; k <= n; k++) {
      const AM = A.map((r, i) => r.map((_, j) => r.reduce((s, a, l) => s + a * Mk[l][j], 0) + (i === j ? c[k - 1] : 0)));
      Mk = AM;
      const AMk = A.map((r, i) => r.map((_, j) => r.reduce((s, a, l) => s + a * Mk[l][j], 0)));
      c.push(-AMk.reduce((s, r, i) => s + r[i], 0) / k);
    }
    const mul = (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]];
    const div = (a, b) => { const d = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d]; };
    const p = z => c.reduce((s, ck) => { const m = mul(s, z); return [m[0] + ck, m[1]]; }, [0, 0]);
    let z = [[1, 0]];
    for (let k = 1; k < n; k++) z.push(mul(z[k - 1], [0.4, 0.9]));
    for (let it = 0; it < 500; it++) {
      z = z.map((zi, i) => { let d = [1, 0]; z.forEach((zj, j) => { if (j !== i) d = mul(d, [zi[0] - zj[0], zi[1] - zj[1]]); }); const q = div(p(zi), d); return [zi[0] - q[0], zi[1] - q[1]]; });
    }
    return z;
  }

  // Top-k eigenvectors of a symmetric n×n matrix (array of rows) by power iteration with deflation, largest first.
  function eig(A, k) {
    const n = A.length, out = [];
    for (let e = 0; e < k; e++) {
      let v = Array.from({ length: n }, (_, i) => Math.sin(i + 1 + e));
      for (let it = 0; it < 300; it++) {
        for (const u of out) { const d = v.reduce((s, x, i) => s + x * u[i], 0); v = v.map((x, i) => x - d * u[i]); }
        const w = A.map(r => r.reduce((s, a, j) => s + a * v[j], 0)), nv = Math.hypot(...w) || 1;
        v = w.map(x => x / nv);
      }
      out.push(v);
    }
    return out;
  }
  // The plane the readout reads most: top two right singular vectors of Wo, and the angle of a 16-vector in it.
  function plane(W) {
    const A = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => { let s = 0; for (let k = 0; k < T; k++) s += W.Wo[k * N + i] * W.Wo[k * N + j]; return s; }));
    const [e1, e2] = eig(A, 2);
    const coords = h => [h.reduce((s, x, i) => s + x * e1[i], 0), h.reduce((s, x, i) => s + x * e2[i], 0)];
    return { e1, e2, coords, angle: h => { const [a, b] = coords(h); return Math.atan2(b, a); }, row: k => W.Wo.subarray(k * N, k * N + N) };
  }

  window.M = { N, T, base: weights(window.DATA.weights), weights, clone, run, readout, order, label, argmax, margin, votes, rng, bank, accuracy, margins,
    meanState, pushes, runnerUpParts, eigvals, eig, plane };
})();
