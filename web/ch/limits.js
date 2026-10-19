// Chapter 13: the limit. The clip pattern (which cells are switched on at each step) of a near-tie input against its
// reference route, and the cascade one early sign flip sets off (research/findings/circuit.md, the cascade near a tie).
(function () {
  const ndtr = x => { const z = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.3275911 * z); const e = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z); return 0.5 * (1 + (x < 0 ? -e : e)); };
  const ndtri = p => { let lo = -8, hi = 8; for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (ndtr(m) < p) lo = m; else hi = m; } return (lo + hi) / 2; };
  // Same top three; the seven lower inputs, in position order, at evenly spaced quantiles of the law below the 3rd.
  function reference(x) { const o = M.order(x), T = x[o[2]], r = x.slice(); o.slice(3).sort((a, b) => a - b).forEach((t, j) => { r[t] = ndtri(ndtr(T) * (j + 0.5) / 7); }); return r; }
  const gates = x => M.run(M.base, x).pre.map(z => Array.from(z, v => v > 0));

  Story.chapter('limits', {
    title: 'What we can\'t explain yet',
    hint: { text: 'Press New near-tie input', icon: 'controls' },
    steps: [
      `<p>We can say what every neuron does. Can we say, for any single input, <i>why</i> the network answers what it does?</p>
       <p>Mostly. The grid shows which neurons are switched on (not clipped) at each step. Once you know that pattern, the network is plain multiply-and-add. Guess the pattern from where the largest number and the runner-up sit, and that arithmetic matches the network's answer on <b data-fact="route_agree"></b> of inputs. Near ties, where the mistakes are, it drops to <b data-fact="route_near"></b>. Outlined squares are clips that differ from the guessed pattern.</p>`,
      `<p>Near a tie, small things decide. Flip just the sign of the first input. Before anything else has arrived the memory is empty, so that one sign decides every clip of the first step.</p>
       <p>Then the change spreads: different clips change the memory, which changes later clips, step after step. Outlined squares are clips the flip changed. This chain reaction is the <b>cascade</b>.</p>`,
      `<p>Across near-tie inputs, the first input's sign flips every first-step clip on <b data-fact="sign_share"></b> of them, and then about <b data-fact="sign_with"></b> later clips change, against <b data-fact="sign_without"></b> without it.</p>
       <p>And there is no short list of what matters. Count the clips each input flips directly, by its own value, before any knock-on: the most common <b data-fact="cascade_sets50"></b> combinations of such flips cover half the near-tie inputs, and covering nine in ten takes <b data-fact="cascade_sets90"></b>.</p>`,
      `<p>That is the honest state of things. Every neuron has a job. The decision has a shape: a record, comparators, a clock, and timers that vote. The failures have causes: ties, and a shortcut. But predicting one near-tie decision still means running the cascade, clip by clip, which is almost the same as running the network.</p>
       <p>So the challenge this network was posted to pose is still open, even at 432 numbers: an explanation short enough to be an explanation, yet exact enough to call the close ones.</p>`,
    ],
    mount(fig, bar) {
      let x = [0.31, -1.2, 1.54, 0.1, -0.4, 1.05, -0.9, 1.02, -0.1, 0.4], stage = 0;
      const g = M.rng(51);
      const cv = V.canvas(fig, (ctx, w, h) => {
        const G = gates(x), other = stage >= 1 ? gates(x.map((v, t) => (t === 0 ? -v : v))) : gates(reference(x));
        const left = 84, top = 84, s = Math.min((w - left - 40) / 10, (h - top - 110) / 16);
        V.title(ctx, 'Which neurons are switched on after each input', left, top - 48);
        V.text(ctx, stage >= 1 ? `Outlined: clips changed by flipping ${V.xi(0)} from ${V.fmt(x[0])} to ${V.fmt(-x[0])}` : 'Outlined: clips that differ from the fixed pattern', left, top - 28, { size: 13, face: 'serif', style: 'italic' });
        for (let t = 0; t < 10; t++) V.text(ctx, V.xi(t), left + t * s + s / 2, top - 8, { size: 11.5, align: 'center', color: V.C.ink2 });
        let diff = 0;
        for (let i = 0; i < 16; i++) {
          V.text(ctx, `n${i}`, left - 10, top + i * s + s / 2, { size: 11.5, align: 'right', base: 'middle', color: V.C.ink });
          ctx.fillStyle = V.cellColor(i); ctx.beginPath(); ctx.arc(left - 40, top + i * s + s / 2, 4, 0, 2 * Math.PI); ctx.fill();
          for (let t = 0; t < 10; t++) {
            ctx.fillStyle = G[t][i] ? V.alpha(V.cellColor(i), 0.8) : V.C.surface2;
            V.rect(ctx, left + t * s + 2, top + i * s + 2, s - 4, s - 4);
            if (G[t][i] !== other[t][i]) { diff++; ctx.strokeStyle = V.C.ink; ctx.lineWidth = 2; ctx.strokeRect(left + t * s + 1, top + i * s + 1, s - 2, s - 2); }
          }
        }
        const o = M.order(x), ans = M.argmax(M.run(M.base, x).logits);
        V.text(ctx, `${diff} of 160 clips differ`, left, top + 16 * s + 30, { size: 13, color: V.C.ink, weight: 600 });
        V.text(ctx, `Runner-up ${V.fmt(x[o[1]])} at position ${o[1]}, third ${V.fmt(x[o[2]])} at ${o[2]}`, left, top + 16 * s + 52, { size: 12.5, color: V.C.ink2 });
        V.chip(ctx, `Network says ${ans} ${ans === o[1] ? '✓' : '✗'}`, w - 40, top + 16 * s + 40, { color: ans === o[1] ? V.C.right : V.C.wrong, align: 'right', size: 12 });
      });
      V.button(bar, 'New near-tie input', () => {
        x = Array.from({ length: 10 }, g); const o = M.order(x); x[o[2]] = x[o[1]] - 0.02; cv.redraw();
      });
      return { step(k) { stage = k; cv.redraw(); } };
    },
  });
})();
