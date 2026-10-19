// Chapter 9: ties. Slide one input: the scores move smoothly but the right answer jumps at each tie, so the network
// must be wrong in a sliver beside it. Slide two: errors hug the tie lines.
Story.chapter('ties', {
  title: 'Why it must fail at ties',
  hint: { text: 'Press New slice for other inputs', icon: 'controls', from: 2 },
  steps: [
    `<p>Take one input and slide it from low to high, keeping the other nine fixed. The curves are the ten answer scores. The highest curve is the network's answer at that value.</p>
     <p>Every curve is <b>unbroken</b>: a clip can bend it, but nothing makes it jump. The network is built from multiplying, adding, and clipping, so a tiny nudge to an input only nudges the scores.</p>`,
    `<p>But the right answer does jump. The moment the sliding number passes the runner-up, the runner-up <i>changes</i>, and the right answer <b>jumps</b> to a new position.</p>
     <p>Unbroken curves cannot jump. They need some room to cross, and in that room the network is wrong: the red sliver, where the margin is on the wrong side of zero. No network whose scores move continuously can avoid it. The best it can do is make the sliver thin.</p>`,
    `<p>Now slide two inputs at once. Each pixel is one input: pale where the network is right, red where it's wrong. The thin lines mark ties, where two of the numbers are equal.</p>
     <p>The errors hug the ties. Away from them the network is almost never wrong.</p>`,
    `<p>Across random inputs, when the runner-up and the third-largest are within 0.05 of each other, the network is right only <b data-fact="acc_st_near"></b> of the time. When they're far apart, <b data-fact="acc_st_far"></b>. Of all its mistakes, <b data-fact="err_third"></b> point to the <span class="third">third-largest</span>.</p>
     <p>Remember trying to fool it in <a data-ch="game"></a>? This is why near-equal numbers worked. Press <b>New slice</b> to change the other inputs.</p>`,
    `<p>Notice what the network is <i>not</i> confused about: order. Take ten typical numbers, well spread out, and feed them in every possible order, all <b data-fact="ord_total"></b> of them. It is wrong on only <b data-fact="ord_wrong"></b>, and every one of those has the runner-up first and the largest last.</p>
     <p>So most of its mistakes come from how close two numbers are, not from the order they arrive in.</p>`,
  ],
  mount(fig, bar) {
    let x = [0.31, -1.2, 1.54, 0.1, -0.4, 1.05, -0.9, 0.66, -0.1, 0.4];
    let stage = 0, grow = 1, map = null;
    const tok = {}, g = M.rng(9), RES = 160;
    const pick = () => { const o = M.order(x); return { a: o[2], b: o[1] }; };
    let { a, b } = pick();
    const lo = -1.5, hi = 2.5;

    function slice1() {
      const n = 400, out = [];
      for (let i = 0; i <= n; i++) {
        const v = lo + (hi - lo) * i / n, y = x.slice(); y[a] = v;
        const L = M.run(M.base, y).logits;
        out.push({ v, L: Array.from(L), ans: M.argmax(L), lab: M.label(y) });
      }
      return out;
    }
    function slice2() {
      const img = new Uint8Array(RES * RES);
      for (let i = 0; i < RES; i++) for (let j = 0; j < RES; j++) {
        const y = x.slice(); y[a] = lo + (hi - lo) * (i + 0.5) / RES; y[b] = lo + (hi - lo) * (j + 0.5) / RES;
        img[j * RES + i] = M.argmax(M.run(M.base, y).logits) === M.label(y) ? 0 : 1;
      }
      return img;
    }
    let s1 = slice1();

    const cv = V.canvas(fig, (ctx, w, h) => {
      if (stage < 2) {
        const left = 104, right = 64, top = 66, bottom = 122, pw = w - left - right, ph = h - top - bottom;
        const all = s1.flatMap(o => o.L), Lmin = Math.min(...all), Lmax = Math.max(...all);
        const X = v => left + (v - lo) / (hi - lo) * pw, Y = L => top + (1 - (L - Lmin) / (Lmax - Lmin)) * ph;
        const upto = Math.floor(grow * (s1.length - 1));
        if (stage >= 1) for (let i = 0; i < upto; i++) if (s1[i].ans !== s1[i].lab) {
          ctx.fillStyle = V.alpha(V.C.wrong, 0.45); ctx.fillRect(X(s1[i].v), top, X(s1[i + 1].v) - X(s1[i].v) + 0.5, ph);
        }
        const o = M.order(x), hot = { [o[0]]: V.C.max, [b]: V.C.second, [a]: V.C.third };
        V.title(ctx, 'The ten scores', left, top - 26);
        for (let k = 0; k < 10; k++) {
          ctx.strokeStyle = hot[k] || V.alpha(V.C.ink2, 0.25); ctx.lineWidth = hot[k] ? 2.5 : 1;
          ctx.beginPath(); for (let i = 0; i <= upto; i++) { const px = X(s1[i].v), py = Y(s1[i].L[k]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); } ctx.stroke();
          if (hot[k] && upto > 0) V.text(ctx, `score of ${k}`, X(s1[upto].v) + 5, Y(s1[upto].L[k]), { size: 12, color: V.C.ink, base: 'middle', weight: 600 });
        }
        // the two tie lines, labelled at opposite ends so the labels cannot collide
        for (const [k, c] of [[o[0], V.C.max], [b, V.C.second]]) {
          ctx.strokeStyle = c; ctx.setLineDash([4, 4]); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(x[k]), top); ctx.lineTo(X(x[k]), top + ph); ctx.stroke(); ctx.setLineDash([]);
          V.text(ctx, k === b ? 'tie with the runner-up' : 'tie with the largest', X(x[k]), k === b ? top - 8 : top + ph - 6, { size: 12, color: V.C.ink, align: 'center', face: 'serif', style: 'italic' });
        }
        // the right answer and the network's as strips under the plot
        const sy = top + ph + 20;
        V.text(ctx, 'Right answer', left - 8, sy + 8, { size: 11.5, align: 'right', base: 'middle' });
        V.text(ctx, 'Network says', left - 8, sy + 32, { size: 11.5, align: 'right', base: 'middle' });
        for (let i = 0; i < upto; i++) {
          const x0 = X(s1[i].v), x1 = X(s1[i + 1].v) + 0.5;
          ctx.fillStyle = s1[i].lab === b ? V.C.second : s1[i].lab === a ? V.C.third : V.C.max; ctx.fillRect(x0, sy, x1 - x0, 16);
          ctx.fillStyle = s1[i].ans === b ? V.C.second : s1[i].ans === a ? V.C.third : s1[i].ans === o[0] ? V.C.max : V.C.other; ctx.fillRect(x0, sy + 24, x1 - x0, 16);
          // under the network's strip, a mark wherever it disagrees with the right answer: a sliver too thin to see above
          if (stage >= 1 && s1[i].ans !== s1[i].lab) { ctx.fillStyle = V.C.wrong; ctx.fillRect(x0 - 2, sy + 44, x1 - x0 + 4, 4); }
        }
        if (stage >= 1 && s1.slice(0, upto).some(o2 => o2.ans !== o2.lab)) V.text(ctx, 'red mark: the network is wrong ✗', left + pw, sy + 62, { size: 12, align: 'right', color: V.C.ink });
        V.text(ctx, `Sliding ${V.xi(a)}, the third-largest, from ${V.fmt(lo, 1)} to ${V.fmt(hi, 1)}`, left + pw / 2, h - 14, { size: 13, align: 'center', color: V.C.ink2, face: 'serif', style: 'italic' });
        V.legend(ctx, left + 150, top - 30, [['largest', V.C.max], ['runner-up', V.C.second], ['third-largest', V.C.third], ['wrong', V.C.wrong]]);
      } else {
        if (!map) map = slice2();
        const side = Math.min(w - 130, h - 110), ox = (w - side) / 2 + 24, oy = 36, cell = side / RES;
        for (let j = 0; j < RES; j++) for (let i = 0; i < RES; i++) {
          ctx.fillStyle = map[j * RES + i] ? V.C.wrong : V.C.surface2;
          ctx.fillRect(ox + i * cell, oy + side - (j + 1) * cell, cell + 0.5, cell + 0.5);
        }
        const P = v => (v - lo) / (hi - lo) * side, o = M.order(x), others = [o[0], ...o.slice(3)].filter(k => k !== a && k !== b);
        ctx.lineWidth = 1;
        const vline = (v, c) => { ctx.strokeStyle = c; ctx.beginPath(); ctx.moveTo(ox + P(v), oy); ctx.lineTo(ox + P(v), oy + side); ctx.stroke(); };
        const hline = (v, c) => { ctx.strokeStyle = c; ctx.beginPath(); ctx.moveTo(ox, oy + side - P(v)); ctx.lineTo(ox + side, oy + side - P(v)); ctx.stroke(); };
        for (const k of others.slice(0, 3)) { vline(x[k], V.alpha(V.C.ink2, 0.35)); hline(x[k], V.alpha(V.C.ink2, 0.35)); }
        ctx.strokeStyle = V.alpha(V.C.ink, 0.7); ctx.beginPath(); ctx.moveTo(ox, oy + side); ctx.lineTo(ox + side, oy); ctx.stroke();
        let wrong = 0; for (const v of map) wrong += v;
        V.text(ctx, `${V.xi(a)}, the third-largest`, ox + side / 2, oy + side + 24, { size: 13, align: 'center', color: V.C.ink2, face: 'serif', style: 'italic' });
        ctx.save(); ctx.translate(ox - 20, oy + side / 2); ctx.rotate(-Math.PI / 2); V.text(ctx, `${V.xi(b)}, the runner-up`, 0, 0, { size: 13, align: 'center', color: V.C.ink2, face: 'serif', style: 'italic' }); ctx.restore();
        V.text(ctx, `Red: wrong, ${(100 * wrong / map.length).toFixed(1)}% of this square. Lines: ties.`, ox + side / 2, oy + side + 48, { size: 12.5, align: 'center' });
      }
    });

    V.button(bar, 'New slice', () => { x = Array.from({ length: 10 }, g); ({ a, b } = pick()); s1 = slice1(); map = null; cv.redraw(); });
    V.button(bar, 'Replay', () => { if (stage < 2) V.tween(tok, 1500, u => { grow = u; cv.redraw(); }); });
    return {
      step(k) {
        const enter = stage === 0 && k === 0; stage = k;
        if (enter) V.tween(tok, 1500, u => { grow = u; cv.redraw(); }); else cv.redraw();
      },
    };
  },
});
