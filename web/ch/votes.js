// Chapter 7: votes. The final memory is exactly the sum of one piece per input; read out, each input votes for each
// position. The margin between the answer and the third-largest's position splits among the top three, differently
// on every input; on average it is a tug-of-war between the 2nd and the 3rd (results/experiments/margin_by_rank.txt).
Story.chapter('votes', {
  title: 'Ten votes that add up',
  hint: { text: 'Push the third-largest up with the slider', icon: 'controls', from: 3 },
  steps: [
    `<p>Here's a sharper lens. Once you know which neurons were clipped at each step, the rest is only multiplying and adding, with no constant. So after the last step the memory splits <b>exactly</b> into ten pieces, one left behind by each input. Read each piece out through the output weights and you get each input's <b>votes</b> for each answer position.</p>
     <p>Rows are inputs, columns are answer positions; blue is a vote for, red a vote against. Add up a column and you get that position's score, exactly.</p>`,
    `<p>Now look at one race: the right answer against the <span class="third">third-largest's</span> position, where most of the network's mistakes point. The right answer's score minus its rival's is the <b>margin</b>: positive means the network is right, and the bigger it is, the more sure it is.</p>
     <p>Each bar is one input's pull on that margin. Nearly all the pull comes from the three largest numbers, and how they split it changes from input to input: deal a few and watch.</p>`,
    `<p>Average the pulls over many random inputs, each divided by its own number, and a rule appears:</p>
     <p class="rule"><span data-fact="coef_S"></span> × <span class="second">runner-up</span> <span data-fact="coef_T"></span> × <span class="third">third</span> <span data-fact="coef_M"></span> × <span class="max">largest</span></p>
     <p>The runner-up and the third pull almost equally hard in opposite directions: a <b>tug-of-war</b>. The whiskers show how far single inputs stray from the average, and it's a long way. The average is the rule; which neurons each input's clips switched off sets the details.</p>`,
    `<p>Use the slider to push the third-largest up toward the runner-up. Where they meet, the right answer jumps from one position to the other, so the network's answer has to flip somewhere near there. <a data-ch="ties"></a> shows why it can't flip at exactly the right place.</p>
     <details><summary>Why the split changes</summary><p>The votes add up to the scores exactly. But each input's piece depends on which neurons were clipped to zero along the way, and that depends on all the other inputs. Near a tie, those details decide the answer.</p></details>`,
  ],
  mount(fig, bar) {
    let x0 = [0.31, -1.2, 1.54, 0.1, -0.4, 1.05, -0.9, 0.66, -0.1, 0.4], x = x0.slice(), stage = 0, u = 0;
    const g = M.rng(13);
    let avg = null;
    function average() {
      const n = 2000, X = M.bank(n, 17), sum = new Float64Array(10), sq = new Float64Array(10);
      for (let s = 0; s < n; s++) {
        const xi = X.subarray(s * 10, s * 10 + 10), o = M.order(xi), Vt = M.votes(M.base, xi).votes;
        o.forEach((t, rank) => { const c = (Vt[t][o[1]] - Vt[t][o[2]]) / xi[t]; sum[rank] += c; sq[rank] += c * c; });
      }
      return Array.from(sum, (s, k) => ({ mean: s / n, sd: Math.sqrt(Math.max(0, sq[k] / n - (s / n) ** 2)) }));
    }
    // an input's name in ink, with its rank's color as a dot beside it
    const rowLabel = (ctx, t, rank, x, y) => {
      V.text(ctx, V.xi(t), x, y, { size: 12.5, align: 'right', base: 'middle', color: V.C.ink, weight: rank < 3 ? 700 : 400 });
      if (rank < 3) { ctx.fillStyle = V.rankColor(rank); ctx.beginPath(); ctx.arc(x - 26, y, 4, 0, 2 * Math.PI); ctx.fill(); }
    };
    const cv = V.canvas(fig, (ctx, w, h) => {
      const r = M.run(M.base, x), V10 = M.votes(M.base, x, r).votes, o = M.order(x), lab = o[1], rival = o[2];
      const hx = 72, hy = 82, hw = (stage >= 1 ? w * 0.48 : w) - 112, bx = w * 0.52 + 80;  // bx: the right panel's left edge
      if (stage < 2) {
        const s = Math.min(hw / 10, (h - 150) / 10), vmax = Math.max(...V10.flatMap(v => Array.from(v, Math.abs)));
        V.title(ctx, 'Each input\'s vote for each answer position', hx, hy - 40, 15, stage >= 1 ? bx - hx - 16 : Infinity);
        V.legend(ctx, hx, h - 26, [['vote for', V.C.pos], ['vote against', V.C.neg]]);
        for (let k = 0; k < 10; k++) V.text(ctx, String(k), hx + k * s + s / 2, hy - 10, { size: 12, align: 'center', color: k === lab || (k === rival && stage >= 1) ? V.C.ink : V.C.muted, weight: k === lab || (k === rival && stage >= 1) ? 700 : 400 });
        V.text(ctx, 'position', hx + 10 * s + 8, hy - 10, { size: 11.5 });
        for (let t = 0; t < 10; t++) {
          rowLabel(ctx, t, o.indexOf(t), hx - 10, hy + t * s + s / 2);
          for (let k = 0; k < 10; k++) { ctx.fillStyle = V.diverge(V10[t][k], vmax * 0.6); V.rect(ctx, hx + k * s + 1, hy + t * s + 1, s - 2, s - 2); }
        }
        if (stage >= 1) for (const k of [lab, rival]) { ctx.strokeStyle = k === lab ? V.C.second : V.C.third; ctx.lineWidth = 2.5; ctx.strokeRect(hx + k * s, hy, s, 10 * s); }
      } else {
        avg = avg || average();
        const bw = hw / 10, zero = hy + (h - hy - 90) / 2;
        const amax = Math.max(...avg.map(a => Math.abs(a.mean) + a.sd)), sc = (h - hy - 110) / 2 / amax;
        V.title(ctx, 'Average pull, by rank', hx, hy - 40);
        V.text(ctx, 'per unit of each number; 2,000 inputs; whiskers: one sd', hx, hy - 20, { size: 12.5, face: 'serif', style: 'italic', max: bx - hx - 16 });
        ctx.strokeStyle = V.C.axis; ctx.lineWidth = 1; V.hline(ctx, hx, hx + hw, zero);
        avg.forEach((a, k) => {
          const cx = hx + bw * (k + 0.5);
          ctx.fillStyle = V.rankColor(k); V.bar(ctx, cx - bw * 0.3, zero, bw * 0.6, a.mean * sc, 3);
          ctx.strokeStyle = V.C.ink2; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx, zero - (a.mean + a.sd) * sc); ctx.lineTo(cx, zero - (a.mean - a.sd) * sc); ctx.stroke();
          if (Math.abs(a.mean) >= 0.5) V.text(ctx, V.sfmt(a.mean, 0), cx, a.mean >= 0 ? zero - (a.mean + a.sd) * sc - 6 : zero - (a.mean - a.sd) * sc + 16, { size: 12, align: 'center', color: V.C.ink, weight: 600 });
          V.text(ctx, String(k + 1), cx, h - 50, { size: 12, align: 'center', color: k < 3 ? V.C.ink : V.C.muted, weight: k < 3 ? 700 : 400 });
        });
        V.text(ctx, 'rank, largest first', hx + hw / 2, h - 32, { size: 12, align: 'center', face: 'serif', style: 'italic' });
        V.legend(ctx, hx, h - 12, [['largest', V.C.max], ['runner-up', V.C.second], ['third', V.C.third], ['the rest', V.C.other]]);
      }
      if (stage >= 1) {
        const bw = w * 0.48 - 124, mid = bx + bw / 2, contrib = V10.map(v => v[lab] - v[rival]);
        const cmax = Math.max(1, ...contrib.map(Math.abs)), bh = Math.min(26, (h - 170) / 10);
        V.title(ctx, 'Each input\'s pull on the margin', bx, hy - 40);
        V.text(ctx, `score of ${lab} minus score of ${rival}`, bx, hy - 20, { size: 12.5, face: 'serif', style: 'italic' });
        ctx.strokeStyle = V.C.axis; ctx.lineWidth = 1; V.vline(ctx, mid, hy - 4, hy + 10 * bh * 1.25);
        contrib.forEach((c, t) => {
          const y = hy + t * bh * 1.25, len = (c / cmax) * bw / 2, rank = o.indexOf(t);
          rowLabel(ctx, t, rank, bx - 10, y + bh / 2);
          ctx.fillStyle = V.rankColor(rank); V.roundRect(ctx, Math.min(mid, mid + len), y + 2, Math.abs(len), bh - 4, 3); ctx.fill();
          if (Math.abs(c) >= 0.5) V.text(ctx, V.sfmt(c, 0), mid + len + (len >= 0 ? 6 : -6), y + bh / 2, { size: 11.5, align: len >= 0 ? 'left' : 'right', base: 'middle', color: V.C.ink2 });
        });
        const m = r.logits[lab] - r.logits[rival], ans = M.argmax(r.logits), good = ans === lab;
        V.chip(ctx, `Margin ${V.sfmt(m, 1)}: the network says ${ans} ${good ? '✓' : '✗'}`, bx, hy + 10 * bh * 1.25 + 30, { color: good ? V.C.right : V.C.wrong });
      }
    });
    const o0 = () => M.order(x0);
    const read = V.readout(bar);
    const apply = () => {
      const [, q, r3] = o0(); x = x0.slice(); x[r3] = V.lerp(x0[r3], x0[q] + 0.25, u);
      read.textContent = `third ${V.fmt(x[r3])}, runner-up ${V.fmt(x0[q])}`; cv.redraw();
    };
    V.button(bar, 'Deal new numbers', () => { x0 = Array.from({ length: 10 }, g); u = 0; slider.value = 0; apply(); });
    const slider = V.slider(bar, 'Push the third up', { min: 0, max: 1, step: 0.002, value: 0, width: '12rem' }, v => { u = v; apply(); });
    bar.appendChild(read);
    apply();
    return { step(k) { stage = k; cv.redraw(); } };
  },
});
