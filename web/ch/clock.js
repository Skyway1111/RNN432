// Chapter 6: the clock. Two spikes (2 at p, 1 at q); the 16-number memory projected on the readout's top plane is a
// hand; each answer position reads along its own direction, a tick on the rim; the bar on each tick is that position's
// actual score from all 16 neurons, so the winner is always visible even when the flattened hand lands between ticks.
// Then where the turning comes from: the block of W_mem among n2 and the comparators, whose complex eigenvalue pair
// turns the memory by a fixed angle each step (research/findings/circuit.md, why the readout takes its shape).
Story.chapter('clock', {
  title: 'A clock that remembers when',
  hint: { text: 'Choose where the two numbers land', icon: 'controls', from: 2 },
  steps: [
    `<p>Give the network the simplest input there is: all zeros, except a <span class="max">2</span> (the largest) and a <span class="second">1</span> (the runner-up).</p>
     <p>Its memory is 16 numbers, too many to draw. So we flatten it onto a dial: the two directions in memory that the <b>output weights</b> care about most. After each input, the memory is a point on this dial, and the hand points at it.</p>`,
    `<p>Now the ticks. Each answer position reads the memory along its own direction, drawn as a tick on the rim, and positions 0 to 7 march around the dial in order, about <b data-fact="wheel_step"></b> apart.</p>
     <p>The bar on each tick is that position's full <b>score</b>, from all 16 neurons; the tallest bar is the answer. By the last step the hand points close to the <span class="second">runner-up's</span> tick, and that tick's bar is tallest. The network has built a <b>clock</b>: an angle that stores <i>when</i> something happened.</p>`,
    `<p>How does the hand know? If the runner-up arrives <b>after</b> the largest, its arrival starts the hand on a fixed path, so how far the hand has turned says how long ago it came.</p>
     <p>If it arrives <b>before</b>, it was the record at the time. When the largest shows up and beats it, the hand snaps to the angle for "the old record came this long ago." Pick other positions below and watch both cases.</p>`,
    `<p>Where does the turning come from? Look only at the memory weights among the record keeper and the four comparators: a block of 25 of the 256. Apply it once and it acts like a <b>turn</b>: the memory rotates by <b data-live="clock-angle"></b> and shrinks a little, to <b data-live="clock-mod"></b> of its size.</p>
     <p>The figure follows one point through repeated turns. Eight of them go almost all the way around, which is why positions 0 to 7 sit evenly around the dial.</p>
     <details><summary>In linear-algebra terms</summary><p>The block's eigenvalues are drawn as dots. Two of them are a complex pair, r·e<sup>±iθ</sup>; a matrix with such a pair turns the plane they span by θ each time it is applied, and scales it by r.</p></details>`,
    `<p>It's close, not exact: the flattened hand often lands tens of degrees from its tick, and the other 14 directions of memory finish the job. On these clean inputs the network is right <b data-fact="wheel_acc"></b> of the time.</p>
     <p>Positions 8 and 9 don't fit on the dial: one more turn would land position 8 on position 0. So the network gives them their own neurons: <span class="timer">n5</span> lights up for position 8, and <span class="core">n4</span> vetoes 9. Press <b>Random input</b> to see ten numbers all pushing the hand at once.</p>`,
  ],
  mount(fig, bar) {
    const P = M.plane(M.base), BLK = [2, 7, 6, 1, 8];
    const ev = M.eigvals(BLK.map(i => BLK.map(j => M.base.Wh[i * 16 + j])));
    const turn = ev.reduce((b, z) => (Math.atan2(z[1], z[0]) > Math.atan2(b[1], b[0]) ? z : b));
    const theta = Math.atan2(turn[1], turn[0]), rho = Math.hypot(turn[0], turn[1]);
    const sec = fig.closest('.chapter');
    sec.querySelectorAll('[data-live="clock-angle"]').forEach(e => (e.textContent = `${(theta * 180 / Math.PI).toFixed(0)}°`));
    sec.querySelectorAll('[data-live="clock-mod"]').forEach(e => (e.textContent = `${(100 * rho).toFixed(0)}%`));
    let p = 2, q = 5, x = null, stage = 0, tt = 10, messy = false, grow = 1;
    const tok = {}, g = M.rng(5);
    const setX = () => { if (!messy) { x = new Array(10).fill(0); x[p] = 2; x[q] = 1; } };
    setX();
    let r = M.run(M.base, x);

    function eigenView(ctx, w, h) {
      const cx = w * 0.5, cy = h * 0.53, R = Math.min(w * 0.36, h * 0.36);
      V.title(ctx, 'One point, turned again and again by the block', 28, 40);
      V.text(ctx, 'dots: the block\'s eigenvalues;  numbers: how many turns', 28, 62, { size: 13, face: 'serif', style: 'italic' });
      ctx.strokeStyle = V.C.axis; ctx.lineWidth = 1;
      V.hline(ctx, cx - R * 1.2, cx + R * 1.2, cy); V.vline(ctx, cx, cy - R * 1.2, cy + R * 1.2);
      ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
      V.text(ctx, 'size 1', cx + R * 0.72 + 8, cy - R * 0.72 - 6, { size: 12, face: 'serif', style: 'italic' });
      // the spiral: the pair's own coordinates after k turns, r^k at angle k theta
      const upto = Math.floor(grow * 8 + 1e-9), pts = Array.from({ length: 9 }, (_, k) => [cx + R * rho ** k * Math.cos(k * theta), cy - R * rho ** k * Math.sin(k * theta)]);
      ctx.strokeStyle = V.alpha(V.C.ink, 0.55); ctx.lineWidth = 1.6; ctx.beginPath();
      for (let s = 0; s <= upto * 12; s++) { const k = s / 12, rr = R * rho ** k; const X = cx + rr * Math.cos(k * theta), Y = cy - rr * Math.sin(k * theta); s ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
      ctx.stroke();
      pts.slice(0, upto + 1).forEach(([X, Y], k) => {
        ctx.fillStyle = V.C.ink; ctx.beginPath(); ctx.arc(X, Y, 3.5, 0, 2 * Math.PI); ctx.fill();
        const a = k * theta, lx = cx + (R * rho ** k + 18) * Math.cos(a), ly = cy - (R * rho ** k + 18) * Math.sin(a);
        V.text(ctx, String(k), lx, ly, { size: 14, align: 'center', base: 'middle', color: V.C.ink, weight: 600, face: 'serif' });
      });
      // the arc of one turn, between turns 0 and 1
      ctx.strokeStyle = V.C.core; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(cx, cy, R * 0.32, 0, -theta, true); ctx.stroke();
      V.text(ctx, `${(theta * 180 / Math.PI).toFixed(1)}° per turn`, cx + R * 0.36 * Math.cos(theta / 2) + 8, cy - R * 0.36 * Math.sin(theta / 2), { size: 13, color: V.C.ink, base: 'middle', weight: 600 });
      // the eigenvalues themselves
      for (const [re, im] of ev) {
        const pair = Math.abs(im) > 1e-6;
        ctx.fillStyle = pair ? V.C.core : V.C.timer; ctx.strokeStyle = V.C.surface; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(cx + R * re, cy - R * im, pair ? 7 : 5.5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      }
      V.legend(ctx, 28, h - 26, [['the turning pair', V.C.core], ['the block\'s other eigenvalues', V.C.timer]]);
    }

    const cv = V.canvas(fig, (ctx, w, h) => {
      if (stage === 3) return eigenView(ctx, w, h);
      const o = M.order(x), lab = o[1], step = Math.min(10, Math.floor(tt));
      // the input tape
      const slot = Math.min(52, (w - 60) / 10), tx = (w - slot * 10) / 2;
      V.text(ctx, 'The ten inputs', tx, 16, { size: 13, color: V.C.ink2, face: 'serif', style: 'italic' });
      for (let t = 0; t < 10; t++) {
        const rank = o.indexOf(t), cx = tx + slot * (t + 0.5), on = t < tt, now = t === step - 1 && tt < 10, lit = x[t] || messy;
        ctx.fillStyle = on && lit ? V.rankColor(rank) : V.C.surface2;
        ctx.globalAlpha = on ? 1 : 0.55;
        V.roundRect(ctx, cx - slot * 0.4, 24, slot * 0.8, 30, 5); ctx.fill(); ctx.globalAlpha = 1;
        if (now) { ctx.strokeStyle = V.C.ink; ctx.lineWidth = 2; ctx.strokeRect(cx - slot * 0.4, 24, slot * 0.8, 30); }
        V.text(ctx, messy ? V.fmt(x[t], 1) : String(x[t]), cx, 39, { color: on && lit && rank < 3 ? V.C.surface : V.C.ink2, size: 12, align: 'center', base: 'middle', weight: 600 });
        V.text(ctx, V.xi(t), cx, 70, { size: 11.5, align: 'center', color: V.C.ink2 });
      }
      const ans = M.argmax(r.logits), good = ans === lab;
      if (tt >= 10) V.chip(ctx, `Network says position ${ans} ${good ? '✓' : '✗'}   runner-up at ${lab}`, w / 2, 102, { color: good ? V.C.right : V.C.wrong, align: 'center' });
      else V.text(ctx, step ? `Step ${step} of 10: ${V.xi(step - 1)} has arrived` : 'Before the first input', w / 2, 102, { size: 13, align: 'center', color: V.C.ink2, base: 'middle' });
      // the dial
      const cx = w / 2, cy = 124 + (h - 124 - 44) / 2, R = Math.min(w * 0.3, (h - 124 - 44) / 3.3);
      const at = (a, rr) => [cx + rr * Math.cos(a), cy - rr * Math.sin(a)];
      ctx.strokeStyle = V.C.axis; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI); ctx.stroke();
      const hv = r.h[step], L = step > 0 ? M.readout(M.base, hv) : new Float64Array(10), lmax = Math.max(1e-9, ...Array.from(L, Math.abs)), barMax = R * 0.42;
      if (stage >= 1) for (let k = 0; k < 10; k++) {
        const a = P.angle(P.row(k)), win = tt >= 10 && k === ans, isLab = k === lab;
        const col = win ? (good ? V.C.right : V.C.wrong) : isLab && tt >= 10 ? V.C.second : V.C.muted;
        // the tick, and the score as a bar along it: outward for a positive score, inward for a negative one
        const [x1, y1] = at(a, R - 6), [x2, y2] = at(a, R + 6);
        ctx.strokeStyle = col; ctx.lineWidth = win || isLab ? 3 : 2; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        if (step > 0) {
          const len = (L[k] / lmax) * barMax, [bx0, by0] = at(a, R + (len >= 0 ? 6 : 0)), [bx1, by1] = at(a, R + 6 + len);
          ctx.strokeStyle = len >= 0 ? V.alpha(col, win ? 1 : 0.6) : V.alpha(V.C.neg, 0.35); ctx.lineWidth = win ? 9 : 6; ctx.lineCap = 'round';
          ctx.beginPath(); ctx.moveTo(bx0, by0); ctx.lineTo(bx1, by1); ctx.stroke(); ctx.lineCap = 'butt';
        }
        const [lx, ly] = at(a, R + 12 + barMax + 14);
        V.text(ctx, String(k), lx, ly, { color: win || isLab ? V.C.ink : V.C.ink2, size: win || isLab ? 17 : 13.5, align: 'center', base: 'middle', weight: win || isLab ? 700 : 400 });
      }
      // the hand's path so far, on an inner ring, and the hand
      const ang = r.h.slice(1).map(v => (Math.hypot(...v) > 1e-9 ? P.angle(v) : null));
      const ring = R * 0.6;
      let prev = null;
      for (let t = 0; t < step; t++) {
        if (ang[t] === null) continue;
        const [px, py] = at(ang[t], ring);
        if (prev) { ctx.strokeStyle = V.alpha(V.C.ink2, 0.35); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(px, py); ctx.stroke(); }
        ctx.fillStyle = V.alpha(V.C.ink2, 0.65); ctx.beginPath(); ctx.arc(px, py, 2.5, 0, 2 * Math.PI); ctx.fill();
        prev = [px, py];
      }
      const last = ang.slice(0, step).filter(a => a !== null).pop();
      if (last !== undefined) {
        const [hx, hy] = at(last, R * 0.86);
        ctx.strokeStyle = V.C.ink; ctx.fillStyle = V.C.ink; ctx.lineWidth = 3; V.arrow(ctx, cx, cy, hx, hy, 11);
      }
      ctx.fillStyle = V.C.ink; ctx.beginPath(); ctx.arc(cx, cy, 4, 0, 2 * Math.PI); ctx.fill();
      if (h < 520) return;  // a phone's short figure: the dial alone
      if (stage >= 1) V.text(ctx, 'Rim: the ten answer positions, each with its score', 16, h - 34, { size: 12.5, face: 'serif', style: 'italic' });
      V.text(ctx, 'Inside: where the hand has pointed after each input', 16, h - 14, { size: 12.5, face: 'serif', style: 'italic' });
    });

    function play() { r = M.run(M.base, x); V.tween(tok, 3200, u => { tt = u * 10; cv.redraw(); }); }
    const opts = Array.from({ length: 10 }, (_, k) => [k, String(k)]);
    const sp = V.select(bar, 'Largest at', opts, p, v => { p = +v; if (q === p) { q = (p + 1) % 10; sq.value = q; } messy = false; setX(); play(); });
    const sq = V.select(bar, 'runner-up at', opts, q, v => { q = +v; if (q === p) { p = (q + 1) % 10; sp.value = p; } messy = false; setX(); play(); });
    const replay = V.button(bar, 'Replay', () => (stage === 3 ? V.tween(tok, 2400, u => { grow = u; cv.redraw(); }) : play()));
    V.button(bar, 'Random input', () => { messy = true; x = Array.from({ length: 10 }, g); play(); });

    return {
      step(k) {
        const was = stage, enter = was === 0 && k === 0;
        stage = k;
        replay.textContent = k === 3 ? 'Turn again' : 'Replay';
        if (enter) play();
        else if (k === 3 && was !== 3) V.tween(tok, 2400, u => { grow = u; cv.redraw(); });
        else { if (was === 3) tt = 10; cv.redraw(); }
      },
    };
  },
});
