// The scroll engine. A chapter registers {title, steps: [html], hint, mount(fig, bar) -> {step(k)}, after(el)}; a part
// registers {kicker, title, html} and sits between chapters. Story.start() lays everything out in registration order,
// builds the hero index and the rail, mounts each figure when it nears the screen, and calls step(k) as the step whose
// text crosses the middle of the screen changes. A chapter's hint names its gesture on the figure until the first
// touch; its after(el), if any, fills a full-width section below its steps when it mounts.
// <a data-ch="ID"> in prose becomes a link reading "chapter N", so chapters can be reordered without retyping numbers.
// ?final mounts everything at its last step (for web/check.py).
(function () {
  const items = [], chapters = [];
  const final = new URLSearchParams(location.search).has('final');
  const ICON = { hover: '<svg viewBox="0 0 24 24"><path d="M4 4l7 17 2.5-6.5L20 12z"/></svg>',
    drag: '<svg viewBox="0 0 24 24"><path d="M12 3v18M8 7l4-4 4 4M8 17l4 4 4-4"/></svg>',
    spin: '<svg viewBox="0 0 24 24"><path d="M4 12a8 8 0 1 0 3-6.2M4 4v5h5"/></svg>',
    controls: '<svg viewBox="0 0 24 24"><path d="M4 8h10M18 8h2M4 16h4M12 16h8"/><circle cx="15" cy="8" r="2"/><circle cx="9" cy="16" r="2"/></svg>' };

  function chapter(id, spec) { const ch = { id, kind: 'chapter', ...spec }; items.push(ch); chapters.push(ch); }
  function part(id, spec) { items.push({ id, kind: 'part', ...spec }); }

  function facts(root) {
    for (const el of root.querySelectorAll('[data-fact]')) {
      const v = window.DATA.facts[el.dataset.fact];
      if (v === undefined) console.error('missing fact: ' + el.dataset.fact);
      el.textContent = v ?? '??';
    }
    for (const el of root.querySelectorAll('[data-ch]')) {
      const ch = chapters.find(c => c.id === el.dataset.ch);
      if (!ch) { console.error('missing chapter: ' + el.dataset.ch); continue; }
      el.textContent = 'chapter ' + ch.n;
      if (el.tagName === 'A') el.href = '#' + ch.id;
    }
  }

  function buildChapter(ch) {
    const sec = document.createElement('section');
    sec.className = 'chapter'; sec.id = ch.id;
    const hint = ch.hint ? `<div class="hint">${ICON[ch.hint.icon || 'drag']}<span>${ch.hint.text}</span></div>` : '';
    sec.innerHTML = `<div class="wrap"><div class="ch-head"><span class="num">${ch.n}</span><h2>${ch.title}</h2></div>
      <div class="ch-body"><div class="prose">${ch.steps.map((s, k) => `<div class="step" data-step="${k}">${s}</div>`).join('')}</div>
      <div class="figure"><div class="fig"></div><div class="fig-bar">${hint}</div></div></div>${ch.after ? '<div class="after"></div>' : ''}</div>`;
    ch.sec = sec; ch.current = -1; ch.ctl = null; ch.touched = false;
    ch.hintEl = sec.querySelector('.hint');
    if (ch.hintEl) {
      const fig = sec.querySelector('.fig'), bar = sec.querySelector('.fig-bar');
      const done = () => { ch.touched = true; ch.hintEl.classList.remove('show'); };
      fig.addEventListener('pointerdown', done);
      bar.addEventListener('input', done); bar.addEventListener('click', done);
      if (ch.hint.icon === 'hover') { let travel = 0, last = null; fig.addEventListener('pointermove', e => { if (last) travel += Math.hypot(e.clientX - last[0], e.clientY - last[1]); last = [e.clientX, e.clientY]; if (travel > 160) done(); }); }
    }
    return sec;
  }
  function buildPart(p) {
    const sec = document.createElement('section');
    sec.className = 'part'; sec.id = p.id;
    sec.innerHTML = `<div class="wrap"><p class="kicker">${p.kicker}</p><h2>${p.title}</h2>${p.html}</div>`;
    return sec;
  }

  function mount(ch) {
    if (ch.ctl) return;
    try { ch.ctl = ch.mount(ch.sec.querySelector('.fig'), ch.sec.querySelector('.fig-bar')) || {}; }
    catch (e) { console.error(`chapter ${ch.id}: ${e.stack || e}`); ch.ctl = {}; }
    if (ch.after) {
      try { ch.after(ch.sec.querySelector('.after')); } catch (e) { console.error(`chapter ${ch.id} after: ${e.stack || e}`); }
    }
    if (final) go(ch, ch.steps.length - 1);
  }

  function go(ch, k) {
    if (k === ch.current) return;
    ch.current = k;
    ch.sec.querySelectorAll('.step').forEach(s => s.classList.toggle('active', +s.dataset.step === k));
    if (ch.hintEl && !ch.touched) ch.hintEl.classList.toggle('show', k >= (ch.hint.from || 0));
    try { ch.ctl.step && ch.ctl.step(k); } catch (e) { console.error(`chapter ${ch.id} step ${k}: ${e.stack || e}`); }
  }

  function index(toc, rail) {
    let group = null, railGroup = false;
    for (const it of items) {
      if (it.kind === 'part') {
        group = document.createElement('div');
        group.innerHTML = `<div class="kicker"><small>${it.kicker}</small>${it.title}</div>`;
        toc.appendChild(group);
        if (railGroup) rail.insertAdjacentHTML('beforeend', '<div class="gap"></div>');
        railGroup = true;
        continue;
      }
      (group || toc).insertAdjacentHTML('beforeend', `<a href="#${it.id}"><span>${it.n}</span>${it.title}</a>`);
      rail.insertAdjacentHTML('beforeend', `<a href="#${it.id}" title="${it.n}. ${it.title}">${it.n}</a>`);
    }
  }

  function start() {
    const root = document.getElementById('story'), toc = document.getElementById('toc');
    const rail = document.createElement('nav'); rail.id = 'rail'; rail.setAttribute('aria-label', 'Chapters'); document.body.appendChild(rail);
    chapters.forEach((ch, i) => (ch.n = i + 1));
    items.forEach(it => root.appendChild(it.kind === 'part' ? buildPart(it) : buildChapter(it)));
    if (toc) index(toc, rail);
    facts(document);
    if (final) { chapters.forEach(mount); return; }
    const near = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) mount(chapters.find(c => c.sec === e.target)); }), { rootMargin: '100% 0px' });
    const mid = new IntersectionObserver(es => es.forEach(e => {
      if (!e.isIntersecting) return;
      const ch = chapters.find(c => c.sec.contains(e.target));
      mount(ch); go(ch, +e.target.dataset.step);
    }), { rootMargin: '-45% 0px -45% 0px' });
    chapters.forEach(ch => { near.observe(ch.sec); ch.sec.querySelectorAll('.step').forEach(s => mid.observe(s)); });
    // the rail lights the chapter under the middle of the screen, and hides over the hero
    const links = [...rail.querySelectorAll('a')];
    const where = new IntersectionObserver(es => es.forEach(e => {
      if (!e.isIntersecting) return;
      const i = chapters.findIndex(c => c.sec === e.target);
      links.forEach((a, k) => a.classList.toggle('on', k === i));
    }), { rootMargin: '-50% 0px -50% 0px' });
    chapters.forEach(ch => where.observe(ch.sec));
    const hero = document.getElementById('hero');
    if (hero) new IntersectionObserver(es => rail.classList.toggle('show', !es[0].isIntersecting), { rootMargin: '-30% 0px 0px' }).observe(hero);
    else rail.classList.add('show');
  }

  window.Story = { chapter, part, start, chapters };
})();
