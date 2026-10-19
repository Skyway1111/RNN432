"""The site gate: rebuild data.js, load the explainer in headless Chrome with every chapter at its last step, and fail
on a console error, a missing fact, a JS forward pass off az.model by more than 1e-9, accuracy over 20,000 inputs
slower than 150 ms, a measured number in the prose outside a data-fact or data-live span, a data-live span the page
left empty, or a field guide without its sixteen plates and their silencing tests. Screenshots of the hero, each
part, each chapter and the field guide go to web/_shots/ (gitignored) for review.

python -m web.check [CHAPTER ...] [--steps]   (--steps also shoots every step of the named chapters, motion reduced)
"""
import re
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
SHOTS = HERE / "_shots"
LITERAL = {"432", "256", "160", "20000", "200000"}  # structural constants the prose may state outright: weights, the page's samples


def numbers_outside_facts(page) -> list[str]:
    texts = page.evaluate("""() => [...document.querySelectorAll('.step, #hero, .part, #glossary, #coda, .guide')].map(s => {
        const c = s.cloneNode(true); c.querySelectorAll('[data-fact], [data-live], code, .mono, nav, footer').forEach(e => e.remove()); return c.textContent; })""")
    bad = []
    for t in texts:
        for tok in re.findall(r"\d[\d,]*(?:\.\d+)?", t):
            digits = tok.replace(",", "").replace(".", "").lstrip("0")
            if len(digits) >= 3 and tok.replace(",", "") not in LITERAL:
                bad.append(tok)
    return bad


def main() -> None:
    subprocess.run([sys.executable, "-m", "web.build"], check=True, stdout=subprocess.DEVNULL)
    steps = "--steps" in sys.argv
    only = {a for a in sys.argv[1:] if not a.startswith("--")}
    SHOTS.mkdir(exist_ok=True)
    errors, ok = [], True
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto((HERE / "index.html").as_uri() + "?final")
        page.wait_for_timeout(3000)  # the hero's bloom, and the field guide's silencing tests
        if not only:
            page.screenshot(path=str(SHOTS / "00-hero.png"))
            for pid in page.evaluate("() => [...document.querySelectorAll('.part')].map(s => s.id)"):
                page.evaluate(f"document.querySelector('#{pid}').scrollIntoView({{block: 'center', behavior: 'instant'}})")
                page.wait_for_timeout(250)
                page.screenshot(path=str(SHOTS / f"00-part-{pid}.png"))
        err = page.evaluate("""() => { let e = 0; DATA.parity.x.forEach((x, i) => M.run(M.base, x).logits.forEach((v, k) =>
            e = Math.max(e, Math.abs(v - DATA.parity.logits[i][k])))); return e; }""")
        ms = page.evaluate("() => { M.accuracy(M.base, 20000); const t = performance.now(); M.accuracy(M.base, 20000); return performance.now() - t; }")
        stray = numbers_outside_facts(page)
        empty = page.evaluate("() => [...document.querySelectorAll('[data-live]')].filter(e => !e.textContent.trim()).map(e => e.dataset.live)")
        plates = page.evaluate("() => [document.querySelectorAll('.plate').length, document.querySelectorAll('.plate .sil b').length]")
        ids = page.evaluate("() => Story.chapters.map(c => c.id)")
        for i, cid in enumerate(ids, 1):
            if only and cid not in only:
                continue
            page.evaluate(f"document.querySelector('#{cid} .step:last-child').scrollIntoView({{block: 'center', behavior: 'instant'}})")
            page.wait_for_timeout(250)
            page.screenshot(path=str(SHOTS / f"{i:02d}-{cid}.png"))
            after = page.locator(f"#{cid} .after")
            if after.count():
                after.screenshot(path=str(SHOTS / f"{i:02d}-{cid}-after.png"))
        if steps:
            page.emulate_media(reduced_motion="reduce")
            page.goto((HERE / "index.html").as_uri())
            page.wait_for_timeout(500)
            for i, cid in enumerate(ids, 1):
                if cid not in only:
                    continue
                for k in range(page.evaluate(f"document.querySelectorAll('#{cid} .step').length")):
                    page.evaluate(f"document.querySelector('#{cid} .step[data-step=\"{k}\"]').scrollIntoView({{block: 'center', behavior: 'instant'}})")
                    page.wait_for_timeout(400)
                    page.screenshot(path=str(SHOTS / f"{i:02d}-{cid}-step{k}.png"))
        browser.close()
    checks = [("console and page errors", not errors, " | ".join(dict.fromkeys(errors))),
              ("JS forward vs az.model", err <= 1e-9, f"max |dlogit| {err:.1e}"),
              ("accuracy on 20,000 inputs", ms <= 150, f"{ms:.0f} ms"),
              ("numbers outside data-fact spans", not stray, ", ".join(stray[:10])),
              ("data-live spans filled", not empty, ", ".join(empty)),
              ("field guide: plates, silencing tests done", plates == [16, 16], f"{plates[0]} plates, {plates[1]} tested")]
    for name, good, detail in checks:
        print(f"{'ok ' if good else 'FAIL'} {name}: {detail or 'none'}")
        ok &= good
    print(f"screenshots: {SHOTS}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
