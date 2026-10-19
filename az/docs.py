"""Doc-structure lint and the generated research index. Run through `python -m az.check [--fix]`.

Rules: relative links resolve; files stay under their line caps; research files, active or parked, open with
`# Title`, a blank line, and a `>` summary of at most 5 lines whose first line, the index entry,
fits in 140 characters; angles carry a known **Status:**;
every active experiment cites an existing finding, is cited by an active research file, and has a stored output.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"
INDEX = RESEARCH / "README.md"
PARKED = ROOT / "parked" / "research"
INDEXES = {INDEX: RESEARCH, ROOT / "parked" / "README.md": PARKED}  # each generated index and the tree it lists
STATUSES = ("active", "proposed", "parked", "dead")
CAP, INDEX_CAP, SUMMARY_CAP, ENTRY_CAP = 150, 80, 5, 140
LINK = re.compile(r"\]\(([^)\s]+)\)")
START, END = "<!-- index:start -->", "<!-- index:end -->"


def _docs():
    return sorted(p for p in ROOT.rglob("*.md") if not {"_ext", "_shots", ".git"} & set(p.relative_to(ROOT).parts))


def _summary(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3 or not lines[0].startswith("# ") or lines[1] or not lines[2].startswith("> "):
        return None
    quote = []
    for line in lines[2:]:
        if not line.startswith("> "):
            break
        quote.append(line[2:])
    return quote


def _status(path):
    m = re.search(r"^\*\*Status:\*\* (\w+)", path.read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else None


def index(readme=INDEX):
    """The tables of one tree's angles, findings and notes, linked from its README."""
    tree = INDEXES[readme]
    link = lambda p: p.relative_to(readme.parent).as_posix()  # noqa: E731
    entry = lambda p: (_summary(p) or ["?"])[0]  # noqa: E731
    angles = sorted((tree / "angles").glob("*.md"), key=lambda p: (STATUSES.index(_status(p)) if _status(p) in STATUSES else 9, p.stem))
    sections = [("Angles", angles), ("Findings", sorted((tree / "findings").glob("*.md"))),
                ("Notes", sorted(p for p in tree.glob("*.md") if p.name != "README.md"))]
    rows = []
    for title, docs in sections:
        if docs:
            status = title == "Angles"
            rows += ["", f"### {title}", "| Doc | Status | Summary |" if status else "| Doc | Summary |", "|---|---|---|" if status else "|---|---|"]
            rows += [f"| [{p.stem}]({link(p)}) |{f' {_status(p)} |' if status else ''} {entry(p)} |" for p in docs]
    return "\n".join(rows[1:])


def _with_index(text, readme=INDEX):
    head, rest = text.split(START, 1)
    return f"{head}{START}\n{index(readme)}\n{END}{rest.split(END, 1)[1]}"


def lint(fix=False):
    errs = []
    for p in _docs():
        rel, text = p.relative_to(ROOT), p.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            if not re.match(r"[a-z]+:", target) and not target.startswith("#"):
                if not (p.parent / target.split("#")[0]).exists():
                    errs.append(f"{rel}: broken link {target}")
        cap = INDEX_CAP if p == INDEX else CAP
        if len(text.splitlines()) > cap:
            errs.append(f"{rel}: {len(text.splitlines())} lines, cap {cap}; split it")
        if RESEARCH in p.parents or PARKED in p.parents:
            s = _summary(p)
            if s is None:
                errs.append(f"{rel}: must open with '# Title', a blank line, then a '> ' summary")
            elif len(s) > SUMMARY_CAP:
                errs.append(f"{rel}: summary is {len(s)} lines, cap {SUMMARY_CAP}")
            elif len(s[0]) > ENTRY_CAP:
                errs.append(f"{rel}: first summary line is {len(s[0])} chars, cap {ENTRY_CAP} (it is the index entry)")
            if p.parent.name == "angles" and _status(p) not in STATUSES:
                errs.append(f"{rel}: needs '**Status:** one of {'/'.join(STATUSES)}'")
    for readme in INDEXES:
        text = readme.read_text(encoding="utf-8")
        if _with_index(text, readme) != text:
            if fix:
                readme.write_text(_with_index(text, readme), encoding="utf-8")
            else:
                errs.append(f"{readme.relative_to(ROOT).as_posix()}: index is stale; run python -m az.check --fix")
    cited = "\n".join(p.read_text(encoding="utf-8") for p in RESEARCH.rglob("*.md"))
    for e in sorted((ROOT / "experiments").glob("*.py")):
        if e.stem in ("__init__", "common"):
            continue
        doc = e.read_text(encoding="utf-8")
        finding = re.search(r"^Finding: (.+)$", doc, re.M)
        if not finding or not all((ROOT / f.strip()).exists() for f in finding.group(1).split(",")):
            errs.append(f"experiments/{e.name}: docstring needs 'Finding: <existing research file>'")
        if e.stem not in cited:
            errs.append(f"experiments/{e.name}: no research file cites it")
        if not (ROOT / "results" / "experiments" / f"{e.stem}.txt").exists():
            errs.append(f"experiments/{e.name}: run it into results/experiments/{e.stem}.txt")
    for r in sorted((ROOT / "results" / "experiments").glob("*.txt")):
        if not (ROOT / "experiments" / f"{r.stem}.py").exists():
            errs.append(f"results/experiments/{r.name}: no experiment produces it")
    return errs
