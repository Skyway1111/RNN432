"""Build web/data.js: the example model's weights, the facts and tables the story quotes, and parity vectors for
web/check.py.

python -m web.build

Every fact is read from a committed raw output in results/, the home its finding links to, and fails loudly when the
pattern is missing, so the story cannot quote a number that no experiment produced. Facts are display strings.
"""
import json
import re
from pathlib import Path

import numpy as np
import torch as th

from az.model import load, trace

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results" / "experiments"


def grab(name: str, pattern: str, path: Path = EXP) -> tuple[str, ...]:
    text = (path / name).read_text(encoding="utf-8")
    m = re.search(pattern, text, re.S | re.M)
    if not m:
        raise SystemExit(f"web.build: {name} has no match for {pattern!r}")
    return m.groups()


def floats(s: str) -> list[float]:
    return [float(v) for v in re.findall(r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?", s)]


def pct(v: float, d: int = 0) -> str:
    return f"{100 * v:.{d}f}%"


def signed(v: float) -> str:
    return f"{'+' if v >= 0 else '−'}{abs(v):.0f}"


def span(vals, fmt) -> str:
    return f"{fmt(min(vals))}–{fmt(max(vals))}"


facts = {}
acc, = grab("accuracy_structure.txt", r"^accuracy ([\d.]+)")
facts["acc"] = pct(float(acc), 1)
err_max, err_third = grab("accuracy_structure.txt", r"errors predicting the max: ([\d.]+)\s+the 3rd: ([\d.]+)")
facts["err_max"], facts["err_third"] = pct(float(err_max)), pct(float(err_third))
near, = grab("accuracy_structure.txt", r"S-T in \[0,0\.05\): mass [\d.]+ accuracy ([\d.]+)")
far, = grab("accuracy_structure.txt", r"S-T in \[0\.5,9\): mass [\d.]+ accuracy ([\d.]+)")
facts["acc_st_near"], facts["acc_st_far"] = pct(float(near)), pct(float(far))

total, wrong = grab("permutation_exact.txt", r"example, mean profile: exactly \d+/(\d+) = [\d.]+; wrong on (\d+) of")
facts["ord_total"], facts["ord_wrong"] = f"{int(total):,}", f"{int(wrong):,}"

steps, two = grab("readout_wheel.txt", r"example:.*?steps: \[([^\]]+)\].*?two-impulse accuracy ([\d.]+)")
facts["wheel_step"] = f"{abs(np.mean(floats(steps)[:7])):.0f}°"
facts["wheel_acc"] = pct(float(two))

coef = floats(grab("margin_by_rank.txt", r"l_q - l_r coefficient mean by rank \[([^\]]+)\]")[0])
facts["coef_M"], facts["coef_S"], facts["coef_T"] = (signed(v) for v in coef[:3])

D, slope = grab("tail_modes.txt", r"example/0:.*?quiet-state D ([\d.]+), measured f/\|u_9\| median ([\d.]+)")
facts["tail_D"], facts["tail_slope"] = f"{float(D):.0f}", f"{float(slope):.0f}"
c8, = grab("tail_modes.txt", r"example/0:.*?c=8 \(\d+ events\): le1pos ([\d.]+)")
facts["tail_share_c8"] = pct(float(c8))

p8 = json.loads((ROOT / "results" / "bench" / "sampling_rare_bench.json").read_text())["truth_median"][4]
facts["rare_c8"] = f"1 in {round(1 / p8, -4):,.0f}"
modes = floats(grab("modes_score.txt", r"bench\s+geometric mean([^;]+);")[0])
ce = floats(grab("ce_score.txt", r"bench\s+salts 0-4([^;]+);")[0])
facts["modes_c8"], facts["ce_c8"] = f"{modes[4]:.0f}", f"{ce[4]:.0f}"

row = floats(grab("weight_whatif.txt", r"Wh\[7,2\]\s+=([^\n]+)")[0])
facts["w72"] = f"{row[0]:.3f}"
facts["w72_drop"] = f"{100 * min(row[4], row[5]):.0f}–{100 * max(row[4], row[5]):.0f} points"
n2, = grab("noise_sensitivity.txt", r"example:.*?sigma 0\.002\s*: mean ([\d.]+)")
wh, = grab("noise_sensitivity.txt", r"example:.*?noise on Wh only, sigma 0\.003: mean ([\d.]+)")
wi, = grab("noise_sensitivity.txt", r"example:.*?noise on Wi only, sigma 0\.003: mean ([\d.]+)")
facts["noise_acc"], facts["noise_wh"], facts["noise_wi"] = pct(float(n2), 1), pct(float(wh), 1), pct(float(wi), 1)

sets50, sets90 = grab("cascade_stations.txt", r"^example:.*?sets for 50 / 90% of draws (\d+) / (\d+)")
facts["cascade_sets50"], facts["cascade_sets90"] = sets50, f"{int(sets90):,}"
share, with_, without = grab("cascade_stations.txt", r"^example:.*?first input's sign: flips every gate at step 0 on ([\d.]+) of draws, with ([\d.]+) gates retuned later against ([\d.]+)")
facts["sign_share"], facts["sign_with"], facts["sign_without"] = pct(float(share)), f"{float(with_):.0f}", f"{float(without):.1f}"

agree, = grab("canonical_routes.txt", r"reference linear map': agreement with the model ([\d.]+)")
near_r, = grab("canonical_routes.txt", r"reference surrogate agreement near a tie ([\d.]+)")
facts["route_agree"], facts["route_near"] = pct(float(agree)), pct(float(near_r))

# Training of the five sibling networks (research/findings/training.md): accuracy and how well all 16 neurons encode
# the record (lagged) and the jump at each checkpoint, after 0, 2^20, ..., 2^30 training sequences.
seeds = []
for block in re.split(r"^seed \d+:", (EXP / "core_emergence.txt").read_text(encoding="utf-8"), flags=re.M)[1:]:
    rows = re.findall(r"^\s+(?:0|2\^\d+)\s+([\d.]+) \|[^|]+\|([^\n]+)$", block, re.M)
    if len(rows) != 12:
        raise SystemExit(f"web.build: core_emergence.txt has a seed with {len(rows)} checkpoints, not 12")
    dec = [floats(r[1]) for r in rows]
    seeds.append(dict(acc=[float(r[0]) for r in rows], record=[d[0] for d in dec], jump=[d[1] for d in dec]))
at = lambda k, i: [s[k][i] for s in seeds]  # noqa: E731
facts["train_acc22"], facts["train_acc24"], facts["train_acc30"] = (span(at("acc", i), lambda v: f"{100 * v:.0f}") + "%" for i in (3, 5, 11))
facts["train_rec22"], facts["train_rec24"] = (span(at("record", i), lambda v: f"{v:.2f}") for i in (3, 5))
clean = {m: int(c) for m, c in re.findall(r"^(\w+): single neurons with a role fit R2 > 0\.9: (\d+)", (EXP / "role_probe.txt").read_text(encoding="utf-8"), re.M)}
words = "zero one two three four five six seven".split()
facts["clean_example"] = words[clean["example"]]
facts["clean_seeds"] = f"{words[min(v for m, v in clean.items() if m != 'example')]} to {words[max(v for m, v in clean.items() if m != 'example')]}"

w = load("example")
x = np.random.default_rng(0).standard_normal((64, 10))
logits = trace(w, th.tensor(x, dtype=th.float64))[0].numpy()
table = np.load(EXP / "weight_whatif.npy")  # 432 weights (Wi, Wh, Wo row-major) x steps -0.5 -0.1 -0.02 +0.02 +0.1 +0.5
drop = (float(acc) - table[:, [1, 4]].mean(1)).round(4)  # mean accuracy lost at +-0.1, the breaking chapter's map
cut_base, = grab("neuron_atlas.txt", r"^accuracy on the 2\^18 scoring inputs: ([\d.]+)")
cuts = np.load(EXP / "neuron_atlas.npy")  # (16 targets, 17 sources: input, n0..n15): accuracy with that link cut
data = dict(weights=dict(Wi=w.Wi.tolist(), Wh=w.Wh.tolist(), Wo=w.Wo.tolist()), facts=facts, whatif=drop.tolist(),
            links=dict(base=float(cut_base), cut=cuts.round(4).tolist()), training=dict(seeds=seeds, clean=clean),
            parity=dict(x=x.tolist(), logits=logits.tolist()))
out = Path(__file__).with_name("data.js")
out.write_text("window.DATA = " + json.dumps(data) + ";\n", encoding="utf-8")
print(f"wrote {out.relative_to(ROOT)}: {len(facts)} facts")
for k, v in facts.items():
    print(f"  {k:14s} {v}")
