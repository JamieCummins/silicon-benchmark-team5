#!/usr/bin/env python3
"""02_process_pages.py

Process intermediate/prompt_pages.csv (gpt-4 prompts split into survey pages,
persona/demographic pages already removed) into:
  - intermediate/stimulus_by_condition.csv : study, condition, ordered stimulus pages (pre-outcome pages)
  - intermediate/outcome_pages.csv         : study, outcome, verbatim outcome page + parsed scale
  - intermediate/study_summary.txt         : human-readable per-study digest used for the
                                             text-stimulus classification (documented in README)
Run: uv run python 02_process_pages.py   (from pipeline/, or anywhere with abs paths)
"""
import csv, re, sys, unicodedata
from collections import defaultdict, Counter
from pathlib import Path

BASE = Path("/Users/jamie/git/silicon-sample-benchmark/pipeline/data/derived/hewitt_slice")
INT = BASE / "intermediate"
csv.field_size_limit(10_000_000)

rows = list(csv.DictReader(open(INT / "prompt_pages.csv", encoding="utf-8")))
print(f"loaded {len(rows)} page rows")

# ---- 1. normalize chooser danglers and re-deduplicate variants ----------------
CHOOSER_RE = re.compile(r"\n*(You choose|Participant X? ?chooses|Participant chooses):\s*'?$")

def norm_page(t: str) -> str:
    t = t.replace("\r\n", "\n")
    t = CHOOSER_RE.sub("", t.strip()).strip()
    # canonicalize the two prompt-template voices (second-person vs "Participant X")
    t = t.replace("Participant X answers the question", "You answer the question")
    t = t.replace("Participant X chooses", "You choose")
    return t

# rebuild per (study, condition, outcome, variant): ordered page list
cells = defaultdict(dict)   # key -> {variant: (n_prompts, [pages])}
for r in rows:
    key = (r["study"], r["condition.name"], r["outcome.name"])
    v = int(r["variant"])
    n = int(r["variant_n_prompts"])
    cells[key].setdefault(v, [n, {}])
    cells[key][v][1][int(r["page_idx"])] = norm_page(r["page_text"])

merged = {}          # key -> [pages] (modal after normalization)
unstable = []
total_prompts = 0
for key, vars_ in cells.items():
    counter = Counter()
    for v, (n, pages) in vars_.items():
        seq = tuple(pages[i] for i in sorted(pages))
        counter[seq] += n
        total_prompts += n
    if len(counter) > 1:
        unstable.append((key, counter))
    merged[key] = list(counter.most_common(1)[0][0])
print(f"cells: {len(merged)}; prompts covered: {total_prompts}; "
      f"cells still >1 variant after normalization: {len(unstable)}")
for key, counter in unstable[:10]:
    print("  UNSTABLE:", key, [ (n, [p[:60] for p in seq]) for seq, n in counter.most_common(3)])

# ---- 2. split into stimulus pages (all but last) and outcome page (last) ------
stim_by_cell = {k: v[:-1] for k, v in merged.items()}
out_by_cell = {k: v[-1] if v else "" for k, v in merged.items()}

# stimulus pages should be identical across outcomes for the same condition
stim_by_cond = {}
stim_mismatch = []
for (study, cond, outc), pages in sorted(stim_by_cell.items()):
    k = (study, cond)
    if k in stim_by_cond and stim_by_cond[k] != pages:
        stim_mismatch.append((k, outc))
    else:
        stim_by_cond.setdefault(k, pages)
print(f"conditions: {len(stim_by_cond)}; conditions whose stimulus pages differ across outcomes: {len(set(k for k,_ in stim_mismatch))}")
for (k, o) in stim_mismatch[:12]:
    print("  STIM-MISMATCH across outcomes:", k, "outcome:", o)

# outcome page should be identical across conditions for the same outcome
out_by_outcome = defaultdict(Counter)
for (study, cond, outc), page in out_by_cell.items():
    out_by_outcome[(study, outc)][page] += 1
out_varies = {k: c for k, c in out_by_outcome.items() if len(c) > 1}
print(f"study-outcomes: {len(out_by_outcome)}; with condition-dependent outcome page: {len(out_varies)}")
for k in list(out_varies)[:15]:
    print("  OUTCOME-VARIES by condition:", k, "->", len(out_varies[k]), "versions")

# ---- 3. write stimulus_by_condition.csv (page-level, keeps ordering) ----------
PAGE_SEP = "\n\n"   # when joining pages later

def unquote(page: str) -> str:
    """Strip the '> ' survey-transcript quoting from a page."""
    out = []
    for line in page.split("\n"):
        if line.startswith("> "):
            out.append(line[2:])
        elif line.strip() == ">":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out).strip()

with open(INT / "stimulus_by_condition.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["study", "condition.name", "n_stim_pages", "page_idx", "page_text"])
    for (study, cond), pages in sorted(stim_by_cond.items()):
        if not pages:
            w.writerow([study, cond, 0, "", ""])
        for i, p in enumerate(pages, 1):
            w.writerow([study, cond, len(pages), i, unquote(p)])

# ---- 4. outcome pages + scale parsing ----------------------------------------
SCALE_RE = re.compile(
    r"Please choose a number from (-?\d+(?:\.\d+)?)\s*\((.*?)\)\s*to\s*(-?\d+(?:\.\d+)?)\s*\((.*?)\)\s*$",
    re.S)
SCALE_RE_NOLAB = re.compile(
    r"Please choose a number from (-?\d+(?:\.\d+)?)\s*to\s*(-?\d+(?:\.\d+)?)\s*$", re.S)

out_rows = []
unparsed = []
for (study, outc), counter in sorted(out_by_outcome.items()):
    # if outcome page varies by condition keep the modal one but flag it
    page, _ = counter.most_common(1)[0]
    varies = len(counter) > 1
    text = unquote(page)
    m = SCALE_RE.search(text)
    if m:
        smin, lab_min, smax, lab_max = m.group(1), m.group(2), m.group(3), m.group(4)
        qtext = text[:m.start()].rstrip()
        scale_labels = f"{smin}={lab_min}; {smax}={lab_max}"
    else:
        m2 = SCALE_RE_NOLAB.search(text)
        if m2:
            smin, smax = m2.group(1), m2.group(2)
            qtext, scale_labels = text[:m2.start()].rstrip(), ""
        else:
            smin = smax = scale_labels = ""
            qtext = text
            unparsed.append((study, outc, text[-200:]))
    out_rows.append({
        "study": study, "outcome.name": outc,
        "outcome_page_varies_by_condition": varies,
        "outcome_text": qtext,
        "scale_min": smin, "scale_max": smax, "scale_labels": scale_labels,
        "outcome_page_full": text,
    })
with open(INT / "outcome_pages.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader(); w.writerows(out_rows)
print(f"outcome pages written: {len(out_rows)}; scale-line unparsed: {len(unparsed)}")
for u in unparsed[:10]:
    print("  UNPARSED SCALE:", u[0], u[1], "|", u[2].replace("\n", " / ")[-150:])

# ---- 5. per-study digest for classification ----------------------------------
feat = {r["study"]: r for r in csv.DictReader(open(
    "/Users/jamie/git/silicon-sample-benchmark/pipeline/data/raw/hewitt2026_llm_prediction/codeocean_capsule/data/RA_study_features.csv", encoding="utf-8"))}

MEDIA_RE = re.compile(r"\b(image|photo|picture|video|watch|audio|recording|graphic|\[img|\.jpg|\.png)\b", re.I)

studies = sorted({k[0] for k in stim_by_cond})
with open(INT / "study_summary.txt", "w", encoding="utf-8") as f:
    for study in studies:
        sf = feat.get(study, {})
        conds = sorted(c for (s, c) in stim_by_cond if s == study)
        outs = sorted(o for (s, o) in out_by_outcome if s == study)
        f.write("=" * 100 + "\n")
        f.write(f"STUDY: {study} | tess={sf.get('study_is_tess','?')} | field={sf.get('study_field','?')}\n")
        f.write(f"TITLE: {sf.get('study_title','?')} | AUTHORS: {sf.get('study_authors','?')}\n")
        f.write(f"CONDITIONS ({len(conds)}): {conds}\n")
        f.write(f"OUTCOMES ({len(outs)}): {outs}\n")
        # do stimulus pages differ across conditions?
        seqs = {c: tuple(stim_by_cond[(study, c)]) for c in conds}
        distinct = len(set(seqs.values()))
        npages = {c: len(seqs[c]) for c in conds}
        f.write(f"STIM: {distinct} distinct page-sequences across {len(conds)} conditions; pages per condition: {npages}\n")
        ovaries = [o for o in outs if (study, o) in out_varies]
        f.write(f"OUTCOME PAGE VARIES BY CONDITION: {ovaries if ovaries else 'no'}\n")
        media = sorted({m.group(0).lower() for c in conds for p in seqs[c] for m in [MEDIA_RE.search(p)] if m})
        f.write(f"MEDIA KEYWORDS IN STIMULI: {media if media else 'none'}\n")
        for c in conds:
            joined = PAGE_SEP.join(unquote(p) for p in seqs[c])
            f.write(f"--- condition: {c!r} ({len(seqs[c])} pages, {len(joined)} chars)\n")
            f.write((joined[:600] + ("...[TRUNC]" if len(joined) > 600 else "")) + "\n")
        for o in outs[:3]:
            page, _ = out_by_outcome[(study, o)].most_common(1)[0]
            f.write(f"~~~ outcome {o!r}: {unquote(page)[:300]}\n")
        f.write("\n")
print("wrote study_summary.txt")
