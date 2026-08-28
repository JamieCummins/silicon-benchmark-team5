#!/usr/bin/env python3
"""03_build_slice.py

Build the retrodiction slice (contrasts.csv, stimuli.csv) from the intermediate
tables produced by 01_extract_from_rds.R and 02_process_pages.py.

Selection rules (documented in README.md):
  1. Universe = 71 studies in rct_responses.RDS.
  2. Drop studies/cells without prompts in llm_responses.RDS (willer845; two outcomes).
  3. Study-level text-stimulus classification (hand-coded from intermediate/study_summary.txt,
     reasons recorded in STUDY_CLASS below).
  4. Reference designation:
       Rule A ("named_control")  : unique control/baseline/placebo condition by name.
       Rule B ("hypothesis_t0")  : all RA hypotheses for the study share one single-cell
                                   t0 side -> that cell is the study-wide reference.
       Rule C ("hypothesis_pair"): otherwise, single-t0 hypothesis pairs only
                                   (t1 side may be pooled -> one row per t1 cell).
     Studies with none of these are excluded (ambiguous reference).
  5. Condition-level exclusion: McCabeS19 'INSURANCE+GOVERNMENT CUE' (corrupted prompt
     in the source archive - contains Howat1039 stimulus).
  6. Sampling to target: keep ALL included Coppock (non-TESS) studies; TESS studies are
     shuffled with seed 42 and added while total contrasts <= 130.

Statistics: per-cell n/mean/sd from participant-level rct_responses (original coding),
ate = mean_treat - mean_ref, Welch SE. Rows are then oriented to the response scale as
shown in outcome_text (the LLM-prompt presentation): when scale_flip is TRUE, means and
ate are flipped (m -> min+max-m, ate -> -ate) so every row is internally consistent.
"""
import csv, math, random, re
from collections import defaultdict
from pathlib import Path

BASE = Path("/Users/jamie/git/silicon-sample-benchmark/pipeline/data/derived/hewitt_slice")
INT = BASE / "intermediate"
RAW = Path("/Users/jamie/git/silicon-sample-benchmark/pipeline/data/raw/hewitt2026_llm_prediction")
csv.field_size_limit(10_000_000)

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

# ---------------------------------------------------------------- load inputs
cell_stats = list(csv.DictReader(open(INT / "cell_stats.csv", encoding="utf-8")))
outcome_pages = {(r["study"], r["outcome.name"]): r
                 for r in csv.DictReader(open(INT / "outcome_pages.csv", encoding="utf-8"))}
scales = {(r["study"], r["outcome.name"]): r
          for r in csv.DictReader(open(INT / "outcome_scales.csv", encoding="utf-8"))}
feat = {r["study"]: r for r in csv.DictReader(open(RAW / "codeocean_capsule/data/RA_study_features.csv", encoding="utf-8"))}

stim_pages = defaultdict(list)   # (study, condition) -> ordered page texts
for r in csv.DictReader(open(INT / "stimulus_by_condition.csv", encoding="utf-8")):
    if r["page_idx"]:
        stim_pages[(r["study"], r["condition.name"])].append(r["page_text"])
    else:
        stim_pages.setdefault((r["study"], r["condition.name"]), [])

hyps = defaultdict(lambda: defaultdict(list))  # (study, outcome, hyp) -> {t: [conds]}
for r in csv.DictReader(open(INT / "hypotheses_long.csv", encoding="utf-8")):
    hyps[(r["study"], r["outcome.name"], r["hypothesis"])][r["t_hypothesis"]].append(r["condition.name"])

stats = {}  # (study, outcome, condition) -> dict
outcomes_by_study = defaultdict(set)
conds_by_study = defaultdict(set)
for r in cell_stats:
    stats[(r["study"], r["outcome.name"], r["condition.name"])] = r
    outcomes_by_study[r["study"]].add(r["outcome.name"])
    conds_by_study[r["study"]].add(r["condition.name"])

# outcomes/conditions that actually have prompts
prompt_outcomes = set(outcome_pages)
prompt_conds = set(stim_pages)

log(f"STEP 0 universe: {len(outcomes_by_study)} studies, "
    f"{len({(r['study'], r['outcome.name']) for r in cell_stats})} study-outcomes, "
    f"{len(cell_stats)} cells, "
    f"{len(cell_stats) - len({(r['study'], r['outcome.name']) for r in cell_stats})} cell contrasts (all-vs-one-reference count)")

# ------------------------------------------------- study-level classification
# category: include | question_wording | task_induction | no_prompts
# (hand-coded after reading intermediate/study_summary.txt; see README)
STUDY_CLASS = {
    # --- TESS studies: text stimulus -> include
    "AnsonBRIEF60": ("include", "news-article information treatment"),
    "Blair1131": ("include", "crisis-bargaining vignette"),
    "Bougher893": ("include", "candidate-positions table (text)"),
    "Braman751": ("include", "news-article vignette factorial"),
    "CalarcoS38": ("include", "person vignette factorial"),
    "Campbell1308": ("include", "news story vs control"),
    "Cohen1099": ("include", "person vignette factorial"),
    "Connors1226": ("include", "social-pressure reminder text"),
    "CorstangeBRIEF69": ("include", "conflict-framing information text"),
    "Craig735": ("include", "press-release text"),
    "FaheyS78": ("include", "news scenario factorial"),
    "FarrowS6": ("include", "imagined-scenario vignette"),
    "Haaland874": ("include", "research-findings information treatment"),
    "HamiltonS31": ("include", "accusation vignette"),
    "HankinsonS22": ("include", "policy-siting vignette"),
    "Harbridge-Yong1032": ("include", "news article factorial"),
    "Iles1294": ("include", "news article (conflicting nutrition info)"),
    "Kennedy1017": ("include", "family vignette factorial"),
    "KuruB67": ("include", "poll-report news story"),
    "McCabeS19": ("include", "health-cost experience vignette"),
    "McGinty730": ("include", "narrative radio-excerpt text"),
    "Melin1066": ("include", "resume/job-description vignette"),
    "Mezzapelle1439": ("include", "incident-report vignette"),
    "MunschS68": ("include", "reference-check email vignette"),
    "SchaadS62": ("include", "narrative vignette"),
    "Schnabel903": ("include", "person-impression vignette"),
    "ShannonS2": ("include", "policy argument text"),
    "ShannonS28": ("include", "persuasive argument text"),
    "Silverman1035": ("include", "corrective-information news text"),
    "Stoker1063": ("include", "equivalency-framed problem description"),
    "Terman1029": ("include", "news article (human-rights shaming)"),
    "ThorsonS42": ("include", "misinformation-coverage exposure vs DVs-first"),
    "WallaceS9": ("include", "audience-cost vignette"),
    "Williamson859": ("include", "tax-information treatment (chart described in text)"),
    "bolsenM6": ("include", "normative message vs distractor baseline"),
    "bucci1408": ("include", "drunk-driving scenario vignette"),
    "KlarS44": ("include", "abortion-law framing text"),
    "Krupnikov719": ("include", "data-transparency notice text (procedural)"),
    "relihan1399": ("include", "news article with elite cues"),
    "senS81": ("include", "population-statistics information (plots described in text)"),
    # --- TESS studies: excluded
    "Enos700": ("question_wording", "target-group swap inside the rated item"),
    "KlarBRIEF70": ("question_wording", "wording variants of the outcome question itself"),
    "KrupnikovS34": ("question_wording", "'feel' vs 'think' wording of partisanship item"),
    "RyanS14": ("question_wording", "party swap inside rumor item"),
    "Howard823": ("task_induction", "expense-estimation writing task (also excluded from predictions by authors)"),
    "Howat1039": ("task_induction", "stereotype-listing writing task"),
    "Levendusky741": ("task_induction", "priming/writing tasks (identity prime, self-affirmation)"),
    "Merolla843": ("task_induction", "emotion-induction writing task"),
    "Rifkin1073": ("task_induction", "busyness reflection induction"),
    "wayne841": ("task_induction", "emotion-induction writing task with photo stimulus"),
    "willer845": ("no_prompts", "absent from llm_responses.RDS - no stimuli available"),
    # --- Coppock (non-TESS) studies
    "immigration": ("include", "pro/anti immigration news article"),
    "death_penalty": ("include", "one-sentence persuasion arguments"),
    "patriot_act": ("include", "pro/con argument summaries"),
    "system_threat": ("include", "census news article vs placebo article"),
    "brandtS1": ("include", "foreclosure vignette"),
    "caprarielloS2": ("task_induction", "recall-own-spending induction"),
    "flavinS4": ("include", "argument with/without definition"),
    "gashS5": ("include", "court-ruling vignette vs no prompt"),
    "melloS6": ("include", "medical-error disclosure vignette"),
    "jacobsenS7": ("include", "school report-card format (text table)"),
    "piazzaS8": ("include", "news report; suspect names varied"),
    "shaferS9": ("include", "marital-name vignette"),
    "thompsonS10": ("include", "high/low fear terrorism-preparedness message"),
    "turagaS11": ("include", "mercury information text"),
    "wallaceS12": ("include", "country promise-keeping vignette"),
    "parmerS15": ("include", "smallpox scenario + recommendation text"),
    "converseS16": ("task_induction", "question-order/perspective-taking task"),
    "dennyS17": ("include", "job-application memo vignette"),
    "pedullaS18": ("include", "resume vignette (name/race varied)"),
    "berganS20": ("include", "news article with party-label swap"),
}
assert set(STUDY_CLASS) == set(outcomes_by_study), (
    set(STUDY_CLASS) ^ set(outcomes_by_study))

included = [s for s, (c, _) in STUDY_CLASS.items() if c == "include"]
excl_counts = defaultdict(list)
for s, (c, why) in STUDY_CLASS.items():
    if c != "include":
        excl_counts[c].append(s)
log(f"STEP 1 no-prompt exclusions: willer845 (whole study); "
    f"Howard823/predictedexpense_nextweek and Melin1066/salary (outcome-level, no prompts)")
log(f"STEP 2 text-stimulus filter: {len(included)} studies kept; excluded: " +
    "; ".join(f"{k}={sorted(v)}" for k, v in sorted(excl_counts.items())))

# ------------------------------------------------------- reference designation
NAMED_CONTROL = {  # verified unique control-like condition names (Rule A)
    "AnsonBRIEF60": "Control group 1",
    "Campbell1308": "Control",
    "Connors1226": "Control",
    "Craig735": "Control",
    "Haaland874": "CONTROL",
    "KlarS44": "CONTROL",
    "Krupnikov719": "Group 1 Control",
    "McCabeS19": "CONTROL",
    "McGinty730": "No narrative",
    "ShannonS2": "Control",
    "Silverman1035": "Control group: No Corrective Information",
    "Williamson859": "CONTROL",
    "bolsenM6": "Baseline",
    "senS81": "Gender-Control",
    "system_threat": "craig_placebo",
    "gashS5": "4 [No Prompt]",
}
CORRUPTED_CONDITIONS = {("McCabeS19", "INSURANCE+GOVERNMENT CUE")}

def hyp_map_outcome(study, o):
    """Map a hypothesis outcome name (e.g. 'coppock_dv_name') to the rct outcome name."""
    if o in outcomes_by_study[study]:
        return o
    if len(outcomes_by_study[study]) == 1:
        return next(iter(outcomes_by_study[study]))
    return None

# study -> list of hypothesis records {outcome, t1:[...], t0:[...]}
study_hyps = defaultdict(list)
for (s, o, h), v in hyps.items():
    study_hyps[s].append({"outcome": hyp_map_outcome(s, o), "hyp": h,
                          "t1": v.get("1", []), "t0": v.get("0", [])})

reference_rule = {}
for s in included:
    if s in NAMED_CONTROL:
        reference_rule[s] = ("named_control", NAMED_CONTROL[s])
        continue
    hs = study_hyps.get(s, [])
    t0_cells = {tuple(sorted(h["t0"])) for h in hs}
    if hs and len(t0_cells) == 1 and len(next(iter(t0_cells))) == 1:
        reference_rule[s] = ("hypothesis_t0", next(iter(t0_cells))[0])
    elif any(len(h["t0"]) == 1 for h in hs):
        reference_rule[s] = ("hypothesis_pair", None)
    else:
        reference_rule[s] = ("ambiguous", None)

rule_counts = defaultdict(list)
for s, (rule, ref) in reference_rule.items():
    rule_counts[rule].append(s)
log("STEP 3 reference designation: " +
    "; ".join(f"{k}={len(v)} ({sorted(v)})" for k, v in sorted(rule_counts.items())))
ambiguous = rule_counts.get("ambiguous", [])

# ------------------------------------------------------------ enumerate contrasts
def direction(study, outcome, cond, ref):
    """Orientation of the RA-coded hypothesis covering (cond, ref); see README."""
    for h in study_hyps.get(study, []):
        if h["outcome"] not in (outcome, None):
            continue
        if cond in h["t1"] and ref in h["t0"]:
            return 1
        if cond in h["t0"] and ref in h["t1"]:
            return -1
    return None

contrast_rows = []   # dicts keyed later
for s in included:
    rule, ref = reference_rule[s]
    if rule == "ambiguous":
        continue
    outs = sorted(outcomes_by_study[s] & {o for (st, o) in prompt_outcomes if st == s})
    pairs = set()
    if rule in ("named_control", "hypothesis_t0"):
        for c in sorted(conds_by_study[s]):
            if c == ref or (s, c) not in prompt_conds or (s, c) in CORRUPTED_CONDITIONS:
                continue
            for o in outs:
                pairs.add((c, ref, o))
    else:  # hypothesis_pair
        for h in study_hyps[s]:
            if len(h["t0"]) != 1:
                continue
            r0 = h["t0"][0]
            for c in h["t1"]:
                if (s, c) not in prompt_conds or (s, r0) not in prompt_conds:
                    continue
                o = h["outcome"]
                if o is None:
                    continue
                if o in outs:
                    pairs.add((c, r0, o))
    for (c, r0, o) in sorted(pairs):
        contrast_rows.append({"study": s, "condition": c, "reference": r0,
                              "outcome": o, "reference_type": rule})

by_study_n = defaultdict(int)
for r in contrast_rows:
    by_study_n[r["study"]] += 1
log(f"STEP 4 eligible contrasts before sampling: {len(contrast_rows)} across {len(by_study_n)} studies")
log("   per study: " + ", ".join(f"{s}={n}" for s, n in sorted(by_study_n.items())))

# --------------------------------------------------------------- sampling to target
coppock = [s for s in by_study_n if feat[s]["study_is_tess"] == "FALSE"]
tess = [s for s in by_study_n if feat[s]["study_is_tess"] == "TRUE"]
n_copp = sum(by_study_n[s] for s in coppock)
TARGET_MAX = 130
rng = random.Random(42)
tess_shuffled = sorted(tess)
rng.shuffle(tess_shuffled)
kept = set(coppock)
total = n_copp
skipped = []
for s in tess_shuffled:
    if total + by_study_n[s] <= TARGET_MAX:
        kept.add(s)
        total += by_study_n[s]
    else:
        skipped.append(s)
log(f"STEP 5 sampling (seed 42): keep all {len(coppock)} Coppock studies ({n_copp} contrasts); "
    f"TESS shuffled order = {tess_shuffled}")
log(f"   kept {len(kept & set(tess))} TESS studies; skipped (budget) = {skipped}; "
    f"final: {len(kept)} studies, {total} contrasts (target <= {TARGET_MAX})")

contrast_rows = [r for r in contrast_rows if r["study"] in kept]

# --------------------------------------------------------------- build contrasts.csv
def fnum(x, nd=6):
    return "" if x is None else f"{x:.{nd}g}"

FLAG_PATTERNS = [
    ("bracket_alternation", re.compile(r"\]\s*/\s*\[")),
    ("html_tags", re.compile(r"</?[uib]>", re.I)),
    ("layout_placeholder", re.compile(r"\[(BLANK )?SPACE\]|\[TEXT BOX|\[Large [Tt]ext box|\[O: small text box\]")),
    ("embedded_response_prompt", re.compile(r"You answer the question|You choose|RESPONSE OPTIONS:|Participants see the following response options|\[participant responds\]", re.I)),
    ("chart_or_image_reference", re.compile(r"\bchart\b|\bplot below\b|\bpictured\b|\bimages?\b|\bgraph\b", re.I)),
]
def flags_for(text):
    return ",".join(name for name, pat in FLAG_PATTERNS if pat.search(text)) if text else ""

out_rows = []
for r in contrast_rows:
    s, c, ref, o = r["study"], r["condition"], r["reference"], r["outcome"]
    st_t = stats[(s, o, c)]
    st_r = stats[(s, o, ref)]
    op = outcome_pages[(s, o)]
    sc = scales[(s, o)]
    flip = sc["scale_flip"] == "TRUE"
    smin, smax = float(sc["outcome_scale_min"]), float(sc["outcome_scale_max"])
    n_t, n_r = int(st_t["n"]), int(st_r["n"])
    m_t, m_r = float(st_t["mean"]), float(st_r["mean"])
    sd_t, sd_r = float(st_t["sd"]), float(st_r["sd"])
    if flip:  # orient to the scale as displayed in outcome_text
        m_t, m_r = (smin + smax) - m_t, (smin + smax) - m_r
    ate = m_t - m_r
    se = math.sqrt(sd_t**2 / n_t + sd_r**2 / n_r)
    d = direction(s, o, c, ref)
    out_rows.append({
        "study_id": s,
        "study_label": feat[s]["study_title"],
        "condition": c,
        "reference": ref,
        "outcome_key": o,
        "outcome_text": op["outcome_text"] if op["outcome_text"] else op["outcome_page_full"],
        "scale_min": fnum(smin), "scale_max": fnum(smax),
        "scale_labels": op["scale_labels"],
        "n_treat": n_t, "n_ref": n_r,
        "mean_treat": fnum(m_t), "mean_ref": fnum(m_r),
        "sd_treat": fnum(sd_t), "sd_ref": fnum(sd_r),
        "ate": fnum(ate), "se": fnum(se),
        "hypothesized_direction": "" if d is None else d,
        "reference_type": r["reference_type"],
        "scale_flip": sc["scale_flip"],
        "outcome_text_is_scale_only": "TRUE" if not op["outcome_text"] or op["outcome_text"].strip() == "" else "FALSE",
    })

out_rows.sort(key=lambda r: (r["study_id"], r["outcome_key"], r["condition"]))
with open(BASE / "contrasts.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader(); w.writerows(out_rows)
log(f"WROTE contrasts.csv: {len(out_rows)} rows, {len({r['study_id'] for r in out_rows})} studies")

# --------------------------------------------------------------- build stimuli.csv
SCAFFOLD_LINE = re.compile(r"^\s*(You answer the question|You choose)\s*$", re.M)
def clean_page(t):
    return SCAFFOLD_LINE.sub("[participant responds]", t).strip()

stim_rows = []
for s in sorted(kept):
    conds = sorted(c for (st, c) in prompt_conds if st == s and (st, c) not in CORRUPTED_CONDITIONS)
    page_sets = [stim_pages[(s, c)] for c in conds]
    common = set(page_sets[0])
    for ps in page_sets[1:]:
        common &= set(ps)
    # study_context: common pages in the order they appear in the first condition that has them
    ordered_common = [p for p in page_sets[0] if p in common]
    context = "\n\n".join(clean_page(p) for p in ordered_common)
    for c in conds:
        own = [p for p in stim_pages[(s, c)] if p not in common]
        text = "\n\n".join(clean_page(p) for p in own)
        stim_rows.append({
            "study_id": s,
            "condition": c,
            "stimulus_text": text,
            "source": "llm_prompts",
            "study_context": context,
            "flags": flags_for(text + "\n" + context),
        })
with open(BASE / "stimuli.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(stim_rows[0].keys()))
    w.writeheader(); w.writerows(stim_rows)
n_empty = sum(1 for r in stim_rows if not r["stimulus_text"])
log(f"WROTE stimuli.csv: {len(stim_rows)} condition rows "
    f"({n_empty} with empty stimulus_text = no condition-specific content)")

# --------------------------------------------------------------- validation vs MOESM4
try:
    import openpyxl
    wb = openpyxl.load_workbook(RAW / "nature_si/MOESM4.xlsx", read_only=True)
    ws = wb["panelA"]
    hdr = None
    panel = {}
    for row in ws.iter_rows(values_only=True):
        if hdr is None:
            hdr = list(row); continue
        d = dict(zip(hdr, row))
        panel[(d["study"], d["reference_condition"], d["condition.name"], d["outcome.name"])] = d["estimate"]
    checked = []
    for r in out_rows:
        key = (r["study_id"], r["reference"], r["condition"], r["outcome_key"])
        if key in panel and len(checked) < 5 and r["study_id"] not in {c[0] for c in checked}:
            smin, smax = float(r["scale_min"]), float(r["scale_max"])
            ate_orig = float(r["ate"]) * (-1 if r["scale_flip"] == "TRUE" else 1)  # back to original coding
            ours01 = ate_orig / (smax - smin)
            checked.append((r["study_id"], r["condition"][:40], r["outcome_key"],
                            round(ours01, 6), round(panel[key], 6), round(ours01 - panel[key], 8)))
    log("VALIDATION vs MOESM4 panelA (estimate on original coding rescaled to [0,1]):")
    for c in checked:
        log(f"   {c[0]} | {c[1]} | {c[2]} | ours={c[3]} theirs={c[4]} diff={c[5]}")
    n_match = 0; n_tot = 0; maxdiff = 0.0
    for r in out_rows:
        key = (r["study_id"], r["reference"], r["condition"], r["outcome_key"])
        if key in panel:
            smin, smax = float(r["scale_min"]), float(r["scale_max"])
            ate_orig = float(r["ate"]) * (-1 if r["scale_flip"] == "TRUE" else 1)
            dd = abs(ate_orig / (smax - smin) - panel[key])
            n_tot += 1; maxdiff = max(maxdiff, dd); n_match += dd < 1e-6
    log(f"   full sweep: {n_tot} slice contrasts found in panelA; {n_match} agree within 1e-6; max abs diff = {maxdiff:.2e}")
except Exception as e:
    log(f"VALIDATION ERROR: {e}")

with open(INT / "build_log.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines) + "\n")
