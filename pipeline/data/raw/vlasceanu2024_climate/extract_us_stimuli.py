"""Extract verbatim US/English intervention stimulus texts from the Qualtrics
QSF file of the Vlasceanu et al. (2024) climate tournament (OSF ytf89,
ClimateManylabs_QSF/usa_2.qsf = "2022_08_22_ClimateCollab-MASTER").

For each Qualtrics Standard block, writes the visible question texts (HTML
stripped) of that block's questions, in block order, to
stimuli_us_extracted/<NN>_<block_name>.txt

The intervention blocks map to condName in data63.xlsx roughly as:
  1. Control Condition IVs + Control Distracter        -> Control
  2. Identity-Social-Norms-Intervention                -> WorkTogetherNorm
  3. Negative-Emotion-Intervention                     -> NegativeEmotions
  4. Scientific Consensus Intervention                 -> SciConsens
  5. Collective Action Intervention_New                -> CollectAction
  6. System Justification Intervention                 -> SystemJust
  7. Decreasing Psychological Distance Intervention    -> PsychDistance
  8. Correcting Pluralistic Ignorance Intervention     -> PluralIgnorance
  9. A Letter to Future GenerationsV2                  -> LetterFutureGen
  10. Dynamic Social Norms                             -> DynamicNorm
  11. Future Self-Continuity Intervention              -> FutureSelfCont
  12. A Binding Moral Foundations Intervention_v1Globe -> BindingMoral
(Verify against the paper / adaptation manual before final use; extraction is
mechanical and includes response-option prompts and comprehension probes.)
"""

import html
import json
import os
import re

QSF = "osf_materials/usa_2.qsf"
OUTDIR = "stimuli_us_extracted"

with open(QSF) as f:
    d = json.load(f)

# question id -> payload
questions = {}
for e in d["SurveyElements"]:
    if e.get("Element") == "SQ":
        p = e["Payload"]
        questions[p["QuestionID"]] = p


def clean(txt):
    txt = re.sub(r"<br\s*/?>", "\n", txt or "", flags=re.I)
    txt = re.sub(r"</p>", "\n", txt, flags=re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = html.unescape(txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n\s*\n+", "\n\n", txt)
    return txt.strip()


os.makedirs(OUTDIR, exist_ok=True)
n_written = 0
for e in d["SurveyElements"]:
    if e.get("Element") != "BL":
        continue
    pay = e["Payload"]
    blocks = pay.values() if isinstance(pay, dict) else pay
    for i, b in enumerate(blocks):
        if not isinstance(b, dict) or b.get("Type") not in ("Standard", "Default"):
            continue
        name = b.get("Description", f"block{i}")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
        parts = []
        for be in b.get("BlockElements", []):
            if be.get("Type") == "Question":
                q = questions.get(be.get("QuestionID"))
                if not q:
                    continue
                txt = clean(q.get("QuestionText", ""))
                choices = q.get("Choices") or {}
                copt = [clean(c.get("Display", "")) for c in choices.values()
                        if isinstance(c, dict)]
                copt = [c for c in copt if c]
                block_txt = txt
                if copt:
                    block_txt += "\n[Response options: " + " | ".join(copt) + "]"
                if block_txt.strip():
                    parts.append(f"### {q.get('DataExportTag','?')}\n{block_txt}")
        if parts:
            with open(os.path.join(OUTDIR, f"{i:02d}_{safe}.txt"), "w") as f:
                f.write(f"BLOCK: {name}\nSOURCE: {QSF}\n\n" + "\n\n".join(parts))
            n_written += 1

print(f"wrote {n_written} block files to {OUTDIR}/")
