# Retrodiction run 2 (Hewitt/Ashokkumar archive slice) — pre-analysis note

Written 2026-08-18, BEFORE elicitation on ground 2. Purpose: freeze the
confirmation test so ground-2 results can't be quietly repurposed as tuning.

## Design

- Ground: message/persuasion-like slice of the Hewitt/Ashokkumar Nature 2026
  archive (target 25-45 studies, 60-130 treatment-vs-reference contrasts,
  US samples, text stimuli). Selection rules documented by the extraction
  script; selection happens blind to whether our pipeline would score well.
- Crowd: luna, maverick, deepseek, gptoss x 6 variants (balanced half-grid:
  indices 0,3,4,7,8,11) + terra x 3 variants (0,7,11). qwen36 dropped
  (unusable in run 1), kimi dropped (cost control; LOMO contribution ~0).
  Caveat acknowledged: roster changes alongside ground.
- Priors protocol (prospective): sign = hypothesized direction from the
  archive's own hypothesis metadata (mechanical, not eyeballed); magnitude
  0.8pp of scale range for attitude outcomes; behavioral analogs (if any
  survive selection) 0.2pp. Epsilon = 0.02 native slider units equivalent.
- Budget: <= $5. Runner prints an estimate before firing.

## Frozen from run 1 (Vlasceanu US) — to be CONFIRMED, not retuned

- best-by-r:    w_prior=0.75, lam=0.3, trim=0.2  (in-sample r=0.855)
- best-by-RMSE: w_prior=0.75, lam=1.0, trim=0.0  (in-sample RMSE=4.45pp)
- Run-1 raw reference points: pooled r=0.563, within-outcome r=0.267,
  directional 82%, MAE 3.89pp, RMSE 4.52pp, calib beta=2.83.

## Hypotheses (stated before seeing ground-2 results)

H1 Transfer: frozen w_prior=0.75 settings beat the raw crowd on pooled r on
   ground 2 (though by less than in-sample, since priors were sized for
   megastudy-style outcomes).
H2 Raw skill: pooled r in 0.35-0.65; within-outcome r (where >=2 contrasts
   share a study-outcome) in 0.1-0.4.
H3 Positivity bias replicates beyond effort outcomes: directional accuracy on
   observed-negative cells < 40% vs > 85% on positives, and the predicted-
   negative rate is under half the observed-negative rate.
H4 Sign-informed priors (prospective, from hypothesis metadata) raise
   directional accuracy on negative cells vs the flat-positive prior baseline.
H5 Underprediction persists but attenuates: calib beta in 1.3-3.0 (archive
   effects average smaller than Vlasceanu's).
H6 Extended-grid RMSE optimum has lam >= 1 (amplification or no shrink), NOT
   lam < 1.

## Metrics (all reported, pp of scale range)

pooled Pearson r (+ cluster bootstrap CI over studies), Spearman rho,
within-study-outcome r, directional % (half credit at zero), MAE, RMSE,
calib beta/alpha, r_adj where SEs allow; all-zero baseline alongside; sign
split (observed +/-) with confusion matrix.

## Decision rule for the benchmark

Hyperparameters adopted for the benchmark = combos near-top on BOTH grounds
by RMSE (tie-break r), with scale chosen conservatively (contamination makes
retro skill an upper bound). If transfer fails badly (frozen settings not
beating raw on ground 2), fall back to conservative defaults (w=0.5, lam=1.0)
and diagnose before any further tuning.
