# Benchmark outcome priors — literature-updated draft for review

Status: DRAFT v2 (2026-08-18), synthesized from four literature sweeps
(trust-in-scientists interventions; climate-messaging meta-analyses; survey-
embedded donation/signup behavior; backfire + policy-role advocacy). Full
agent reports with citations preserved in the session record; key anchors
cited inline. AWAITING JAMIE SIGN-OFF. Doubles as registration-form text.

## How the literature was calibrated to the terminal benchmark

Six adjustments, applied explicitly:

1. **Recency weighting.** 2024-26 large-N studies run smaller than pre-2020
   literature. The two closest published analogs disagree: Vlasceanu et al.
   2024 (US arm, our retro ground 1) averaged ≈ +4.5pp on its targeted
   outcome across 11 selected arms; Voelkel et al. 2026 (Nat. Clim. Change,
   N=13.5k, 10 arms, 0-100 sliders, immediate — structurally the closest
   analog) averaged ≈ +1.2pp, best arms +2.5-3.7, with pilot replications
   mostly null. Meta-analytic centers (Rode g=0.08; van Stekelenburg belief
   g=0.12; Većkalov d=0.06-0.10) side with Voelkel. We center between them,
   nearer Voelkel: **targeted-family average ≈ +1.5pp**.
2. **Retro bracket.** The crowd underpredicted the novel megastudy 2.8x and
   was calibrated (beta=1.0) on the published archive; the benchmark is a
   novel megastudy -> priors sit at/above the raw crowd's typical output
   (crowd pilot mean on trust ≈ +1.1) but well below Vlasceanu-implied 3x.
3. **Proximity gradient.** All 16 arms target trust in climate scientists ->
   trust family gets targeted-outcome magnitudes; climate attitudes are one
   inferential step away (untargeted here, unlike in Vlasceanu/Voelkel);
   behavior most distal. Institutional-trust interventions cluster at
   d=0.05-0.25 (+0.5..+3 points) with frequent nulls; the large effects in
   that literature (7-15% of range) are single-scientist vignette outcomes,
   not class-level trust — we are predicting class-level.
4. **Ceiling/composition.** US trust baseline is high (TISP US ≈3.8/5;
   Pew 76% "fair amount"+), Democrats near ceiling, 22-43pt partisan gaps ->
   sample ATEs are Republican/Independent movement diluted by ~half the
   sample; opt-in panel skews engaged -> slightly higher baselines, less
   headroom.
5. **Damage/repair asymmetry.** Repairing trust is 2-4x harder than damaging
   it (Wingen repair d≈0.1-0.2 vs damage d≈0.5; all repair conditions null)
   -> modest trust gains; distrust falls by less than trust rises.
6. **Backfire base rates.** True mean-level attitudinal backfires are rare
   (P≈2-5%; -1..-4 points when they occur; 0/52 correction issues backfired)
   -> all attitude priors positive; arm-level exceptions (Funding arm's
   COI-salience risk: funding/COI mentions cut integrity perceptions ~d=0.5
   even in defensive framings) are left to the crowd's arm-level deviations.

## Prior table (average-arm ATE vs control, native units)

| # | outcome | prior | pp of range | rationale + anchors |
|---|---|---|---|---|
| 1 | trust_multidimensional | **+1.5** | 1.5 | targeted family; composite slightly stickier than single item; between Voelkel (+1.2 targeted avg) and Vlasceanu (+4.5), recency-weighted; Swiss advocacy field exp +1.5-2% of range |
| 2 | trust_post | **+1.7** | 1.7 | single-item targeted; moves slightly more than 12-item composite |
| 3 | distrust_post | **-1.1** | 1.1 | NEGATIVE. Mirror of trust damped by repair asymmetry (~0.65x trust gain) |
| 4 | funding_perceptions | **+0.8** | 0.8 | trust-adjacent but spending-framed (partisan-anchored); below trust family |
| 5 | policy_role_mean | **+1.0** | 1.0 | legitimacy adjacent to targeted construct; +1..+3 only for explicitly targeted messages (ours mostly aren't); advocacy exposure doesn't hurt trust (Kotcher; Cologna) |
| 6 | inst_trust_mean | **+0.7** | 0.7 | halo to EPA/NASA/NOAA/unis small; federal-government item drags composite |
| 7 | belief_post | **+0.8** | 0.8 | one step from message content; untargeted here (vs Voelkel's targeted +1.16); consensus-arm upside ~+2-3 lives in arm deviations |
| 8 | concern_mean | **+0.7** | 0.7 | affect moves less from calm informational texts; Voelkel concern +1.23 when targeted; one observed backfire (-0.76) keeps this modest |
| 9 | policy_general | **+0.6** | 0.6 | policy = hardest attitude family (Rode g=0.01; Većkalov null; Voelkel +0.76 targeted); single "do more" item is the easier policy ask |
| 10 | policy_specific_mean | **+0.4** | 0.4 | 7-item crystallized policies (taxes etc.); reliably smallest attitude effects |
| 11 | behavior_mean | **+0.4** | 0.4 | intention composite; intentions > behavior but < attitudes; licensing mostly evaporates under bias correction (Rotella g≈0.13->≈0) so not negative |
| 12 | donation_ams | **+$0.04** | 0.4 | real money; behavior-lit central +$0.10 tempered by Voelkel's donation -1.69pp (ns) and trust-texts-are-not-appeals; NEGATIVE TAIL REAL (range -0.10..+0.20) |
| 13 | newsletter_signup | **+0.006** | 0.6 | real click; static-message ATEs on signup-type outcomes +1-3pp, but the offer page is identical across arms (effect flows only through induced interest) -> +0.6pp; range -1..+2pp |

Implied pp-profile ordering (what pooled leaderboard r keys on):
trust_post > trust_multi > distrust(|.|) > policy_role > belief ≈ funding >
concern ≈ inst_trust > newsletter > policy_general > policy_specific ≈
behavior ≈ donation.

## Expected control-group baselines (T1/T2 anchors; refine vs microdata)

| outcome | baseline | source anchors |
|---|---|---|
| trust_multidimensional | ~66 | TISP US ≈3.8/5≈70 general scientists; climate scientists slightly lower, esp. conservatives |
| trust_post | ~64 | single climate-scientist item below composite |
| distrust_post | ~32 | not a perfect mirror (2-D trust/distrust) |
| funding_perceptions | ~58 | "too little" skew on climate research spending |
| policy_role_mean | ~58 | Pew 51% active-role (partisan 36/67); sliders softer than binary |
| inst_trust_mean | ~55 | EPA/NASA/NOAA/unis 55-65; federal gov ~35-40 |
| belief_post | ~66 | Vlasceanu US control 66.5; Voelkel 65.4 |
| concern_mean | ~58 | Voelkel concern 60.4 |
| policy_general | ~66 | Voelkel general policy 68.0 |
| policy_specific_mean | ~63 | popular items (clean water, forests) pull above taxes |
| behavior_mean | ~48 | intention items incl. costly ones (solar, flying) |
| donation_ams | ~$1.85 | 17-20% of $10 endowment; 40-55% zero-spike; modes at 0/5/10 |
| newsletter_signup | ~0.09 | real-link mid-survey opt-in ~10% (3-25% range) |

## Judgment calls for Jamie (ordered by leverage)

1. **The Vlasceanu-Voelkel tension.** The two closest analogs disagree 4x on
   targeted-outcome magnitudes. I centered trust at +1.5 (recency + prereg-
   replication side). If you believe the benchmark resembles Vlasceanu's US
   panel more than Voelkel's, trust priors should rise toward +2.5-3.
2. **Donation sign.** +$0.04 with an explicitly real negative tail
   (Voelkel donations came out negative, ns). Flip to ~0/-0.02 if you think
   passive trust-texts crowd out giving.
3. **Newsletter at +0.6pp.** In pp terms this rivals attitude effects and
   exceeds policy items. Defensible (click behavior is cheap) but if you
   think the identical offer page mutes arm differences, cut to +0.3pp.
4. **Distrust asymmetry at 0.65x.** Literature supports <1x; is 0.65 right?
5. **Anything from your network** (unpublished trust-intervention results)
   that should shift #1 — legal private expertise, not benchmark leakage.

Sign-off on this table locks the aggregation prior vector; w_prior in
[0.25, 0.5] (retro-adopted) then blends it with the crowd at run time.
