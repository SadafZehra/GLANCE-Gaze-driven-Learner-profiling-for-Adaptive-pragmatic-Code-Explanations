# GLANCE

### **G**aze-driven **L**earner profiling for **A**daptive, pragmatic **C**ode **E**xplanations

*Can a few glimpses of how a developer reads code tell us who they are, and can that profile be used to choose the right explanation?*

> **Notebook:** `FewShot_Developer_Profiling.ipynb`
> **Type:** End-to-end feasibility study (eye-tracking, few-shot profiling, pragmatic explanation selection)
> **Status:** Working pipeline on **real eye-tracking data**; the headline findings are largely **null or modest** (see [Results](#results) and [Limitations](#limitations-and-threats-to-validity)). This README reports them as they are.

---

## Abstract

Good explanations depend on the reader: an experienced developer wants a short, precise answer, while a novice may need step-by-step help. GLANCE asks whether this reader profile can be inferred from only a handful of short gaze snippets ("shots") and then used to choose an explanation. The study has two parts.

**Part 1 (profiling).** Using Java trials from the EMIP eye-movement dataset (154 participants, 61,130 fixations), each fixated token is aligned to the exact source code and assigned a surprisal value from two code language models (0.5B and 1.5B, used as proxies for a less and a more experienced reader). Fixations are merged into gaze events and grouped into 20-event windows. We test (a) whether LM surprisal predicts gaze duration separately for novices and experienced developers, (b) whether unseen developers can be **identified** (5-way, 1/3/5 shots) with a learned metric, and (c) whether their **expertise class** can be predicted from a few windows.

**Part 2 (explanation).** A code-LM speaker generates 96 candidate explanations for two Java programs across four styles (concise, stepwise, trace, analogy), with or without retrieval grounding and with or without demonstrations. Each candidate is scored for hallucination, leakage, correctness, informativity (to a listener LM), and a class-specific predicted reading time. A **Rational Speech Acts (RSA)-style** speaker selects the explanation that maximises informativity minus weighted reading cost.

**Findings in brief:** the pipeline runs end to end and unseen developers can be identified above chance (34.5% vs. 20% chance at 5 shots with a learned metric), but **expertise class prediction is at chance level**, LM surprisal does **not** reliably predict gaze duration in this setup, and as a consequence the RSA selection ended up **identical for both classes**. The notebook is therefore best read as a documented first attempt with an honest negative result on class-aware personalisation, plus a reusable pipeline.

---

## Research Questions

| ID | Question | Outcome |
|----|----------|---------|
| **RQ1** | Does code-LM surprisal predict gaze duration, and does the better-matching LM differ between novices and experienced developers? | No reliable effect (held-out gain about 0) |
| **RQ2** | Can unseen developers be identified from K = 1, 3, 5 windows of gaze behaviour? | Yes, modestly above chance; learned metric helps |
| **RQ3** | Can developer expertise class be predicted from K windows? | No, at chance level |
| **RQ4** | Which gaze features separate the classes, and can the prediction be explained? | Features are listed, but with chance-level accuracy they should not be interpreted substantively |
| **RQ5** | Do retrieval grounding and demonstrations change hallucination and leakage in generated explanations? | No significant effect of demonstrations (p = 0.725); small samples |
| **RQ6** | Does profile-aware RSA choose different explanations for different classes? | No, choices were identical for both classes |

---

## Pipeline Overview

```
        EMIP eye-tracking data (Java trials)
                     │
   align every fixated token to the exact source code
                     │
      surprisal of each token under two code LMs
        (0.5B = "less experienced", 1.5B = "more experienced")
                     │
        gaze events  →  20-event windows ("shots")
                     │
   ┌─────────────────┼─────────────────────────────┐
   ▼                 ▼                             ▼
 Surprisal vs.   Few-shot identification      Few-shot class prediction
 gaze duration   (5-way, learned NCA metric)  (prototype / logistic reg.)
 by class                                            │
                                         interpretability (coefficients, tree)
                     │
   class-specific reading-cost model (predicted gaze time)
                     │
  Speaker LM → 96 candidate explanations (style x grounding x demos)
                     │
 hallucination • leakage • correctness • informativity • reading cost
                     │
        RSA choice: informativity − α · reading cost
```

---

## Data

| Item | Value |
|------|-------|
| Source | EMIP eye-movement-in-programming data; corrected fixation file from the EMIP Toolkit repository (`finalset_line_part.csv`), expertise labels from `emip_metadata.csv` |
| Language / stimuli | **Java only**: `Rectangle` and `Vehicle` programs, 4 stimulus images (two variants of each) |
| Fixations | 61,130 from 154 participants |
| Token alignment | All 496 areas of interest (AOIs) aligned to exact source text |
| Gaze events | 36,363 (consecutive fixations on the same token merged; mean 1.68 fixations per event) |
| Expertise labels | `expertise_programming`: none/low → **novice** (37), medium/high → **experienced** (117) |
| Windows | 20 gaze events each, within trials only. Developers with enough windows: **98** (77 experienced, 21 novice), 1,418 windows |
| Split | By developer: 58 training, 40 unseen test developers (stratified, 60/40) |

If the metadata file is missing, the notebook falls back to k-means "slow/fast reader" clusters; in the reported run the real expertise labels were found and used.

---

## Method

### Part 1: Gaze profiling

**Surprisal.** Per-token surprisal (bits) from `Qwen2.5-Coder-0.5B` ("small") and `Qwen2.5-Coder-1.5B` ("large"), computed over the full program with a short `// Java program` context.

**Gaze-event features.** Log duration, fixation count, regression and line regression (eyes move back), jump size, identifier/literal-operator flags, in-`main` flag, token length, normalised position in trial.

**Surprisal test (RQ1).** Per class, a linear model of within-participant log-duration with baseline predictors (length, token type, trial position) is compared with the same model plus surprisal, using **participant-grouped 5-fold cross-validation** and held-out Gaussian log-likelihood gain (Demberg-style test), with bootstrap CIs over participants.

**Window features (12).** `mean_log_dur, sd_log_dur, regression, line_regression, jump, refixation, is_name, is_litop, in_main, mean_surp, surp_sensitivity, length`, standardised on training developers only.

**Few-shot identification (RQ2).** 5-way, K-shot episodes over unseen test developers (600 episodes, 3 query windows per developer), nearest-prototype classification in (a) standardised feature space and (b) a **Neighbourhood Components Analysis (NCA)** metric (8 components) trained on training developers to discriminate individuals.

**Few-shot class prediction (RQ3).** Class prototypes and class-balanced logistic regression on K-window mean profiles (40 random profiles per developer); balanced accuracy over unseen developers with bootstrap CIs.

**Interpretability (RQ4).** Logistic-regression coefficients, permutation importance on unseen developers, a depth-2 decision tree, and a per-developer explanation of one prediction.

### Part 2: Explanation generation and selection

**Reading-cost model.** Per class, absolute log gaze duration regressed on token length and the class-matched LM surprisal; applied word by word to predict reading time (seconds) for a candidate explanation.

**Speaker.** `Qwen2.5-Coder-1.5B-Instruct`, greedy decoding, max 180 new tokens.

**Candidate grid (96 explanations).** 2 programs x 3 questions x 2 grounding conditions (`plain`, retrieval-augmented `rag` using TF-IDF over code lines and a small notes base) x 2 demonstration settings (0 or 2 unrelated-class demonstrations) x 4 styles.

**Metrics per candidate.**

| Metric | Definition |
|--------|-----------|
| Hallucination | Mentions an identifier not in the program, the question, or a Java allow-list |
| Leakage | Mentions an identifier that exists only in the demonstrations (e.g. `Circle`, `deposit`) |
| Correctness | For closed questions, all expected values appear (regex check) |
| Informativity | Increase in listener-LM log-probability of the program given the explanation (nats) |
| Reading cost | Predicted seconds from the class gaze model |

**RSA choice.** Utility = informativity − α · reading cost under the class-matched listener, softmax speaker with λ = 1, main α = 1; compared with α = 0 (literal), "always concise", and "always stepwise".

---

## Results

All numbers come from the saved notebook run (Colab GPU). Chance levels are stated where relevant.

### RQ1: Does code-LM surprisal predict gaze duration?

| Class | LM | Held-out ΔLL / event | 95% CI | Surprisal coef. | Participants |
|-------|----|---------------------:|--------|----------------:|-------------:|
| Experienced | small | -0.0000 | [-0.0001, 0.0000] | 0.0013 | 117 |
| Experienced | large | -0.0000 | [-0.0001, 0.0001] | 0.0016 | 117 |
| Novice | small | -0.0000 | [-0.0005, 0.0006] | -0.0027 | 37 |
| Novice | large | -0.0001 | [-0.0006, 0.0006] | -0.0025 | 37 |

Adding surprisal gives **no measurable improvement on held-out readers** for either class or model, and the intervals include zero. The "reader-matched" LM was `small` for both classes, but this choice rests on differences that are effectively noise. The idea that a smaller LM mirrors a less experienced reader is therefore **not supported** by this analysis.

### RQ2: Few-shot identification of unseen developers (5-way, chance = 0.20)

| Shots | Standardised features | NCA metric |
|------:|----------------------:|-----------:|
| 1 | 0.248 ± 0.009 | 0.281 ± 0.010 |
| 3 | 0.286 ± 0.010 | 0.322 ± 0.010 |
| 5 | 0.311 ± 0.009 | **0.345 ± 0.011** |

Identification improves with more shots and the learned NCA metric beats raw features at every K, but absolute accuracy is moderate (up to 34.5%). Intervals are over random episodes drawn from the same 40 test developers, so they reflect episode sampling, not uncertainty across developers.

### RQ3: Few-shot prediction of expertise class (balanced accuracy, chance = 0.50)

| Method | 1 shot | 3 shots | 5 shots |
|--------|-------:|--------:|--------:|
| Class prototypes | 0.485 [0.416, 0.584] | 0.506 [0.388, 0.619] | 0.424 [0.307, 0.561] |
| Logistic regression | 0.498 [0.420, 0.573] | 0.470 [0.405, 0.556] | 0.446 [0.367, 0.530] |

All intervals include 0.50, and several point estimates are below it. **Expertise class could not be predicted from gaze windows in this setup.**

### RQ4: Interpretability (5-shot profiles)

Largest logistic-regression coefficients (positive = towards *novice*) and permutation importance on unseen developers:

| Feature | Coefficient | Permutation importance |
|---------|------------:|-----------------------:|
| is_name | -1.400 | 0.031 |
| is_litop | -0.817 | 0.001 |
| jump | -0.740 | 0.027 |
| surp_sensitivity | -0.728 | 0.020 |
| line_regression | 0.660 | 0.010 |
| sd_log_dur | -0.496 | 0.011 |

The depth-2 decision tree reached 0.447 accuracy on unseen developers, and one of its splits assigns *novice* to both leaves, a sign of a degenerate fit. **Because the classifier itself is at chance, these coefficients should not be read as evidence of what distinguishes novices from experienced developers.** They are included because the interpretability code is reusable.

A per-developer demo (developer 6, true class *experienced*) was classified correctly with p = 0.68, but a single correct case is not evidence of accuracy given the aggregate results.

### Reading-cost models

| Class | Fitted model (log ms) | Listener LM |
|-------|----------------------|-------------|
| Experienced | 4.848 + 0.029·length + 0.003·surprisal | small |
| Novice | 4.839 + 0.026·length + 0.002·surprisal | small |

The two class models are nearly identical, which limits how different the downstream class-specific costs can be.

### RQ5: Faithfulness of generated explanations (96 candidates)

| Grounding | Demos | Any hallucination | Leakage | Correct (closed Qs) |
|-----------|------:|------------------:|--------:|--------------------:|
| plain | 0 | 0.208 [0.083, 0.375] | 0.167 | 0.562 |
| plain | 2 | 0.042 [0.000, 0.125] | 0.000 | 0.812 |
| rag | 0 | 0.083 [0.000, 0.208] | 0.125 | 0.562 |
| rag | 2 | 0.167 [0.042, 0.333] | 0.042 | 0.688 |

Effect of two demonstrations on hallucination: **-0.042** (paired permutation test, p = 0.725, 48 pairs). No condition is clearly better, and the sample is small (24 candidates per condition, 16 for correctness). The retrieval-grounded 0-shot condition has the lowest-to-moderate hallucination rate, but the intervals overlap heavily.

### RQ6: Profile-aware RSA selection

| Class | Policy | Informativity | Cost (s) | Words | Hallucination | Correct |
|-------|--------|--------------:|---------:|------:|--------------:|--------:|
| Experienced | RSA (α = 1) | 47.85 | 14.15 | 96.5 | 0.167 | 0.5 |
| Experienced | Literal (α = 0) | 48.33 | 16.11 | 109.2 | 0.000 | 0.5 |
| Experienced | Always concise | 21.88 | 8.85 | 60.2 | 0.000 | 1.0 |
| Experienced | Always stepwise | 37.69 | 16.04 | 108.5 | 0.167 | 0.5 |
| Novice | RSA (α = 1) | 47.85 | 13.80 | 96.5 | 0.167 | 0.5 |
| Novice | Literal (α = 0) | 48.33 | 15.70 | 109.2 | 0.000 | 0.5 |
| Novice | Always concise | 21.88 | 8.62 | 60.2 | 0.000 | 1.0 |
| Novice | Always stepwise | 37.69 | 15.64 | 108.5 | 0.167 | 0.5 |

- RSA reduces predicted reading cost relative to the literal speaker (about 14.2 s vs. 16.1 s for experienced readers) at almost the same informativity.
- **Both classes received the same styles** (trace 4 times, stepwise 2 times out of 6 questions) and the same informativity and word counts; only the cost estimate differs slightly. The α-sweep is also identical across classes (mean length falls from 109 to 84 words as α goes from 0 to 4).
- The **concise** style had the highest correctness (1.0 on closed questions) and zero hallucination, but the lowest informativity score, so the informativity metric favours long explanations without necessarily measuring correctness.
- In the end-to-end demo, an *experienced* developer was given the *stepwise* style, which is described in the prompt as aimed at beginners. This is consistent with the finding that no real personalisation occurred.

---

## Limitations and Threats to Validity

1. **Class prediction is at chance.** Everything downstream that depends on the class (interpretability, personalised RSA) inherits this. The notebook demonstrates the pipeline, not a working personalisation effect.
2. **Class imbalance and small test set.** 77 experienced vs. 21 novice developers after filtering, and only 40 unseen test developers (about 30 experienced and 10 novice), which makes class-level estimates noisy.
3. **Coarse labels.** Self-reported programming expertise (none/low vs. medium/high) is a weak proxy for reading behaviour.
4. **Very small stimulus set.** Two Java programs (four images) shared by all participants, and two programs for explanation generation. Findings may not generalise to other code or languages. Python trials in the dataset were not used.
5. **Features and windows.** 12 generic features over 20-event windows may be too coarse to capture expertise-related patterns, and windows from the same developer share trial context.
6. **Surprisal is a weak predictor here.** Held-out gains are about zero, and surprisal was computed with small base models on full programs without natural-language context. The "small LM = novice reader, large LM = expert reader" assumption is not validated.
7. **Metrics are heuristic.** Hallucination is an identifier-based regex check; correctness only checks that expected numbers appear (it can pass explanations whose reasoning is wrong, as seen in a printed example); informativity is listener-LM log-probability gain, which grows with length.
8. **Small candidate sample.** 96 explanations, one generation per cell (greedy), no human judgments of quality or readability.
9. **Reading cost transfer.** Gaze-duration models of code reading are applied to natural-language explanations, a transfer that is not validated.
10. **Identification intervals.** Episode-based CIs treat episodes as independent although they are drawn from the same 40 developers.
11. **Style collapse with demonstrations.** In the printed sample, concise and analogy candidates with 2 demonstrations produced the same 28-word output, which suggests that demonstrations can wash out the style instruction for this small speaker model.

---

## Suggested Extensions

- **Better labels and more data:** use continuous measures (e.g. task performance or years of experience), use all EMIP stimuli including Python, and add a second eye-tracking dataset for out-of-sample validation.
- **Richer profiling:** sequence models or scan-path features (e.g. reading order relative to program structure), and window sizes tuned per class.
- **Check the surprisal assumption:** compare several LM sizes and families, add natural-language context, and test against established reading-time covariates.
- **Stronger evaluation of explanations:** human ratings of clarity and correctness, execution-based correctness checks, and multiple samples per condition.
- **Make RSA personalisation testable:** use classes with demonstrably different reading costs and utilities (e.g. from a user study), and report whether choices differ when profiles differ.
- **Related work and references:** this README does not cite literature. A references section should be added after verifying sources for eye tracking in programming, RSA/pragmatic generation, few-shot metric learning, and faithfulness evaluation.

---

## Reproducibility

### Requirements

- Python 3.9+, GPU recommended (candidate generation took about 590 s on a Colab GPU)
- Packages (auto-installed by the first cell if missing): `transformers`, `accelerate`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `torch`, `requests`
- Network access to clone the EMIP Toolkit repository (`github.com/MyNameBeJohn/EMIP-Toolkit`) and, optionally, to download the full EMIP release to obtain `emip_metadata.csv`. If the metadata cannot be downloaded, upload `emip_metadata.csv` manually; otherwise the notebook uses k-means gaze clusters instead of real expertise labels.

### Run

1. Open `FewShot_Developer_Profiling.ipynb` in Jupyter, VS Code, or Google Colab (GPU runtime recommended).
2. Run all cells from top to bottom.
3. Outputs are saved to `w2935_work/outputs/`.

### Key configuration

| Parameter | Default | Meaning |
|-----------|--------:|---------|
| `SEED` | 42 | Global seed |
| `SMALL_LM` / `LARGE_LM` | Qwen2.5-Coder 0.5B / 1.5B | Listener (reader proxy) models |
| `SPEAKER_LM` | Qwen2.5-Coder-1.5B-Instruct | Explanation generator |
| `WINDOW` | 20 | Gaze events per shot |
| `SHOTS` | `[1, 3, 5]` | Few-shot budgets |
| `N_WAY`, `N_QUERY` | 5, 3 | Episode design |
| `N_EPISODES` | 600 | Random identification episodes |
| `PROFILE_SAMPLES` | 40 | Random K-window profiles per developer |
| `MAX_NEW_TOKENS` | 180 | Explanation length cap |
| `ALPHAS`, `MAIN_ALPHA` | 0 to 4, 1.0 | RSA cost weights |
| `RSA_LAMBDA` | 1.0 | Speaker rationality |
| `N_BOOT` | 1000 | Bootstrap repetitions |

### Output files (`w2935_work/outputs/`)

| File | Content |
|------|---------|
| `surprisal_fit_by_class.csv` | Held-out surprisal gain by class and LM |
| `fewshot_developer_identification.csv` | Identification accuracy by shots and space |
| `fewshot_class_prediction.csv` | Class prediction by method and shots |
| `interpretability_coefficients.csv` | Coefficients and permutation importance |
| `candidate_explanations.csv`, `candidate_explanations_scored.csv` | Raw and scored explanations |
| `faithfulness_by_condition.csv` | Hallucination, leakage, correctness rates |
| `rsa_choices.csv`, `rsa_alpha_sweep.csv` | RSA selections and α sensitivity |
| `summary.json` | Headline numbers |
| `fig1` to `fig5` (`.png`) | Figures for each analysis |

---

## Suggested Citation

```
GLANCE: Gaze-driven Learner profiling for Adaptive, pragmatic Code Explanations.
Few-shot developer profiling from eye movements and profile-aware explanation selection.
[Your Name], [Year]. Code and notebook: FewShot_Developer_Profiling.ipynb
```

## License

Add your preferred license here (e.g. MIT). Note that the EMIP data and the EMIP Toolkit have their own licenses and terms of use.
