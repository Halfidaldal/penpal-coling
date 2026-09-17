# PENPAL — COLING: creativity ratings and story structure

Human creativity annotations for co-written stories, paired with turn-agnostic
structural measures computed from sentence embeddings. Every measure here describes
the completed story artifact; no turn boundaries are used anywhere.

## Layout

```
data/
├── interim/
│   ├── annotations/       human ratings and raw annotation tables
│   ├── stories/           full story text for the whole corpus
│   └── embeddings/        sentence embeddings, index, and interim arrays
└── processed/
    ├── surprisal/         story-level and window-level surprisal NTR metrics
    ├── embeddings/        embedding metadata
    └── rqa/               story recurrence metrics, matrices, and lag profiles
scripts/                00 build story table -> 01 embed -> 02 metrics -> 03 surprisal
                        00a simulates the cross-model LLM-LLM condition
src/story_recurrence/   the recurrence metric implementations
src/surprisal_ntr/      the surprisal and stylometry implementations
src/cross_sim/          the cross-model simulation
```

## The corpus

Three conditions, using the short codes the annotation file uses:

| code | condition | stories | annotated |
|---|---|---|---|
| `ha` | human + LLM | 100 | 91 |
| `hh` | human + human | 36 | 36 |
| `aa` | LLM + LLM, each model with itself | 80 | **20** |
| `aa_cross` | LLM + LLM, full model × model grid | 112 | – |

`aa_cross` is generated here rather than inherited from the EMNLP corpus; see
**The cross-model condition** below. It is absent until you run
`scripts/00a_simulate_cross_model.py`, and every step skips it cleanly when it is.

The annotation set covers 147 stories. The remaining 60 LLM-LLM stories were never
rated but their text is included, so structural measures can be computed on the full
216-story corpus and only the rating-linked analyses are restricted to the annotated
subset.

`scripts/00_build_story_table.py` assembles `data/interim/stories/full_stories_all.csv`
(216 rows: `conversation_id`, `full_story`, `condition`) from the per-condition interim
files.

## The cross-model condition

The EMNLP LLM-LLM condition paired every model with itself, which confounds two
things: being written by an LLM, and being written by *one* LLM with its own
prose to continue. `scripts/00a_simulate_cross_model.py` separates them by
running the full model × model grid.

With M models there are M² ordered cells. For the four PenPal models that is 16:
4 self-pairs on the diagonal, 12 cross-model cells off it. Order matters off the
diagonal, because the opener writes into an empty story and the responder always
writes into someone else's prose, so *gpt-4.1 opens to claude* and *claude opens
to gpt-4.1* are different cells. On the diagonal the distinction is vacuous, but
the cell is kept so the grid stays a clean M × M factorial and the same-model
condition is regenerated under identical settings rather than borrowed.

Default is 7 stories per cell: **112 stories, 1120 turns, 2240 generation
calls**. `stories_per_pair: 6` gives 96 instead; both are balanced. `--n-stories
100` hits an exact total by giving the first four cells one extra story, which
costs the balance.

```bash
python scripts/00a_simulate_cross_model.py --dry-run   # grid and call count, no API
python scripts/00a_simulate_cross_model.py             # the real run
python scripts/00a_simulate_cross_model.py --resume    # continue after an interruption
```

Generation is unchanged from PenPal-EMNLP `src/nes/simulation.py`: both sides get
the same prompt, per-provider context handling matches the experiment's adapters,
responses that echo the partner are stripped, and both outputs are truncated by
2–5 words before being saved and passed on. `src/cross_sim/providers.py` and
`prompts.py` are verbatim copies, so the new stories stay comparable with the
existing 80; changes to how a turn is generated belong upstream first.

Needs `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY` in the
environment or a `.env` file. Cells whose models have no key are skipped with a
warning rather than failing the run, and each finished story is appended to the
raw CSV immediately, so an interrupted run loses at most one story.

### What it writes

| file | contents |
|---|---|
| `data/interim/stories/aa_cross_turns.csv` | one row per turn; the checkpoint `--resume` reads |
| `data/interim/stories/ai-ai-cross_stories_full_text_filtered.csv` | one row per story, same shape as the other conditions |
| `data/interim/stories/aa_cross_id_map.csv` | `story_id` → `conversation_id` |
| `data/interim/stories/aa_cross_run_metadata.json` | settings, grid, per-cell counts |

Story-level model columns: `model_starter` and `model_responder` are the ones to
analyse by — they are invariant to the author-column counterbalancing. The
`author_1`/`author_2` labels are swapped for a seeded ~50% of stories, exactly as
the EMNLP pipeline did, so `author_1` does not become a synonym for "went first";
`starter` records which column holds the opener, and `model_author_1` /
`model_author_2` follow the swap. `pair_id` is the ordered cell
(`starter>responder`), `dyad_id` collapses the two directions of a cross pair,
and `model_id` mirrors `pair_id` because no single model id is meaningful for a
cross-model story.

Both AA conditions get `llmness: 2` in `config.yaml` — both sides are LLMs, only
the pairing differs. The tie means the monotone trend test should run on one AA
condition at a time; compare the two with the same-vs-cross contrast instead.

## Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

`scipy` is optional; without it the exponential decay fit returns NaN and the
power-law and linear fits still work.

## Run

Three steps. Step 01 is GPU-bound and slow; step 02 is CPU-only and fast, so
thresholds can be re-tuned without re-embedding.

Step 00a is optional and upstream of the rest: it generates the cross-model
condition by calling the model APIs, so run it only when you want that condition
(see **The cross-model condition**).

```bash
python scripts/00a_simulate_cross_model.py   # optional, costs API calls
python scripts/00_build_story_table.py
```

```bash
python scripts/01_compute_embeddings.py \
  --input data/interim/annotations/penpal_annotations_final.csv \
  --id-col id --text-col text \
  --outdir data/interim/embeddings --batch-size 16
```

```bash
python scripts/02_compute_metrics.py \
  --indir data/interim/embeddings \
  --outdir data/processed/rqa
```

```bash
python scripts/03_compute_surprisal_metrics.py --config config.yaml
```

`data/interim/annotations/penpal_annotations_final.csv` is the master 1-row-per-story table containing all 216 unique stories (80 LLM-LLM, 100 Human-AI, 36 Human-Human). Running step 01 and step 02 on this file automatically generates all sentence embeddings, recurrence metrics, and attaches all human quality ratings into `data/processed/rqa/story_metrics.csv`.

Embedding model defaults to `Kingsoft-LLM/QZhou-Embedding`, matching the main PENPAL
pipeline. It is decoder-based, so left padding and `trust_remote_code` are set
automatically. On CUDA it loads in bfloat16; raise `--batch-size` as memory allows.

## Novelty, Transience, and Resonance Analysis

The novelty, transience, and resonance pipeline consists of:
1. `scripts/03_compute_surprisal_metrics.py`: Computes stylometric features and LM surprisal-based Novelty, Transience, and Resonance (NTR) metrics across sliding word windows. It outputs story-level aggregates (`data/processed/surprisal/penpal_measures_all.csv`) and window-level metrics (`data/processed/surprisal/surprisal_window_level.csv`).
2. `analysis/novelty_transience_resonance.Rmd`: R Markdown report performing condition comparisons, inter-annotator reliability, predictor correlations with human judgments, and **Resonance ~ Novelty * Condition** interaction modeling to analyze local innovation bias across conditions.


## Outputs of the metric pipeline

| location | file | contents |
|---|---|---|
| `data/processed/rqa/` | `story_metrics.csv` | **main deliverable** — one row per story, all scalars |
| `data/processed/rqa/` | `lag_profiles.csv` | long format `story_id, lag, mean_sim, n_pairs` |
| `data/interim/embeddings/` | `sentence_index.csv` | every sentence with story id and position |
| `data/interim/embeddings/` | `stories.csv` | story id, condition, sentence and word counts |
| `data/interim/embeddings/` | `sentence_embeddings.npz` | per-story arrays `(n_sentences, dim)` |
| `data/processed/rqa/` | `similarity_matrices.npz` / `distance_matrices.npz` | per-story cosine matrices |
| `data/processed/rqa/` | `recurrence_matrices_rr.npz` / `_eps.npz` | per-story binary recurrence matrices |
| `data/processed/surprisal/` | `penpal_measures_all.csv` | story-level surprisal & stylometry table |
| `data/processed/surprisal/` | `surprisal_window_level.csv` | window-level surprisal terms |

## Why these measures

They replace the LDA topic-KL family, which was confounded with sentence length
(`topic_novelty` ↔ mean sentence length ρ = +.76; `topic_novelty` ↔ `topic_entropy`
ρ = −.88; `topic_novelty` ↔ `topic_transience` ρ = +.98). Fixed-dimension sentence
embeddings do not have LDA's length-dependent posterior concentration, so that
confound is largely removed.

Three families:

| family | question | module |
|---|---|---|
| Recurrence (RQA) | does the story echo its own earlier material? | `rqa.py` |
| Trajectory geometry | what shape is the story's path through meaning space? | `trajectory.py` |
| Semantic decay | how fast does the story forget what it said? | `decay.py` |

### Thresholding: read this before interpreting RQA

Binarising the distance matrix requires a threshold ε, and the choice drives every RQA
scalar. Both standard strategies are computed and exported side by side.

**`rr_*` — fixed recurrence rate (use these for cross-condition structure).**
ε is set per story as the `--target-rr` quantile of that story's own distances, so RR is
constant by construction and `DET`, `LAM`, `L`, `ENTR` describe how recurrences are
*organised* rather than how many there are. This is the right default here: conditions
differ in baseline self-similarity, and with a single fixed ε a globally more repetitive
condition scores higher on every RQA measure at once — the same "one underlying factor
wearing several names" problem that made topic-novelty uninterpretable.

**`eps_*` — one corpus-wide ε.** Here `eps_RR` is itself the informative variable (how
self-similar is this story at all), but the structure measures are no longer independent
of it, so do not read `eps_DET` and `eps_RR` as separate findings.

`--theiler` excludes the band `|i−j| <= theiler`. `0` removes only the trivial
self-match; `1` also removes adjacent sentences, whose similarity is a local cohesion
effect already measured by `sim_lag1` and by the stylometric
`adjacent_sentence_similarity`.

## Metric reference

### RQA (prefixes `rr_` and `eps_`)

| metric | definition | narrative reading |
|---|---|---|
| `RR` | recurrent cells ÷ eligible cells | how often the story revisits earlier states |
| `DET` | share of recurrent points on diagonal lines ≥ `l_min` | ordered repetition — the story re-runs *sequences* |
| `L` | mean diagonal line length | typical length of an echo, in sentences |
| `L_MAX` | longest diagonal | the single longest echoed run |
| `DIV` | 1 / `L_MAX` | divergence |
| `ENTR` | Shannon entropy (bits) of diagonal-length distribution | variety of echo lengths; low = all echoes alike |
| `RATIO` | `DET` / `RR` | structure per unit of recurrence |
| `LAM` | share of recurrent points on vertical lines ≥ `v_min` | dwelling — one sentence matches a stretch of consecutive others |
| `TT` | mean vertical line length | how long it dwells |
| `V_MAX` | longest vertical line | |

`eps_story` is the per-story fixed-RR threshold; low values mean the story needed a
tight threshold to reach the target RR, i.e. it is globally self-similar.

### Trajectory geometry

| metric | definition |
|---|---|
| `path_length` | Σ‖eᵢ₊₁ − eᵢ‖ — total ground covered |
| `mean_step`, `sd_step`, `max_step` | step-size distribution; `sd_step` high = a few big leaps among small moves |
| `net_displacement` | ‖eₙ − e₁‖ |
| `straightness` | `net_displacement / path_length` ∈ [0,1]; 1 = consistently directed, ~0 = wanders or returns |
| `radius_gyration` | RMS distance from the story centroid — order-free spread |
| `centroid_norm` | ‖mean(eᵢ)‖ ∈ [0,1]; high = every sentence points the same way |
| `msd_alpha` | slope of log MSD(lag) vs log lag. ≈1 random-walk, >1 directed, <1 confined |
| `msd_alpha_r2` | fit quality for `msd_alpha` |
| `circularity` | cosine between mean of first and last `edge_frac` of sentences; high = the ending returns to the opening |
| `first_last_cos` | cosine between the first and last sentence |
| `mean_pairwise_sim`, `sd_pairwise_sim` | order-free self-similarity |

### Semantic decay

Profile: `sim(k) = mean_i cos(eᵢ, eᵢ₊ₖ)`, over lags with ≥ `--decay-min-pairs`
supporting pairs, capped at `--decay-max-lag-frac` × story length.

| metric | definition |
|---|---|
| `decay_lambda` | rate in `sim(k) = C + A·exp(−λk)`; higher = faster forgetting |
| `decay_C` | asymptotic floor — persistent global similarity the story never leaves |
| `decay_A` | decaying amplitude — the locally-bound part |
| `decay_half_life` | ln2 / λ, in sentences |
| `decay_exp_r2` | exponential fit quality |
| `decay_fit_at_bound` | 1 = a fitted parameter sat on its bound, i.e. the fit is degenerate — **drop these rows before modelling λ** |
| `decay_beta` | power-law exponent from log sim ~ −β log k |
| `decay_pow_r2` | power-law fit quality |
| `decay_lin_slope`, `decay_lin_r2` | assumption-free linear slope; always defined |
| `decay_half_distance` | fit-free: interpolated lag where sim falls halfway from `sim_lag1` to the profile minimum |
| `sim_lag1`, `sim_maxlag` | raw anchors |
| `decay_max_lag`, `decay_n_lags` | profile extent (use to screen short stories) |

Prefer `decay_lambda` when `decay_exp_r2` is high; fall back to `decay_half_distance`
and `decay_lin_slope` otherwise. `lag_profiles.csv` lets you model the profile directly.

## Tuning `--target-rr` after the first run

Step 02 prints how many stories have NaN `rr_L` / `rr_ENTR` — stories where no diagonal
line of length ≥ `l_min` formed, so there was nothing to average. If that count is large
(say >25% of stories), the target recurrence rate is too low for these story lengths;
re-run step 02 with `--target-rr 0.10` or `0.15`. Step 02 is CPU-only and takes seconds,
so sweeping this is cheap and needs no re-embedding.

Likewise check the `decay_fit_at_bound` count. A handful is normal; many means the lag
profiles are close to flat and `decay_lambda` should not be the headline decay measure.

## Notes for analysis

- Stories with fewer than 3 sentences yield NaN throughout. Screen on `n_sentences` and
  `decay_n_lags` before modelling.
- Many metrics scale with story length. `n_words` and `n_sentences` are exported for use
  as covariates — worth adjusting for, given how the length confound behaved in the
  topic-KL family.
- These measures are not independent of each other (`RR`, `mean_pairwise_sim` and
  `decay_C` all index global self-similarity). Choose a small confirmatory set with
  directional hypotheses rather than correlating everything against every rating.

## Related repositories

Independent, not submodules — the analyses share a corpus but nothing else.

- `penpal-emnlp` — turn-level alignment and narrative agency across the three conditions. Its
  `src/nes/simulation.py` is the source the cross-model simulation here was ported from; it is
  the paper's replication pipeline and is not modified by this work
- `penpal-nlp4dh` — the earlier NLP4DH paper (Human–AI condition only)
