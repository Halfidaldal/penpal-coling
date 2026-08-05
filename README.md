# PENPAL — COLING: creativity ratings and story structure

Human creativity annotations for co-written stories, paired with turn-agnostic
structural measures computed from sentence embeddings. Every measure here describes
the completed story artifact; no turn boundaries are used anywhere.

## Layout

```
data/annotations/       human ratings + the story-level measure table
data/stories/           full story text for the whole corpus
data/embeddings/        per-condition full-story embeddings from the PENPAL pipeline
notebooks/              creativity / novelty / reader-appreciation analysis
scripts/                00 build story table -> 01 embed -> 02 metrics
src/story_recurrence/   the metric implementations
output/                 pipeline outputs (gitignored)
```

## The corpus

Three conditions, using the short codes the annotation file uses:

| code | condition | stories | annotated |
|---|---|---|---|
| `ha` | human + LLM | 100 | 91 |
| `hh` | human + human | 36 | 36 |
| `aa` | LLM + LLM | 80 | **20** |

The annotation set covers 147 stories. The remaining 60 LLM-LLM stories were never
rated but their text is included, so structural measures can be computed on the full
216-story corpus and only the rating-linked analyses are restricted to the annotated
subset.

`scripts/00_build_story_table.py` assembles `data/stories/full_stories_all.csv`
(216 rows: `conversation_id`, `full_story`, `condition`) from the per-condition interim
files. `data/annotations/combined_data_2.csv` is the older 156-story text table, kept
for provenance — it truncated the LLM-LLM condition to the 20 annotated stories.

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

```bash
python scripts/00_build_story_table.py
```

```bash
python scripts/01_compute_embeddings.py \
  --input data/stories/full_stories_all.csv \
  --id-col conversation_id --text-col full_story \
  --outdir output --batch-size 16
```

```bash
python scripts/02_compute_metrics.py --outdir output --target-rr 0.05 --theiler 0
```

To restrict to the annotated subset instead, point `--input` at
`data/annotations/penpal_annotations_final.csv` and drop the `--id-col`/`--text-col`
flags — that file already uses the default `id` / `text` column names.

Embedding model defaults to `Kingsoft-LLM/QZhou-Embedding`, matching the main PENPAL
pipeline. It is decoder-based, so left padding and `trust_remote_code` are set
automatically. On CUDA it loads in bfloat16; raise `--batch-size` as memory allows.

## The notebook

`notebooks/PENPAL_creativity_analysis.ipynb` covers stylometry, topic- and word-level
KL novelty/transience/resonance (Barron et al.), surprisal-based versions with a
window-size robustness sweep, inter-annotator reliability, and which measures predict
which judgments. It reads `data/annotations/penpal_annotations_final.csv` and writes
`penpal_measures_all.csv` and `surprisal_window_sweep.csv` back into the same
directory. Paths resolve relative to the repository root, so run it from `notebooks/`
or from the root.

The surprisal section defaults to `google/gemma-4-31b` and needs a GPU. Set
`RUN_SURPRISAL = False` in the config cell to skip it.

## Outputs of the metric pipeline

| file | contents |
|---|---|
| `story_metrics.csv` | **main deliverable** — one row per story, all scalars |
| `lag_profiles.csv` | long format `story_id, lag, mean_sim, n_pairs` |
| `sentence_index.csv` | every sentence with story id and position |
| `stories.csv` | story id, condition, sentence and word counts |
| `sentence_embeddings.npz` | per-story arrays `(n_sentences, dim)` |
| `similarity_matrices.npz` / `distance_matrices.npz` | per-story cosine matrices |
| `recurrence_matrices_rr.npz` / `_eps.npz` | per-story binary recurrence matrices |
| `*_metadata.json` | model, parameters, versions, timestamp |

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

- `penpal-emnlp` — turn-level alignment and narrative agency across the three conditions
- `penpal-nlp4dh` — the earlier NLP4DH paper (Human–AI condition only)
