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


## Outputs of the Metric Pipeline & Data Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                             story_metrics.csv                           │
│                                 (df_raw)                                │
│    One row per story — metadata, RQA (rr_ & eps_), Trajectory, Decay,   │
│                   Surprisal summaries, and Human Ratings               │
└─────────────────────────────────────────────────────────────────────────┘
        │                                         │
        ▼                                         ▼
┌───────────────────────┐             ┌───────────────────────┐
│   lag_profiles.csv    │             │  sentence_index.csv   │
│       (df_lag)        │             │  (df_sentence_index)  │
│ Long format: sim vs   │             │ Long format: sentence │
│ sentence distance lag │             │ text & position index │
└───────────────────────┘             └───────────────────────┘
        │                                         │
        └────────────────────┬────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│                surprisal_unit_level.csv                 │
│ Sentence-level Surprisal & History-Aware Geometry joins  │
│ (s_novelty, s_transience, step_in, novelty_centroid,    │
│                     path_detour)                        │
└─────────────────────────────────────────────────────────┘
```

| location | file | contents |
|---|---|---|
| `data/processed/rqa/` | `story_metrics.csv` | **main deliverable (`df_raw`)** — one row per story, all scalars |
| `data/processed/rqa/` | `lag_profiles.csv` | **`df_lag`** — long format `story_id, lag, mean_sim, n_pairs` |
| `data/interim/embeddings/` | `sentence_index.csv` | **`df_sentence_index`** — every sentence with story id and position |
| `data/interim/embeddings/` | `stories.csv` | story id, condition, sentence and word counts |
| `data/interim/embeddings/` | `sentence_embeddings.npz` | per-story sentence arrays `(n_sentences, 3584)` |
| `data/processed/rqa/` | `similarity_matrices.npz` / `distance_matrices.npz` | per-story $N \times N$ cosine matrices |
| `data/processed/rqa/` | `recurrence_matrices_rr.npz` / `_eps.npz` | per-story binary recurrence matrices |
| `data/processed/surprisal/` | `penpal_measures_all.csv` | story-level surprisal & stylometry table |
| `data/processed/surprisal/` | `surprisal_window_level.csv` | 14-word fixed window level surprisal terms |
| `data/processed/surprisal/` | `surprisal_unit_level.csv` | sentence-level surprisal terms (`s_novelty`, `s_transience`, `s_resonance`) |

---

## Methodological Rationale: Why Are There So Many Variable Variants?

The core structural concepts expand into multiple specific columns across the dataframes for **four fundamental methodological reasons**:

1. **Two Complementary RQA Thresholding Schemes (`rr_` vs `eps_`)**:
   - Every RQA metric (`RR`, `DET`, `L`, `L_MAX`, `DIV`, `ENTR`, `RATIO`, `LAM`, `TT`, `V_MAX`) is computed **twice**:
     - **`rr_*` (Fixed-RR)**: Uses a per-story threshold tuned so every story achieves a target recurrence rate (e.g. 5% or 10%). This isolates **structural pattern organization** independent of story length or overall repetition volume.
     - **`eps_*` (Fixed-Epsilon)**: Uses a single fixed global threshold ($\varepsilon$) across the whole corpus. This measures **raw thematic density** and baseline repetitiveness.

2. **Three Functional Fitting Models for Semantic Decay (`decay_*`)**:
   - Because story memory decay may not follow a single functional shape, three separate decay models are fitted side-by-side:
     - **Exponential Fit**: `decay_lambda` (decay rate), `decay_A` (amplitude), `decay_C` (floor), `decay_half_life`.
     - **Power-law Fit**: `decay_beta` (scale-free exponent).
     - **Linear Fit**: `decay_lin_slope` (assumption-free slope).
     - **Empirical Anchors**: `sim_lag1`, `sim_maxlag`, `decay_half_distance`.

3. **Multi-Aspect Human Quality Ratings (`overall_*`, `mean_*`)**:
   - `story_metrics.csv` (`df_raw`) merges structural metrics with human evaluation scores covering overall enjoyment, coherence, and creativity, as well as specific sub-aspects (originality, surprisingness, logical progression, element consistency).

4. **Local vs. History-Aware Trajectory & Surprisal Metrics**:
   - Local single-step measures (`step_in`, `step_out`) look only at adjacent sentence pairs ($i-1 \rightarrow i$ or $i \rightarrow i+1$).
   - History-aware trajectory measures (`novelty_centroid`, `path_detour`) incorporate the entire preceding narrative history ($\mathbf{c}_{<i}$) up to sentence $i$, properly capturing global contextual departure and transient detours.

---

## Comprehensive Variable Reference Guide

### 1. `df_raw` (`data/processed/rqa/story_metrics.csv`)
The primary 1-row-per-story master dataframe. Loaded via `load_story_metrics()` in [`analysis/comparison_utils.R`](file:///Users/halfidaldal/Documents/Research/penpal-coling/analysis/comparison_utils.R).

#### **A. Story Metadata & Identifiers**
- `story_id` / `conversation_id`: Unique story identifier string.
- `cond` / `condition`: Condition factor (`HH` = Human-Human, `H-LLM` = Human-AI, `LLM-LLM` = AI-AI).
- `n_sentences`: Total sentence count in the story.
- `n_words`: Total word count in the story.
- `n_annotators`: Number of human annotators who evaluated the story.

#### **B. Fixed-RR RQA Metrics (`rr_*` prefix)**
- `rr_RR`: Achieved recurrence rate (e.g. $\approx 0.05$ or $0.10$).
- `rr_DET`: Determinism (share of recurrent points forming diagonal lines $\ge l_{\min}$). Measures **ordered sequence repetition**.
- `rr_L`: Average diagonal line length (in sentences).
- `rr_L_MAX`: Longest repeated diagonal sequence length.
- `rr_DIV`: Divergence ($1 / \text{rr\_L\_MAX}$).
- `rr_ENTR`: Shannon entropy (in bits) of diagonal line length distribution. High = varied echo lengths; low = uniform echoes.
- `rr_RATIO`: Ratio of determinism to recurrence rate (`rr_DET / rr_RR`).
- `rr_LAM`: Laminarity (share of recurrent points forming vertical lines $\ge v_{\min}$). Measures **thematic stasis / dwelling**.
- `rr_TT`: Trapping time (average vertical line length / dwelling duration).
- `rr_V_MAX`: Longest vertical line (max consecutive sentences dwelling on one topic).

#### **C. Fixed-Epsilon RQA Metrics (`eps_*` prefix)**
- `eps_RR`: Unconstrained recurrence rate under the fixed global threshold $\varepsilon$.
- `eps_DET`, `eps_L`, `eps_L_MAX`, `eps_DIV`, `eps_ENTR`, `eps_RATIO`, `eps_LAM`, `eps_TT`, `eps_V_MAX`: Same RQA metrics as above, computed under the single global epsilon.
- `eps_story`: The per-story fixed-RR threshold value used; low values mean the story required a tight threshold to reach the target RR (globally self-similar).

#### **D. Trajectory Geometry Metrics**
- `path_length`: Sum of consecutive sentence step distances ($\sum \|\mathbf{e}_{i+1} - \mathbf{e}_i\|$; total semantic ground covered).
- `mean_step`: Average step distance between adjacent sentences (pacing jumpiness).
- `sd_step`: Standard deviation of step distances (pacing unevenness / irregular leaps).
- `max_step`: Largest single step distance between adjacent sentences.
- `net_displacement`: Straight-line distance from sentence 1 to sentence $N$ ($\|\mathbf{e}_N - \mathbf{e}_1\|$).
- `straightness`: Ratio $\text{net\_displacement} / \text{path\_length} \in [0, 1]$ ($1 =$ laser-straight directed path, $\approx 0 =$ wandering/circling).
- `radius_gyration`: RMS distance of sentences from story centroid (semantic volume occupied).
- `centroid_norm`: Norm of mean sentence vector ($\|\mathbf{c}\| \in [0, 1]$; high = topically concentrated, low = dispersed).
- `msd_alpha`: Diffusion exponent from $\log \text{MSD}(\text{lag}) \sim \alpha \log(\text{lag})$ ($\alpha \approx 1$ random walk, $>1$ directed superdiffusive, $<1$ confined).
- `msd_alpha_r2`: $R^2$ fit quality of the MSD diffusion exponent fit.
- `circularity`: Cosine similarity between opening sentences (first 20%) and ending sentences (last 20%). High = narrative callback / closure.
- `first_last_cos`: Cosine similarity between sentence 1 and sentence $N$.
- `mean_pairwise_sim`: Mean cosine similarity across all sentence pairs.
- `sd_pairwise_sim`: Standard deviation of pairwise sentence similarities.

#### **E. Semantic Decay Metrics (`decay_*` prefix)**
- **Exponential Model**:
  - `decay_lambda`: Decay rate ($\lambda$; higher = faster forgetting/drifting).
  - `decay_A`: Decaying amplitude ($A$).
  - `decay_C`: Asymptotic similarity floor ($C$; persistent background topic).
  - `decay_half_life`: Half-life in sentences ($\ln(2) / \lambda$).
  - `decay_exp_r2`: $R^2$ fit quality of exponential decay.
  - `decay_fit_at_bound`: Flag ($1$ or $0$) indicating if a fitted parameter sat on its bound (screen these out before modeling $\lambda$).
- **Power-law Model**:
  - `decay_beta`: Scale-free decay exponent ($\beta$).
  - `decay_pow_r2`: $R^2$ fit quality of power-law decay.
- **Linear Model**:
  - `decay_lin_slope`: Assumption-free linear decay slope ($b$).
  - `decay_lin_r2`: $R^2$ fit quality of linear decay.
- **Empirical Anchors**:
  - `sim_lag1`: Mean cosine similarity at distance lag 1 (adjacent sentences).
  - `sim_maxlag`: Mean cosine similarity at maximum fitted lag.
  - `decay_half_distance`: Empirical lag where similarity drops to half of `sim_lag1`.

#### **F. Human Quality Ratings**
- `overall_coherence`: Mean human rating for overall story coherence (1–5 scale).
- `overall_creativity`: Mean human rating for overall story creativity (1–5 scale).
- `overall_likeability`: Mean human rating for overall story likeability/enjoyment (1–5 scale).
- `mean_coherence_element_consistency`: Rating for narrative element consistency.
- `mean_coherence_logical_progression`: Rating for logical progression.
- `mean_creativity_originality`: Rating for story originality.
- `mean_creativity_surprisingness`: Rating for story surprisingness.
- `mean_likeability_enjoyability`: Rating for reader enjoyability.
- `mean_likeability_quality`: Rating for general writing quality.
- `mean_lead_time`: Average time (seconds) taken by annotators to complete evaluation.

---

### 2. Sentence-Level Surprisal & Geometry (`data/processed/surprisal/surprisal_unit_level.csv`)
Primary file for sentence-level coupling analyses (e.g. [`analysis/surprisal_embedding_analysis.qmd`](file:///Users/halfidaldal/Documents/Research/penpal-coling/analysis/surprisal_embedding_analysis.qmd)).

- **Identifiers & Metadata:**
  - `id`: Story conversation ID (matches `story_id`).
  - `sent_idx`: Zero-based sentence index ($0, 1, 2, \dots$).
  - `cond`: Story condition (`HH`, `H-LLM`, `LLM-LLM`).
  - `unit_text`: Raw sentence text string.
  - `n_unit_words`: Word count of the sentence.
- **Gemma LLM Surprisal Metrics:**
  - `s_base`: Base sentence surprisal ($-\log_2 P(\text{Sentence} \mid \text{BOS})$).
  - `s_ctx`: Contextual sentence surprisal ($-\log_2 P(\text{Sentence} \mid \text{Story History})$).
  - `s_novelty`: Sentence Novelty ($s_{ctx} - s_{base}$; backward-looking surprisal reduction).
  - `s_transience`: Sentence Transience (forward-looking marginal surprisal on sentence $i+1$).
  - `s_resonance`: Sentence Resonance ($s_{novelty} - s_{transience}$).
- **Joined Embedding Step & History Metrics:**
  - `step_in`: Cosine distance between sentence $i-1$ and sentence $i$.
  - `step_out`: Cosine distance between sentence $i$ and sentence $i+1$.
  - `novelty_centroid`: Cosine distance between sentence $i$ and the running history centroid $\mathbf{c}_{<i}$ ($1 - \cos(\mathbf{c}_{<i}, \mathbf{e}_i)$).
  - `path_detour`: Triangle detour distance ($\text{dist}(\mathbf{c}_{<i}, \mathbf{e}_i) + \text{dist}(\mathbf{e}_i, \mathbf{e}_{i+1}) - \text{dist}(\mathbf{c}_{<i}, \mathbf{e}_{i+1})$).

---

### 3. `df_lag` (`data/processed/rqa/lag_profiles.csv`)
A long-format dataframe holding the empirical similarity decay curve for each story:

- `story_id`: Unique story identifier.
- `lag`: Sentence distance lag $k$ ($k = 1, 2, 3, \dots$).
- `mean_sim`: Mean cosine similarity of all sentence pairs separated by distance $k$.
- `n_pairs`: Number of sentence pairs evaluated at lag $k$.
- `cond` / `condition`: Story condition (`HH`, `H-LLM`, `LLM-LLM`).

---

### 4. `df_sentence_index` (`data/interim/embeddings/sentence_index.csv`)
A long-format dataframe indexing every individual sentence in the corpus:

- `story_id`: Unique story identifier.
- `sent_idx`: Zero-based sentence position within the story ($0, 1, 2, \dots$).
- `n_words`: Word count of that specific sentence.
- `sentence`: Raw sentence text string.

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
