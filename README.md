# AL vs PM: a foundations-first neuron decoder

Can we tell which visual area a single neuron lives in — **VISal (AL)** or **VISpm (PM)** — from
how it responds to drifting gratings? This notebook answers that with the *simplest, most
interpretable* method possible, prioritizing understanding of the data over chasing accuracy.

**Notebook:** `AL_vs_PM_Foundations.ipynb`
**Data:** Allen Brain Observatory (via AllenSDK), one Cre line, one imaging depth (175 µm),
drifting-gratings stimulus.

## The data pipeline
`dff` (raw calcium traces) → `trial_response` (per-trial means, carries nuisance variability)
→ `tuning_array` (trial-averaged signal, `8 orientations × 5 temporal frequencies`)
→ flattened to one **40-value tuning vector per neuron**. Session id is kept per neuron for
grouped cross-validation.

## What the notebook does
1. **Know the data** — inspect `dff`, `stim_table`, `trial_response`, `tuning_array`.
2. **Understand one neuron** — tuning heatmaps and direction/TF curves.
3. **Build the dataset** — flatten to `n_neurons × 40`, with area labels and session ids.
4. **Signal vs nuisance** — mean AL−PM tuning shows where the areas differ.
5. **Look before classifying** — PCA / t-SNE (by area and by session as a leakage check).
6. **Simplest decoder, built up** — one hand-thresholded feature → logistic regression
   (session-grouped CV, `class_weight="balanced"`, tunable L1/L2) → all 40 features →
   confusion matrix with per-area recall.
7. **One control** — permutation test that the score beats chance.

## Key methodological choices
- **Session-grouped CV:** a whole session is in train or test, never split — guards against
  batch/session effects.
- **Balanced accuracy & AUC vs chance = 0.5:** classes are unequal, so plain accuracy misleads.
- **`class_weight="balanced"`:** stops the model defaulting to the majority area, giving honest
  per-area recall.

## Headline result
AL and PM are only **weakly separable at the single-neuron level** (balanced accuracy ≈ 0.55,
AUC ≈ 0.59) — a real biological result, not a modelling failure. The single interpretable feature
(weighted temporal frequency) is about as good as the full 40-vector.

## Next steps
Add **static-grating spatial-frequency (SF) features** (PM prefers high SF / low TF) and move
toward **population-level decoding** for a genuine signal boost.

## Run
Designed for Google Colab. Run cells top-to-bottom; the first cell installs/loads AllenSDK.
Requires: `allensdk`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
