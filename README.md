# Predicting Brain Regions: AL vs PM from visual tuning

Can we identify which higher visual area a neuron belongs to — **VISal (AL)** or
**VISpm (PM)** based on a stimulus?

### Scientific questions
1. **Do neurons in these regions have distinct responses?**
2. **If so, can we distinguish the regions based on those responses?**

**Main notebook:** `Predicting_Brain_Regions_final.ipynb`
**Data:** Allen Brain Observatory (via AllenSDK) — Cux2 line, layer 2/3 (@175 µm) and
layer 4 (@275 µm), drifting- and static-grating stimuli.

## Data pipeline
`dff` (raw calcium traces) → `trial_response` (per-trial means) → `tuning_array`
(trial-averaged, `8 directions × 5 temporal frequencies` for drifting; `orientations ×
spatial frequencies` for static) → flattened to one **tuning vector per neuron**
(40-D drifting, 30-D static). Session id is kept per neuron for grouped cross-validation.

## What the notebook does (Sections 1–12)
1. **Know the data** — inspect `dff`, `stim_table`, `trial_response`, `tuning_array`.
2. **Build the dataset** — flatten to `n_neurons × 40`, with area labels + session ids.
3. **Signal vs nuisance** — mean AL−PM tuning shows where the areas differ (differences are small).
4. **Look before classifying** — PCA / t-SNE (heavy overlap, no clean clusters).
5. **Decoders, built up** — one thresholded feature → logistic regression → all 40 features →
   confusion matrix with per-area recall.
6. **TF-only features** — average out direction; keep the 5-number TF curve.
7. **Tuned-neuron selection** — keep neurons that are **strong AND reliable**:
   `depth = (R_max − R_min)/R_max ≥ 0.25` AND one-way ANOVA `p < 0.05`.
8. **Layers** — L2/3 vs L4 vs combined.
9. **Static gratings** — repeat with spatial-frequency (SF) and orientation features.
10. **Temporal 1D-CNN** — sequence model across layers (drifting + static).

## Key methodological choices
- **Session-grouped CV:** whole sessions are in train or test, never split — guards against
  session/batch leakage (each session is entirely one area).
- **Balanced accuracy & AUC (chance = 0.5):** classes are unequal and symmetric, so plain
  accuracy / F1 would mislead; balanced accuracy weights both areas equally.
- **`class_weight="balanced"`, StandardScaler fit on train only.**
- **Label-agnostic tuned selection:** depth/ANOVA never look at area, so filtering is denoising,
  not leakage.

## Headline results
- Areas are **weakly but systematically distinguishable** — balanced accuracy ≈ **0.57–0.69**
  (above chance, far from perfect); single neurons overlap heavily.
- A compact **5-number TF/SF tuning vector decodes area as well as the full 30–40-D response**,
  and generalizes across sessions better — **best in layer 4**.
- Selecting "tuned" neurons **barely changes accuracy** (flat across thresholds): the area
  signal is **distributed**, not carried by a few standout cells.
- L4 gives **more balanced** PM/AL classification than L2/3.

## Conclusions
- **Q1:** Yes — the reliable difference lives in the **shape of TF/SF tuning** (AL favors
  fast/coarse = high TF, low SF; PM favors slow/fine = high SF, low TF), not in cell type.
- **Q2:** Yes — area is readable above chance from a **single neuron's tuning**, a weak but
  systematic, distributed signal that peaks in L4.

## Limitations
Modest accuracy; few sessions limit CV folds; the tuning threshold is chosen on the same CV
score (optimistic); depth ignores the circular structure of orientation/direction; and this is
single-neuron (not simultaneous population) decoding, limited to two areas / one Cre line.

## File
- `tf_tuned_selection_slide.html` — depth-based TF-tuned selection + accuracy-vs-threshold sweep.

## Run
Designed for Google Colab. Run cells top-to-bottom; the first cell installs/loads AllenSDK.
Requires: `allensdk`, `numpy`, `pandas`, `matplotlib`, `scikit-learn` (+ `scipy`).