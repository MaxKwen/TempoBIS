# Methodology and design decisions

Each choice below is a starting point, picked before seeing real results. If a pilot shows a
choice is wrong, change it in the config **before** running the full experiment, and record why.

## 1. Data and splits

* **Index:** signer, label and sample are parsed from file names
  (`signer{S}_label{L}_sample{N}`), as in the official WL-BISINDO split metadata.
* **Audit first** (`tempobis audit`): video count, glosses, signers, videos per signer, fps,
  decoded frame counts (container counts can be wrong), unreadable files, duplicate ids and
  byte-identical files. The official metadata suggests videos are **not** evenly spread across
  signers; the audit confirms the real distribution.
* **Main protocol: leave-one-signer-out (LOSO).** Each of the 5 signers is the test set once.
  This measures generalization to an unseen person, which is what matters for sign recognition.
  The official signer-independent split (test signer 4) is one of these folds.
* **Validation:** stratified 15% of the *training signers'* videos, used for early stopping only.
  This is mildly optimistic (val signers are seen in training) but never touches the test signer.
* **No leakage:** perturbations are applied only at test time; speed augmentation for the
  comparator is generated from training videos after the split.

## 2. Features (`features.py`)

* MediaPipe Tasks: PoseLandmarker (lite) + HandLandmarker (2 hands), VIDEO mode.
* Hands are assigned to the signer's left/right by matching hand wrists to pose wrists, because
  the handedness label assumes a mirrored image.
* Per frame: (x, y) of nose, shoulders, elbows, wrists and 2 × 21 hand points + 2 "hand
  detected" flags → 100 features. z is dropped (noisy from a single camera).
* Normalization per video: x scaled by aspect ratio; origin = median shoulder midpoint;
  unit = median shoulder width. Using per-video medians (not per-frame) keeps body motion.
* Missing hands: gaps of ≤ 6 frames are interpolated; otherwise the hand collapses onto the pose
  wrist, so a lowered or out-of-view hand still moves with the arm and is flagged as not detected.

## 3. Temporal normalization (`temporal.py`)

All methods output T = 32 timesteps by linear interpolation at fractional frame positions.

**Motion energy.** For each inter-frame step, the mean displacement of the 22 hand points
(21 landmarks + pose wrist), summed over both hands, smoothed with a 5-frame moving average,
minus its 10th percentile (landmark jitter floor), clipped at 0.

**Motion-aware sampling.** Segment density
`d_t = (1 − α) e_t / Σe + α / (N − 1)`; the cumulative density is inverted at T evenly spaced
levels. Properties checked by unit tests:
* α = 1 reproduces uniform sampling exactly;
* positions are monotonic (temporal order is preserved);
* a larger share of timesteps falls inside the signing segment.

α = 0.3 keeps at least 30% of the timestep budget spread evenly, so static hand-shape holds are
not skipped. α is a hyperparameter; if it is tuned, tune it on validation data only.

**Trim + uniform (ablation)** isolates the "ignore idle frames" part: cut where cumulative motion
energy crosses 2% and 98%, then sample uniformly inside.

**Uniform + speed augmentation (comparator)** asks whether a smarter normalization is needed at
all, or whether training on varied speeds is enough. Each training video gets two extra copies
with a random global or core speed factor in [0.6, 1.6] (log-uniform). This range overlaps the
test factors, so this comparator sees test-like perturbations during training. Interpret it as a
strong reference, not a like-for-like competitor.

## 4. Stress test

Factors: 0.5, 0.75, 1.0, 1.25, 1.5 (s > 1 = faster = fewer frames).

* `global`: resample the whole sequence to round((N − 1) / s) + 1 frames.
* `core`: only the active segment is resampled; idle head and tail are copied unchanged.
  Active segment = first to last frame where a wrist is clearly above its own rest height
  (rest = 90th percentile of wrist height; threshold = max(0.3, 0.25 × range) shoulder widths).
  This uses wrist **position**, while the proposed method uses hand **motion**, so the test is
  not built on the method's own signal. The two are still correlated (raising a hand is motion),
  which is a limitation to state in the report.

Expected behavior to verify, not assume:
* Uniform sampling is almost invariant to `global` speed (the unit test
  `test_uniform_absorbs_global_slowdown` shows this in input space). A small drop at 1.5× can
  come from losing frames. A flat curve here is a meaningful finding, not a failure.
* `core` speed changes the share of timesteps that land on the sign under uniform sampling.
  This is where differences between methods should appear, if there are any.

## 5. Model and training

Bidirectional GRU (2 layers, 128 units) over a linear + LayerNorm input projection, mean + max
pooling, dropout 0.3. AdamW (lr 1e-3, wd 1e-4), cosine schedule, label smoothing 0.1, up to 80
epochs, early stopping on validation macro-F1 (patience 15). Light feature-space augmentation
(global scale ±5%, Gaussian noise) is identical for all methods. An MLP on flattened sequences
(`model.type: mlp`) is available as a simpler model.

## 6. Reporting

* Primary metric: macro-F1 on the held-out signer; accuracy is also stored.
* Mean ± std over folds (seeds averaged within each fold first).
* Robustness per run: clean F1, mean perturbed F1, **drop** = clean − mean perturbed, worst case.
* Method vs baseline: paired differences over (fold, seed) runs with a Wilcoxon signed-rank test.
  With 5 folds × 3 seeds the runs within a fold are not independent, so treat p-values as a guide
  and report per-fold results next to them.
* Error analysis: per-class F1, most confused pairs, accuracy vs hand-detection rate.

## 7. Known risks

* **Transitions are high-motion.** Raising and lowering the hands produces strong motion energy
  but is not the sign itself. Motion-aware sampling may oversample these transitions. Check this
  with the pilot plot (motion energy vs. active segment) and the trim ablation.
* **Signer imbalance.** Folds with a small test signer give noisier scores; report per-fold numbers.
* **Synthetic speed.** Interpolation produces smooth in-between poses; real fast signing also
  shortens or merges movements. Results answer "robust to time-scaling", not "robust to every
  real speed difference".
