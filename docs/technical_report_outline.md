# Technical report outline (4–6 pages)

Fill every number from `outputs/results/*.md|csv`. Do not round in your favor or drop folds.

1. **Introduction**: signing speed varies between people; fixed-length sampling is the default;
   question and contributions (a controlled speed stress test for BISINDO + a motion-aware
   normalization + ablation and augmentation comparison).
2. **Related work**: isolated SLR with pose (SPOTER, Siformer); WL-BISINDO baselines; temporal
   sampling and keyframe selection; speed/time-warp augmentation.
3. **Dataset and preprocessing**: audit table (from `audit_summary.json`), landmark detection
   rates, feature definition, normalization.
4. **Methods**: uniform, trim + uniform, motion-aware (equation for the density, role of α),
   uniform + speed augmentation; GRU classifier.
5. **Experimental setup**: LOSO × 3 seeds, validation rule, stress-test definitions (global vs
   core, how the active segment is found), metrics, paired test.
6. **Results**: robustness curves (both perturbations), table at each factor, clean accuracy,
   drop, per-fold table. Compare with WL-BISINDO published SI numbers only as context (different
   models and protocol).
7. **Error analysis and limitations**: hardest glosses, confused pairs, effect of hand detection,
   transition oversampling, synthetic speed, single regional variant.
8. **Conclusion**: answer the research question in one paragraph, including if the answer is "no".
