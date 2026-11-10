# Analysis

Built in phase 7 of `PLAN.md`. Input: the survey export (`survey/export/`).

1. **Rater reliability:** split each group in half and correlate per-car mean ratings (per trait).
2. **Trait alignment (main result):** per trait, Spearman correlation across cars between Group A's car ratings and Group B's face ratings.
3. **Matching:** accuracy vs 50% chance (binomial test, then a mixed-effects logistic model with participant and car as random effects).
4. **Realistic vs cartoon (exploratory):** compare matching accuracy and trait correlations on the shared ~20 cars.
5. **Automatic checks:** face-detector landmarks on generated faces vs headlight/grille positions; latent distance to real faces for fronts vs rears.
