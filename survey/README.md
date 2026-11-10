# Survey (Claude artifact)

A single web page published as a Claude artifact; responses saved in the artifact's built-in database. Built in phase 6 of `PLAN.md`.

## Flow (~15 min)

1. Consent note (anonymous, no personal data) → random assignment to **Group A** or **Group B**.
2. **Trait ratings** on 4 traits (1–7):
   - Group A rates the **cars**.
   - Group B rates the **generated faces** (never sees the cars).
3. **Matching test** (both groups): a car + two faces (its own and another car's) → "Which face belongs to this car?" ~30 realistic-model trials + ~10 cartoon-model trials, in random order with random left/right placement.

## Files (to create)

- `stimuli/`: images exported from `outputs/` (cars, faces), plus `manifest.json` listing items, pairs and distractors.
- `index.html`: the survey page.
- `export/`: responses pulled from the artifact database for analysis.

## Before the real study

- Pilot with 1–2 people: check that the link opens, works on phones, and saves responses.
- Freeze the stimuli and manifest before collecting data.
