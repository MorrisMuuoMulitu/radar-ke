# RADAR Kenya — Tester Guide

Thanks for testing the RADAR CT review workspace. It is a **research
prototype**, not a medical device. Do **not** upload real patient data — use
the example case (or ask for a test scan).

## What to try

1. **Example case** — click `Open example case` in the sidebar (instant, no GPU wait).
   - Scan explorer: switch Layout / plane, try Window presets (Abdomen, Liver, Bone…),
     use Previous/Next slice, enable the organ overlay, save a slice PNG.
   - Findings: search (e.g. `liver`, `cyst`, `calcification`), filter by anatomy,
     move the score threshold, export CSV.
2. **Analyze a scan** (if you have a `.nii` / `.nii.gz` / DICOM zip):
   upload, click `Analyze scan`, wait for the model (a few minutes on GPU).
   Then explore the viewer with true HU windowing and the new segmentation.
3. **Review & export** — mark a finding `Likely present`, shortlist it, add
   clinical context + notes, set review status, and read the generated report
   draft (Clinical context / Findings by organ / Impression / Review limitations).
   Click `Save case to history`, then open it from the sidebar.
4. **Case worklist** — switch `View → Case worklist`: search, filter by status,
   open a case row, download a report TXT from a row, export the worklist CSV.
5. **Validation** — paste a radiologist report snippet into the Validation tab,
   `Extract findings`, confirm present/absent lists, and read the agreement
   metrics (accuracy, sensitivity, specificity, F1). Save the case; the
   adjudication is stored with it.

## Expected behavior notes

- Model scores are **not disease probabilities** — they compare predefined
  positive/negative prompts. All output needs qualified review.
- Inference runs one scan at a time; if another tester is analyzing, yours
  waits.
- Uploads and saved reviews stay on the test machine. Use `Clear case and
  delete uploads` to remove temporary files.

## Report issues

Please report: crashes, wrong-looking segmentation/organ labels, viewer
glitches, confusing report text, or anything surprising about scores.