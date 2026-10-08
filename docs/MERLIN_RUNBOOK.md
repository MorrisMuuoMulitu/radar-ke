# MERLIN Evaluation Runbook

Reproduce RADAR's published external performance (**AvgAUC 0.8835**,
`docs/INFERENCE.md`) on your own hardware, using the public
[MERLIN dataset](https://stanfordaimi.azurewebsites.net/datasets/60b9c7ff-877b-48ce-96c3-0194c8205c40).

You do **not** need any local DICOM export or hospital data for this. It is an
implementation check — "does our deployment reproduce the published numbers?" —
not a new clinical claim.

---

## 0. What you need

| Item | Detail |
|---|---|
| MERLIN access | Registration + data-use agreement on the Stanford AIMI portal (institutional email). Approval typically takes a day or two. |
| GPU | Any CUDA GPU. This deployment runs an 8 GB RTX 4060 with compact windows (`ROI_SIZE=64,192,288`). |
| Disk | ≈200 MB per study. n=200 ≈ 40 GB; the full 5,125-study test split ≈ 1 TB (see §5). |
| Python | The repo venv. `openpyxl` (for `reports_final.xlsx`) is in `requirements.txt`. |
| Checkpoints | `ckpt/checkpoint_radar_pretrain.pth` + `ckpt/infer_text_embedding_merlin.pt` (already present). |

---

## 1. Download the portal files

From the MERLIN dataset page, download and unpack into **one folder** (nesting is
fine — the tooling searches recursively):

```
~/Downloads/merlinabdominalctdataset/
├── reports_final.xlsx                     # study id, Findings, Split, Few Shot
├── zero_shot_findings_disease_cls.csv     # ground-truth disease labels (0/1/-1)
└── <anything>/…/<patient-id>.nii.gz       # CT volumes (test split included)
```

Only the **test split** is scored; the runbook matches it against the volumes you have.

---

## 2. Readiness check (no GPU work, ~10 s)

```bash
cd /home/mulitu/Desktop/damo-radar
python RADAR_inference/merlin_run.py --portal-dir ~/Downloads/merlinabdominalctdataset --check
```

It locates the files, generates `ckpt/merlin_report.json` and
`ckpt/merlin_labels.json` if missing, and prints:

```
  reports_final.xlsx                 : …/reports_final.xlsx
  zero_shot_findings_disease_cls.csv : …/zero_shot_findings_disease_cls.csv
  volumes found (*.nii.gz)           : 5125
  test-split studies in report       : 5125
  test-split volumes present on disk : 200
Plan
  selected                           : 200 studies (seed 0)
  source size of selection           : 40.1 GB
  est. GPU time                      : 1.4 h (8 GB card, compact windows)
  compute device                     : NVIDIA GeForce RTX 4060 Laptop GPU
```

If "test-split volumes present" is 0, check `--portal-dir` (and that the volumes
really are `.nii.gz`).

---

## 3. Run a sample

```bash
MODEL_ROOT=$PWD/ckpt CONFIGS_ROOT=$PWD/ckpt ROI_SIZE=64,192,288 \
python RADAR_inference/merlin_run.py \
    --portal-dir ~/Downloads/merlinabdominalctdataset \
    --n 200 --seed 0 --chunk 50
```

- **Chunked**: 50 studies are staged (symlinked) and scored per pass, so RAM/VRAM
  stay bounded on an 8 GB card. Tune `--chunk` (lower = safer, more overhead).
- **Resumable**: studies already in `--out/results.csv` are skipped, so an
  interrupted or overnight run continues where it stopped.
- **`--delete-source`** removes each source volume after it is scored (destructive,
  for bounded disk on the full split).

---

## 4. Read the results

Outputs in `results/merlin_subset/`:

| File | Contents |
|---|---|
| `results.csv` | Master per-study scores (21 MERLIN finding columns) — the resumable record. |
| `selection.txt` | Exactly which studies were scored (provenance for the write-up). |
| `merlin_comparison.md` | **The result**: computed vs published AUC per finding with deltas. |
| `merlin_comparison.json` | Same, machine-readable (includes mean AUC). |

Example row:

```
| finding      | published | computed | delta   | n   | positives |
|--------------|----------:|---------:|--------:|----:|----------:|
| gallstones   |    0.9193 |   0.9102 | -0.0091 | 198 |        41 |
| splenomegaly |    0.9682 |   0.9733 | +0.0051 | 200 |        22 |
```

`n` is the number of studies with a usable label (`-1` = "not evaluated" is
skipped, per the MERLIN protocol), `positives` the number of positive cases.
Missing organ scores count as `0` (organ not intact), and
`surgically_absent_gallbladder` is derived from segmentation, exactly as in the
upstream metric script.

---

## 5. Scale up

| Scope | Command addition | Data | Time (8 GB GPU) |
|---|---|---|---|
| 200-study sample | `--n 200` | ≈40 GB | ≈1.4 h |
| 500-study sample | `--n 500` | ≈100 GB | ≈3.5 h |
| Everything downloaded | `--all` | as downloaded | ≈26 s/study |
| Full 5,125 split | `--all --delete-source` in download batches | ≈1 TB total, ≈40 GB at a time | ≈37 h |

For the full split, chunk the *download*: fetch a few hundred studies, run
`--all --delete-source` (or `--n` on the new ones), then fetch the next batch —
`results.csv` accumulates. A cloud GPU VM with a large disk is the comfortable
option (see `docs/DEPLOYMENT_CLOUD.md`).

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Use compact windows: `ROI_SIZE=64,192,288`. Lower `--chunk` to 10–20. Close other GPU users. |
| `No MERLIN test volumes … were found` | Wrong `--portal-dir`, volumes not `.nii.gz`, or `ckpt/merlin_report.json` doesn't match this download (regenerate with `--rebuild-json`). |
| `missing ckpt/merlin_report.json` | `reports_final.xlsx` not found under `--portal-dir`, or run with `--rebuild-json`. |
| `compute device: CPU only` | The process can't see `/dev/nvidia*` (container/sandbox without GPU access). CPU inference of the full split is impractical. |
| Run stopped mid-way | Just re-run the same command — it resumes from `results.csv`. |
| Disk filling up | Add `--delete-source`, reduce `--n`, or shrink `--chunk`. |
| Metrics all `—` | `ckpt/merlin_labels.json` missing/regenerated with wrong CSV; check §2 output. |
| `openpyxl` import error | `pip install openpyxl` (now in `requirements.txt`). |

---

## 7. Interpretation and caveats

- **Subset ≠ full split.** A 200-study sample reproduces the *level* of
  performance; per-finding AUCs carry wide confidence intervals (use `n`/`positives`
  to judge). Report the sample size alongside the numbers.
- **Different item set.** MERLIN evaluation uses RADAR's **21** MERLIN findings
  (`infer_text_embedding_merlin.pt`), not the app's **146** findings. Don't mix
  the two when quoting results.
- **Implementation check, not a new claim.** This validates that this deployment
  reproduces published behaviour; it says nothing about local patient populations.
- Quoting results? Use `selection.txt` + `merlin_comparison.md` and cite the seed.

---

## Appendix — manual equivalent

`merlin_run.py` wraps these steps; run them directly if you prefer:

```bash
# JSON inputs (equivalent to ckpt/transform_report_to_json.py / transform_label_to_json.py)
python - <<'PY'
import sys; sys.path.insert(0, 'RADAR_inference')
import merlin_eval
merlin_eval.build_report_json('<portal>/reports_final.xlsx', 'ckpt/merlin_report.json')
merlin_eval.build_labels_json('<portal>/zero_shot_findings_disease_cls.csv', 'ckpt/merlin_labels.json')
PY

# plan, then score, then metrics
python RADAR_inference/merlin_subset_eval.py --data-dir <flat-or-nested-volumes> --n 200 --dry-run --out results/merlin_subset
MODEL_ROOT=$PWD/ckpt CONFIGS_ROOT=$PWD/ckpt ROI_SIZE=64,192,288 \
python RADAR_inference/merlin_subset_eval.py --data-dir <volumes> --n 200 \
    --out results/merlin_subset --labels ckpt/merlin_labels.json --results results/merlin_subset/results.csv
python RADAR_inference/merlin_subset_eval.py --metrics-only results/merlin_subset/results.csv \
    --labels ckpt/merlin_labels.json --out results/merlin_subset
```

Smoke-tested end to end with synthetic fixtures (a fake portal folder, nested
volumes, generated xlsx/csv): readiness → chunked scoring → comparison report →
resume. See `webapp/tests/test_merlin_eval.py`.
