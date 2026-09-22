# RADAR Kenya Review Workspace

RADAR Kenya is a local, radiologist-centered review workspace for contrast-enhanced abdominal CT. It builds on the upstream RADAR research code from Alibaba DAMO Academy and adds a Streamlit interface (`webapp/`) for scan upload, model finding scores, segmentation overlays, structured review, report generation, case-worklist management, and clinical validation of model outputs against a radiologist reference.

This repository keeps the original RADAR training, preprocessing, and inference code available while adding the practical web app in `webapp/`.

## Where to look first

This repository contains two distinct bodies of work.

**Original work in this repository:**
- `webapp/app.py` — the Streamlit review workspace: viewer, findings explorer, review notebook, report export, worklist.
- `webapp/review.py` — Streamlit-independent domain logic. **Start here** for the clinical validation harness (`validation_table`, `validation_summary`, `match_findings_to_report`), which computes agreement against a radiologist reference over adjudicated findings only.
- `webapp/tests/` — unit tests for the review workflow, report template, and validation metrics.
- `RADAR_inference/inference_service.py` — device-agnostic inference service used by the app. **Not part of upstream**; written here to add GPU auto-detection, compact inference windows on low-VRAM cards, and a CPU fallback.
- `deploy/` — Docker Compose stack (GPU webapp + Caddy reverse proxy with basic auth and TLS), persistent case volume, and the tester onboarding pack.
- `docs/DEPLOYMENT_CLOUD.md`, `docs/deployment-agent-prompt.md`, `DEPLOYMENT_REPORT.md` — cloud deployment path, deployment agent brief, and the pilot deployment record.

**Upstream research code — Alibaba DAMO Academy, Apache 2.0:**
- `RADAR_train/` — training and preprocessing.
- `RADAR_inference/` **except** `inference_service.py` — `inference_demo.py`, `inference_merlin_testset.py`, `calc_metrics_merlin_testset.py`, and `dynamic_network_architectures/`.
- `docs/INFERENCE.md`, `docs/PREPROCESS.md`, `docs/TRAINING.md`, `download_scripts/`, and the `ckpt/` helper scripts.

Model checkpoints are not committed — see [Model Files](#model-files).

> **Status: research and product prototype.** Not a certified medical device and must not be used as an autonomous diagnosis system. Outputs require qualified radiologist review.

## Feature Overview

- **Case workspace** — upload a CT scan (NIfTI or DICOM zip), run RADAR inference (GPU, with a CPU fallback), or open the bundled example case instantly.
- **Three-plane volume explorer** — axial / coronal / sagittal views, radiology CT window presets (Abdomen, Liver, Soft tissue, Lung, Bone) with level/width controls, slice stepping, organ segmentation overlays, and finding→anatomy navigation jumps.
- **Finding explorer** — 146 predefined finding scores with search, anatomy filtering, score thresholding, sorting, and CSV export.
- **Review notebook** — per-finding review states (`Needs review`, `Likely present`, `Likely absent`, `Ignore`), shortlist for follow-up, review status (`Not started`, `In progress`, `Reviewed`), clinical context / indication, reviewer notes, and export of review JSON, report text, and all finding scores.
- **Radiology-style report draft** — auto-generated with `Clinical context`, `Findings by organ`, `Impression`, `Review summary`, `Reviewer notes`, and `Review limitations` sections.
- **Case worklist** — dashboard of all saved cases with status KPIs, filename search, status filter, sorting, per-row open / report / delete, and CSV export.
- **Clinical validation harness** — per-case agreement between RADAR model scores and a radiologist reference standard (TP/FP/TN/FN, accuracy, sensitivity, specificity, precision, F1), with report-text finding extraction and downloadable validation summaries.
- **Case history** — saved review snapshots (metadata, notes, shortlist, review states, scores, validation) stored locally; the worklist and report work with saved cases even without the original scan pixels.

Fresh analyses save a HU display volume aligned to the model output when possible, so the viewer can use radiology-style window width/level presets. Older saved outputs and some example paths fall back to normalized display data.

---

## Workspace Guide

### 1. Case workspace (sidebar)

| Control | Behavior |
|---|---|
| **Import CT scan** | Accepts `.nii`, `.nii.gz`, or a `.zip` containing **one DICOM series** (see [Input Formats](#input-formats)). |
| **Analyze scan** | Runs RADAR segmentation + finding scoring on the uploaded scan. Enabled once a file is attached; runs on CUDA automatically when a GPU is available, otherwise on CPU (slow). |
| **Open example case** | Loads the bundled example volume (`data/demo_cases/AC423ccbe.nii.gz`) with its previously computed scores — works with no GPU, instantly. |
| **Case history** | Quick-open any saved review from the history store. |
| **View switch** | Toggle between `Review workspace` (current case) and `Case worklist` (all saved cases). |

### 2. Scan explorer

- Layout: three planes at once, or a single plane.
- **Window presets**: Abdomen (60/400), Liver (70/150), Soft tissue (40/350), Lung (−600/1500), Bone (300/1500) — level and width are adjustable.
- Previous/next slice stepping, per-plane slice sliders, and PNG export of any slice.
- **Organ overlay**: highlights the 36 segmented anatomy structures (turbo colormap) with adjustable opacity; center the view on any segmented organ.
- The example case displays normalized data (no true HU windowing) unless it is reanalyzed; fresh analyses store HU display volumes.

### 3. Findings

- Search the 146 findings (literal, case-insensitive), filter by anatomy, set a model-score threshold, sort by score or anatomy, and export the filtered table as CSV.
- Scores compare predefined positive/negative prompts and are **not calibrated disease probabilities**; missing predictions are shown blank.

### 4. Review & export

- **Finding review states**: mark findings `Needs review`, `Likely present`, `Likely absent`, or `Ignore` — these drive the report and the worklist "likely present" counts.
- **Shortlist findings for follow-up**; set the **review status**; record **clinical context / indication** and **reviewer notes**.
- **Save case to history** persists a review snapshot (see [Case history data](#case-history-data)).
- Downloads: review JSON, report text (TXT), and all finding scores (CSV).

### 5. Case worklist

The worklist (`Case worklist` view) turns the single-case tool into a review pipeline:

- KPI header: total cases and counts per status.
- **Search cases** (filename), **filter by status** (Not started / In progress / Reviewed), **sort** (Newest, Oldest, Filename, Status).
- Each row shows filename, saved date, case ID, status, shortlist count, and **likely-present findings** (count + names).
- Per-row actions: **Open case** (loads the full saved review), **Report TXT** (download the structured report without opening), **Delete**.
- **Download worklist CSV** of the filtered list.

### 6. Validation

The **Validation** tab computes agreement between RADAR and a radiologist reference per case:

1. **Paste radiologist report** → **Extract findings** suggests matching findings (term + synonym matching; ambiguous terms like "cyst" only match when the organ is also mentioned).
2. Adjust the **Reference — present** and **Reference — absent** multiselects (this is the saved ground truth).
3. Set the **prediction threshold** (model score ≥ threshold ⇒ predicted positive; default 0.5).
4. Metrics are computed over **adjudicated findings only**: TP / FP / TN / FN, accuracy, sensitivity, specificity, precision (PPV), F1, plus a per-finding agreement table (`Liver / Cyst: 0.810, reference Present, predicted Positive — TP`).
5. Download the **validation summary TXT**; adjudication is saved with the review and reloaded when the case is reopened.

> Caveat: unmarked findings are neither positive nor negative — metrics cover only the adjudicated set, and the summary states this explicitly to avoid inflated accuracy.

---

## Report Template

The structured report draft (`structured_report()` in `webapp/review.py`) is a radiology-style text document:

```
==============================================================
 RADAR-AIDED ABDOMINAL CT REPORT (DRAFT)
==============================================================
Study / Reviewed on / Review status / Reviewer

CLINICAL CONTEXT      — indication + analysis method
FINDINGS BY ORGAN     — findings grouped by anatomy in a fixed
                        radiology organ order, each with score and
                        review state (Likely present / Needs review /
                        Shortlisted / High model score); absent and
                        ignored findings are excluded
IMPRESSION            — findings marked likely present, shortlisted
                        for follow-up, or a no-findings fallback
REVIEW SUMMARY        — counts per review state + shortlist
REVIEWER NOTES        — free text
REVIEW LIMITATIONS    — research-use and methodology caveats
```

- Organs are ordered (Liver → Gallbladder → Pancreas → Spleen → Kidney → Adrenal gland → Stomach → Duodenum → Large/Small bowel → Esophagus → Aorta → Portal vein → Heart → Lung → Bladder → Rib → Sacrum), with any remaining organs after.
- Unmarked findings with model score ≥ 0.5 are surfaced as `High model score`.
- The report is research-only and must not be signed as a clinical document.

---

## Input Formats

| Format | Handling |
|---|---|
| `.nii` / `.nii.gz` | Single NIfTI volume → straight to inference (auto-resampled to 1×1×5 mm, HU clipped to −300…400). |
| `.zip` | One DICOM series → extracted and converted with `dcm2niix` (must yield exactly one `.nii.gz`). |

Limits:

- Upload cap **2 GB** per file (`maxUploadSize` in `.streamlit/config.toml`).
- DICOM zip: uncompressed total **≤ 8 GB**; archive member paths are validated; conversion timeout 600 s.
- Not accepted: raw/loose DICOM files or folders, multi-series or multi-phase zips, `.mha`, `.npy`, other formats.
- `dcm2niix` must be on `PATH` for DICOM zips (installed via the Dockerfile; `sudo apt install dcm2niix` on Debian/Ubuntu).

---

## Data

### Bundled example
`data/demo_cases/AC423ccbe.nii.gz` + precomputed scores in `results/RADAR_infer_results_demo_8gb.csv` (fallback `RADAR_infer_results_demo.csv`). Open via **Open example case** — no GPU required.

### Downloaded test scans (`data/test_scans/`)
A small sample of whole-body CT volumes from the public **CT-ORG** dataset (CC-BY-4.0) can be placed under `data/test_scans/ct_org/` for pipeline testing (directory is git-ignored; download rather than commit):

```bash
mkdir -p data/test_scans/ct_org
python - <<'PY'
from huggingface_hub import hf_hub_download
for v in ["volume-105.nii.gz", "volume-94.nii.gz", "volume-83.nii.gz"]:
    hf_hub_download("MedOtter/ct-org", f"volumes/{v}", repo_type="dataset",
                    local_dir="data/test_scans/ct_org")
PY
```

Matching organ masks (6 structures: liver, bladder, lungs, kidneys, bone, brain) are in `labels/` of the same repo. Caveat: CT-ORG is whole-body with uncontrolled contrast phase — fine for exercising the pipeline, out-of-domain for calibrated findings.

### In-domain external test set (MERLIN)
The official external test set RADAR was evaluated on (AvgAUC 0.8835) is the **MERLIN** dataset from the [Stanford AIMI Shared Datasets portal](https://stanfordaimi.azurewebsites.net/datasets/60b9c7ff-877b-48ce-96c3-0194c8205c40) (registration required). It ships radiologist reports and disease/finding labels; the repo includes RADAR's published results in `results/RADAR_infer_results_MerlinTestset.csv` and the reprocessing scripts under `ckpt/` (`transform_report_to_json.py`, `transform_label_to_json.py`). See `docs/INFERENCE.md` for the full external-evaluation workflow.

---

## Repository Setup

Create a Python environment and install dependencies:

```bash
conda create -n radar python=3.10
conda activate radar
pip install -r requirements.txt
```

This machine's existing local venv:

```bash
source .venv/bin/activate
```

## Model Files

The RADAR checkpoints and support files are large and are **not committed** to this repository. Expected local layout:

```text
ckpt/
  checkpoint_radar_pretrain.pth        # RADAR pretrained on RAD-CT (webapp inference)
  checkpoint_unet.pth                  # anatomy segmentation
  checkpoint_radar_plus.pth            # RADAR+ variant
  checkpoint_radar_plus_finetuned_on_merlin.pth
  infer_text_embedding_radar.pt        # fixed prompt embeddings
  infer_text_embedding_merlin.pt
  bert-base-chinese/  bert-base-uncased/   # BERT tokenizer + model
  merlin_report_organ_normal_v1.json
  merlin_report_organ_report_v1.json
```

Download from the RADAR Hugging Face organization via the helper scripts:

```bash
cd download_scripts
python download_checkpoints.py        # radar-generalist/RADAR  (model)
python download_auxiliary_data.py     # radar-generalist/RADAR-auxiliary-data (dataset)
```

## Run The App

From the repository root:

```bash
streamlit run webapp/app.py --server.address 127.0.0.1 --server.port 8501
# or, with the repo venv:
.venv/bin/python -m streamlit run webapp/app.py --server.address 127.0.0.1 --server.port 8501
```

Then open <http://127.0.0.1:8501>.

Behavior notes:

- **GPU auto-detection**: with CUDA available the app uses the GPU (on ≤ 10 GB VRAM GPUs it automatically selects compact inference windows via `ROI_SIZE=64,192,288` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid OOM). Without CUDA it falls back to CPU mode — functional but very slow for full scans (tens of minutes to hours).
- Uploads are processed locally; temporary files live under the system temp dir (`radar_web_*`) and are removed via **Clear case and delete uploads**.
- If the host GPU is present but the app reports *"No CUDA GPU detected"*, the process likely lacks access to `/dev/nvidia*` (e.g. inside a restricted sandbox/container) — relaunch with GPU device access.

## Case History Data

Saved reviews are stored as JSON under `~/.radar_ke/cases` (override with the `RADAR_CASE_HISTORY_DIR` environment variable). Each snapshot contains: case id, file name, saved timestamp, review status, shortlist, finding review states, reviewer notes, clinical context, reviewer name, validation adjudication (present/absent/threshold), finding scores, and the example flag. **Volumes/pixels are not archived** — reopening a saved case shows the review metadata; image slices require reopening or reanalyzing the scan.

## Docker

The Dockerfile builds the app image with `dcm2niix`; checkpoints should be mounted, not baked in.

```bash
docker build -t radar-ke .
docker run --gpus all -p 8501:8501 -v "$PWD/ckpt:/app/ckpt" radar-ke
```

## Deployment

A ready-to-run deployment stack lives in [`deploy/`](deploy/README.md):

- **Path A — LAN/VPN pilot:** Docker Compose (GPU webapp + Caddy reverse
  proxy with basic auth and TLS), persistent case data volume, tester guide
  (`deploy/TESTERS.md`), and a password-hash helper.
- **Path B — GPU cloud VM:** [`docs/DEPLOYMENT_CLOUD.md`](docs/DEPLOYMENT_CLOUD.md)
  covers providers, provisioning, Let's Encrypt on a real domain, operations,
  cost control, and data/privacy rules.

Same Compose stack for both — start locally, move to the cloud later without
rework.

## Tests

```bash
python -m unittest discover -s webapp/tests -v
```

Coverage includes: findings filtering/export, review-state workflow (AppTest), window-preset math, case-history save/open (AppTest), worklist rows/filter/delete, worklist UI open-flow (AppTest), report template sections/content, and the validation harness (report-text extraction, confusion matrix, metrics, persistence).

`git diff --check` before committing to catch whitespace errors.

## Architecture

| File | Role |
|---|---|
| `webapp/app.py` | Streamlit UI: sidebar (import/analyze/example/history/view switch), scan explorer, findings, review & export, validation, case worklist; GPU detection and CPU fallback. |
| `webapp/review.py` | Pure, Streamlit-independent logic: score tables, filters, case records + JSON history store, worklist aggregation, structured (radiology) report, validation metrics and report-text extraction. |
| `webapp/style.css` | Styling for the workspace shell. |
| `webapp/.streamlit/config.toml` | Server settings (headless, upload cap). |
| `RADAR_inference/inference_service.py` | **Original.** Device-agnostic inference (`_infer_device()`): NIfTI load → resample 1×1×5 mm → sliding-window segmentation → per-organ finding scoring → `.npz` (image/HU/mask/scores) + CSV. Not part of upstream. |
| `deploy/` | **Original.** Docker Compose stack: GPU webapp + Caddy reverse proxy (basic auth, internal TLS), persistent case volume, tester onboarding pack, password-hash helper. |
| `docs/DEPLOYMENT_CLOUD.md`, `docs/deployment-agent-prompt.md` | **Original.** GPU cloud deployment path, and the self-contained deployment agent brief. |
| `DEPLOYMENT_REPORT.md` | **Original.** Pilot deployment record: hardening steps, verification results, issues found and their fixes. |
| `RADAR_train/`, `RADAR_inference/*` (except `inference_service.py`), upstream `docs/` | Upstream RADAR training / inference / preprocessing code and documentation (Apache 2.0). |
| `download_scripts/` | Hugging Face helpers for checkpoints and auxiliary data. |
| `data/test_scans/` | Optional downloaded test scans (git-ignored; see [Data](#data)). |

## Git Remotes

```text
origin   https://github.com/MorrisMuuoMulitu/radar-ke.git   # this product
upstream https://github.com/alibaba-damo-academy/damo-radar.git  # original RADAR
```

Push product work to `origin`; pull upstream changes from `upstream`.

## Roadmap

- **Case metadata (patient-driven)**: patient ID, study date, modality, reviewer, institution — staged next; the report header already has placeholders.
- **Cohort validation view**: adjudicate across all saved cases and roll up per-finding sensitivity/specificity at cohort level.
- **Commercial readiness**: privacy statement surfaced in-app, audit log (who changed what/when), user login, branding polish.
- **Clinical validation path**: define test cases, compare RADAR outputs with radiologist reports, track agreement — the per-case harness is the foundation.

## Privacy And Clinical Use

- Uploaded scans stay on the local machine during the current workflow; the app does not send scans to a cloud service by default; review exports are generated locally.
- The interface is a radiologist support tool. Clinical deployment requires privacy, audit, security, validation, regulatory, and institution-specific review (see Roadmap).

## License And Attribution

The upstream RADAR code is released under Apache License 2.0, with third-party components under their own licenses. Model weights, datasets, and supporting files may have separate terms (e.g. the CT-ORG test scans and MERLIN data). Review upstream licenses and asset terms before commercial use.

If RADAR is useful in research work, cite the upstream paper:

```bibtex
@article{damo-radar-2026,
    author = {Qi Zhang and Jianpeng Zhang and Weiwei Cao and Zilin Lu and Wanxing Chang and Haonan Ding and Cao Chen and Zhi Li and Xing Xue and Sinuo Wang and Shaoteng Zhang and Yutong Xie and Yong Xia and Qi Wu and Zhongyi Shui and Xi Li and Zhilin Zheng and Yanjie Zhou and Tony C.W. Mok and Yingda Xia and Hongkan Wang and Xianghua Ye and Tao Ma and Jie Peng and Xiaoguang Wang and Jian Ding and Yuming Gao and Huazhen Ye and Yiping Liu and Dongjie Chen and Zhaomin Ni and Jianwen Ning and Wei Zhang and Jian Liu and Chaohui Yu and Shenghong Ju and Jianfeng Zhang and Wenbo Xiao and Ling Zhang and Tingbo Liang },
    title = {An expert-level generalist AI for abdominal CT diagnosis},
    journal = {Science},
    volume = {393},
    number = {6817},
    pages = {eaec6129},
    year = {2026},
    doi = {10.1126/science.aec6129},
    URL = {https://www.science.org/doi/abs/10.1126/science.aec6129}
}
```