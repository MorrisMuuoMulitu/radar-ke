# RADAR Kenya Review Workspace

RADAR Kenya is a local review workspace for contrast-enhanced abdominal CT. It builds on the upstream RADAR research code from Alibaba DAMO Academy and adds a Streamlit interface for radiologist-centered scan review, model finding scores, segmentation overlays, notes, and exports.

This repository keeps the original RADAR training, preprocessing, and inference code available while adding a practical web app in `webapp/`.

## Current Status

- Local Streamlit app for CT upload and review.
- NIfTI upload support (`.nii`, `.nii.gz`).
- DICOM zip upload support through `dcm2niix`.
- GPU-backed RADAR inference when checkpoints are available.
- Low-VRAM inference window selection for 8 GB GPUs.
- Three-plane CT viewer with segmentation overlay controls.
- Finding table with search, anatomy filtering, thresholding, and CSV export.
- Review notebook with shortlist, status, notes, JSON export, report text export, and full score export.
- Local case-history snapshots for saved review metadata.

This is a research and product prototype. It is not a certified medical device and must not be used as an autonomous diagnosis system. Outputs require qualified radiologist review.

## Upstream Project

This work started from:

https://github.com/alibaba-damo-academy/damo-radar

The upstream project describes RADAR as a generalist vision-language model trained on contrast-enhanced abdominal CT examinations with anatomy-aware image-text pairs. The original documentation remains useful for model training, preprocessing, and evaluation:

- `docs/TRAINING.md`
- `docs/INFERENCE.md`
- `docs/PREPROCESS.md`

## Repository Setup

Create a Python environment and install dependencies:

```bash
conda create -n radar python=3.10
conda activate radar
pip install -r requirements.txt
```

If using the local virtual environment that already exists on this machine:

```bash
source .venv/bin/activate
```

## Model Files

The RADAR model checkpoints and supporting files are large and are not intended to be committed directly into this repository.

Expected local layout:

```text
ckpt/
  checkpoint_radar_pretrain.pth
  infer_text_embedding_radar.pt
  infer_text_embedding_merlin.pt
  merlin_report_organ_normal_v1.json
  merlin_report_organ_report_v1.json
```

Download the required model files from the RADAR Hugging Face organization or with the upstream helper scripts:

```bash
cd download_scripts
python download_checkpoints.py
python download_auxiliary_data.py
```

## Run The App

From the repository root:

```bash
streamlit run webapp/app.py --server.address 127.0.0.1 --server.port 8501
```

Then open:

```text
http://127.0.0.1:8501
```

The app processes uploads locally. Temporary uploaded case files are written under the system temp directory with a `radar_web_` prefix and can be cleared from the sidebar. Saved case-history snapshots store review metadata, notes, shortlist, and scores under `~/.radar_ke/cases` by default; they do not archive CT pixel data.

## Docker

The Dockerfile copies the Streamlit app and `.streamlit` configuration, but checkpoints should be mounted instead of baked into the image.

Build:

```bash
docker build -t radar-ke .
```

Run with local checkpoint and data directories mounted as needed:

```bash
docker run --gpus all -p 8501:8501 \
  -v "$PWD/ckpt:/app/ckpt" \
  radar-ke
```

## Development

Run the lightweight app tests:

```bash
python -m unittest webapp.tests.test_review webapp.tests.test_app
```

Check whitespace errors before committing:

```bash
git diff --check
```

## Git Remotes

Recommended remote setup:

```text
origin   https://github.com/MorrisMuuoMulitu/radar-ke.git
upstream https://github.com/alibaba-damo-academy/damo-radar.git
```

Use `origin` for this Kenya-focused product work. Keep `upstream` for pulling future changes from the original RADAR project.

## Privacy And Clinical Use

- Uploaded scans stay on the local machine during the current workflow.
- The app does not send scans to a cloud service by default.
- Review exports are generated locally.
- The interface should be treated as a radiologist support tool.
- Clinical deployment requires privacy, audit, security, validation, regulatory, and institution-specific review.

## License And Attribution

The upstream RADAR code is released under Apache License 2.0, with third-party components under their own licenses. Model weights, datasets, and supporting files may have separate terms. Review upstream licenses and asset terms before commercial use.

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
