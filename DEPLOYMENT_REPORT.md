# DEPLOYMENT_REPORT — RADAR Kenya CT Review Webapp (LAN Pilot)

## Summary

**Verdict: Deployed with caveats** — the stack (`webapp` + `caddy`) is running and
healthy, GPU access is proven inside the container, the HTTPS/basic-auth gate
returns 401/200 as designed, and end-to-end CUDA inference succeeds on a public
CT-ORG scan. Caveats: the CLI smoke test only passes on a **single** scan on this
host (multi-scan folder mode OOMs — see Issues), the model loads **lazily** so the
`Initialize done` line is not present in startup logs, and two **deployment-config
fixes** were required in `deploy/docker-compose.yml` (documented below). No
application source code was modified. A browser walkthrough was not performed by
the agent and still needs a human.

## Environment

| Item | Value |
|---|---|
| Host | `Asgard` |
| OS / kernel | Ubuntu 26.04.1 LTS / `7.0.0-31-generic` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB |
| Driver | 580.173.02 (CUDA 13.0) |
| Docker | 29.1.3 (build 29.1.3-0ubuntu4.1) |
| Docker Compose | 2.40.3+ds1-0ubuntu1 |
| Repo / branch / commit | `/home/mulitu/Desktop/damo-radar`, `main`, `c3d16dc` |
| Image | `radar-ke:latest` (built locally) |
| Container runtime | `nvidia` (installed in Phase 1) |

## Steps & results

### Phase 0 — Pre-flight (all pass)
- `nvidia-smi` → RTX 4060 Laptop, driver 580.173.02. ✅
- `docker info | grep Runtimes` → only `io.containerd.runc.v2 runc` (**no `nvidia`**) → Phase 1 required.
- Checkpoints present: `ckpt/checkpoint_radar_pretrain.pth` (1.46 GB), `ckpt/checkpoint_unet.pth` (199 MB). ✅
- Ports 80/443 free (`ss -tlnp`). ✅
- `deploy/.env` present with a valid bcrypt `RADAR_AUTH_HASH` (value never printed). ✅
- `git status --short` clean at `c3d16dc`.

### Phase 1 — NVIDIA container toolkit (required)
The toolkit was in a half-removed state (`nvidia-container-toolkit-base` status
`rc`, empty apt list, keyring present). Installed from the official repo:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit     # 1.20.1-1
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
- `docker info` now lists the `nvidia` runtime. ✅
- **Gate:** `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` → exit 0, printed RTX 4060. ✅

### Phase 2 — Build
```bash
docker compose -f deploy/docker-compose.yml build   # exit 0
```
Base image `nvidia/cuda:12.4.1-runtime-ubuntu22.04` + pip deps (~28 min; torch
2.14.0+cu130, monai, streamlit, transformers, etc.). Image `radar-ke:latest`.

### Phase 3 — Start + health
```bash
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml ps
```
- Startup logs contain `Uvicorn server started on 0.0.0.0:8501`. ✅
- Caddy obtained its internal certificate for `radar.localhost`. ✅
- **CUDA gate:** `docker compose exec -T webapp python3 -c "import torch; …"` →
  `cuda_available=True`, `device=NVIDIA GeForce RTX 4060 Laptop GPU`, `torch=2.14.0+cu130`. ✅
- `docker compose exec -T webapp nvidia-smi` → exit 0, RTX 4060. ✅
- **Deviation:** the model is loaded **lazily** (`webapp/app.py:168` →
  `inference_service.get_model()`), so `--> Initialize done (device: cuda)` is
  **not** emitted at startup. It was captured during Phase 5 (below). This is a
  design property, not a fault.

### Phase 4 — Protected endpoint (Caddy)
```bash
curl -ks -o /dev/null -w '%{http_code}\n' https://radar.localhost/                       # 401
curl -ks -u "radiologist:$RADAR_PILOT_PASSWORD" -o /dev/null -w '%{http_code}\n' https://radar.localhost/   # 200
```
- No credentials → **401**; wrong password → **401**; valid → **200** (7260-byte
  Streamlit page). `radar.localhost` resolves to `::1`. ✅
- Password was supplied only for this check (from the brief, per operator
  authorization) and was not written to any file or to this report.

### Phase 5 — End-to-end GPU inference smoke test
The brief's command uses `python` (not present in the image) and, in folder mode,
loads all 3 CT-ORG volumes concurrently via `num_workers=12`, which OOM-killed the
host. Final working invocation (single scan, matching the live app's per-case path):
```bash
mkdir -p deploy/smoke-out
docker compose -f deploy/docker-compose.yml run --rm --no-deps \
  -e ROI_SIZE=64,192,288 \
  -v "$PWD/data/test_scans/ct_org/volumes/volume-105.nii.gz:/smoke-in/volume-105.nii.gz:ro" \
  -v "$PWD/deploy/smoke-out:/app/smoke-out" \
  webapp python3 /app/RADAR_inference/inference_demo.py \
  --img_dir /smoke-in --save_dir /app/smoke-out --save_tag container_smoke
```
- Exit **0**; log shows `--> Initialize done (device: cuda)` and
  `evaluate done, save result_csv to /app/smoke-out/RADAR_infer_results_container_smoke.csv`. ✅
- Inference time ≈ 26 s on the GPU for the single volume.

### Phase 6 — Report
This document.

## Verification evidence

**GPU in container (Phase 1 gate):**
```
$ docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi   # EXIT=0
| NVIDIA-SMI 580.173.02   Driver Version: 580.173.02   CUDA Version: 13.0
|   0  NVIDIA GeForce RTX 4060 ...   1145MiB / 8188MiB
```

**CUDA visible to the app container:**
```
cuda_available= True
device= NVIDIA GeForce RTX 4060 Laptop GPU
torch= 2.14.0+cu130
```

**Model init on GPU (from Phase 5 log):**
```
--> Start initializing...
--> ckpt_path:  /app/ckpt/checkpoint_radar_pretrain.pth
--> Initialize done (device: cuda)
--> Start inference.
Infer: 100%|██████████| 1/1 [00:25<00:00, 25.82s/it]
evaluate done, save result_csv to /app/smoke-out/RADAR_infer_results_container_smoke.csv.
```

**Auth gate (Phase 4):**
```
no-creds:   401
wrong-pw:   401
valid-creds:200
valid-body-bytes:7260
```

**Inference CSV** `deploy/smoke-out/RADAR_infer_results_container_smoke.csv`:
- data rows = **1**, columns = **147** (1 `file_name` + 146 finding scores)
- non-null scores = **146 / 146**
- masked score sample: min ≈ **0.00**, max ≈ **0.98** (e.g. top finding ≈ 0.97)

**Final `docker compose ps`:**
```
NAME                IMAGE             STATUS
radar-ke-caddy-1    caddy:2-alpine    Up (healthy)   0.0.0.0:80->80/tcp, 0.0.0.0:443->443/tcp
radar-ke-webapp-1   radar-ke:latest   Up (healthy)   8501/tcp
```

## Issues & resolutions

1. **Docker had no `nvidia` runtime (GPU invisible).**
   *Cause:* toolkit half-removed. *Fix:* reinstalled `nvidia-container-toolkit`
   1.20.1-1, `nvidia-ctk runtime configure --runtime=docker`, restarted Docker.
   *Result:* GPU gate passed.

2. **`dpkg` blocked on a leftover conffile prompt** (`nvidia-cdi-refresh.env`,
   from the prior partial install) during non-interactive install.
   *Fix:* reinstalled with `-o Dpkg::Options::="--force-confold"` + `dpkg --configure -a`.
   *Result:* toolkit configured cleanly.

3. **Smoke test `python: not found` (exit 127).**
   *Cause:* the image ships `python3`, not `python`; the brief's command uses
   `python`. *Fix:* ran the same script with `python3`. (Brief bug; no source change.)

4. **`RuntimeError: unable to allocate shared memory(shm) … No space left on
   device (28)`.**
   *Cause:* Docker default `/dev/shm` = 64 MB; `inference_service.py:423` uses
   `DataLoader(num_workers=12)` and passes full CT volumes between processes via
   shm. *Fix:* added `shm_size: "4gb"` to the `webapp` service in
   `deploy/docker-compose.yml` and recreated the stack. *Result:* shm error gone.

5. **Host OOM-kill (exit 137) in folder mode with all 3 volumes.**
   *Cause:* 12 DataLoader workers prefetch the 3 large CT-ORG volumes
   concurrently; the inference process reached ~13 GB RSS. With host swap already
   ~100 % full, the kernel/`systemd-oomd` killed the process. *Mitigation:*
   ran the smoke test on a **single** scan (the same path the live app uses —
   `run_case` stages exactly one file, so only one worker loads). *Result:* exit 0,
   CSV produced. **This is a host-capacity caveat, not a GPU/auth defect.**

6. **`caddy` reported `unhealthy` though the proxy worked.**
   *Cause:* the healthcheck fetched `http://127.0.0.1:80/`, which Caddy 308-redirects
   to HTTPS; wget then hit `https://127.0.0.1/`, where `tls internal` resets the
   handshake (cert is for `radar.localhost`) → false negative.
   *Fix:* changed the healthcheck to probe the admin API
   `http://127.0.0.1:2019/config/` (returns 200). *Result:* `caddy` now `healthy`.

### Deployment-config changes (explicit)
Only `deploy/docker-compose.yml` was changed (two lines + comments); **no
application source** was touched:
- `webapp`: added `shm_size: "4gb"`.
- `caddy`: healthcheck probe changed to the admin API.
These are **uncommitted** (`git status` shows ` M deploy/docker-compose.yml`).
`deploy/.env` was neither read into the report nor modified. `deploy/smoke-out/`
is a new untracked artifact directory.

## Access & usage

- **URL:** `https://radar.localhost/` on this host. The Caddy site block is bound
  to `radar.localhost` only, so a **bare-IP** URL (`https://192.168.1.5/`) fails
  the TLS handshake. From other LAN machines, either:
  - add an `/etc/hosts` entry on the tester machine — `192.168.1.5 radar.localhost`
    (or `.7` for the Wi-Fi interface) — then open `https://radar.localhost/`; **or**
  - set `CADDY_DOMAIN` in `deploy/.env` to the LAN name/IP and
    `docker compose -f deploy/docker-compose.yml up -d caddy` to re-provision.
  Verified: `--resolve radar.localhost:443:192.168.1.5` → **401** (auth gate works
  over LAN), bare IP → TLS error `exit 35`.
- Testers must accept the **self-signed certificate** once (expected for the LAN pilot).
- **Login:** basic-auth user `radiologist`; password is the one you chose
  (rotate any time with `deploy/hash-password.sh`; the hash lives in
  `deploy/.env`). The password is deliberately **not** recorded in this report.
- **Testers:** hand them [`deploy/TESTERS.md`](deploy/TESTERS.md).
- **Maintenance** (from `deploy/README.md`):
  ```bash
  docker compose -f deploy/docker-compose.yml ps                 # status
  docker compose -f deploy/docker-compose.yml logs -f --tail=50  # logs
  docker compose -f deploy/docker-compose.yml down               # stop (keeps data)
  docker compose -f deploy/docker-compose.yml down -v            # stop + DELETE data
  ```
- **Data/privacy:** saved reviews and uploads live in the `radar-ke_radar-data`
  volume; weights are mounted read-only from `./ckpt`. Keep real patient data out
  of the pilot — only the bundled example and the public CT-ORG samples were used.
- **Multi-user:** one GPU inference at a time (analyses queue on the model lock).

## Outstanding items

1. **Browser walkthrough not performed by the agent** — the Streamlit UI itself
   (upload → review → save) has not been exercised end-to-end in a browser. The
   underlying `run_case` engine is proven by the CLI smoke test. **Recommended
   next action:** open `https://radar.localhost/` (or the LAN IP), accept the cert,
   log in, and run the bundled example / one CT-ORG volume through the UI.
2. **Multi-scan (folder) inference is not usable on this host as-is** — 12
   DataLoader workers × full volumes exhaust RAM/swap. Single-case use (the app's
   actual path) works. If batch/folder inference is ever needed, reduce
   `num_workers` or raise host RAM/swap (would be an app-source change, so not
   made here).
3. **Host memory pressure** — swap was ~100 % used during testing; closing other
   heavy apps (e.g. Chrome) is advisable before live use.
4. **Changes are uncommitted** — review and commit `deploy/docker-compose.yml`
   when ready (not done, per instructions).
5. Cosmetic, non-blocking: Caddy logs a `basicauth` deprecation warning (known,
   functional) and PyTorch emits non-tuple-indexing `UserWarning`s in
   `inference_service.py:485,502,503`.
