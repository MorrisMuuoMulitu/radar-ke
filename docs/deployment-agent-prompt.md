# Deployment Agent Brief — RADAR Kenya Pilot (LAN)

You are an automated deployment engineer with shell (bash) access on the target
machine. Your job is to deploy the RADAR CT review webapp as a protected LAN
pilot, verify it end-to-end, and write a deployment report. Work step by step,
verify each step before moving on, and follow the issue-handling policy below.

## Goal (immutable)

Get the Docker-based pilot stack **running and verified on this machine**:

1. NVIDIA GPU available to Docker containers.
2. `webapp` (Streamlit) container healthy with the RADAR model loaded on CUDA.
3. `caddy` reverse proxy serving HTTPS with basic auth in front of the app.
4. End-to-end GPU inference proven on one test scan (no patient data).
5. A written `DEPLOYMENT_REPORT.md` documenting everything.

## Machine & repo context (verify each yourself, do not assume)

- OS: Ubuntu 22.04+; shell user `mulitu`; hostname `Asgard`.
- GPU: NVIDIA GeForce RTX 4060 Laptop (8 GB); host driver 580.173.02 — confirm
  with `nvidia-smi` (may need `sudo`).
- Docker 29.x + Docker Compose v2 installed (`docker --version`,
  `docker compose version`).
- Repo: `/home/mulitu/Desktop/damo-radar` (git branch `main`, remote `origin`).
- The deployment stack already exists and was previously validated:
  - `deploy/docker-compose.yml` — `webapp` (GPU reservation, `../ckpt` read-only
    mount, `radar-data` volume for case history/uploads) + `caddy`
    (basic auth + TLS, ports 80/443).
  - `deploy/Caddyfile` — `tls internal` (self-signed), `basicauth` with env
    placeholders; validated against `caddy:2`.
  - `deploy/.env` — ALREADY configured by the user:
    `RADAR_AUTH_USER=radiologist`, `RADAR_AUTH_HASH=<bcrypt hash, $$-escaped>`,
    `CADDY_DOMAIN=radar.localhost`. **Do not regenerate or "fix" this hash.**
  - `deploy/.env.example`, `deploy/hash-password.sh`, `deploy/README.md`,
    `deploy/TESTERS.md`, `docs/DEPLOYMENT_CLOUD.md`.
- Checkpoints present: `ckpt/checkpoint_radar_pretrain.pth` and friends.
- Test data (no patient data): `data/demo_cases/AC423ccbe.nii.gz`,
  `data/test_scans/ct_org/volumes/*.nii.gz`.
- Known quirk (do NOT "fix" it): Docker Compose interpolates `$` in `.env`
  values, so the bcrypt hash is deliberately stored with `$$` escaping. Compose
  renders `$$` → `$`; the container receives the correct 60-char `$2a$…` hash.

## Security & safety rules (never violate)

- **Never print, log, or commit** the contents of `deploy/.env` or any password.
  Mask secrets in the report. `deploy/.env` is git-ignored — never `git add` it.
- **Never upload, download, or process real patient data.** Only the bundled
  example and `data/test_scans` (public CT-ORG) are allowed.
- **Do not expose the app beyond the LAN.** Self-signed TLS + basic auth is by
  design. Do not open cloud ports or bypass auth.
- **Do not run destructive commands** (`docker compose down -v` deletes stored
  cases; `git reset --hard`, `git push --force` etc.) unless explicitly asked.
- If a step needs the real password (basic-auth check), read it from
  `$RADAR_PILOT_PASSWORD` if set; otherwise ask the user for it. Never store it
  in a file.
- Prefer read-only inspection (`docker compose ps`, `logs`, `config --format json`,
  `git status`) before any change.

## Steps (do in order; verify each gate before continuing)

### Phase 0 — Pre-flight
1. `nvidia-smi` → expect RTX 4060 + driver 580.x.
2. `docker info | grep Runtimes` → if `nvidia` is missing, the toolkit is not
   installed (Phase 1 needed).
3. Checkpoints: `ls ckpt/checkpoint_radar_pretrain.pth ckpt/checkpoint_unet.pth`.
4. Ports: `ss -tlnp | grep -E ':80 |:443 '` → both free.
5. `test -f deploy/.env && grep -c '^RADAR_AUTH_HASH=..2a..14' deploy/.env`
   (verify present; do not print the value).
6. `git status --short` — working tree may have changes; do not commit anything
   unless asked.

### Phase 1 — NVIDIA container toolkit (only if `nvidia` runtime absent)
Add the nvidia-container-toolkit apt repo (Ubuntu 22.04), install, register,
restart:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Sudo policy:** try `sudo -n true` first. If sudo needs a password and you
cannot supply it non-interactively, stop and ask the user to paste the exact
sudo lines above into a terminal, wait for their confirmation, then continue.
If `gpg` asks "Overwrite?" for the existing keyring, answer `y` (regenerating
the official key is idempotent).

**Gate:** `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
must print the RTX 4060. If it instead says the driver can't communicate,
re-check `sudo nvidia-ctk runtime configure --runtime=docker` + restart, then
retry once; if it still fails, STOP and report blocked (see Issues).

### Phase 2 — Build
From the repo root:

```bash
docker compose -f deploy/docker-compose.yml build
```

First build downloads the CUDA base image + Python deps (a few GB, several
minutes). Use a generous timeout. If a network failure occurs, retry once or
twice; if it keeps failing, report infrastructure/network as the blocker.

### Phase 3 — Start + wait for health
```bash
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml ps
```
Wait for `webapp` and `caddy` to be `running` and (eventually) `healthy`
(webapp start grace is 180 s for model load). Follow logs:

```bash
docker compose -f deploy/docker-compose.yml logs -f webapp
```
**Gate:** logs must show `Uvicorn server started on 0.0.0.0:8501` and
`--> Initialize done (device: cuda)`. If `device: cpu` — the container cannot
see the GPU; go back to Phase 1 (do not proceed silently).
Also confirm inside the container: `docker compose exec webapp nvidia-smi`
(expect RTX 4060 with ~4-5 GB used by the loaded model).

### Phase 4 — Verify the protected endpoint (Caddy)
```bash
curl -ks -o /dev/null -w '%{http_code}\n' https://radar.localhost/            # expect 401 (no credentials)
curl -ks -u "radiologist:$RADAR_PILOT_PASSWORD" -o /dev/null -w '%{http_code}\n' https://radar.localhost/   # expect 200
```
(`radar.localhost` resolves to 127.0.0.1; if not, use
`curl -ks --resolve radar.localhost:443:127.0.0.1 …`.)
If `RADAR_PILOT_PASSWORD` is unset, ask the user for the password for this check
only. If the 200 check fails, inspect `docker compose logs caddy` — the most
likely cause is the `$$` escaping being wrong (container receiving `$$2a…`
instead of `$2a…`); report it, do not weaken auth to work around it.

### Phase 5 — End-to-end GPU inference smoke test (no browser needed)
Run the CLI inference inside the container on ONE public test scan, mounting
the scan read-only and results to the host:

```bash
mkdir -p deploy/smoke-out
docker compose run --rm --no-deps \
  -v "$PWD/data/test_scans/ct_org/volumes:/test_scans:ro" \
  -v "$PWD/deploy/smoke-out:/app/smoke-out" \
  webapp python /app/RADAR_inference/inference_demo.py \
  --img_dir /test_scans --save_dir /app/smoke-out --save_tag container_smoke
```

- **Gate:** exit 0, and `deploy/smoke-out/RADAR_infer_results_container_smoke.csv`
  exists with one row per scan + non-empty scores. Time: several minutes on the
  GPU (compact windows).
- If this step fails with CUDA OOM, retry with `ROI_SIZE=64,192,288` env passed
  to the run; if it still fails, record it as a caveat (not necessarily a
  deployment blocker) and continue to the report.

### Phase 6 — Report
Write `DEPLOYMENT_REPORT.md` at the repo root with EXACTLY these sections:

1. **Summary** — verdict line: `Deployed & verified` | `Deployed with caveats`
   | `Blocked` (+ one-sentence reason, if blocked).
2. **Environment** — OS/kernel, GPU + driver, Docker/Compose versions, repo
   commit (`git rev-parse --short HEAD`).
3. **Steps & results** — each phase above with the command run and its outcome.
4. **Verification evidence** — the key outputs: nvidia-smi container test,
   `device: cuda` log line, 401/200 curl results, inference CSV existence +
   scan count + a masked score sample (e.g. `max 0.94`), `docker compose ps`.
5. **Issues & resolutions** — everything that went wrong, exactly what you tried,
   and the outcome (mask any secrets).
6. **Access & usage** — how the user connects (URL, self-signed-cert note,
   login user; password never written in the report — say "password you chose,
   rotate with deploy/hash-password.sh"), what testers should do
   (`deploy/TESTERS.md`), and the maintenance commands (from `deploy/README.md`).
7. **Outstanding items** — anything not verified (e.g., browser walkthrough
   still needs a human), plus the recommended next action.

Do not modify application source code unless fixing a deployment bug is
unavoidable; if you do, call it out explicitly in the report.

## Issue-handling policy

1. **Diagnose before changing.** Read the full error, then inspect
   (`docker compose logs webapp`, `logs caddy`, `docker inspect`), consult
   `deploy/README.md` (Troubleshooting) and `docs/DEPLOYMENT_CLOUD.md`, then fix.
2. **Known gotchas** (handle without asking):
   - GPU invisible in container / `device: cpu` → toolkit missing or Docker not
     restarted after `nvidia-ctk runtime configure`.
   - Login/auth broken → check `docker compose logs caddy`; verify the container
     receives `$2a$…` (60 chars, single `$`) — never store the hash without
     `$$` escaping.
   - Ports busy → report; do not kill unrelated processes without asking.
   - Slow first build → network; retry up to twice.
   - Browser TLS warning → expected (self-signed); not a bug.
3. **Retry policy:** after applying an obvious fix, retry once. If the same
   blocker persists after a second attempt, or you hit an error you cannot
   explain, **STOP** and write the report with verdict `Blocked`, the exact
   error text, what you tried, and your best next step. Do not guess-and-poke
   endlessly and do not silently skip gates.
4. **Unknown errors:** capture the full stack trace/logs into the report.

## Deliverables

- The running stack (webapp + caddy healthy, GPU inference verified).
- `DEPLOYMENT_REPORT.md` (repo root) per Phase 6.
- Do not commit code unless the user asks.