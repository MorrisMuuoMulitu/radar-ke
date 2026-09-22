# RADAR Kenya — Deployment Guide (Path A: LAN/VPN pilot)

This stack runs the RADAR review webapp on a GPU host with a protected
reverse proxy (Caddy) in front of it. It is designed for a small pilot:
a handful of trusted testers over LAN or VPN, with basic-auth protection
and TLS. The same Compose file runs on a cloud GPU VM (see
[docs/DEPLOYMENT_CLOUD.md](../docs/DEPLOYMENT_CLOUD.md)).

```
                                                      ┌─────────────┐
  tester browser ──https──► Caddy (443, basic-auth) ──► webapp:8501 │
                          (TLS + auth)                │  Streamlit  │
                                                      └──────┬──────┘
                                                             │ GPU (CUDA)
                                        ./ckpt (weights, ro) │
                                        radar-data volume (cases/uploads)
```

## 1. Prerequisites (host)

- Docker Engine **with the NVIDIA container toolkit** so containers can use
  the GPU:

  ```bash
  # after installing the NVIDIA driver:
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
  # verify:
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```

  (Ubuntu quick install: `sudo apt install -y nvidia-container-toolkit`.)
  Without the toolkit the `webapp` service fails to start with no GPU device.

- RADAR checkpoints present locally at `./ckpt` (see `download_scripts/`):

  ```bash
  cd download_scripts && python download_checkpoints.py
  ```

- Ports **80/443** free on the host.

## 2. Configure

```bash
cp deploy/.env.example deploy/.env
deploy/hash-password.sh 'choose-a-strong-password'   # prints the bcrypt hash
# put the hash into deploy/.env as RADAR_AUTH_HASH
```

Edit `deploy/.env`:

- `CADDY_DOMAIN` — for LAN testing keep `radar.localhost` (or the host IP /
  a name that resolves to this machine); TLS will be self-signed.
- `RADAR_AUTH_USER` / `RADAR_AUTH_HASH` — the shared tester login.

## 3. Build and start

From the repository root:

```bash
docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml logs -f webapp   # watch model init
```

Wait for `webapp` to become healthy (model load takes ~1–3 min; first
analysis triggers segmentation + scoring on the GPU).

## 4. Testers connect

- **LAN**: `https://<host-ip-or-radar.localhost>` — accept the self-signed
  certificate once, then log in with the basic-auth credentials.
- **VPN** (Tailscale/ZeroTier): give testers the machine's VPN IP; no public
  ports are opened.
- Give each tester [deploy/TESTERS.md](TESTERS.md).

### TLS certificate (recommended: trust the local CA)

Caddy uses its own internal CA (`tls internal`), so browsers warn with
`ERR_CERT_AUTHORITY_INVALID` until you click through. To remove the warning,
extract and trust Caddy's root CA once per machine:

```bash
# extract (already done in this repo → deploy/tls/caddy-root.crt)
docker run --rm -v radar-ke_caddy-data:/data:ro alpine \
  sh -c 'cat /data/caddy/pki/authorities/local/root.crt' > deploy/tls/caddy-root.crt

# Ubuntu/Debian system store (Chrome/Edge after browser restart)
sudo cp deploy/tls/caddy-root.crt /usr/local/share/ca-certificates/caddy-radar.crt
sudo update-ca-certificates

# Firefox (separate store): Settings → Privacy & Security → Certificates →
# View Certificates… → Authorities → Import… → deploy/tls/caddy-root.crt
# Windows: double-click the .crt → Install Certificate → Local Machine →
# Trusted Root Certification Authorities.
```

Only trust this CA on machines you control. Deleting the `radar-ke_caddy-data`
volume regenerates a new CA (re-extract and re-install).

## 5. Day-to-day

```bash
docker compose -f deploy/docker-compose.yml ps                # status
docker compose -f deploy/docker-compose.yml logs -f --tail=50 # logs
docker compose -f deploy/docker-compose.yml down              # stop (keeps volumes)
docker compose -f deploy/docker-compose.yml down -v           # stop + delete data
```

## 6. Data

- Saved reviews, uploads, and HF cache live in the `radar-ke_radar-data`
  named volume (`/data` in the container). It survives restarts and `down`.
- Backup:

  ```bash
  docker run --rm -v radar-ke_radar-data:/data -v "$PWD/backup:/backup" \
    alpine tar czf /backup/radar-data-$(date +%F).tar.gz -C /data .
  ```

- **Privacy:** keep real patient data out of the pilot. Use the bundled
  example and the public CT-ORG sample (`data/test_scans/`, see root README).
- Model weights are mounted read-only from `./ckpt`; they are not copied
  into images.

## 7. Multi-user behavior (pilot expectations)

- One GPU inference runs at a time; concurrent analyses queue on the model
  lock (second tester waits). The app is a research pilot, not yet
  multi-user production software.
- Case history is a single shared store for the pilot team. Per-user stores
  and real login are on the roadmap (see root README → Roadmap).

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `webapp` exits: no GPU device | Install nvidia-container-toolkit + restart docker (step 1). |
| CUDA OOM during analysis | Automatic on ≤10 GB GPUs (compact windows). On 8 GB cards it is expected and handled. |
| Slow analysis (CPU) | Container sees no GPU — same fix as the first row, or run on a GPU host. |
| `RADAR_AUTH_HASH: ... is required` | Set it in `deploy/.env` (step 2). |
| Self-signed warning in browser | Expected for LAN; accept once, or use a VPN IP + `tls internal`. |
| Ports already in use | Change the `ports` mapping in `deploy/docker-compose.yml` (e.g. `8443:443`). |

Cloud/path B: see [docs/DEPLOYMENT_CLOUD.md](../docs/DEPLOYMENT_CLOUD.md).