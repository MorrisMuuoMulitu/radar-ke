# RADAR Kenya — Deployment Path B: GPU Cloud VM

Extends the LAN pilot ([deploy/README.md](../deploy/README.md)) to a
single GPU VM so **external testers** can reach the app over HTTPS. The
same `deploy/` Compose stack is used; the differences are a real domain
(Let's Encrypt via Caddy), a firewall, persistent disk, and cost control.

> **Before you go cloud:** only de-identified / synthetic data (e.g. the
> bundled example or the public CT-ORG sample). Real patient data on third-
> party infrastructure requires institutional privacy/security review.

## 1. Choose a provider

| Provider | Instance | GPU | Notes |
|---|---|---|---|
| AWS | `g5.xlarge` | T4 (16 GB) | mature; uses EBS, snapshots |
| AWS | `g4dn.xlarge` | T4 (16 GB) | older, cheaper |
| Azure | `Standard_NC4as_T4_v3` | T4 (16 GB) | |
| Lambda / RunPod / Vast.ai | various | RTX 3090/4090, A10 | cheapest hourly; often BYO-image friendly |

Recommended spec: **≥ 16 GB VRAM**, 4 vCPU, 16 GB RAM, 100 GB+ disk
(the model + pip deps + case data fit comfortably). 8 GB cards work too
(compact windows) but are tighter.

## 2. Provision checklist (any provider)

1. Create the VM with an Ubuntu 22.04+ **GPU image** (driver preinstalled),
   a public IP, and a security group **allowing only 22 (SSH) and 443**.
2. Install Docker + NVIDIA container toolkit:

   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo apt install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

3. Get the code and weights:

   ```bash
   git clone https://github.com/MorrisMuuoMulitu/radar-ke.git && cd radar-ke
   cd download_scripts && python download_checkpoints.py   # → ../ckpt
   cd ..
   ```

4. **DNS**: point `radar.example.com` (A record) at the VM's public IP.

5. Configure the stack for a public domain:

   ```bash
   cp deploy/.env.example deploy/.env
   # deploy/.env:
   #   CADDY_DOMAIN=radar.example.com
   #   CADDY_EMAIL=you@example.com     (for Let's Encrypt)
   #   RADAR_AUTH_USER=... RADAR_AUTH_HASH=...   (deploy/hash-password.sh)
   ```

   Edit `deploy/Caddyfile`: **remove the `tls internal` line** so Caddy
   provisions Let's Encrypt certificates for the real domain automatically.

6. Start:

   ```bash
   docker compose -f deploy/docker-compose.yml up -d --build
   docker compose -f deploy/docker-compose.yml logs -f webapp
   ```

   Testers use `https://radar.example.com`.

## 3. Operations

```bash
docker compose -f deploy/docker-compose.yml ps
docker compose -f deploy/docker-compose.yml logs -f --tail=50
docker compose -f deploy/docker-compose.yml down -v     # full reset
```

- **Persistence**: case history/uploads live in the `radar-ke_radar-data`
  volume; back it up (command in deploy/README.md §6) and snapshot the VM
  disk weekly.
- **Updates**: `git pull && docker compose -f deploy/docker-compose.yml up -d --build`.
- **Monitoring**: `nvidia-smi` for GPU, `docker stats` for memory/CPU, and
  the compose logs.

## 4. Cost control

- GPU VMs are billed by the hour. **Stop (not terminate) the VM** when the
  pilot is idle; EBS keeps the data.
- Add an idle schedule: `sudo systemctl` timer or cloud "stop at X" rules.
- Prefer T4/L4-class instances; inference runs on one GPU at a time.

## 5. When the pilot outgrows one VM (later)

- Multi-worker inference (queue + GPU pool), per-user stores + real login,
  audit log, and object storage for scans — tracked in the root README
  Roadmap. The Compose structure keeps the webapp/caddy split so fronting a
  scalable backend later does not change the proxy layer.

## Alternative: public demo (no GPU needed)

If the goal is only a **marketing demo** (example case, no scan analysis),
the app already runs without CUDA. A CPU-only container (or HF Spaces
CPU instance) with `Open example case` works — but scan analysis will be
impractically slow, so this is a demo, not a test environment.