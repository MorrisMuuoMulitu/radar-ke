# RADAR web app image (GPU required at runtime: docker compose / --gpus all)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/mpl \
    RADAR_CASE_HISTORY_DIR=/data/cases

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip dcm2niix \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "numpy<2"

# ffmpeg powers H.264 cine-clip export (browser- and WhatsApp-friendly MP4).
# Kept in its own layer after pip so code changes reuse the dependency cache.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY RADAR_train RADAR_train
COPY RADAR_inference RADAR_inference
COPY webapp webapp
COPY .streamlit .streamlit
COPY download_scripts download_scripts

# Files created with restrictive modes on the host (e.g. 600) must still be
# readable by the non-root runtime user, otherwise imports fail at session start.
RUN chmod -R a+rX /app

# Checkpoints + BERT dirs are large: mount them, do not bake in.
#   -v /host/ckpt:/app/ckpt   (compose mounts ./ckpt read-only)
# Persistent case history / uploads land in the /data volume.
VOLUME ["/app/ckpt", "/data"]

RUN useradd --create-home --uid 1000 radar \
    && mkdir -p /data/cases \
    && chown -R radar:radar /data /home/radar

USER radar

EXPOSE 8501
# Also import the app modules: a permission/import fault must fail the health
# check, not just look healthy while every session crashes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python3 -c "import sys, urllib.request; \
sys.path.insert(0, '/app/webapp'); import cine, review; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).status == 200 else 1)" \
    || exit 1

CMD ["streamlit", "run", "webapp/app.py", "--server.address", "0.0.0.0"]