# RADAR web app image (GPU required at runtime: docker run --gpus all ...)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip dcm2niix \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "numpy<2"

COPY RADAR_train RADAR_train
COPY RADAR_inference RADAR_inference
COPY webapp webapp
COPY .streamlit .streamlit
COPY download_scripts download_scripts

# Checkpoints + BERT dirs are large: mount them, do not bake in.
#   -v /host/ckpt:/app/ckpt
VOLUME ["/app/ckpt"]

EXPOSE 8501
CMD ["streamlit", "run", "webapp/app.py", "--server.address", "0.0.0.0"]
