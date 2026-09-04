FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/app \
    TASK=photon-ct \
    INFERENCE_BATCH_SIZE=4 \
    CONTROL_POINT_CHUNK_SIZE=32 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# gcc lets triton build torch.compile kernels on the grading GPU; without it
# inference silently falls back to eager, which is slower and uses more VRAM.
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
WORKDIR /opt/app

COPY submission/requirements.txt /opt/app/requirements.txt
RUN python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements.txt

COPY submission/app.py submission/inference.py /opt/app/
COPY src/doserad_photon_ct /opt/app/doserad_photon_ct

RUN chown -R user:user /opt/app
USER user

LABEL org.grand-challenge.api-method="invoke" \
      org.grand-challenge.build="v6-resubmit"
ENTRYPOINT ["python", "app.py"]
