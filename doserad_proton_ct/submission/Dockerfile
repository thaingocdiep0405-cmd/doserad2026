# Grand Challenge runs one dose map per job, so torch.compile never amortises:
# the graded logs show 20s of startup compilation plus a second recompilation
# inside invoke once the batch shape drops to one, against 1.5s of eager
# inference for the same map. Disabling it turns 26s per job into under 2s.
FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ARG TASK=proton-ct
# "network" runs the trained U-Net, "pencilbeam" the pyRadPlan engine.
ARG DOSE_ENGINE=network
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/app \
    TASK=$TASK \
    INFERENCE_OVERLAP=0.0 \
    INFERENCE_PATCH_SIZE=96,96,96 \
    RAY_GATE_THRESHOLD=1e-4 \
    ROI_MODE=bbox \
    TORCH_COMPILE=0 \
    DOSE_ENGINE=$DOSE_ENGINE

RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
WORKDIR /opt/app

COPY doserad_proton_ct/submission/requirements.txt doserad_proton_ct/submission/requirements-pb.txt /opt/app/
RUN python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements.txt && \
    if [ "$DOSE_ENGINE" = "pencilbeam" ]; then \
        python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements-pb.txt; \
    fi

COPY doserad_proton_ct/submission/app.py doserad_proton_ct/submission/inference.py doserad_proton_ct/submission/inference_pb.py /opt/app/
COPY doserad_proton_ct/submission/beam_parameters.json /opt/app/beam_parameters.json
COPY doserad_proton_ct/src/doserad_proton /opt/app/doserad_proton
COPY doserad_photon_ct/src/doserad_photon_ct /opt/app/doserad_photon_ct

RUN chown -R user:user /opt/app
USER user

LABEL org.grand-challenge.api-method="invoke"
ENTRYPOINT ["python", "app.py"]
