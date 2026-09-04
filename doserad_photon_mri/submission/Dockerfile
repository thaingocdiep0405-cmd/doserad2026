FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/app \
    TASK=photon-mri

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
WORKDIR /opt/app

COPY submission/requirements.txt /opt/app/requirements.txt
RUN python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements.txt

COPY submission/app.py submission/inference.py /opt/app/
COPY src/doserad_photon_mri /opt/app/doserad_photon_mri

RUN chown -R user:user /opt/app
USER user

LABEL org.grand-challenge.api-method="invoke"
ENTRYPOINT ["python", "app.py"]
