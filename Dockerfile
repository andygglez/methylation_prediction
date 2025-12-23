FROM pytorchlightning/pytorch_lightning:2.5.6-py3.12-torch2.5-cuda12.1.1

ENV DEBIAN_FRONTEND=noninteractive

RUN pip install --upgrade pip

RUN pip install wandb matplotlib numpy
