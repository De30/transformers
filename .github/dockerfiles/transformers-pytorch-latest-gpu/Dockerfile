FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip

RUN git clone https://github.com/huggingface/transformers

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir -e ./transformers[dev]
RUN python3 -m pip install --no-cache-dir -U torch~=1.10
RUN python3 -m pip uninstall -y torch flax
