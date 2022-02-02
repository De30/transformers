FROM nvcr.io/nvidia/pytorch:21.03-py3
LABEL maintainer="Hugging Face"

RUN apt update
RUN apt install -y git libaio-dev


RUN git clone https://github.com/huggingface/transformers

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install ./transformers[testing,deepspeed]
