FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG CACHE_SKIPER=$(date +%s)
RUN echo "Cloning repo with sha"${CACHE_SKIPER}

RUN git clone https://github.com/huggingface/transformers

RUN python3 -m pip install --no-cache-dir -e ./transformers[dev,onnxruntime]
RUN python3 -m pip install --no-cache-dir -U tensorflow~=2.7
RUN python3 -m pip uninstall -y torch flax

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop
