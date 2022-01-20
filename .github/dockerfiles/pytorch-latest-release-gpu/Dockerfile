FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
LABEL maintainer="Hugging Face"

RUN apt update
RUN apt install -y git


RUN git clone https://github.com/huggingface/transformers

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install ./transformers[dev]

RUN apt-get -y update && apt-get install -y libsndfile1-dev && apt install -y tesseract-ocr

RUN python3 -m pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "from torch import version; print(version.__version__.split('+')[0])")+cpu.html
RUN python3 -m pip install --no-cache-dir \
    torchvision \
    git+https://github.com/facebookresearch/detectron2.git \ 
    pytesseract \
    https://github.com/kpu/kenlm/archive/master.zip
