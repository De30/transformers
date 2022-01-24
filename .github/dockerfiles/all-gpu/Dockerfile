FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
LABEL maintainer="Hugging Face"

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng

RUN git clone https://github.com/huggingface/transformers

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir ./transformers[integrations,sklearn,testing,onnx,onnxruntime,sentencepiece,torch-speech,vision,timm,tf-speech]

RUN python3 -m pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-$(python -c "from torch import version; print(version.__version__.split('+')[0])")+cpu.html
RUN python3 -m pip install --no-cache-dir torchvision git+https://github.com/facebookresearch/detectron2.git pytesseract https://github.com/kpu/kenlm/archive/master.zip