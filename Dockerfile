FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
COPY transformers /transformers
RUN rm -Rf /transformers/.venv
RUN cd /transformers && pip install -e ".[dev]"
RUN apt-get -y update && apt-get -y install git &&  apt-get -y install gcc 
RUN cd $HOME && git clone --recursive https://github.com/facebookresearch/vissl.git && cd $HOME/vissl/ && \ 
    # Optional, checkout stable v0.1.6 branch. While our docs are versioned, the tutorials
    # use v0.1.6 and the docs are more likely to be up-to-date.
    git checkout v0.1.6 && \ 
    git checkout -b v0.1.6 && \
    # install vissl dependencies
    pip install opencv-python &&  pip install cython && \
    # update classy vision install to commit stable for vissl.
    # Note: If building from vissl main, use classyvision main.
    pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/4785d5ee19d3bcedd5b28c1eb51ea1f59188b54d && \
    # update fairscale install to commit stable for vissl.
    pip install fairscale==0.4.6 && \
    # install vissl dev mode (e stands for editable)
    pip install -e ".[dev]"

CMD ["python", "/transformers/src/transformers/models/regnet/convert_regnet_to_pytorch.py" , "--pytorch_dump_folder_path", "/models", "--model_name", "regnet-y-320-seer"]