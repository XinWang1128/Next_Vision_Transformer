FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH="/usr/local/cuda-11.8/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"

COPY sources.list /etc/apt/sources.list

RUN apt-get update -y
RUN apt-get install -y build-essential

RUN python -m pip install --upgrade pip 

RUN pip install mmcv 
RUN pip install -U openmim
RUN mim install mmcv-full
RUN pip install timm
RUN pip install einops
RUN pip install mmdet
RUN pip install opencv-python-headless

