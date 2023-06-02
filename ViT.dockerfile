FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime


RUN apt-get update -y 
RUN apt-get install -y build-essential
RUN pip install mmcv 
RUN pip install -U openmim
RUN mim install mmcv-full
RUN pip install timm
RUN pip install einops
RUN pip install mmdet
RUN pip install opencv-python-headless

