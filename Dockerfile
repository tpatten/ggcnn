# docker build -t ubuntu1604py36
#FROM ubuntu:16.04
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN rm -rf /var/lib/apt/lists/* \
    /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \
    apt-get install -y software-properties-common vim && \
    add-apt-repository ppa:jonathonf/python-3.6
    
RUN apt-get update -y

RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv && \
    apt-get install -y git

# ==================================================================
# update pip
# ------------------------------------------------------------------ 
RUN python3.6 -m pip install pip --upgrade && \
    python3.6 -m pip install wheel
    
# ==================================================================
# python
# ------------------------------------------------------------------    
RUN python3.6 -m pip install numpy \
                             opencv-python \
                             matplotlib \
                             scikit-image \
                             imageio \
                             torch \
                             torchvision \
                             torchsummary \
                             tensorboardX
                             
# ==================================================================
# tools
# ------------------------------------------------------------------    
RUN apt-get install -y libsm6 libxext6 libxrender-dev
    
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
