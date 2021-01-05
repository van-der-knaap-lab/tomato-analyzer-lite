FROM ubuntu:18.04

LABEL maintainer="Wes Bonelli"

COPY . /opt/tomato-analyzer-lite

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
    build-essential \
    python3-setuptools \
    python3-pip \
    python3-numexpr \
    python3.8 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1

RUN python3.8 -m pip install --upgrade pip && \
    python3.8 -m pip install -r /opt/tomato-analyzer-lite/requirements.txt
