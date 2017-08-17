# Start with Ubuntu base image
FROM ubuntu:16.04
MAINTAINER nakkat

# configure environment
ENV SHELL /bin/bash
ENV NB_USER py-deep
ENV NB_UID 1000
ENV WORK_DIR /src
ENV LANG C.UTF-8

# Install
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        less \
        # install python3
        python3 \
        python3-dev \
        python3-setuptools \
        python3-pip \
        python3-venv \
        python3-wheel \
        # requirements for numpy
        python3-numpy \
        python3-scipy \
        python3-pandas \
        # requirements for keras
        python3-h5py \
        python3-yaml \
        python3-pydot \
        # requirements for matplotlib
        python3-tk \
        libpng12-dev \
        libfreetype6-dev \
        python3-matplotlib \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip six
RUN pip3 --no-cache-dir install -U jupyter

# Create py-deep user with UID=1000 and in the 'users' group
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER \
    && mkdir -p /src \
    && chown $NB_USER /src

RUN mkdir -p /src/py-deep-samples \
    && git clone https://github.com/nakkat/py-deep-samples.git /src/py-deep-samples \
    && pip install git+https://github.com/nakkat/py-deep-samples.git

WORKDIR $WORK_DIR
USER $NB_USER
