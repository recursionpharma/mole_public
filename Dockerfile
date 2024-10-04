ARG IMAGE_TAG=23.09-py3

FROM nvcr.io/nvidia/tensorrt:${IMAGE_TAG} as base

RUN python3 -m pip install -U pip && python3 -m pip install --no-cache-dir pip-tools

FROM base as training

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    make gcc gcc-multilib \
    daemontools \
    openssh-server \
    git-all \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install training requirements
COPY requirements/main_3.10_gpu.txt main_3.10_gpu.txt
COPY requirements/constraints_3.10.txt constraints_3.10.txt
RUN sed -i "s/torch==2.1.0/torch==2.1.0+cu118/" constraints_3.10.txt
RUN pip-sync constraints_3.10.txt
RUN pip-sync main_3.10_gpu.txt
RUN git clone https://github.com/recursionpharma/mole_public.git && \
    cd mole_public && \
    pip install -e .
