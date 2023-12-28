FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y \
        nano python3-pip python3-mock libpython3-dev \
        libpython3-all-dev python-is-python3 wget curl cmake \
        software-properties-common sudo \
    && sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pip -U \
    && pip install onnx==1.15.0 \
    && pip install onnxsim==0.4.33 \
    && pip install nvidia-pyindex \
    && pip install onnx_graphsurgeon \
    && pip install onnx2tf \
    && pip install onnx2tf \
    && pip install simple_onnx_processing_tools \
    && pip install tensorflow==2.15.0 \
    && pip install protobuf==3.20.3 \
    && pip install h5py==3.7.0 \
    && pip install psutil==5.9.5 \
    && pip install onnxruntime==1.16.3 \
    && pip install ml_dtypes==0.2.0

# Re-release flatc with some customizations of our own to address
# the lack of arithmetic precision of the quantization parameters
# https://github.com/PINTO0309/onnx2tf/issues/196
RUN wget https://github.com/PINTO0309/onnx2tf/releases/download/1.16.31/flatc.tar.gz \
    && tar -zxvf flatc.tar.gz \
    && chmod +x flatc \
    && mv flatc /usr/bin/

ENV USERNAME=user
RUN echo "root:root" | chpasswd \
    && adduser --disabled-password --gecos "" "${USERNAME}" \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && echo "%${USERNAME}    ALL=(ALL)   NOPASSWD:    ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}
USER ${USERNAME}
ARG WKDIR=/workdir
WORKDIR ${WKDIR}
RUN sudo chown ${USERNAME}:${USERNAME} ${WKDIR}

RUN echo 'export CUDA_VISIBLE_DEVICES=-1' >> ${HOME}/.bashrc \
    && echo 'export TF_CPP_MIN_LOG_LEVEL=3' >> ${HOME}/.bashrc
