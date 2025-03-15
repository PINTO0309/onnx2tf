FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ARG BUILD_ARCH="linux/amd64"

RUN apt-get update \
    && apt-get install -y \
        nano python3-pip python3-mock libpython3-dev \
        libpython3-all-dev python-is-python3 wget curl cmake \
        software-properties-common sudo pkg-config libhdf5-dev \
    && sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pip -U \
    && pip install onnx==1.17.0 \
    && pip install onnxsim==0.4.33 \
    && pip install nvidia-pyindex \
    && pip install onnx_graphsurgeon \
    && pip install onnx2tf \
    && pip install onnx2tf \
    && pip install simple_onnx_processing_tools \
    && pip install tensorflow==2.19.0 \
    && pip install protobuf==4.25.5 \
    && pip install h5py==3.11.0 \
    && pip install psutil==5.9.5 \
    && pip install onnxruntime==1.18.1 \
    && pip install ml_dtypes==0.5.1 \
    && pip install tf-keras==2.19.0 \
    && pip install flatbuffers>=23.5.26

# Re-release flatc with some customizations of our own to address
# the lack of arithmetic precision of the quantization parameters
# https://github.com/PINTO0309/onnx2tf/issues/196
RUN if [ "${BUILD_ARCH}" = "linux/amd64" ]; then \
        wget -O flatc.tar.gz https://github.com/PINTO0309/onnx2tf/releases/download/1.16.31/flatc.tar.gz; \
    elif [ "${BUILD_ARCH}" = "linux/arm64" ]; then \
        wget -O flatc.tar.gz https://github.com/PINTO0309/onnx2tf/releases/download/1.26.6/flatc_arm64.tar.gz; \
    else \
        echo "Unsupported architecture: ${BUILD_ARCH}" && exit 1; \
    fi \
    && tar -zxvf flatc.tar.gz \
    && chmod +x flatc \
    && mv flatc /usr/bin/

ENV USERNAME=user
ARG WKDIR=/workdir

RUN echo "root:root" | chpasswd \
    && adduser --disabled-password --gecos "" "${USERNAME}" \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && echo "%${USERNAME}    ALL=(ALL)   NOPASSWD:    ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && mkdir -p ${WKDIR} \
    && chown ${USERNAME}:${USERNAME} ${WKDIR}

USER ${USERNAME}
WORKDIR ${WKDIR}

RUN echo 'export CUDA_VISIBLE_DEVICES=-1' >> ${HOME}/.bashrc \
    && echo 'export TF_CPP_MIN_LOG_LEVEL=3' >> ${HOME}/.bashrc
