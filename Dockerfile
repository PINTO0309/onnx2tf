FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ARG BUILD_ARCH="linux/amd64"
ARG ONNX2TF_REF="main"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10 \
    PIP_PROGRESS_BAR=off

RUN apt-get update \
    && apt-get install -y \
        nano python3-pip python3-mock libpython3-dev git \
        libpython3-all-dev python-is-python3 wget curl cmake \
        software-properties-common sudo pkg-config libhdf5-dev \
    && sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    for i in 1 2 3; do \
        python -m pip install --break-system-packages --prefer-binary \
            --timeout "${PIP_DEFAULT_TIMEOUT}" --retries "${PIP_RETRIES}" \
            onnx==1.20.1 \
            onnxsim-prebuilt==0.4.39.post2 \
            onnxoptimizer==0.4.2 \
            onnxruntime==1.24.3 \
            sne4onnx \
            sng4onnx \
            tensorflow==2.19.0 \
            ai_edge_litert==2.1.2 \
            protobuf==4.25.5 \
            h5py==3.12.1 \
            psutil==5.9.5 \
            ml_dtypes==0.5.1 \
            tf-keras==2.19.0 \
            'flatbuffers==25.12.19' && break; \
        if [ "${i}" -eq 3 ]; then \
            exit 1; \
        fi; \
        sleep 10; \
    done

# Install onnx2tf from GitHub source (branch/tag/commit can be overridden by ONNX2TF_REF).
RUN set -eux; \
    for i in 1 2 3; do \
        python -m pip install --break-system-packages --no-deps \
            --timeout "${PIP_DEFAULT_TIMEOUT}" --retries "${PIP_RETRIES}" \
            "git+https://github.com/PINTO0309/onnx2tf.git@${ONNX2TF_REF}" && break; \
        if [ "${i}" -eq 3 ]; then \
            exit 1; \
        fi; \
        sleep 10; \
    done

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
