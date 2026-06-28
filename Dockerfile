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
        libgl1 libglib2.0-0 libgomp1 \
    && sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    for i in 1 2 3; do \
        python -m pip install --break-system-packages --prefer-binary \
            --timeout "${PIP_DEFAULT_TIMEOUT}" --retries "${PIP_RETRIES}" \
            numpy==2.2.6 \
            onnx==1.20.1 \
            onnxruntime==1.26.0 \
            opencv-python==4.13.0.92 \
            onnxsim==0.6.5 \
            onnxoptimizer==0.4.2 \
            onnxscript==0.6.2 \
            ai-edge-litert==2.1.2 \
            sne4onnx==2.0.1 \
            sng4onnx==2.0.1 \
            psutil==5.9.5 \
            protobuf==7.35.1 \
            h5py==3.14.0 \
            ml_dtypes==0.5.4 \
            setuptools==81.0.0 \
            tensorflow==2.21.0 \
            tf-keras==2.21.0 \
            keras==3.15.0 \
            tqdm==4.67.1 \
            pytest==9.0.2 \
            'flatbuffers==25.12.19' && break; \
        if [ "${i}" -eq 3 ]; then \
            exit 1; \
        fi; \
        sleep 10; \
    done

RUN set -eux; \
    for i in 1 2 3; do \
        python -m pip install --break-system-packages --prefer-binary \
            --index-url https://download.pytorch.org/whl/cpu \
            --timeout "${PIP_DEFAULT_TIMEOUT}" --retries "${PIP_RETRIES}" \
            torch==2.11.0 && break; \
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
