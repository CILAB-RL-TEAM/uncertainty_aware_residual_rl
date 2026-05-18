# =============================================================================
# uncertainty_aware_residual_rl
# Base: CUDA 12.1 devel (JAX cuda12-plugin, PyTorch 2.4.0+cu121 호환)
# =============================================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# ── 빌드 타임 환경 변수 ────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    CONDA_DIR=/root/miniconda3 \
    MUJOCO_DIR=/root/.mujoco \
    MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210 \
    DPPO_DIR=/root/dppo \
    PROJECT_DIR=/workspace \
    DPPO_LOG_DIR=/tmp/dppo_logs

# ── 런타임 환경 변수 (학습 스크립트와 동일) ────────────────────────────────
ENV PYOPENGL_PLATFORM=egl \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all

# =============================================================================
# 1. 시스템 패키지
# =============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 빌드 도구
    build-essential gcc g++ cmake git wget curl ca-certificates unzip \
    # OpenGL / EGL (MuJoCo 오프스크린 렌더링)
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglew-dev \
    libglfw3 libglfw3-dev \
    libosmesa6-dev \
    # X11 헤더 (headless 환경 호환)
    libxrender1 libxext6 libx11-6 \
    libxi-dev libxrandr-dev libxxf86vm-dev \
    # mujoco-py C 빌드 의존성
    patchelf \
    libxml2-dev libxslt1-dev \
    # 미디어 처리 (moviepy 사용)
    ffmpeg \
    # Python 빌드
    python3.10 python3-pip python3.10-dev \
    libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# 2. Miniforge (Python 3.10 환경 관리)
# Miniconda 대신 Miniforge 사용: conda-forge 채널 기본 사용, Anaconda TOS 없음
# =============================================================================
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
         -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && ${CONDA_DIR}/bin/conda clean -afy

ENV PATH="${CONDA_DIR}/bin:${PATH}"

# uarl conda 환경 (Python 3.10) 생성
# libstdcxx-ng: mujoco-py EGL 초기화 시 발생하는 GLIBCXX 버전 오류 방지
# (docs/sim_quick_start.md 트러블슈팅 항목 참고)
# Miniforge는 conda-forge가 기본 채널이므로 -c conda-forge 불필요
RUN conda create -n uarl python=3.10 -y \
    && conda run -n uarl \
       conda install libstdcxx-ng -y \
    && conda clean -afy

# 이후 pip / python 명령이 uarl 환경을 사용하도록 PATH 우선순위 설정
ENV PATH="${CONDA_DIR}/envs/uarl/bin:${PATH}"

# =============================================================================
# 3. MuJoCo 2.1 바이너리 (mujoco-py==2.1.2.14 빌드에 필수)
# =============================================================================
RUN mkdir -p ${MUJOCO_DIR} \
    && wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
             -O /tmp/mujoco210.tar.gz \
    && tar -xf /tmp/mujoco210.tar.gz -C ${MUJOCO_DIR} \
    && rm /tmp/mujoco210.tar.gz

ENV LD_LIBRARY_PATH="${MUJOCO_DIR}/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}"

# =============================================================================
# 4. mujoco-py 선행 설치
# requirements.txt 설치 전에 먼저 빌드해 두어야 d4rl 등 의존 패키지가 정상 설치됨
# =============================================================================
RUN pip install mujoco-py==2.1.2.14

# =============================================================================
# 5. PyTorch 2.4.0 (CUDA 12.1 빌드)
# requirements.txt 설치 전에 CUDA 빌드 버전을 명시적으로 설치
# (PyTorch 공식 인덱스 미사용 시 CPU 버전이 설치될 수 있음)
# =============================================================================
RUN pip install \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# =============================================================================
# 6. DPPO (Diffusion Policy Policy Optimization)
# README 설치 순서 2번: dppo를 프로젝트 외부에 clone 후 pip install
# =============================================================================
RUN git clone https://github.com/irom-princeton/dppo.git ${DPPO_DIR} \
    && pip install -e ${DPPO_DIR} \
    && pip install -e "${DPPO_DIR}[robomimic]"

ENV PYTHONPATH="${DPPO_DIR}:${PYTHONPATH}"

# =============================================================================
# 7. Python 의존성 (requirements.txt)
# Docker 레이어 캐시 활용: requirements.txt만 먼저 복사 후 설치
# =============================================================================
COPY requirements.txt ${PROJECT_DIR}/requirements.txt

RUN pip install -r ${PROJECT_DIR}/requirements.txt

# =============================================================================
# 8. 프로젝트 코드 복사 및 serl_launcher 설치
# =============================================================================
COPY . ${PROJECT_DIR}
WORKDIR ${PROJECT_DIR}

RUN pip install -e serl_launcher/

# =============================================================================
# 9. 로그 / 체크포인트 디렉토리 생성
# =============================================================================
RUN mkdir -p /root/test_checkpoints/logs \
    && mkdir -p ${DPPO_LOG_DIR} \
    && ln -s ${PROJECT_DIR} /root/uncertainty_aware_residual_rl

# =============================================================================
# 10. 대화형 셸 진입 시 환경 자동 활성화
# =============================================================================
RUN echo "source ${CONDA_DIR}/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate uarl" >> ~/.bashrc \
    && echo "export PYTHONPATH=${DPPO_DIR}:\$PYTHONPATH" >> ~/.bashrc \
    && echo "export LD_LIBRARY_PATH=${MUJOCO_DIR}/mujoco210/bin:/usr/lib/nvidia:\$LD_LIBRARY_PATH" >> ~/.bashrc \
    && echo "export DPPO_LOG_DIR=${DPPO_LOG_DIR}" >> ~/.bashrc

# =============================================================================
# 기본 작업 디렉토리: 학습 스크립트 위치
# =============================================================================
WORKDIR ${PROJECT_DIR}/examples/residual_sim

CMD ["/bin/bash"]
