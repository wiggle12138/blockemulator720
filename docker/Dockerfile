# 使用Python官方slim镜像
FROM python:3.10-slim

# 设置环境变量
ENV TZ=Asia/Shanghai \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# 先初始化APT环境
RUN apt-get update --allow-releaseinfo-change && \
    # 安装必要的包确保sources.list文件存在
    apt-get install -y --no-install-recommends apt-utils

# 更换为阿里云APT源
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装系统级依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 设置时区
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 设置工作目录
WORKDIR /app

# 设置国内pip源 (清华源)
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn && \
    pip config set global.timeout 300 && \
    pip config set global.retries 5

# 安装基础科学计算库
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    matplotlib \
    scikit-learn

# 安装数据处理和网络库
RUN pip install --no-cache-dir \
    networkx \
    tqdm \
    pyyaml \
    requests \
    joblib

# 安装日期时间相关库
RUN pip install --no-cache-dir \
    python-dateutil \
    pytz

# 安装PyTorch 2.2.2 CPU版本 (修正版)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    torchaudio==2.2.2+cpu

# 修正 PyG 安装命令
RUN pip install --no-cache-dir \
    torch-scatter==2.1.2+pt22cpu \
    torch-sparse==0.6.18+pt22cpu \
    torch-cluster==1.6.3+pt22cpu \
    torch-spline-conv==1.2.2+pt22cpu \
    torch-geometric==2.5.0 \
    torch-geometric-temporal==0.56.2 \
    -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

# 复制项目文件 - 按依赖关系从稳定到变化排序
COPY Files/blockEmulator_linux_Precompile ./blockEmulator_linux_Precompile
COPY Files/paramsConfig.json ./paramsConfig.json
COPY Files/ipTable.json ./ipTable.json

COPY partition/ ./partition/
COPY muti_scale/ ./muti_scale/
COPY evolve_GCN/ ./evolve_GCN/
COPY feedback/ ./feedback/

COPY complete_sharding_go_interface.py ./
COPY complete_integrated_sharding_system.py ./
COPY evolvegcn_preload_service.py ./
COPY evolvegcn_preload_service_safe.py ./

# # 创建非root用户并设置权限
# RUN useradd -m -d /app appuser && \
#     chown -R appuser:appuser /app && \
#     chmod +x /app/blockEmulator_linux_Precompile

# 创建非root用户
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -ms /bin/bash -g appgroup appuser

# 赋予可执行文件权限并更改所有者
RUN chown appuser:appgroup /app/blockEmulator_linux_Precompile && \
    chmod +x /app/blockEmulator_linux_Precompile

# 赋予/app目录给appuser写权限
RUN chown -R appuser:appgroup /app

# 切换到非root用户
USER appuser

# 验证Python环境
RUN python --version && \
    python -c "import torch; print('PyTorch:', torch.__version__)" && \
    python -c "import torch_geometric; print('PyG version:', torch_geometric.__version__)"