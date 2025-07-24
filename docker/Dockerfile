# 使用ubuntu镜像作为基础镜像，并安装Python 3.9
FROM ubuntu:22.04

# # 设置时区
# ENV TZ=Asia/Shanghai

# 更换为国内镜像源 (阿里云) 以解决网络问题
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 更新apt包列表并安装必要的工具和Python环境
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    tzdata \
    software-properties-common \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 设置时区
ENV TZ=Asia/Shanghai

# 创建非root用户
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -ms /bin/bash -g appgroup appuser

# 设置工作目录
WORKDIR /app

# 使用国内PyPI镜像源加速Python包安装
RUN python3 -m pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 安装Python依赖包（基于environment.yml，使用国内镜像源）
RUN python3 -m pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    torch==1.10.0 \
    torchvision==0.11.0 \
    torchaudio==0.10.0 \
    torch-geometric==2.0.3 \
    torch-geometric-temporal==0.56.0 \
    torch-cluster==1.5.9 \
    torch-scatter==2.0.9 \
    torch-sparse==0.6.12 \
    torch-spline-conv==1.2.1 \
    numpy==1.19.2 \
    pandas==1.1.5 \
    scikit-learn==0.24.2 \
    scipy==1.5.4 \
    matplotlib==3.3.4 \
    seaborn==0.11.2 \
    networkx==2.5.1 \
    tqdm==4.64.1 \
    pyyaml==6.0.1 \
    requests==2.27.1 \
    charset-normalizer==2.0.12 \
    colorama==0.4.5 \
    cython==3.0.12 \
    decorator==4.4.2 \
    joblib==1.1.1 \
    jinja2==3.0.3 \
    markupsafe==2.0.1 \
    pyparsing==3.1.4 \
    python-dateutil==2.9.0.post0 \
    pytz==2025.2 \
    threadpoolctl==3.1.0 \
    urllib3==1.26.20

# 复制四步算法的所有Python模块
COPY partition/ ./partition/
COPY muti_scale/ ./muti_scale/
COPY evolve_GCN/ ./evolve_GCN/
COPY feedback/ ./feedback/

# 复制预热服务和集成接口（按您的要求保持不变）
COPY evolvegcn_preload_service_safe.py .
COPY complete_sharding_go_interface.py .
COPY complete_integrated_sharding_system.py .

# 复制现有的Python文件
COPY python_config.json .

# 创建数据交换目录（仅supervisor节点需要）
RUN mkdir -p data_exchange outputs step1_outputs

# 关键步骤：将编译好的Linux程序和配置文件复制到镜像中
COPY ./Files/blockEmulator_linux_Precompile .

# 赋予可执行文件权限并更改所有者
RUN chown appuser:appgroup /app/blockEmulator_linux_Precompile && \
    chmod +x /app/blockEmulator_linux_Precompile

# 赋予/app目录给appuser写权限
RUN chown -R appuser:appgroup /app

# 切换到非root用户
USER appuser

# 定义容器启动时要执行的命令
ENTRYPOINT ["./blockEmulator_linux_Precompile"]