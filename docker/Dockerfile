# 使用轻量级的ubuntu镜像作为运行环境
FROM ubuntu:22.04

# 更换为国内镜像源 (阿里云) 以解决网络问题
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 更新apt包列表并安装必要的工具（包括Python环境）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    tzdata \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 设置时区
ENV TZ=Asia/Shanghai

# 创建非root用户
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -ms /bin/bash -g appgroup appuser

# 设置工作目录
WORKDIR /app

# 现在全部改用挂载逻辑，减少重新生成镜像的麻烦，不行，要加上执行权限，只能是复制
# 关键步骤：将编译好的Linux程序和配置文件复制到镜像中
COPY ./Files/blockEmulator_linux_Precompile .
# COPY ./Files/paramsConfig.json .
# COPY ./Files/ipTable.json .

# 复制EvolveGCN相关文件到容器（文件已复制到当前目录）
COPY evolvegcn_go_interface.py .
COPY integrated_four_step_pipeline.py .
COPY python_config.json .

# 安装Python依赖
RUN python3 -m pip install --no-cache-dir torch numpy pandas scikit-learn matplotlib seaborn tqdm

# 赋予可执行文件权限并更改所有者
RUN chown appuser:appgroup /app/blockEmulator_linux_Precompile && \
    chmod +x /app/blockEmulator_linux_Precompile

# 赋予/app目录给appuser写权限
RUN chown -R appuser:appgroup /app

# 切换到非root用户
USER appuser

# 定义容器启动时要执行的命令
ENTRYPOINT ["./blockEmulator_linux_Precompile"]