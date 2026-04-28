# 使用 Python 3.11 slim 基础镜像
FROM python:3.11-slim-bullseye

# 暴露端口
EXPOSE 8000

# 设置工作目录
WORKDIR /app

ENV PADDLE_PDX_CACHE_HOME=/app/.paddlex

# 复制 uv 依赖文件
COPY pyproject.toml uv.lock /app/

# 换源并安装系统依赖
RUN sed -i "s@http://deb.debian.org@http://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 换源并安装 uv，然后根据锁文件安装 Python 依赖
RUN python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir --upgrade pip uv && \
    uv sync --frozen --no-dev --no-install-project

# 复制项目文件
COPY . /app

# 创建项目内模型目录并解压模型文件
RUN mkdir -p /app/.paddleocr && \
    tar xf /app/pp-ocrv4/ch_ppocr_mobile_v2.0_cls_infer.tar -C /app/.paddleocr 2>/dev/null && \
    tar xf /app/pp-ocrv4/ch_PP-OCRv4_det_infer.tar -C /app/.paddleocr && \
    tar xf /app/pp-ocrv4/ch_PP-OCRv4_rec_infer.tar -C /app/.paddleocr && \
    rm -rf /app/pp-ocrv4/*.tar

# 启动命令
CMD ["uv", "run", "--no-sync", "uvicorn", "main:app", "--host", "0.0.0.0", "--workers", "2", "--log-config", "./log_conf.yaml"]
