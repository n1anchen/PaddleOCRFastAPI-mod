# PaddleOCRFastAPI

一个可 Docker (Compose) 部署的, 基于 `FastAPI` 的简易版 Paddle OCR Web API.

## 版本选择

| PaddleOCR | Branch |
| :--: | :--: |
| v2.5 | [paddleocr-v2.5](https://github.com/cgcel/PaddleOCRFastAPI/tree/paddleocr-v2.5) |
| v2.7 | [paddleocr-v2.7](https://github.com/cgcel/PaddleOCRFastAPI/tree/paddleocr-v2.7) |

## 接口功能

- [x] 局域网范围内路径图片 OCR 识别
- [x] Base64 数据识别
- [x] 上传文件识别

## 部署方式

### 直接部署

1. 复制项目至部署路径

   ```shell
   git clone https://github.com/cgcel/PaddleOCRFastAPI.git
   ```

   > *master 分支为项目中支持的 PaddleOCR 的最新版本, 如需安装特定版本, 请克隆对应版本号的分支.*

2. 安装 `uv`，并按 `pyproject.toml` 与 `uv.lock` 同步依赖

   ```shell
   uv sync --frozen
   ```

   > 如果之前用过 `pip` 安装依赖，建议先删除旧的 `.venv`，避免新旧包混装。

   > 默认情况下，PaddleOCR 首次运行自动下载的模型缓存现在会写入项目目录下的 `.paddlex/`，不再落到当前用户目录，便于整体打包和内网迁移。

   > 如需完全离线部署，可将已解压的中文模型放到项目目录下的 `.paddleocr/`，默认目录结构如下：

   ```text
   .paddleocr/
   ├─ ch_PP-OCRv4_det_infer/
   ├─ ch_PP-OCRv4_rec_infer/
   └─ ch_ppocr_mobile_v2.0_cls_infer/
   ```

   > 也可以通过环境变量覆盖：`OCR_MODEL_DIR`、`OCR_TEXT_DETECTION_MODEL_DIR`、`OCR_TEXT_RECOGNITION_MODEL_DIR`、`OCR_TEXTLINE_ORIENTATION_MODEL_DIR`。

3. 运行 FastAPI

   ```shell
   uv run uvicorn main:app --host 0.0.0.0
   ```

### Docker 部署

在 `Centos 7`, `Ubuntu 20.04`, `Ubuntu 22.04`, `Windows 10`, `Windows 11` 中测试成功, 需要先安装好 `Docker`.

1. 复制项目至部署路径

   ```shell
   git clone https://github.com/cgcel/PaddleOCRFastAPI.git
   ```

   > *master 分支为项目中支持的 PaddleOCR 的最新版本, 如需安装特定版本, 请克隆对应版本号的分支.*

2. 制作 Docker 镜像

   ```shell
   cd PaddleOCRFastAPI
   # 手工下载模型，避免程序第一次运行时自动下载。实现完全离线，加快启动速度
   cd pp-ocrv4/ && sh download_det_cls_rec.sh

   # 返回Dockfile所在目录，开始build
   cd ..
   # 使用宿主机网络build
   # 可以用宿主机上的http_proxy和https_proxy
   docker build -t paddleocrfastapi:latest --network host .
   ```

   > 构建镜像时，下载到 `pp-ocrv4/` 的模型包会被自动解压到项目目录 `/app/.paddleocr/`，容器内运行时不会再依赖 `/root` 或其他用户目录。

3. 编辑 `docker-compose.yml`

    ```yaml
    version: "3"

    services:

       paddleocrfastapi:
          container_name: paddleocrfastapi # 自定义容器名
          image: paddleocrfastapi:latest # 第2步自定义的镜像名与标签
          environment:
             - TZ=Asia/Hong_Kong
             - OCR_LANGUAGE=ch
             - OCR_MODEL_DIR=/app/.paddleocr
             - PADDLE_PDX_CACHE_HOME=/app/.paddlex
          ports:
             - "8000:8000" # 自定义服务暴露端口, 8000 为 FastAPI 默认端口, 不做修改
          restart: unless-stopped
    ```

4. 生成 Docker 容器并运行

   ```shell
   docker compose up -d --build
   ```

5. Swagger 页面请访问 localhost:\<port\>/docs

### 内网离线部署建议

如果需要将当前环境迁移到内网电脑，可按下面方式准备：

1. 在一台可联网电脑上先把项目依赖与模型准备好。

2. 确认项目目录下至少包含以下内容：

   ```text
   .venv/                  # 可选：如需整目录迁移 Python 环境
   .paddlex/               # 首次运行自动下载后的 PaddleX / PaddleOCR 缓存
   .paddleocr/             # 可选：手工预置的中文 OCR 模型目录
   pp-ocrv4/               # 可选：Docker 构建前使用的模型压缩包
   pyproject.toml
   uv.lock
   docker-compose.yml
   Dockerfile
   ```

3. 若目标机器也使用 `uv` 在线下安装依赖，建议至少保留项目代码、`pyproject.toml`、`uv.lock`、`.paddlex/` 或 `.paddleocr/`；若目标机器完全不能安装依赖，则建议直接用 Docker 镜像或自行打包 Python 运行环境。

4. 迁移到内网机器后：

   - 直接运行方式：保持 `.paddlex/` 或 `.paddleocr/` 与项目同级即可；
   - Docker 方式：保持 `pp-ocrv4/` 中模型包存在，执行构建时会自动解压到 `/app/.paddleocr/`；
   - 如需自定义目录，可通过环境变量 `OCR_MODEL_DIR` 或 `PADDLE_PDX_CACHE_HOME` 指定。

5. 如需最稳妥的内网交付方式，推荐在外网机器先构建完成镜像，再通过 `docker save` / `docker load` 迁移镜像到内网。

## Change language

1. 将此仓库克隆至本地.
2. 修改环境变量 `OCR_LANGUAGE`，例如在 `docker-compose.yml` 中设置：

   ```yaml
   environment:
     - OCR_LANGUAGE=en
   ```

   修改前, 先阅读 [supported language list](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/multi_languages_en.md#5-support-languages-and-abbreviations).

3. 如使用项目内预置模型，请同时准备对应语言模型；否则程序会在项目目录下的 `.paddlex/` 中自动下载对应模型。

4. 重新创建 docker 镜像, 或直接运行 `main.py`.

## 运行截图
API 文档：`/docs`

![Swagger](https://raw.githubusercontent.com/cgcel/PaddleOCRFastAPI/dev/screenshots/Swagger.png)

## Todo

- [ ] support ppocr v4
- [ ] GPU mode
- [x] Image url recognition

## License

**PaddleOCRFastAPI** is licensed under the MIT license. Refer to [LICENSE](https://github.com/cgcel/PaddleOCRFastAPI/blob/master/LICENSE) for more information.
