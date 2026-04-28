# PaddleOCRFastAPI

![GitHub](https://img.shields.io/github/license/cgcel/PaddleOCRFastAPI)

[中文](./README_CN.md)

A simple way to deploy `PaddleOCR` based on `FastAPI`.

## Support Version

| PaddleOCR | Branch |
| :--: | :--: |
| v2.5 | [paddleocr-v2.5](https://github.com/cgcel/PaddleOCRFastAPI/tree/paddleocr-v2.5) |
| v2.7 | [paddleocr-v2.7](https://github.com/cgcel/PaddleOCRFastAPI/tree/paddleocr-v2.7) |

## Features

- [x] Local path image recognition
- [x] Base64 data recognition
- [x] Upload file recognition

## Deployment Methods

### Deploy Directly

1. Copy the project to the deployment path

   ```shell
   git clone https://github.com/cgcel/PaddleOCRFastAPI.git
   ```

   > *The master branch is the most recent version of PaddleOCR supported by the project. To install a specific version, clone the branch with the corresponding version number.*

2. Install `uv` and sync dependencies from `pyproject.toml` and `uv.lock`

   ```shell
   uv sync --frozen
   ```

   > If you previously installed dependencies with `pip`, remove the old `.venv` first to avoid mixed package versions.

   > By default, models downloaded on first run are now cached under the project directory in `.paddlex/` instead of the current user's home directory, which makes packaging and offline migration easier.

   > For fully offline deployment, place the extracted Chinese OCR models under `.paddleocr/` in the project root using this layout:

   ```text
   .paddleocr/
   ├─ ch_PP-OCRv4_det_infer/
   ├─ ch_PP-OCRv4_rec_infer/
   └─ ch_ppocr_mobile_v2.0_cls_infer/
   ```

   > You can also override these paths with `OCR_MODEL_DIR`, `OCR_TEXT_DETECTION_MODEL_DIR`, `OCR_TEXT_RECOGNITION_MODEL_DIR`, and `OCR_TEXTLINE_ORIENTATION_MODEL_DIR`.

3. Run FastAPI

   ```shell
   uv run uvicorn main:app --host 0.0.0.0
   ```

### Docker Deployment

Test completed in `Centos 7`, `Ubuntu 20.04`, `Ubuntu 22.04`, `Windows 10`, `Windows 11`, requires `Docker` to be installed.

1. Copy the project to the deployment path

   ```shell
   git clone https://github.com/cgcel/PaddleOCRFastAPI.git
   ```

   > *The master branch is the most recent version of PaddleOCR supported by the project. To install a specific version, clone the branch with the corresponding version number.*

2. Building a Docker Image

   ```shell
   cd PaddleOCRFastAPI
   # 手工下载模型，避免程序第一次运行时自动下载，实现完全离线，加快启动速度
   cd pp-ocrv4/ && sh download_det_cls_rec.sh
   
   # 返回Dockfile所在目录，开始build
   cd ..
   # 使用宿主机网络
   # 可直接使用宿主机上的代理设置，例如在build时，用宿主机上的代理
   # docker build -t paddleocrfastapi:latest --network host --build-arg HTTP_PROXY=http://127.0.0.1:8888 --build-arg HTTPS_PROXY=http://127.0.0.1:8888 .
   docker build -t paddleocrfastapi:latest --network host .
   ```

   > During image build, the model archives downloaded into `pp-ocrv4/` are extracted into `/app/.paddleocr/`, so the container no longer depends on `/root` or any per-user model directory.

3. Edit `docker-compose.yml`

    ```yaml
    version: "3"

    services:

       paddleocrfastapi:
          container_name: paddleocrfastapi # Custom Container Name
          image: paddleocrfastapi:latest # Customized Image Name & Label in Step 2
          environment:
             - TZ=Asia/Hong_Kong
             - OCR_LANGUAGE=ch # support 80 languages. refer to https://github.com/Mushroomcat9998/PaddleOCR/blob/main/doc/doc_en/multi_languages_en.md#language_abbreviations
             - OCR_MODEL_DIR=/app/.paddleocr
             - PADDLE_PDX_CACHE_HOME=/app/.paddlex
          ports:
             - "8000:8000" # Customize the service exposure port, 8000 is the default FastAPI port, do not modify
          restart: unless-stopped
    ```

4. Create the Docker container and run

   ```shell
   docker compose up -d --build
   ```

5. Swagger Page at `localhost:<port>/docs`

### Offline deployment notes

If you plan to move this service to an isolated intranet machine, prepare it like this:

1. On a machine with Internet access, install dependencies and download or preload the OCR models first.

2. Make sure the project directory contains at least the following items:

   ```text
   .venv/                  # optional: if you want to move the Python environment as-is
   .paddlex/               # PaddleX / PaddleOCR cache downloaded on first run
   .paddleocr/             # optional: preloaded extracted Chinese OCR models
   pp-ocrv4/               # optional: model archives used before Docker build
   pyproject.toml
   uv.lock
   docker-compose.yml
   Dockerfile
   ```

3. If the target machine can still install dependencies offline from your prepared environment, keep the project code together with `pyproject.toml`, `uv.lock`, and either `.paddlex/` or `.paddleocr/`. If it cannot install dependencies at all, prefer shipping a Docker image or a fully packed Python runtime.

4. After copying the project to the offline machine:

   - for direct runs, keep `.paddlex/` or `.paddleocr/` under the project root;
   - for Docker builds, keep the model archives in `pp-ocrv4/`, and the build will extract them into `/app/.paddleocr/` automatically;
   - if you need a different location, override it with `OCR_MODEL_DIR` or `PADDLE_PDX_CACHE_HOME`.

5. For the most predictable intranet delivery, build the Docker image on an Internet-connected machine first, then move it with `docker save` / `docker load`.

## Change language

1. Clone this repo to localhost.
2. Update the `OCR_LANGUAGE` environment variable, for example in `docker-compose.yml`:

   ```yaml
   environment:
     - OCR_LANGUAGE=en
   ```

   Before changing it, read the [supported language list](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/multi_languages_en.md#5-support-languages-and-abbreviations).

3. If you preload models under the project directory, make sure the preloaded model matches the selected language; otherwise the application will auto-download the required model into `.paddlex/` under the project root.

4. Rebuild the Docker image, or run `main.py` directly.

## Screenshots
API Docs: `/docs`

![Swagger](https://raw.githubusercontent.com/cgcel/PaddleOCRFastAPI/dev/screenshots/Swagger.png)

## Todo

- [x] support ppocr v4
- [ ] GPU mode
- [x] Image url recognition

## License

**PaddleOCRFastAPI** is licensed under the MIT license. Refer to [LICENSE](https://github.com/cgcel/PaddleOCRFastAPI/blob/master/LICENSE) for more information.
