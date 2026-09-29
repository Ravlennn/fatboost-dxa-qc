FROM python:3.12.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    PYTHONPATH=/app/src DXAQC_MODEL_DIR=/app/models/release
WORKDIR /app
COPY requirements-runtime.txt ./
RUN python -m pip install --no-cache-dir --no-deps -r requirements-runtime.txt \
    && python -m pip install --no-cache-dir --no-deps --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0 \
    && python -m pip check
COPY src ./src
COPY models/release ./models/release
RUN python -m dxaqc.cli doctor
USER 65532:65532
ENTRYPOINT ["python", "-m", "dxaqc.cli"]
CMD ["--help"]
