# this Dockerfile should be built from the repo root
FROM python:3.13.1-slim-bookworm

ENV PYTHONUNBUFFERED=True
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

COPY pyproject.toml .
COPY uv.lock .
RUN uv sync --frozen --no-install-workspace --package=backend

COPY . .
RUN uv sync --frozen --package=backend

CMD ["/workspace/.venv/bin/python", "/workspace/apps/backend/src/backend/run_api.py"]
