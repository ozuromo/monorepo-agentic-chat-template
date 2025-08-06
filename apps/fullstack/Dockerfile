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
RUN uv sync --frozen --all-packages --no-install-workspace

COPY . .
RUN uv sync --frozen --all-packages

RUN chmod +x apps/fullstack/src/fullstack/start.sh
CMD ["./apps/fullstack/src/fullstack/start.sh"]
