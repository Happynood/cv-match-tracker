FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libgl1 \
      git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN uv sync --no-dev

ENTRYPOINT ["uv", "run", "matchtracker"]
CMD ["--help"]
