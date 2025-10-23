# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    TZ=Asia/Shanghai \
    RDC_DUCKDB_PATH=/app/var/data/market.duckdb \
    RDC_DATA_CACHE_DIR=/app/var/cache

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        curl \
        git \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir .[monitor] \
    && chmod +x deploy/monitoring/*.sh \
    && mkdir -p /app/var/data /app/var/cache

VOLUME ["/app/var/data", "/app/var/cache"]

EXPOSE 19555

CMD ["rdc", "--help"]
