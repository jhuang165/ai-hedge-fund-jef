# The reporter: this repo installed into a slim Python image.
# Build context is the repository root (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
# Dependencies first, against a stub package, so a code change does not
# re-download every wheel: only the final --no-deps reinstall reruns.
COPY pyproject.toml README.md ./
RUN mkdir -p hedge_fund && touch hedge_fund/__init__.py && pip install . && rm -rf hedge_fund
COPY hedge_fund ./hedge_fund
RUN pip install --no-deps . && rm -rf /root/.cache

RUN useradd --uid 1000 --create-home reporter \
    && mkdir -p /data && chown reporter:reporter /data
USER reporter
ENV HOME=/data
VOLUME ["/data"]

EXPOSE 8788
HEALTHCHECK --interval=60s --timeout=5s --start-period=20s \
    CMD curl -fsS http://127.0.0.1:8788/healthz >/dev/null || exit 1

CMD ["aihf", "reporter", "--host", "0.0.0.0", "--port", "8788"]
