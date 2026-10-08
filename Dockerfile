FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca AS runtime

# The pinned uv base lags Debian security updates (2026-10-03: its gnutls and
# openssl carried CRITICAL CVEs that bookworm-security had already fixed), and
# CI's image scan fails on any CRITICAL with a fix available. Apply the
# distro's security updates at build time so a stale base cannot ship them.
# APT_REFRESH (CI passes the UTC date) changes this layer's cache key daily:
# the Artifact Registry build reuses a GitHub Actions layer cache, and without
# it a cached upgrade layer kept serving packages that bookworm-security had
# since fixed (2026-10-08: perl-base deb12u3, three CRITICALs).
ARG APT_REFRESH=unset
RUN echo "apt refresh: ${APT_REFRESH}" \
    && apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
# Ship the config files so gnosis auto-loads configs/default.yaml (the preferred
# config) from the working directory when GNOSIS_CONFIG_FILE is unset.
COPY configs ./configs

RUN uv sync --locked --no-dev --no-cache

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "gnosis.main:app", "--host", "0.0.0.0", "--port", "8080"]
