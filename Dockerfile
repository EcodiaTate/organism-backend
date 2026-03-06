# syntax=docker/dockerfile:1.5

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=0 \
    VIRTUAL_ENV=/opt/venv \
    ELAN_HOME="/opt/elan" \
    PATH="/opt/venv/bin:/opt/elan/bin:${PATH}"

WORKDIR /app

# Build deps (only in builder)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# ---- Lean toolchain (cached layer unless version changes) ----
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- \
    --default-toolchain leanprover/lean4:v4.14.0 \
    --no-modify-path \
    -y \
 && lean --version \
 && lake --version

# ---- Create venv ----
RUN python -m venv "$VIRTUAL_ENV"

# ---- 1) Copy ONLY dependency descriptors ----
COPY pyproject.toml README.md ./

# ---- 2) Install deps into venv (layer cached + pip cache mounted) ----
# Stub packages to allow "pip install ." without copying your whole repo yet
RUN for pkg in api clients core database infrastructure interfaces primitives prompts systems telemetry utils; do \
    mkdir -p $pkg && touch $pkg/__init__.py; \
done

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip install ".[dev]"

# ---- 3) Copy actual source AFTER deps are installed ----
COPY api/ api/
COPY clients/ clients/
COPY core/ core/
COPY database/ database/
COPY infrastructure/ infrastructure/
COPY interfaces/ interfaces/
COPY primitives/ primitives/
COPY prompts/ prompts/
COPY systems/ systems/
COPY telemetry/ telemetry/
COPY utils/ utils/
COPY main.py simula_worker.py skia_worker.py config.py ./
COPY config/ config/

# Reinstall project code without re-resolving deps
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps --force-reinstall .

# ---- 4) Pre-download embedding model (optional, but you asked for it) ----
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"


# ======================================================================
# Dev stage: editable install so /app bind-mount is live source
# ======================================================================

FROM python:3.12-slim AS dev

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    ELAN_HOME="/opt/elan" \
    PATH="/opt/venv/bin:/opt/elan/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copy venv + lean toolchain from builder (deps already installed)
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/elan /opt/elan
COPY --from=builder /root/.cache /root/.cache

# Copy pyproject so pip knows the package metadata
COPY pyproject.toml README.md ./

# Editable install: registers /app as the source tree.
# The bind-mount in docker-compose.dev.yaml overlays /app at
# runtime, so any file Simula writes on the host is immediately visible
# to the running process (the NeuroplasticityBus handles hot-reload via Redis pub/sub).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -e .

EXPOSE 8000 8001 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop"]

# ======================================================================

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    ELAN_HOME="/opt/elan" \
    PATH="/opt/venv/bin:/opt/elan/bin:${PATH}"

WORKDIR /app

# Runtime OS deps (keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copy venv + lean toolchain + app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/elan /opt/elan
COPY --from=builder /app /app
COPY --from=builder /root/.cache /root/.cache

# Non-root user
RUN groupadd -r ecodiaos && useradd -r -g ecodiaos -m ecodiaos \
 && cp -r /root/.cache /home/ecodiaos/.cache \
 && chown -R ecodiaos:ecodiaos /home/ecodiaos/.cache \
 && chmod -R a+rX /opt/elan

USER ecodiaos

EXPOSE 8000 8001 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop"]
