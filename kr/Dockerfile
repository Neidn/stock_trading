# ── Stage 1: compile Python deps (TA-Lib 0.6.x bundles its own C source) ─────
FROM python:3.13-slim AS builder

# gcc/g++ needed for TA-Lib to compile its bundled C source during pip install
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# --mount=type=cache persists pip download cache between builds.
# If requirements.txt unchanged → layer cache skips this entirely.
# If requirements.txt changed  → already-downloaded wheels reused.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.13-slim

# Copy compiled Python packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages \
                    /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app
COPY src/ src/

# Default: signal-engine (override via command: in Deployment YAML)
CMD ["python", "-m", "src.signal.main"]
