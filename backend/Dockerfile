# ── Stage 1: dependency layer (cached separately from app code) ───────────────
FROM python:3.11-slim AS deps

WORKDIR /app

# System deps for asyncpg + torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: application ──────────────────────────────────────────────────────
FROM deps AS app

WORKDIR /app

COPY . .

RUN mkdir -p logs

EXPOSE 8000

ENTRYPOINT ["python"]
CMD ["main.py", "docker"]
