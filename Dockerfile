FROM python:3.11-slim

WORKDIR /app

# Install system deps + uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"

# Copy dependency files first (Docker layer cache)
COPY requirements.txt pyproject.toml uv.lock ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt || true
RUN pip install --no-cache-dir openenv-core || true

# Copy rest of project
COPY . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

CMD ["python", "-u", "main.py", "--mode", "server"]
