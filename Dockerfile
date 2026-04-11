FROM python:3.11-slim

LABEL maintainer="openenv-community"
LABEL org.opencontainers.image.description="License Compliance Scanner — OpenEnv AI Agent Environment"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Verify imports compile cleanly (no DB connection needed at build time)
RUN python -c "from env.environment import LicenseComplianceEnv; print('✓ Import OK')"

# Run only unit tests that don't need a live DB
RUN python -m pytest tests/ -q -k "not server" && echo "✓ All unit tests pass"

# DATABASE_URL must be injected at runtime — fail fast if missing
# Example: postgresql://user:pass@host:5432/dbname?sslmode=require
ENV DATABASE_URL=""
ENV PORT=7860
EXPOSE 7860

RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
