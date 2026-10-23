# syntax=docker/dockerfile:1

FROM python:3.14-slim AS backend

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    PREDICTION_DEVICE=auto \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 tzdata \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv

# Serving dependencies only; training, notebooks and MLflow are not required.
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY src/backend/requirements.txt /tmp/backend-requirements.txt
RUN uv pip install --system torch --index-url "${PYTORCH_INDEX_URL}" \
    && uv pip install --system -r /tmp/backend-requirements.txt

COPY src/__init__.py ./src/__init__.py
COPY src/backend ./src/backend
COPY src/ai/__init__.py src/ai/lstm.py ./src/ai/

RUN groupadd --gid 10001 hydrology \
    && useradd --uid 10001 --gid hydrology --no-create-home hydrology \
    && mkdir -p /app/model /app/reports \
    && chown hydrology:hydrology /app/reports

USER hydrology
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]

FROM node:24-alpine AS frontend-build

WORKDIR /app
COPY src/frontend/package.json src/frontend/package-lock.json ./
RUN npm ci
COPY src/frontend/ ./
RUN npm run build

FROM nginx:stable-alpine AS frontend

COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=frontend-build /app/dist /usr/share/nginx/html
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
