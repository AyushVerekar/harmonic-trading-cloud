FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.api.txt .
RUN pip install --no-cache-dir -r requirements.api.txt

COPY harmonic ./harmonic
COPY adaptive_trading ./adaptive_trading
COPY service ./service
COPY main.py runtime.py harmonic_lifetime_data.csv ./
COPY artifacts ./artifacts

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn service.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
