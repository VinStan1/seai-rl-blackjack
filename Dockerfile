FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --requirement requirements.txt

COPY requirements-dqn.txt .
RUN pip install --no-cache-dir --requirement requirements-dqn.txt

COPY src ./src
COPY tests ./tests

CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]