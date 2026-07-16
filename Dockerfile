FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

COPY . .

# Create runtime directories
RUN mkdir -p logs data

ENV PYTHONUNBUFFERED=1
ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Default: classic strategy bot
# Override with: docker run ... ig-bot python runners/run_ai_autonomous.py
CMD ["python", "runners/run_live.py"]
