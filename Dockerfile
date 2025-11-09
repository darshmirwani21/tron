# Participant testing Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy participant code
COPY . .

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
CMD ["python", "agent.py"]
