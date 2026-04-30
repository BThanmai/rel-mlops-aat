FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (separate layer = faster rebuilds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY api/        ./api/
COPY src/        ./src/
COPY mlops/      ./mlops/
COPY templates/  ./templates/
COPY data/       ./data/

# Create folders that must exist at runtime
RUN mkdir -p models/archive mlruns data

# Startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 5000

CMD ["/start.sh"]
