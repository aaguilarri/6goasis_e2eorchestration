# Use official Python 3.10 slim image (or full if you need extra tools)
FROM python:3.10-slim

# (Optional) Set env vars to avoid Python buffering issues
ENV PYTHONUNBUFFERED=1

# Install any system dependencies you might need — e.g. if using librdkafka backend
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        # for confluent-kafka you may need librdkafka headers:
        #librdkafka-dev \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy your requirements file if you have one
#OPY requirements.txt .

# Install Python dependencies (kafka client, etc.)
RUN pip install --upgrade pip \
    && pip install kafka-python

WORKDIR home
# Copy your application code
ADD kafka_test.py .

# (Optional) Expose ports if your client listens, etc.
# EXPOSE 8080  (or whichever ports your app uses)

# Entrypoint / command
CMD ["python kafka_test.py"]
