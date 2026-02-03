FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the consumer script
COPY consumer.py .

# Run the consumer
CMD ["python", "consumer.py"]
