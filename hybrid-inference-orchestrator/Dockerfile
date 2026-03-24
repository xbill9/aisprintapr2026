# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

WORKDIR $APP_HOME

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server code
COPY server.py .

# Expose the port (Cloud Run uses 8080 by default)
EXPOSE 8080

# Run the MCP server
CMD ["python", "server.py"]
