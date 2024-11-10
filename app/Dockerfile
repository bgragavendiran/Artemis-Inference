FROM python:3.9-slim
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn firebase-admin requests torch pillow transformers timm python-multipart

# Expose the port on which FastAPI will run
EXPOSE 8080

# Run the FastAPI app with uvicorn on port 8080
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
