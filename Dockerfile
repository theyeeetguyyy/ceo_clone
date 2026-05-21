# Use the official Python image
FROM python:3.12-slim

# Create a non-root user for security (Hugging Face Spaces best practice)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy the requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .

# Expose port 7860 (default for Hugging Face Spaces)
EXPOSE 7860

# Run the FastAPI application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
