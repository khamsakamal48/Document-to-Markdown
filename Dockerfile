# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:latest

# Set the working directory in the container
WORKDIR /app

# Copy Data
COPY requirements.txt .
COPY app.py .
COPY ocr_pipeline.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip
RUN apt update && apt -y upgrade && apt install -y libgl1-mesa-glx tesseract-ocr-all

# Expose port 8501 for streamlit app access from outside of docker container
EXPOSE 8501

# Run streamlit app on container start (optional: you can also use CMD)
CMD ["streamlit", "run", "app.py"]
