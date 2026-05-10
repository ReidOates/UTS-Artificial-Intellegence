# Gunakan image Python yang ringan
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Tentukan working directory di dalam container
WORKDIR /code

# Install system dependencies yang mungkin dibutuhkan oleh library ML
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt dan install library
COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh file proyek ke dalam container
COPY . /code/

# Buat __init__.py jika belum ada agar folder app bisa diakses sebagai package
RUN touch /code/app/__init__.py

# Ekspos port 7860 untuk Hugging Face Spaces
EXPOSE 7860

# Jalankan aplikasi menggunakan Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "app.app:app"]
