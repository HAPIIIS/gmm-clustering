FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && apt-get clean

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies, including GDAL Python bindings
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run database migrations
RUN python manage.py migrate

# Start the application with Gunicorn
CMD ["gunicorn", "--config", "gunicorn-cfg.py", "core.wsgi"]
