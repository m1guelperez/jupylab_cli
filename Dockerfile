# Pull the Python base image
FROM python:3.10-slim-bullseye

# Set environment variables
ENV HOME=/root
ENV APP_HOME=$HOME/notebook-labeling

# Set work directory
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && \
    apt-get -y install git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the current directory contents into the container
COPY . $APP_HOME

# Install the application
RUN pip install -e .

# Keep the container alive
CMD ["bash"]
