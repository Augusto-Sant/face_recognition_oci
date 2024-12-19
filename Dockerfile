# Use a Python base image
FROM python:3.8

# Set the working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt to install dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY /src .

EXPOSE 8000

# Default command
# CMD ["ls", "-a"]
CMD ["python", "main.py"]
