# Use Python 3.11 slim version
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies for OpenGL, pygame, and nes_py
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libgl1-mesa-glx \
    libglu1-mesa \
    xvfb \
    python3-opengl \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy all local files, including videogamearena folder (with ROMs)
COPY . .

# Ensure the ROMs directory is present
RUN mkdir -p /app/videogamearena/roms

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Import ROMs during the build process
RUN python -m retro.import videogamearena/roms

# Expose a range of ports so AWS can assign a dynamic one
EXPOSE 8000-9000

# Run the server with a virtual display and dynamic port
CMD ["xvfb-run", "-s", "-screen 0 1400x900x24", "python", "server.py"]