# Use Python 3.11 slim version
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies for OpenGL, pygame, nes_py, and health checks
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libgl1-mesa-glx \
    libglu1-mesa \
    xvfb \
    python3-opengl \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy all local files, including videogamearena folder (with ROMs)
COPY . .

# Ensure the ROMs directory is present
RUN mkdir -p /app/videogamearena/roms

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Import ROMs during the build process
RUN python -m retro.import videogamearena/roms

# Expose ports for WebSocket and health check
EXPOSE 8000
EXPOSE 8001

# Add health check to verify the /health endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the server with a virtual display and dynamic port
CMD xvfb-run -s "-screen 0 1400x900x24" python server.py