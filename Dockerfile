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

# Define environment variables with default values (optional)
ENV NEXT_PUBLIC_SUPABASE_URL="https://ztitbrotmhiybmpzdnam.supabase.co"
ENV NEXT_PUBLIC_SUPABASE_ANON_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp0aXRicm90bWhpeWJtcHpkbmFtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDA5MzY2NzgsImV4cCI6MjA1NjUxMjY3OH0.1shgHVLONnFas6S9-SDyHeP11Z-JdUBmDCM0w5ITTr4"
ENV WEBSOCKET_PORT=8000
ENV HEALTH_CHECK_PORT=8001

# Run the server with a virtual display and dynamic port
CMD xvfb-run -s "-screen 0 1400x900x24" python server.py