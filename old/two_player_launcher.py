#!/usr/bin/env python3
"""
Breakout Two-Player Game Launcher
---------------------------------
This script launches both the server and two client instances for a two-player Breakout game.
"""

import subprocess
import time
import os
import sys
import signal
import atexit

# Store the processes so we can terminate them when the script exits
processes = []

def cleanup():
    """Terminate all child processes when the script exits."""
    for p in processes:
        try:
            p.terminate()
            print(f"Terminated process {p.pid}")
        except:
            pass

# Register the cleanup function to run on exit
atexit.register(cleanup)

# Handle Ctrl+C gracefully
def signal_handler(sig, frame):
    print("Ctrl+C detected, shutting down...")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def main():
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths to the server and client scripts
    server_script = os.path.join(script_dir, "server", "main.py")
    client_script = os.path.join(script_dir, "client", "main.py")
    
    # Check if the scripts exist
    if not os.path.exists(server_script):
        print(f"Error: Server script not found at {server_script}")
        return
    
    if not os.path.exists(client_script):
        print(f"Error: Client script not found at {client_script}")
        return
    
    print("Starting Breakout Two-Player Game")
    print("=================================")
    
    # Start the server
    print("Starting game server...")
    server_process = subprocess.Popen([sys.executable, server_script])
    processes.append(server_process)
    
    # Wait for the server to start up
    print("Waiting for server to initialize...")
    time.sleep(2)
    
    # Start the first client (Player 1)
    print("Starting Player 1 client...")
    client1_process = subprocess.Popen([sys.executable, client_script, "--player", "player1"])
    processes.append(client1_process)
    
    # Give a moment for the first client to initialize
    time.sleep(1)
    
    # Start the second client (Player 2)
    print("Starting Player 2 client...")
    client2_process = subprocess.Popen([sys.executable, client_script, "--player", "player2"])
    processes.append(client2_process)
    
    print("\nGame started! Players can now compete.")
    print("Close this terminal window to stop all game processes.")
    
    # Keep the script running until manually terminated
    try:
        while True:
            # Check if any process has terminated
            for p in processes[:]:
                if p.poll() is not None:
                    print(f"Process {p.pid} terminated with code {p.returncode}")
                    processes.remove(p)
            
            if not processes:
                print("All processes have terminated. Exiting.")
                break
                
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...")
        cleanup()

if __name__ == "__main__":
    main()