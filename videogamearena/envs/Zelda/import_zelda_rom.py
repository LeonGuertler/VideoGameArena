#!/usr/bin/env python3
import os
import sys
import shutil
import glob

def find_retro_data_dir():
    """Find the stable-retro data directory by checking common locations"""
    
    # Get the Python site-packages directory
    import site
    site_packages = site.getsitepackages()
    
    possible_paths = []
    
    # Add common paths to check
    for sp in site_packages:
        possible_paths.append(os.path.join(sp, 'stable_retro', 'data', 'stable'))
        possible_paths.append(os.path.join(sp, 'retro', 'data', 'stable'))
    
    # Add home directory paths
    home = os.path.expanduser('~')
    possible_paths.append(os.path.join(home, '.local', 'lib', 'python3*', 'site-packages', 'stable_retro', 'data', 'stable'))
    possible_paths.append(os.path.join(home, '.local', 'lib', 'python3*', 'site-packages', 'retro', 'data', 'stable'))
    
    # Expand glob patterns
    expanded_paths = []
    for path in possible_paths:
        if '*' in path:
            expanded_paths.extend(glob.glob(path))
        else:
            expanded_paths.append(path)
    
    # Check each possible path
    for path in expanded_paths:
        if os.path.exists(path):
            print(f"Found retro data directory: {path}")
            return path
    
    # If not found, ask the user
    print("Could not automatically find the retro data directory.")
    user_path = input("Please enter the full path to your stable_retro data directory: ")
    
    if os.path.exists(user_path):
        return user_path
    
    print(f"The path {user_path} does not exist.")
    return None

def import_rom_manually():
    """Manually import a ROM file into stable-retro"""
    
    # ROM filename
    rom_filename = "Legend of Zelda, The - A Link to the Past (USA).sfc"
    
    # Check if ROM exists
    if not os.path.exists(rom_filename):
        print(f"ROM file not found: {rom_filename}")
        return False
    
    # Find the retro data directory
    data_dir = find_retro_data_dir()
    if not data_dir:
        print("Could not find retro data directory. Aborting.")
        return False
    
    # Create necessary directories
    game_dir = os.path.join(data_dir, "SuperNintendoEntertainmentSystem")
    os.makedirs(game_dir, exist_ok=True)
    
    # Copy ROM with the standardized name
    dest_filename = "TheLegendOfZelda-ALinkToThePast-Snes.sfc"
    dest_path = os.path.join(game_dir, dest_filename)
    
    print(f"Copying ROM from: {rom_filename}")
    print(f"Copying ROM to: {dest_path}")
    
    try:
        shutil.copy2(rom_filename, dest_path)
        print(f"Successfully copied ROM to: {dest_path}")
        
        # Create the metadata directories
        meta_dir = os.path.join(data_dir, "metadata", "SuperNintendoEntertainmentSystem")
        os.makedirs(meta_dir, exist_ok=True)
        
        # Create a basic metadata JSON file for the game
        meta_path = os.path.join(meta_dir, "TheLegendOfZelda-ALinkToThePast-Snes.json")
        
        with open(meta_path, 'w') as f:
            f.write("""
{
  "default_player": "Link",
  "default_state": "Start",
  "players": [
    "Link"
  ],
  "states": {
    "Start": {
      "runs": [
        {
          "done": false,
          "reward": 0
        }
      ]
    }
  }
}
""")
        
        print(f"Created metadata file at: {meta_path}")
        
        # Create a basic data directory structure
        data_subdir = os.path.join(data_dir, "data", "SuperNintendoEntertainmentSystem", "TheLegendOfZelda-ALinkToThePast-Snes")
        os.makedirs(data_subdir, exist_ok=True)
        
        # Create a basic scenario.json file
        scenario_path = os.path.join(data_subdir, "scenario.json")
        
        with open(scenario_path, 'w') as f:
            f.write("""
{
  "reward": {
    "variables": {
      "health": {
        "reward": 1.0
      }
    }
  },
  "done": {
    "variables": {
      "health": {
        "op": "equal",
        "reference": 0
      }
    }
  }
}
""")
        
        print(f"Created scenario file at: {scenario_path}")
        
        # Create a basic data.json file
        data_json_path = os.path.join(data_subdir, "data.json")
        
        with open(data_json_path, 'w') as f:
            f.write("""
{
  "info": {
    "health": {
      "address": 8309248,
      "type": "|u1"
    }
  }
}
""")
        
        print(f"Created data file at: {data_json_path}")
        
        return True
        
    except Exception as e:
        print(f"Error copying ROM file: {str(e)}")
        return False

if __name__ == "__main__":
    print("Attempting to manually import Zelda: A Link to the Past into stable-retro")
    
    if import_rom_manually():
        print("\nROM import appears to be successful!")
        print("Try running your Zelda environment with game ID: 'TheLegendOfZelda-ALinkToThePast-Snes'")
    else:
        print("\nROM import failed. Please check the error messages above.")