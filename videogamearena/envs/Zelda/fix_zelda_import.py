#!/usr/bin/env python3
import os
import shutil
import json

def fix_rom_import():
    """Fix the ROM import by ensuring all necessary files are properly configured"""
    
    # Base retro data directory
    retro_dir = "/home/guertlerlo/.local/lib/python3.10/site-packages/retro"
    
    # Check if retro directory exists
    if not os.path.exists(retro_dir):
        print(f"Retro directory not found: {retro_dir}")
        return False
    
    # Game-specific paths
    rom_name = "TheLegendOfZelda-ALinkToThePast-Snes"
    system = "SuperNintendoEntertainmentSystem"
    
    # Path to the ROM file
    rom_dir = os.path.join(retro_dir, "data", "stable", system)
    rom_path = os.path.join(rom_dir, f"{rom_name}.sfc")
    
    # Check if ROM file exists
    if not os.path.exists(rom_path):
        print(f"ROM file not found: {rom_path}")
        print("Trying to copy it again from current directory...")
        
        # Try to copy from current directory
        src_rom = "Legend of Zelda, The - A Link to the Past (USA).sfc"
        if os.path.exists(src_rom):
            os.makedirs(rom_dir, exist_ok=True)
            shutil.copy2(src_rom, rom_path)
            print(f"Copied ROM to: {rom_path}")
        else:
            print(f"Source ROM not found: {src_rom}")
            return False
    
    # Fix: Add an entry to the system json file (critical for game discovery)
    system_json_path = os.path.join(retro_dir, "data", "stable", f"{system}.json")
    if os.path.exists(system_json_path):
        try:
            with open(system_json_path, 'r') as f:
                system_data = json.load(f)
        except json.JSONDecodeError:
            system_data = {}
    else:
        system_data = {}
    
    # Add our ROM to the system data if it's not already there
    if rom_name not in system_data:
        system_data[rom_name] = {
            "system": system,
            "file_sha1": "3d81ee6d537e253c7e3a428c201f9f86f70554c9",  # Dummy hash
            "supported_states": ["Start"],
            "extensions": ["sfc"]
        }
        
        # Save updated system data
        os.makedirs(os.path.dirname(system_json_path), exist_ok=True)
        with open(system_json_path, 'w') as f:
            json.dump(system_data, f, indent=2)
        print(f"Updated system JSON file: {system_json_path}")
    
    # Fix: Make sure metadata directory and files exist
    meta_dir = os.path.join(retro_dir, "data", "stable", "metadata", system)
    meta_path = os.path.join(meta_dir, f"{rom_name}.json")
    
    os.makedirs(meta_dir, exist_ok=True)
    
    # Create metadata if it doesn't exist or update it
    meta_data = {
        "default_player": "Link",
        "default_state": "Start",
        "players": ["Link"],
        "states": {
            "Start": {
                "runs": [
                    {
                        "done": False,
                        "reward": 0
                    }
                ]
            }
        }
    }
    
    with open(meta_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    print(f"Created/updated metadata file: {meta_path}")
    
    # Fix: Make sure data directory and files exist
    data_dir = os.path.join(retro_dir, "data", "stable", "data", system, rom_name)
    os.makedirs(data_dir, exist_ok=True)
    
    # Create scenario.json
    scenario_path = os.path.join(data_dir, "scenario.json")
    scenario_data = {
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
    
    with open(scenario_path, 'w') as f:
        json.dump(scenario_data, f, indent=2)
    print(f"Created/updated scenario file: {scenario_path}")
    
    # Create data.json
    data_json_path = os.path.join(data_dir, "data.json")
    data_json = {
        "info": {
            "health": {
                "address": 8309248,
                "type": "|u1"
            }
        }
    }
    
    with open(data_json_path, 'w') as f:
        json.dump(data_json, f, indent=2)
    print(f"Created/updated data file: {data_json_path}")
    
    # Create/update the rom_list
    rom_list_path = os.path.join(retro_dir, "data", "stable", "rom_list.json")
    if os.path.exists(rom_list_path):
        try:
            with open(rom_list_path, 'r') as f:
                rom_list = json.load(f)
        except json.JSONDecodeError:
            rom_list = {}
    else:
        rom_list = {}
    
    # Add our ROM to the system data if it's not already there
    rom_key = f"{system}/{rom_name}"
    if rom_key not in rom_list:
        rom_list[rom_key] = {"skip": False}
        
        # Save updated rom list
        with open(rom_list_path, 'w') as f:
            json.dump(rom_list, f, indent=2)
        print(f"Updated ROM list file: {rom_list_path}")
    
    # Create a "Start.state" file (empty state file)
    state_dir = os.path.join(data_dir, "Start.state")
    if not os.path.exists(state_dir) or os.path.getsize(state_dir) == 0:
        # Create a minimal valid state file (even an empty file can work)
        with open(state_dir, 'wb') as f:
            f.write(b'\x00' * 128)  # Minimal state file with some nulls
        print(f"Created state file: {state_dir}")
    
    return True

def test_game_loading():
    """Test if the game can be loaded after fixing the import"""
    try:
        import retro
        
        # Force retro to reload its data
        if hasattr(retro.data, '_init'):
            retro.data._init()
        
        # Get available games
        games = retro.data.list_games()
        zelda_games = [g for g in games if 'zelda' in g.lower()]
        
        print("\nAvailable Zelda games:")
        if zelda_games:
            for game in zelda_games:
                print(f"  - {game}")
        else:
            print("  No Zelda games found")
        
        # Try to create the environment
        game_id = "TheLegendOfZelda-ALinkToThePast-Snes"
        print(f"\nTrying to load game: {game_id}")
        
        try:
            env = retro.make(game=game_id)
            print(f"Success! Game loaded: {game_id}")
            return True
        except Exception as e:
            print(f"Failed to load game: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Error testing game loading: {str(e)}")
        return False

if __name__ == "__main__":
    print("Fixing Zelda: A Link to the Past ROM import for retro")
    
    if fix_rom_import():
        print("\nROM import fix complete!")
        test_game_loading()
        
        print("\nIf the game still doesn't load, try the following steps:")
        print("1. Make sure the ROM file extension is '.sfc' (not .smc or other)")
        print("2. Try restarting your Python interpreter")
        print("3. Try 'python -c \"import retro; print(retro.data.list_games())\"' to see available games")
    else:
        print("\nFailed to fix ROM import")