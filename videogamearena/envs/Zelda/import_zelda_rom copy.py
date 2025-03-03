#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil

def import_rom_manually():
    """
    Manual method to import a ROM into stable-retro
    """
    # Path to your ROM file - adjust as needed
    rom_path = "Legend of Zelda, The - A Link to the Past (USA).sfc"
    
    if not os.path.exists(rom_path):
        print(f"ROM file not found: {rom_path}")
        print("Make sure the ROM file is in the current directory or provide full path")
        return False
    
    # Get the stable-retro data directory
    try:
        import retro
        
        # Get the system directory
        system_dir = os.path.join(retro.data.get_system_directory(), "stable")
        game_dir = os.path.join(system_dir, "SuperNintendoEntertainmentSystem")
        
        # Create game directory if it doesn't exist
        os.makedirs(game_dir, exist_ok=True)
        
        # The destination filename should be standardized
        dest_filename = "TheLegendOfZelda-ALinkToThePast-Snes.sfc"
        dest_path = os.path.join(game_dir, dest_filename)
        
        # Copy the ROM file
        shutil.copy2(rom_path, dest_path)
        print(f"Successfully copied ROM to: {dest_path}")
        
        # Create the metadata directory
        meta_dir = os.path.join(system_dir, "metadata", "SuperNintendoEntertainmentSystem")
        os.makedirs(meta_dir, exist_ok=True)
        
        # Create a simple metadata JSON file
        meta_file = os.path.join(meta_dir, "TheLegendOfZelda-ALinkToThePast-Snes.json")
        with open(meta_file, 'w') as f:
            f.write('{\n  "default_state": "Start",\n  "states": {\n    "Start": {\n      "runs": [\n        {\n          "done": false,\n          "reward": 0\n        }\n      ]\n    }\n  }\n}')
        
        print(f"Created metadata file at: {meta_file}")
        
        # Create the data directory
        data_dir = os.path.join(system_dir, "data", "SuperNintendoEntertainmentSystem", "TheLegendOfZelda-ALinkToThePast-Snes")
        os.makedirs(data_dir, exist_ok=True)
        
        # Create a simple scenario JSON file
        scenario_file = os.path.join(data_dir, "scenario.json")
        with open(scenario_file, 'w') as f:
            f.write('{\n  "done": {\n    "variables": {\n      "lives": {\n        "op": "equal",\n        "reference": 0\n      }\n    }\n  },\n  "reward": {\n    "variables": {\n      "score": {\n        "reward": 1\n      }\n    }\n  }\n}')
        
        print(f"Created scenario file at: {scenario_file}")
        
        return True
    
    except Exception as e:
        print(f"Error importing ROM: {str(e)}")
        return False

def import_using_module_call():
    """
    Try to import using retro.import_rom_py
    """
    try:
        rom_path = "Legend of Zelda, The - A Link to the Past (USA).sfc"
        
        if not os.path.exists(rom_path):
            print(f"ROM file not found: {rom_path}")
            return False
        
        # Try to import using the module
        cmd = [sys.executable, "-c", 
              "import retro.import_rom_py as irp; irp.main(['%s'])" % rom_path]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Successfully imported ROM using retro.import_rom_py")
            print(result.stdout)
            return True
        else:
            print("Failed to import ROM using retro.import_rom_py")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error using import_rom_py: {str(e)}")
        return False

def try_data_merge_rom():
    """
    Try to use the data.merge_rom function if available
    """
    try:
        rom_path = "Legend of Zelda, The - A Link to the Past (USA).sfc"
        
        if not os.path.exists(rom_path):
            print(f"ROM file not found: {rom_path}")
            return False
        
        import retro as retro
        
        if hasattr(retro.data, 'merge_rom'):
            retro.data.merge_rom(rom_path)
            print("Successfully imported ROM using retro.data.merge_rom")
            return True
        else:
            print("retro.data.merge_rom is not available")
            return False
            
    except Exception as e:
        print(f"Error using merge_rom: {str(e)}")
        return False

if __name__ == "__main__":
    print("Attempting to import Zelda: A Link to the Past into stable-retro")
    
    # Try all methods in sequence
    if try_data_merge_rom():
        print("Import successful using retro.data.merge_rom")
    elif import_using_module_call():
        print("Import successful using module call")
    elif import_rom_manually():
        print("Import successful using manual method")
    else:
        print("All import methods failed")
        
    # Try to list available games after import
    try:
        import retro as retro
        games = retro.data.list_games()
        
        zelda_games = [game for game in games if 'zelda' in game.lower()]
        if zelda_games:
            print("\nAvailable Zelda games after import:")
            for game in zelda_games:
                print(f"  - {game}")
        else:
            print("\nNo Zelda games found after import. The import may have failed.")
    except Exception as e:
        print(f"Error listing games: {str(e)}")