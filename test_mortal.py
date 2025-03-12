import os
import time
import json
import gzip
import cv2
import base64
import numpy as np
from videogamearena.envs.MortalKombatII.env import MortalKombatIIEnv
from videogamearena.agents.basic_agents import OpenRouterMKAgent, Agent 

# Create output folder
FOLDER = "mk2_games"
os.makedirs(FOLDER, exist_ok=True)

# Special system prompts tailored for different fighter styles
AGGRESSIVE_PROMPT = """
You are an aggressive professional Mortal Kombat II player. Your fighting style is offensive and relentless.
Your goal is to constantly pressure your opponent and deliver maximum damage.

Available buttons:
- Movement: [u] (up), [d] (down), [l] (left), [r] (right)
- Attack: [a] (punch), [b] (kick), [c] (block)
- No action: [n]

Special moves you should use frequently:
- Ice Blast: [d][r][a]
- Flying Kick: [r][r][b]
- Teleport & Punch: [d][u][a]

Your strategy:
1. Be aggressive - prioritize attacking over blocking
2. Use special moves as often as possible
3. Move forward and maintain pressure
4. Only block when absolutely necessary

Respond ONLY with button commands in brackets like [d][r][a] with no explanation.
"""

DEFENSIVE_PROMPT = """
You are a defensive professional Mortal Kombat II player. Your fighting style focuses on counter-attacks and blocking.
Your goal is to punish your opponent's mistakes while minimizing damage to yourself.

Available buttons:
- Movement: [u] (up), [d] (down), [l] (left), [r] (right)
- Attack: [a] (punch), [b] (kick), [c] (block)
- No action: [n]

Special moves you should use:
- Ground Freeze: [d][l][a]
- Block: [c]
- Teleport: [d][u]
- Sweep: [d][b]

Your strategy:
1. Use block [c] frequently to defend against attacks
2. Keep some distance when possible
3. Use teleport to escape corner pressure
4. Wait for openings, then counter-attack with special moves

Respond ONLY with button commands in brackets like [d][l][a] with no explanation.
"""

BALANCED_PROMPT = """
You are a balanced professional Mortal Kombat II player. Your fighting style uses a mix of offense and defense.
Your goal is to adapt to your opponent's strategy and exploit weaknesses.

Available buttons:
- Movement: [u] (up), [d] (down), [l] (left), [r] (right)
- Attack: [a] (punch), [b] (kick), [c] (block)
- No action: [n]

Special moves you should use:
- Ice Blast: [d][r][a]
- Spear: [l][l][a]
- Flying Kick: [r][r][b]
- Block: [c]
- Teleport: [d][u]

Your strategy:
1. Mix up offense and defense
2. Use special moves to punish opponent's mistakes
3. Block when opponent is attacking
4. Vary your approach - don't be predictable

Respond ONLY with button commands in brackets like [d][r][a] with no explanation.
"""

class HumanMKAgent(Agent):
    """Human agent class for Mortal Kombat II using console input."""
    
    def __init__(self, player_num=1):
        """
        Initialize the human agent for Mortal Kombat II.
        
        Args:
            player_num: 1 for Player 1 (lowercase), 2 for Player 2 (uppercase)
        """
        super().__init__()
        self.player_num = player_num
        self.case_func = str.lower if player_num == 1 else str.upper
        
        print(f"\n===== Player {player_num} Controls (Human) =====")
        print(f"Enter inputs as: [d][r][a] for Down+Right+Punch")
        print(f"Special moves:")
        print(f"  Ice Blast: [d][r][a]")
        print(f"  Spear: [l][l][a]")
        print(f"  Flying Kick: [r][r][b]")
        print(f"  Teleport: [d][u]")
        print(f"  Press Enter for no action ([n])")
        print("=" * 30)
    
    def __call__(self, observation):
        """Get human input for the action."""
        try:
            action_input = input(f"Player {self.player_num} action: ")
            
            # Default to no-op if empty input
            if not action_input:
                return self.case_func("[n]")
            
            # Ensure proper formatting with brackets if the user didn't provide them
            if '[' not in action_input:
                # Split into individual characters and add brackets
                action_input = ''.join([f"[{c}]" for c in action_input.strip()])
            
            # Convert to proper case based on player number
            return self.case_func(action_input)
            
        except Exception as e:
            print(f"Input error: {e}")
            return self.case_func("[n]")  # Default to no-op on error


def run_mk2_match(p1_type="openrouter", p2_type="openrouter", 
                  p1_model="anthropic/claude-3.5-sonnet", 
                  p2_model="google/gemini-1.5-pro",
                  p1_style="balanced", p2_style="aggressive",
                  speed_mode="slow", max_frames=2000,
                  save_data=True):
    """
    Run a Mortal Kombat II match with the specified agent types.
    
    Args:
        p1_type: "openrouter" or "human" for Player 1
        p2_type: "openrouter" or "human" for Player 2
        p1_model: OpenRouter model for Player 1
        p2_model: OpenRouter model for Player 2
        p1_style: Fighting style for P1 ("balanced", "aggressive", "defensive")
        p2_style: Fighting style for P2 ("balanced", "aggressive", "defensive")
        speed_mode: Game speed ("human", "slow", "super-slow")
        max_frames: Maximum number of frames to run
        save_data: Whether to save game data
    """
    # Select system prompts based on fighting styles
    p1_prompt = None
    if p1_style == "aggressive":
        p1_prompt = AGGRESSIVE_PROMPT
    elif p1_style == "defensive":
        p1_prompt = DEFENSIVE_PROMPT
    else:  # balanced
        p1_prompt = BALANCED_PROMPT
        
    p2_prompt = None
    if p2_style == "aggressive":
        p2_prompt = AGGRESSIVE_PROMPT
    elif p2_style == "defensive":
        p2_prompt = DEFENSIVE_PROMPT
    else:  # balanced
        p2_prompt = BALANCED_PROMPT
    
    try:
        # Create Player 1 agent
        if p1_type == "human":
            p1_agent = HumanMKAgent(player_num=1)
            p1_name = "Human"
        else:
            p1_agent = OpenRouterMKAgent(
                model_name=p1_model,
                player_num=1,
                system_prompt=p1_prompt+f"You are controlling the player on the left hand side.",
                verbose=True
            )
            p1_name = f"{p1_model.split('/')[-1]}-{p1_style}"
        
        # Create Player 2 agent
        if p2_type == "human":
            p2_agent = HumanMKAgent(player_num=2)
            p2_name = "Human"
        else:
            p2_agent = OpenRouterMKAgent(
                model_name=p2_model,
                player_num=2,
                system_prompt=p2_prompt+f"You are controlling the player on the right hand side.",
                verbose=True
            )
            p2_name = f"{p2_model.split('/')[-1]}-{p2_style}"
        
        print(f"\n=== Match: {p1_name} vs {p2_name} at {speed_mode} speed ===\n")
        
        # Create the environment
        env = MortalKombatIIEnv(speed_mode=speed_mode)
        
        # Initialize game
        obs = env.reset()
        done = False
        raw_obs_data = []  # Store raw data
        frame_count = 0
        
        # Character selection phase
        print("Selecting characters and starting match...")
        
        # Select characters for both players
        # for _ in range(3):
        #     obs, done, info = env.step("[r][a]", "[L][A]")
        #     env.render()
        # time.sleep(0.5)
        
        # Confirm character selection and start match
        obs, done, info = env.step("[s]", "[S]")
        # obs, done, info = env.step("[s]", "[S]")
        # obs, done, info = env.step("[s]", "[S]")
        # time.sleep(1)
        # obs, done, info = env.step("[s]", "[S]")
        # obs, done, info = env.step("[s]", "[S]")
        # time.sleep(1)
        # obs, done, info = env.step("[s]", "[S]")
        env.render()
        time.sleep(1)
        
        print("Match started! Now fighting...")
        
        # Main game loop
        match_start_time = time.time()
        
        while not done and frame_count < max_frames:
            # Render the current frame
            env.render()
            
            # Create separate observations for each player
            p1_obs = {
                "visual": obs["visual"],
                "text": "Player 1: Choose your next move. Use lowercase buttons like [a][b][l]."
            }
            
            p2_obs = {
                "visual": obs["visual"],
                "text": "Player 2: Choose your next move. Use UPPERCASE buttons like [A][B][L]."
            }
            
            # Get player actions
            p1_action = p1_agent(p1_obs)
            print(f"Player 1: {p1_action}")
            
            p2_action = p2_agent(p2_obs)
            print(f"Player 2: {p2_action}")
            
            # Step the environment with both actions
            obs, done, info = env.step(p1_action, p2_action)
            
            # Store observations
            raw_obs_data.append({
                "frame": frame_count,
                "visual": obs["visual"],
                "info": {key: obs[key] for key in obs if key != "visual"},
                "p1_action": p1_action,
                "p2_action": p2_action,
            })
            
            frame_count += 1
            
            # Small delay to control game pace
            time.sleep(0.1)
        
        # Match finished
        match_duration = time.time() - match_start_time
        
        # Get final game info
        p1_score = info.get("p1_score", 0)
        p2_score = info.get("p2_score", 0)
        p1_health = info.get("p1_health", 0)
        p2_health = info.get("p2_health", 0)
        
        # Close the environment
        env.close()
        
        # Stop the agent threads
        if p1_type == "openrouter":
            p1_agent.stop()
        if p2_type == "openrouter":
            p2_agent.stop()
        
        # Print match results
        print("\n" + "=" * 40)
        print(f"Match complete after {frame_count} frames ({match_duration:.1f} seconds)")
        print(f"Player 1 ({p1_name}): Score={p1_score}, Health={p1_health}")
        print(f"Player 2 ({p2_name}): Score={p2_score}, Health={p2_health}")
        
        if p1_health > p2_health:
            print(f"Player 1 ({p1_name}) WINS!")
        elif p2_health > p1_health:
            print(f"Player 2 ({p2_name}) WINS!")
        else:
            print("DRAW!")
        print("=" * 40)
        
        # Save game data if requested
        if save_data and FOLDER:
            compressed_obs_data = []
            
            for frame_data in raw_obs_data:
                frame_id = frame_data["frame"]
                raw_image = frame_data["visual"]
                
                # Convert image to JPEG
                _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR), 
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                # Convert to base64
                img_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
                
                # Prepare metadata
                metadata = {
                    "frame": frame_id,
                    "visual": img_base64,
                    "info": frame_data["info"],
                    "p1_action": frame_data["p1_action"],
                    "p2_action": frame_data["p2_action"]
                }
                compressed_obs_data.append(metadata)
            
            # Generate filename with timestamp
            timestamp = int(time.time())
            file_prefix = f"{p1_name}_vs_{p2_name}"
            # Remove special characters for filename
            file_prefix = re.sub(r'[^\w\s-]', '_', file_prefix)
            file_name = os.path.join(FOLDER, f"{file_prefix}_{timestamp}.json.gz")
            
            # Save compressed game data
            with gzip.open(file_name, "wt", encoding="utf-8") as f:
                json.dump(compressed_obs_data, f, indent=4)
            
            # Save match summary
            summary_info = {
                "p1_name": p1_name,
                "p2_name": p2_name,
                "p1_type": p1_type,
                "p2_type": p2_type,
                "p1_model": p1_model if p1_type == "openrouter" else None,
                "p2_model": p2_model if p2_type == "openrouter" else None,
                "p1_style": p1_style,
                "p2_style": p2_style,
                "speed_mode": speed_mode,
                "total_frames": frame_count,
                "match_duration": match_duration,
                "p1_final_score": p1_score,
                "p2_final_score": p2_score,
                "p1_final_health": p1_health,
                "p2_final_health": p2_health,
                "winner": "Player 1" if p1_health > p2_health else "Player 2" if p2_health > p1_health else "Draw",
                "timestamp": timestamp
            }
            
            # Save summary in a separate file
            with open(file_name.replace(".json.gz", "_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary_info, f, indent=4)
            
            print(f"Game data saved to {file_name}")
        
        return frame_count, p1_score, p2_score, match_duration

    except Exception as e:
        print(f"Error during gameplay: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0, 0


if __name__ == "__main__":
    import re
    
    # Run a match between OpenRouter agents
    run_mk2_match(
        p1_type="openrouter",  # "openrouter" or "human"
        p2_type="openrouter",  # "openrouter" or "human"
        p1_model="google/gemini-2.0-flash-001",
        # p2_model="google/gemini-2.0-flash-001",
        p2_model="openai/gpt-4o-mini",
        p1_style="aggressive",
        p2_style="aggressive",
        speed_mode="human",     # "human", "slow", or "super-slow"
        max_frames=10000,
        save_data=True
    )
    
    # Uncomment to run a human vs AI match
    """
    run_mk2_match(
        p1_type="human", 
        p2_type="openrouter",
        p2_model="anthropic/claude-3.5-sonnet",
        p2_style="defensive",
        speed_mode="human",
        max_frames=1000,
        save_data=True
    )
    """