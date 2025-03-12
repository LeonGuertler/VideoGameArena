import retro
import pygame
import numpy as np

# Initialize pygame for keyboard input
pygame.init()

# Player 1 controls (indices 0-11)
PLAYER1_KEYS = {
    pygame.K_w: 4,      # Up
    pygame.K_s: 5,      # Down
    pygame.K_a: 6,      # Left
    pygame.K_d: 7,      # Right
    pygame.K_g: 0,      # A (Low Punch)
    pygame.K_h: 1,      # B (High Punch)
    pygame.K_j: 2,      # C (Block)
    pygame.K_b: 8,      # X (Low Kick)
    pygame.K_n: 9,      # Y (High Kick)
    pygame.K_m: 10,     # Z (Unused)
    pygame.K_RETURN: 3, # Start
    pygame.K_SPACE: 11  # Select
}

# Player 2 controls (indices 12-23)
PLAYER2_KEYS = {
    pygame.K_UP: 16,    # Up
    pygame.K_DOWN: 17,  # Down
    pygame.K_LEFT: 18,  # Left
    pygame.K_RIGHT: 19, # Right
    pygame.K_i: 12,     # A (Low Punch)
    pygame.K_o: 13,     # B (High Punch)
    pygame.K_p: 14,     # C (Block)
    pygame.K_k: 20,     # X (Low Kick)
    pygame.K_l: 21,     # Y (High Kick)
    pygame.K_SEMICOLON: 22, # Z (Unused)
    pygame.K_RETURN: 15, # Start
    pygame.K_SPACE: 23  # Select
}

def get_key_action():
    """Map keyboard inputs to Genesis controller actions for both players."""
    keys = pygame.key.get_pressed()
    action = [False] * 24  # 12 buttons per player
    
    # Player 1 input
    for key, button in PLAYER1_KEYS.items():
        if keys[key]:
            action[button] = True
    
    # Player 2 input
    for key, button in PLAYER2_KEYS.items():
        if keys[key]:
            action[button] = True
    
    return action

def main():
    # Load Mortal Kombat II for Genesis
    env = retro.make(game="MortalKombatII-Genesis", players=2)
    obs = env.reset()

    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("Mortal Kombat II - 2 Player Versus Mode")

    running = True
    clock = pygame.time.Clock()
    debug = True  # Set to True to see input debug
    p2_joined = False  # Track if P2 has joined

    print("""
    Mortal Kombat II (Genesis) - 2 Player Versus Mode Instructions:
    
    Player 1 (P1):
    - Move: W (Up), S (Down), A (Left), D (Right)
    - Attacks: G (Low Punch), H (High Punch), J (Block)
    - Kicks: B (Low Kick), N (High Kick)
    - Start: RETURN | Select: SPACE
    
    Player 2 (P2):
    - Move: UP, DOWN, LEFT, RIGHT (arrow keys)
    - Attacks: I (Low Punch), O (High Punch), P (Block)
    - Kicks: K (Low Kick), L (High Kick)
    - Start: RETURN | Select: SPACE
    
    Steps to Start PvP:
    1. P1: Press RETURN to start from title screen
    2. P1: Use WASD to navigate to 'Vs Mode', press G to select
    3. P2: Press RETURN to join (activates P2 controls)
    4. P1 (WASD) and P2 (Arrows) pick characters
    5. Press G (P1 A) and I (P2 A) to confirm
    
    Debug: Check console for P2 input confirmation
    Exit: Press ESC to quit
    """)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        action = get_key_action()

        # Ensure P2 joins by pressing Start (index 15)
        if not p2_joined:  # P2 Start pressed
            print("Player 2 joined!")
            action[15] = True
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            action[15] = False
            p2_joined = True

        # Debug output to confirm inputs
        if debug:
            if any(action[0:12]):
                print(f"P1 action: {action[0:12]}")
            if any(action[12:24]):
                print(f"P2 action: {action[12:24]}")

        # Step the environment
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()

        if done:
            obs = env.reset()
            p2_joined = False  # Reset P2 join state
            print("Game reset - navigate to Versus mode again")

        clock.tick(60)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()