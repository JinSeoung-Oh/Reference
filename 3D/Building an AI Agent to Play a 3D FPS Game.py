### From https://levelup.gitconnected.com/building-an-ai-agent-to-play-a-3d-fps-game-using-ursina-deepseek-and-mistral-c4996168ed65

!pip install openai
!pip install pynput
!pip install mss
!pip install ursina

from openai import OpenAI  # The official OpenAI library to interact with the AI model's API.

# Initialize the OpenAI client.
# This configures the connection to the AI model's API endpoint.
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",  # The specific API endpoint URL.
    api_key="YOUR_API_KEY_HERE" # IMPORTANT: PASTE YOUR API KEY HERE
)

# Import all classes and functions from the ursina game engine
from ursina import *

# Create the main application window
app = Ursina()  # Initialize the Ursina application

###########################
# All the game logic and entities will be defined here
###########################

# Run the application
app.run()  # Start the Ursina game loop


# Setting the ground
# The ground is a large plane that the player can walk on.
ground = Entity(  # Create a large ground plane for the player to walk on.
    model='plane',  # 'model' is the shape
    scale=64,       # 'scale' is the size
    texture='grass',# 'texture' gives it a grassy look
    collider='box'  # 'collider' makes it solid so the player can't fall through
)

Sky()

# Importing first-person controller for player movement
from ursina.prefabs.first_person_controller import FirstPersonController

# Create the player, a first-person controller.
# Move with W, A, S, D and look around with the mouse.
player = FirstPersonController(
    y=2,          # Spawn slightly above the ground.
    origin_y=-.5, # Adjust camera height relative to the player's origin.
    speed=8       # Set the player's movement speed.
)

# Create a gun entity and attach it to the camera.
gun = Entity( # Create a gun entity.
    model='cube', # Set the model to a simple cube.
    parent=camera, # Attach the gun to the camera so it moves with the player's view.
    position=Vec3(.5, -0.25, .25), # Position the gun in front of the camera.
    scale=Vec3(.3, .2, 1), # Set the size of the gun.
    origin_z=-.5, # Adjust the gun's origin point for proper rotation.
    color=color.red, # Set the gun's color to red.
    on_cooldown=False # Add a custom attribute to track the gun's firing cooldown.
)

gun = Entity(
    parent=camera,  # Attach the gun to the camera so it moves with the player's view.
    model='assets/ak_47.obj',  # Use a custom AK-47 model.
    color=color.dark_gray,  # Set the gun's color to dark gray (for models without textures).
    scale=0.9,  # Set the size of the gun.
    position=Vec3(0.7, -0.9, 1.5),  # Position the gun in front of the camera.
    rotation=Vec3(0, 90, 0),  # Rotate the gun to face the correct direction.
    on_cooldown=False,  # Add a custom attribute to track the gun's firing cooldown.
    texture='assets/Tix_1.png',  # Apply a texture to the gun model.
)

# Create a muzzle flash effect as a child of the gun.
# It's initially disabled and will be shown when the gun fires.
gun.muzzle_flash = Entity(  # Create muzzle flash entity as child of gun
    parent=gun,              # Set the parent to the gun entity
    z=1,                     # Position the muzzle flash in front of the gun
    world_scale=.5,          # Set the size of the muzzle flash
    model='quad',            # Use a flat quad model for the flash
    color=color.yellow,      # Set the color to yellow for a flash effect
    enabled=False            # Initially disable the muzzle flash
)

# This function is called when the player shoots
# It checks if the gun is not on cooldown, plays a sound, and shows the muzzle flash.
# It also checks if the mouse is hovering over an entity with 'hp' attribute to damage
def shoot():
    if not gun.on_cooldown:  # Only shoot if gun is not on cooldown
        gun.on_cooldown = True  # Set gun on cooldown
        gun.muzzle_flash.enabled = True  # Enable muzzle flash
        from ursina.prefabs.ursfx import ursfx  # Import sound effect
        ursfx(  # Play shooting sound
            [(0.0, 0.0), # Sound wave points for the gunshot sound
             (0.1, 0.9), # Initial burst of sound
             (0.15, 0.75), # Followed by a decay
             (0.3, 0.14), # Further decay
             (0.6, 0.0)],  # End of sound
            volume=0.5, # Set volume of the sound
            wave='noise', # Use noise wave for a gunshot effect
            pitch=random.uniform(-13, -12), # Random pitch for variation
            pitch_change=-12, # Change pitch over time
            speed=3.0 # Speed of the sound effect
        )
        invoke(gun.muzzle_flash.disable, delay=.05)  # Disable muzzle flash after delay
        invoke(setattr, gun, 'on_cooldown', False, delay=.15)  # Reset cooldown after delay
        if mouse.hovered_entity and hasattr(mouse.hovered_entity, 'hp'):  # If mouse is over an enemy
            mouse.hovered_entity.blink(color.red)  # Blink enemy red
            mouse.hovered_entity.hp -= 10  # Decrease enemy HP

# Creating an update function that runs every frame
def update():
    if held_keys['left mouse']:  # Check if left mouse button is held
        shoot()                  # Call the shoot function


# Import random for placing the cubes randomly
import random

# Loop to create random cubes as obstacles in the scene
# This creates a grid of cubes with random heights
for i in range(16):
    Entity(
        model='cube',                          # Set the model to a cube
        origin_y=-.5,                          # Set the origin to align the cube with the ground
        scale=2,                               # Set the base size of the cube
        texture='brick',                       # Apply a brick texture to the cube
        texture_scale=(1, 2),                  # Stretch the texture vertically
        x=random.uniform(-8, 8),               # Place the cube at a random x position between -8 and 8
        z=random.uniform(-8, 8) + 8,           # Place the cube at a random z position, offset by 8 units forward
        collider='box',                        # Enable collision for the cube
        scale_y=random.uniform(2, 3),          # Randomize the height of the cube between 2 and 3
        color=color.hsv(0, 0, random.uniform(.9, 1))  # Assign a random shade of gray color
    )

# Importing shader for lighting and shadows
from ursina.shaders import lit_with_shadows_shader

# Set default shader for all entities
Entity.default_shader = lit_with_shadows_shader

# Creating sun light for the scene
sun = DirectionalLight()
# Set the direction of the sun light
sun.look_at(Vec3(1, -1, -1))

class Enemy(Entity):  # Enemy entity class
    def __init__(self, **kwargs):  # Initialize enemy
        super().__init__(
            parent=shootables_parent,  # Set parent to shootables
            model='cube',  # Use cube model
            scale_y=2,  # Set vertical scale
            origin_y=-.5,  # Set origin
            color=color.light_gray,  # Set color
            collider='box',  # Set collider
            **kwargs  # Additional arguments
        )
        self.health_bar = Entity(  # Create health bar entity
            parent=self,  # Set parent to enemy
            y=1.2,  # Position above enemy
            model='cube',  # Use cube model
            color=color.red,  # Set color to red
            world_scale=(1.5, .1, .1)  # Set scale
        )
        self.max_hp = 100  # Set max HP
        self.hp = self.max_hp  # Initialize HP

    def update(self):  # Update enemy each frame
        dist = distance_xz(player.position, self.position)  # Calculate distance to player
        if dist > 40:  # Skip if too far
            return

        self.health_bar.alpha = max(0, self.health_bar.alpha - time.dt)  # Fade health bar

        self.look_at_2d(player.position, 'y')  # Face player
        hit_info = raycast(  # Raycast forward
            self.world_position + Vec3(0, 1, 0),  # Start above enemy
            self.forward,  # Direction forward
            30,  # Ray length
            ignore=(self,)  # Ignore self
        )
        if hit_info.entity == player:  # If player is hit
            if dist > 2:  # If not close
                self.position += self.forward * time.dt * 5  # Move towards player

    @property
    def hp(self):  # HP property getter
        return self._hp

    @hp.setter
    def hp(self, value):  # HP property setter
        self._hp = value  # Set HP
        if value <= 0:  # If HP depleted
            destroy(self)  # Destroy enemy
            return

        self.health_bar.world_scale_x = self.hp / self.max_hp * 1.5  # Update health bar scale
        self.health_bar.alpha = 1  # Show health bar

# Create a health bar for the player, set its color and initial values
player_health_bar = HealthBar(
    bar_color=color.lime.tint(-.25),  # Health bar color
    roundness=.5,                     # Rounded corners
    value=player.hp,                  # Initial health value
    max_value=player.max_hp           # Maximum health value
)

class Enemy(Entity):
    def __init__(self, **kwargs):
        ...

    def update(self):
        ...

        self.health_bar.alpha = max(0, self.health_bar.alpha - time.dt)  # Fade health bar (line 19)
        self.look_at_2d(player.position, 'y')                            # Face player (line 20)

        hit_info = raycast(
            self.world_position + Vec3(0, 1, 0),  # Raycast from enemy head (line 21)
            self.forward,                         # In forward direction (line 22)
            30,                                   # Raycast distance (line 23)
            ignore=(self,)                        # Ignore self (line 24)
        )

        if hit_info.entity == player:             # If player is hit (line 25)
            if dist > 2:                          # Move towards player if not close (line 26)
                self.position += self.forward * time.dt * 5
            else:
                # Attack logic (line 27)
                if not self.attack_cooldown:
                    player.hp -= 20               # Damage player (line 28)
                    if player_health_bar:         # Update health bar if exists (line 29)
                        player_health_bar.value = player.hp
                    self.attack_cooldown = True   # Set cooldown (line 30)
                    invoke(setattr, self, 'attack_cooldown', False, delay=1)  # Reset cooldown after 1s (line 31)

    @property
    ...

    @hp.setter
    ...


import math  # Import the math module for mathematical functions.

# Get the player's current rotation on the y-axis (in degrees)
player_y_rotation = player.rotation_y

# Calculate the direction vector from the player to the enemy
direction_vector = enemy.position - player.position

# Calculate the angle to the enemy using atan2 for accuracy, in radians
angle_to_enemy_rad = math.atan2(direction_vector.x, direction_vector.z)

# Convert the angle from radians to degrees
angle_to_enemy_deg = math.degrees(angle_to_enemy_rad)

# Calculate the difference between the player's view angle and the angle to the enemy
aiming_error = angle_to_enemy_deg - player_y_rotation

# Normalize the aiming error to be between -180 and 180 degrees for easier interpretation
if aiming_error > 180:
        aiming_error -= 360
if aiming_error < -180:
        aiming_error += 360

# Use a raycast from the player's camera to check for line of sight to the enemy.
hit_info = raycast(player.world_position + player.camera_pivot.up, 
                   camera.forward, distance=100, ignore=(player,))

# The enemy is visible if the raycast hits the enemy entity.
is_enemy_visible = True if hit_info.entity == enemy else False

# Importing json module
import json

# Create a dictionary to hold all the relevant game state data.
game_data = {
        "player_health": player.hp,  # Player's current health
        "player_rotation_y": player_y_rotation,  # Player's Y-axis rotation
        "enemy_health": enemy.hp,  # Enemy's current health
        "distance_to_enemy": distance_xz(player.position, enemy.position),  # Distance to enemy
        "is_enemy_visible": is_enemy_visible,  # Is the enemy visible
        "angle_to_enemy_error": aiming_error,  # Aiming error angle to enemy
        "game_status": game_state,  # Current game status
}

# Write the game state data to a JSON file.
with open("game_state.json", "w") as f:
        json.dump(game_data, f)

def read_game_state():
    """Reads the current game state from a JSON file."""
    with open("game_state.json", "r") as f:
        return json.load(f)


# Import keyboard control classes from pynput
from pynput.keyboard import Key, Controller as KeyboardController

# Import mouse control classes from pynput
from pynput.mouse import Button, Controller as MouseController

# Create an instance of the keyboard controller to simulate key presses
keyboard = KeyboardController()

# Create an instance of the mouse controller to simulate mouse movements and clicks
mouse = MouseController()

# Set the model ID for the "Lieutenant" agent, a faster text-only model for quick tactical decisions.
text_model_id = "deepseek-ai/DeepSeek-V3"

def execute_command(command, game_state):
    """(The Muscle) Translates low-level commands from the AI into actual keyboard and mouse actions."""
    # Get the current aiming error from the game state, defaulting to 0 if it's not available.
    aim_error = game_state.get('angle_to_enemy_error', 0)
    
    # If the command is to AIM or ATTACK and an enemy is currently visible...
    if command in ['AIM', 'ATTACK'] and game_state.get('is_enemy_visible', False):
        # Calculate the necessary mouse movement to correct the aim.
        mouse_movement = -int(aim_error * 2.5) # A multiplier acts as sensitivity.
        # Move the mouse horizontally to adjust the aim.
        mouse.move(mouse_movement, 0)

    # If the command is ATTACK and the aim is already accurate (error is small)...
    if command == "ATTACK" and abs(aim_error) < 5:
        # Press and hold the left mouse button to fire.
        mouse.press(Button.left)
    else:
        # Otherwise, release the left mouse button to stop firing.
        mouse.release(Button.left)

    # If the command is to perform a defensive maneuver...
    if command == "DEFENSIVE_MANEUVER":
        # Press 's' and 'a' keys to move backward and left.
        keyboard.press('s'); keyboard.press('a')
        # Hold the keys for a short duration.
        time.sleep(0.5)
        # Release the keys.
        keyboard.release('s'); keyboard.release('a')

    # If the command is to search for the enemy...
    if command == "SEARCH":
        # Move the mouse horizontally to look around.
        mouse.move(80, 0)
    
    # If the command is to advance...
    if command == "ADVANCE":
        # Press 'w' to move forward briefly.
        keyboard.press('w'); time.sleep(0.3); keyboard.release('w')

 def get_tactical_action_from_llm(strategy, game_state):
    """(The Lieutenant) Uses the text-only model to choose an immediate action based on the Commander's strategy."""
    # Convert the game state dictionary into a JSON string.
    state_report = json.dumps(game_state, indent=2)
    # Send a request to the OpenAI API with the current strategy and game state.
    response = client.chat.completions.create(
        model=text_model_id,  # Use the specified fast text model.
        messages=[
            {
                "role": "system",
                "content": f"""
                You are a tactical AI Lieutenant. Your Commander has issued the strategic order: '{strategy}'.
                Your job is to choose the best IMMEDIATE action based on this order and real-time data.

                IF STRATEGY IS 'ENGAGE_AGGRESSIVELY':
                 - If enemy is visible and aim is good (error < 5), command `ATTACK`.
                 - If enemy is visible but aim is bad, command `AIM`.
                 - If enemy is far (>15m), command `ADVANCE`.
                
                IF STRATEGY IS 'REPOSITION_DEFENSIVELY':
                 - Command `DEFENSIVE_MANEUVER` to get to safety immediately.
                
                IF STRATEGY IS 'HUNT_THE_ENEMY':
                 - If enemy is not visible, command `SEARCH`. If they suddenly become visible, command `AIM`.

                Choose ONE command: `ATTACK`, `AIM`, `ADVANCE`, `DEFENSIVE_MANEUVER`, `SEARCH`.
                """
            },
            {"role": "user", "content": f"Strategy: '{strategy}'.\nReal-time data:\n{state_report}\n\nYour command:"}
        ],
        max_tokens=10,  # Limit the response length.
        temperature=0.0  # Set temperature to 0 for deterministic, non-creative responses.
    )
    # Extract the action command from the AI's response.
    action = response.choices[0].message.content.strip().replace("'", "").replace('"', "")
    print(f"Lieutenant Action for '{strategy}': {action}")
    # Return the chosen action.
    return action

import base64  # Used for encoding the screenshot image into a text format for the API.
import mss  # A fast tool for taking screenshots.
import mss.tools  # Additional tools for the mss library, like saving the image.

# Set the model ID for the "Commander" agent, which uses vision capabilities for high-level strategy.
vision_model_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


# Initialize the agent's current strategic goal. It starts in an "INITIALIZING" state.
current_strategic_goal = "INITIALIZING"
# Keep track of the last time the high-level strategy was updated. Initialized to 0.
last_strategic_update_time = 0
# Set the interval (in seconds) for how often the "Commander" (vision model) should re-evaluate the strategy.
strategic_update_interval = 1.0 # The Commander thinks every 1 seconds

def capture_screen_as_base64():
    """Captures the entire screen and returns it as a base64 encoded string."""
    # Use the mss library as a context manager for efficient resource handling.
    with mss.mss() as sct:
        # Get information about the primary monitor.
        monitor = sct.monitors[1] # Assumes main monitor is 1
        # Grab the image data from the monitor.
        sct_img = sct.grab(monitor)
        # Convert the raw image data (RGB) into PNG format as bytes.
        img_bytes = mss.tools.to_png(sct_img.rgb, sct_img.size)
        # Encode the PNG bytes into a base64 string and decode it to a standard UTF-8 string.
        return base64.b64encode(img_bytes).decode('utf-8')

  def get_strategic_goal_from_vlm(game_state, screenshot_base64):
    """(The Commander) Uses the vision model and game data to decide on a high-level strategy."""
    print("\n--- Commander is Thinking (VLM)... ---")
    # Convert the game state dictionary into a nicely formatted JSON string for the prompt.
    state_report = json.dumps(game_state, indent=2)
    # Send a request to the OpenAI API with the system prompt, user text, and the screenshot.
    response = client.chat.completions.create(
        model=vision_model_id,  # Use the specified vision model.
        messages=[
            {
                "role": "system",
                "content": """
                You are a strategic AI Commander. You see the big picture using both an image and precise data. Your job is to set the overall strategy, not the immediate action.
                Analyze the visual environment and the data report.

                Choose ONE of these STRATEGIC GOALS:
                - `ENGAGE_AGGRESSIVELY`: The situation is favorable. I have good health, and the enemy is in a killable position.
                - `REPOSITION_DEFENSIVELY`: The situation is dangerous. I have low health, am in a bad position (too open, too close), or just took damage. Survival is key.
                - `HUNT_THE_ENEMY`: I cannot see the enemy. My goal is to find them.
                
                Provide ONLY the command word for the chosen strategy.
                """
            },
            {
                "role": "user",
                "content": [
                    # The user prompt includes both the text data and the image.
                    {"type": "text", "text": f"Analyze the scene and this data to set the strategy.\nDATA:\n{state_report}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}}
                ]
            }
        ],
        max_tokens=10  # Limit the response length to get just the command.
    )
    # Extract the strategy text from the AI's response.
    strategy = response.choices[0].message.content.strip().replace("'", "").replace('"', "")
    print(f"--- New Strategy from Commander: {strategy} ---")
    # Return the chosen strategy.
    return strategy

# importing the necessary libraries
import time

# Start the main control loop that runs continuously.
while True:
    # Read the latest game state from the JSON file.
    game_state = read_game_state()
    # If the game state is missing or indicates the game has ended, stop the agent.
    if not game_state or game_state.get('game_status') in ['won', 'lost']:
        print(f"Game over or state unreadable.")
        break

    # --- COMMANDER'S TURN (Strategic Thinking) ---
    # Check if enough time has passed since the last strategic update.
    if time.time() - last_strategic_update_time > strategic_update_interval:
        # If so, capture a new screenshot.
        screenshot = capture_screen_as_base64()
        # Ask the Commander (VLM) for a new strategic goal.
        current_strategic_goal = get_strategic_goal_from_vlm(game_state, screenshot)
        # Update the timestamp for the last strategic update.
        last_strategic_update_time = time.time()

    # --- LIEUTENANT'S TURN (Tactical Execution) ---
    # Ensure the Commander has provided an initial strategy.
    if current_strategic_goal != "INITIALIZING":
        # Ask the Lieutenant (LLM) for a specific tactical action based on the current strategy.
        tactical_action = get_tactical_action_from_llm(current_strategic_goal, game_state)
        # Execute the chosen action.
        execute_command(tactical_action, game_state)
    else:
        # If still initializing, just print a waiting message.
        print("Waiting for initial strategy from Commander...")

    # Pause briefly before the next loop iteration to control the action frequency.
    time.sleep(0.3) # Tactical loop speed

