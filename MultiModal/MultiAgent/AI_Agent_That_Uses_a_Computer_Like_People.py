### From https://levelup.gitconnected.com/creating-an-ai-agent-that-uses-a-computer-like-people-do-288f7ad97169

# Cloning the reo with submodule omniparser
git clone --recursive https://github.com/FareedKhan-dev/ai-desktop
cd ai-desktop/OmniParser

# Installing the dependencies
pip install -r requirements.txt

"""
Setting OmniParser

# Clone OmniParser Repo and navigating to the directory
git clone https://github.com/microsoft/OmniParser/
cd OmniParser

# Download the model checkpoints to the local directory OmniParser/weights/
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} \
         icon_caption/{config.json,generation_config.json,model.safetensors}; do
    huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights
done

# Rename the downloaded 'icon_caption' directory
mv weights/icon_caption weights/icon_caption_florence
"""

import logging  # Logging events.
import os       # OS interactions.
import sys      # System functions.
import re       # Regular expressions.
import math     # Math operations.

from dataclasses import dataclass, field  # Data classes.
from typing import List, Optional, Dict, Any, Tuple  # Type hinting.

import base64   # Base64 encoding/decoding.
import io        # Data streams.
from PIL import Image  # Image processing.
import pyautogui # GUI automation.
import torch   # PyTorch (deep learning).
import asyncio  # Asynchronous programming.
from pathlib import Path # file paths

# OmniParser custom modules (they are in the OmniParser submodule)
from OmniParser.utils.display import get_yolo_model, 
                                     get_caption_model_processor, 
                                     get_som_labeled_img, check_ocr_box


# Define configuration parameters
config = {
    "som_model_path": "weights/icon_detect/model.pt",
    "caption_model_name": "microsoft/OmniParser-v2.0",
    "caption_model_path": "weights/icon_caption_florence",
    "BOX_TRESHOLD": 0.5,  # Adjust threshold for bounding box detection
}

def load_models(config: Dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the Structured Object Model (SOM) for UI detection
    som_model = get_yolo_model(model_path=config["som_model_path"])

    # Load the captioning model processor
    caption_model_processor = get_caption_model_processor(
        model_name=config["caption_model_name"],
        model_name_or_path=config["caption_model_path"],
        device=device
    )

    print("OmniParser models loaded successfully!")
    return som_model, caption_model_processor

def parse_screen(image_base64: str, som_model, caption_model_processor) -> Tuple[str, List[Dict[str, Any]]]:
    # Decode base64 image
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes))
    print("Image size:", image.size)

    # Define scaling for overlay
    box_overlay_ratio = max(image.size) / 3200
    draw_bbox_config = {
        "text_scale": 0.8 * box_overlay_ratio,
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding": max(int(3 * box_overlay_ratio), 1),
        "thickness": max(int(3 * box_overlay_ratio), 1),
    }

    # Perform OCR and get bounding boxes
    (text, ocr_bbox), _ = check_ocr_box(
        image,
        display_img=False,
        output_bb_format="xyxy",
        easyocr_args={"text_threshold": 0.8},
        use_paddleocr=False
    )

    # Process the screen using the SOM model and captioning model
    dino_labeled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image, 
        som_model, 
        BOX_TRESHOLD=config["BOX_TRESHOLD"], 
        output_coord_in_ratio=True, 
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config, 
        caption_model_processor=caption_model_processor, 
        ocr_text=text,
        use_local_semantics=True, 
        iou_threshold=0.7, 
        scale_img=False, 
        batch_size=128
    )

    return dino_labeled_img, parsed_content_list

def get_screenshot() -> Tuple[Image.Image, Path]:

    # Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Capture the entire screen
    img = pyautogui.screenshot()  # Returns a PIL Image
        
    # Define the screenshot file path
    screenshot_path = OUTPUT_DIR / "screenshot.png"

    # Save the screenshot
    img.save(screenshot_path)
    
    # returning the bounded image and bounded boxes
    return img, screenshot_path

# tool to perform actions on the computer
class ComputerActions_Tool:
    tool_name = "computer_actions" # We'll name our tool 'computer_actions'

    def __init__(self):
        # We can set up any initial configurations here if needed
        # For now, we might just need to handle special key names
        self.key_name_mapping = {
            "PageDown": "pagedown",  # Mapping friendly names to pyautogui names
            "PageUp": "pageup",
            "WindowsKey": "win",
            "EscapeKey": "esc",
            "EnterKey": "enter"
        }

    async def perform_action(self, action_type, action_details=None):
        # This function will take the action type and details and execute it

        try:
            if action_type == "move_mouse":
                # Move the mouse to a given coordinate
                if action_details is None or 'coordinates' not in action_details:
                    raise Exception("Mouse movement needs coordinates (x, y).")
                x_coord, y_coord = action_details['coordinates']
                pyautogui.moveTo(x_coord, y_coord, duration=0.2) # Smooth mouse movement
                return {"action_result": f"Mouse moved to ({x_coord}, {y_coord})"}

            elif action_type == "click_left_mouse":
                # Perform a left mouse click
                pyautogui.click()
                return {"action_result": "Left mouse click performed."}

            # ... we would add similar 'elif' blocks for other actions like:
            # 'click_right_mouse', 'click_double_mouse', 'type_text', 'key_press',
            # 'capture_screenshot', 'get_cursor_position', 'pause_briefly',
            # 'scroll_up_page', 'scroll_down_page', 'hover_mouse_over'

            else:
                raise Exception(f"Unknown action type: {action_type}")

        except Exception as action_error:
            return {"action_error_message": f"Error during action: {action_error}"}

# class to orchestrate the actions
class ActionOrchestrator_Agent:

    def __init__(self, feedback_callback_function):
        # Needs a function to provide feedback (e.g., for logging or display)
        self.computer_tool = ComputerActions_Tool() # Initialize our ComputerActions tool
        self.feedback_callback = feedback_callback_function # Store the feedback function

    async def execute_agent_action(self, tool_name, action_input_details):
        # This function takes the tool name and action details and executes

        self.feedback_callback(f"Attempting to use tool: {tool_name}, with details: {action_input_details}", "agent_message") # Informative message

        try:
            if tool_name == "computer_actions":
                # If the tool is 'computer_actions', use our ComputerActions_Tool
                action_type_to_use = action_input_details.get("action") # Get the action type
                action_parameters = action_input_details.get("parameters", None) # Get parameters if any

                action_result = await self.computer_tool.perform_action( # Execute the action using the tool
                    action_type=action_type_to_use, action_details=action_parameters
                )

                if action_result.get("action_error_message"): # Check for errors
                    self.feedback_callback(f"Tool reported an error: {action_result['action_error_message']}", "agent_error") # Report error
                else:
                    if action_result.get("action_result"): # Report success output
                        self.feedback_callback(action_result["action_result"], "agent_output") # Report action output
                return action_result # Return the result of the action

            else:
                return {"action_error_message": f"Unknown tool requested: {tool_name}"} # Error for unknown tool

        except Exception as orchestration_error:
            return {"action_error_message": f"Unexpected problem during action orchestration: {orchestration_error}"} # Error for unexpected problems

class VLMAgent_Agent:
    # ... (Initialization of OmniParser, ActionOrchestrator, LLM config, etc. would go here) ...

    async def run_agent_task(self, user_query):
        # Main method to run the agent for a given user query

        previous_agent_messages = [] # To keep track of conversation history (for context)

        while True: # Keep looping until task is completed or agent decides to stop
            # Step 1: Capture Screenshot
            screenshot_image, screenshot_path = get_screenshot() # Use our screenshot function

            # Step 2: Parse Screen using OmniParser
            labeled_image_base64, parsed_screen_content = parse_screen( # Use our OmniParser function
                screenshot_image, self.omni_parser_models, self.omni_parser_config
            )

            # Step 3: Formulate Prompt for LLM
            prompt_to_llm = create_llm_prompt( # Function to create prompt (we'll define this)
                user_query=user_query,
                screen_content=parsed_screen_content,
                previous_messages=previous_agent_messages # Pass history for context
            )

            # Step 4: Send Prompt to LLM and Get Response
            llm_response_text, _ = send_prompt_to_llm_api( # Function to interact with LLM API
                prompt=prompt_to_llm,
                llm_api_config=self.llm_config # LLM API configuration
            )
            self.output_feedback_function(f"LLM Response: {llm_response_text}", "llm_response") # Show LLM response

            # Step 5: Parse LLM Response to Get Action Command
            action_command_from_llm = parse_llm_response_for_action(llm_response_text) # Function to parse JSON

            # Step 6: Execute Action using ActionOrchestrator
            if action_command_from_llm and action_command_from_llm.get("next_action"): # Check if action is suggested
                action_description = generate_action_description(action_command_from_llm) # For output
                self.output_feedback_function(action_description, "agent_action") # Show action description

                action_name = "computer_actions" # For now, we only have computer actions
                action_input = prepare_action_input(action_command_from_llm) # Prepare input for ActionOrchestrator

                action_result = await self.action_orchestrator.execute_agent_action( # Execute the action
                    tool_name=action_name, tool_input=action_input
                )

                # Step 7: Handle Feedback and Update Conversation History
                previous_agent_messages = update_conversation_history( # Function to update history
                    previous_messages=previous_agent_messages,
                    llm_response=llm_response_text,
                    action_description=action_description,
                    action_result=action_result
                )
                user_query = "Continue with the previous task" # Agent asks LLM to continue in next loop

                if action_result.get("error"): # If there is an error, maybe LLM can fix it in next turn
                    previous_agent_messages.append({"role": "user", "content": action_result["error"]}) # Feedback error to LLM
            else:
                self.output_feedback_function("Task completed or no action needed.", "agent_status") # Task finished
                break # Exit the loop if no action needed

        self.output_feedback_function("Agent task finished.", "agent_status") # Final status message


# Load models
som_model, caption_model_processor = load_models(config)

# Load a desktop screenshot image (convert it to base64)
with open("desktop_screenshot.jpg", "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

# Run OmniParser
labeled_img, parsed_content = parse_screen(image_base64, som_model, caption_model_processor)

# Output directory where temporary screenshot are saved
OUTPUT_DIR = "temp/"
# Step 1: Capture the screenshot
screenshot, screenshot_path = get_screenshot()

# Step 2: Convert the screenshot to base64 format for processing
with open(screenshot_path, "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

# Step 3: Load the OmniParser models
som_model, caption_model_processor = load_models(config)

# Step 4: Parse the screen using OmniParser
labeled_img, parsed_content = parse_screen(image_base64, som_model, caption_model_processor)

possible_actions = [
    "key_press",       # To simulate pressing keys on the keyboard
    "type_text",      # For typing out words and sentences
    "move_mouse",      # To move the cursor around the screen
    "click_left_mouse", # For standard selections and interactions
    "click_right_mouse",# To open context menus and more
    "click_double_mouse",# For actions that require a double click
    "capture_screenshot",# For the agent to re-examine the screen
    "get_cursor_position", # To know where the mouse is currently located
    "pause_briefly",   # To allow time for actions to register
    "scroll_up_page",  # To navigate content vertically
    "scroll_down_page",# Also for vertical navigation
    "hover_mouse_over" # To highlight or trigger interface changes
]

if action_type == "move_mouse":
    if action_details is None or 'coordinates' not in action_details:
        raise Exception("Mouse movement needs coordinates (x, y).")
    x_coord, y_coord = action_details['coordinates']
    pyautogui.moveTo(x_coord, y_coord, duration=0.2)
    return {"action_result": f"Mouse moved to ({x_coord}, {y_coord})"}

elif action_type == "click_left_mouse":
                pyautogui.click()
                return {"action_result": "Left mouse click performed."}

async def main_function_agent(user_query):
    # ... (Set up configurations, API keys, model paths, etc.) ...

    executor = LocalExecutor_Agent(output_callback_function=print)
    
    agent = VLMAgent_Agent(
        omni_parser_config=omni_parser_config,
        llm_config=llm_config,
        action_orchestrator=executor,
        output_feedback_function=print
    )

    print("Waiting for 5 seconds before execution...")
    await asyncio.sleep(5)  # Add a 5-second delay

    await agent.run_agent_task(user_query=user_query)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])  # Combine all arguments as a single string
    else:
        user_input = input("Enter your command: ")

    asyncio.run(main_function_agent(user_input))
