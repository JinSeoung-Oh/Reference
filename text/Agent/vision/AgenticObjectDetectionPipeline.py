### From https://ai.gopubby.com/building-an-agentic-object-detection-pipeline-34e1f3a47323
### This code is just part of pipline. So have to understand the flow and rebuild the code to run it

"""
1. Traditional Computer Vision and Its Limitations
   Historically, computer vision (CV) tasks such as image classification, object detection, and semantic segmentation were framed 
   as closed-set problems, where models could only recognize the classes they were trained on. 
   Even with performance improvements from CNN-based deep learning, adapting to new domains or recognizing unseen objects 
   still required retraining or fine-tuning.

2. Paradigm Shift with CLIP
   The arrival of transformer-based architectures—especially CLIP (Contrastive Language-Image Pretraining)—marked a major turning point.
   -a. Jointly trained text and image encoders on hundreds of millions of captioned images
   -b. Built a shared embedding space for text and images, enabling the model to associate arbitrary text prompts 
       (new class names or descriptive phrases) with relevant visual features—even for objects never seen during training
   -c. Enabled Open-Vocabulary Detection & Segmentation

3. Advancements After CLIP
   -a. DINO, OWL-ViT: Localize phrases like “blue recycling bin” or “stop sign with graffiti” without retraining
   -b. Multimodal LLMs (e.g., GPT-4V, Gemini, Claude-Sonnet): Combine CLIP-like image encoders with text decoders to achieve multimodal capabilities
   -c. Result: Shift from fixed-category CV models to language-command-driven, adaptive VLM systems capable of real-time object detection, 
               segmentation, and scene interpretation
   -d. Reduced need for label engineering and retraining, fostering agentic frameworks

4. Emergence of Fully Agentic Vision Systems
   -a. OpenAI’s o3 model: Goes beyond generating bounding boxes
       -1. Writes Python code to zoom, rotate, or adjust image contrast
       -2. Conducts web searches for context, feeds results back for reasoning, and iterates
   -b. Combines perception, tool use, and self-critique in a single inference pass—enhancing autonomy

5. Proposed Agentic Object Detection Framework
   Combines open-vocabulary object detectors with VLM-based critique and verification.
   -a. Full Workflow
       -1. Concept Inference
           -1) Input: User image and request
           -2) If specific objects are mentioned, extract them directly
           -3) If not, VLM (GPT-4o) identifies salient objects in the image
       -2. Initial Detection
           -1) Send extracted concepts to Grounding DINO for bounding box generation
       -3. Visualization
           -1) Annotate detections with arrows, unique numeric labels, and white borders
       -4. Query Critique & Concept Abstraction
           -1) VLM reasoner (OpenAI o1) + Chain-of-Thought reasoning
           -2) Detects misclassifications or overly specific categories, refining them to higher-level concepts 
               (e.g., poodle → dog, cricketers → people) and refining the request
       -5. Refined Detection
           -1) Run detector again on the original image with the updated concept list
       -6. Final Filtering
           -1) VLM cross-checks updated detections with user intent, discarding irrelevant boxes
       -7. Output
           -1) Final annotated image with validated bounding boxes aligned precisely to user intent

6. Inference-Time Compute and Verification
   -a. Role of Verifiers: LLMs/VLMs inspect and refine outputs from primary models
   -b. Inference-Time Compute: Improves performance by allocating more computational steps at inference without updating weights
       -1. Examples: Multi-step Chain-of-Thought reasoning, self-consistency sampling
   -c. Benefits:
       -1. Model-Agnostic: Can swap in new detectors/reasoners without retraining
       -2. Explainability: CoT logs provide reasons for rejecting predictions

7. Core Value of the Proposed Pipeline
   -a. Delivers high-precision, user-aligned open-set detection without retraining or new data collection
   -b. Enhances detection quality through multi-step VLM-driven verification and critique
   -c. Enables real-time, adaptive, and high-accuracy agentic object detection systems
"""

import base64
import json
from openai import OpenAI

def encode_image(image_path):
    """
    Encodes an image file into a base64-encoded string.
    Args:
        image_path (str): Path to the image file to be encoded.
    Returns:
        str: Base64-encoded string of the image content.
        None: If an error occurs during encoding.
    Raises:
        Exception: If the file cannot be opened or encoded.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

from utils.image_utils import encode_image


class VLMTool:
    def __init__(self, api_key):
        pass

    def chat_completion(self, messages, model="o1", max_tokens=300, temperature=0.1, response_format=None):
        pass

    def extract_objects_from_request(self, image_path, user_text, model="gpt-4o"):
        pass

def chat_completion(
    self,
    messages,
    model="o1",
    max_tokens=300,
    temperature=0.1,
    response_format=None
):
    """Calls GPT for chat completion."""
    try:
        if model in ["gpt-4o", "gpt-4o-mini"]:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format if response_format else {"type": "text"}
            )
        elif model in ["o1"]:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format if response_format else {"type": "text"}
            )
        else:
            raise NotImplementedError("This model is not supported")

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

def extract_objects_from_request(self, image_path, user_text, model="gpt-4o"):
    """
    Asks the LLM to parse user request for which objects to detect/segment.
    Returns a list of objects in plain text.
    """
    base64_image = encode_image(image_path)
    if not base64_image:
        return None

    prompt = (
        "You are an AI vision assistant that extracts objects to be identified from a user's request."
        "If the user wants to detect or semantically segment all objects in the image, return a comma-separated list of objects you can see. "
        "If the user wants to detect or semantically segment specific objects, extract only those mentioned explicitly in their request. "
        "Respond ONLY with the list of objects, separated by commas, and NOTHING ELSE."
        "The objective here is only to understand the objects of interest that can be extracted from the image and the user's request."
        "You are not actually required to perform or execute the user's request."    
        )

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    result = self.chat_completion(messages, model=model)
    if result:
        detected_objects = [
            obj.strip().lower()
            for obj in result.split(",")
            if obj.strip()
        ]
        return detected_objects

    return []

class ObjectDetectionTool:
    """
    Performs object detection using GroundingDINO or OWL-ViT,
    plus an optional 'critique' (refinement) step with a VLM
    to yield a refined set of objects to detect.
    """
    def __init__(self, model_id, device, vlm_tool, confidence_threshold=0.2, concept_detection_model="gpt-4o", initial_critique_model="o1", final_critique_model="gpt-4o"):
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.device = device
        self.vlm_tool = vlm_tool  # The LLMTool that can handle vision (GPT-4V) or similar
        self.confidence_threshold = confidence_threshold
        self.concept_detection_model = concept_detection_model
        self.initial_critique_model = initial_critique_model
        self.final_critique_model = final_critique_model
        
        # We store bounding boxes for potential usage later (e.g., for SAM).
        self.last_detection_bboxes = []
        self.last_filtered_objects = []

def draw_arrows_and_numbers(image_path, detected_objects):
    """
    Draws arrows and numbers on an image to label detected objects.

    This function dynamically places numbers near the borders with arrows pointing 
    from object centers to the borders. Arrows are dashed for clarity, and the 
    numbering avoids overlap when possible.

    Args:
        image_path (str): Path to the input image.
        detected_objects (list): List of tuples containing object information in 
            the format (number, object_name, bounding_box), where bounding_box is 
            (x1, y1, x2, y2).

    Returns:
        str: Path to the saved labeled image ('labeled_objects_optimized.jpg').

    Note:
        - Arrows are drawn from object centers to the nearest border.
        - Numbers are displayed with semi-transparent backgrounds for readability.
    """
    img = cv2.imread(image_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    used_positions = []

    # Pad the image with a white border
    top, bottom, left, right = 50, 50, 50, 50
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    height, width, _ = img.shape

    for i, (num, obj, box) in enumerate(detected_objects):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of the bounding box

        # Adjust coordinates for padded image
        x1 += left
        y1 += top
        x2 += left
        y2 += top
        cx += left
        cy += top

        # Determine arrow direction towards the nearest border
        distances = {'top': cy, 'bottom': height - cy, 'left': cx, 'right': width - cx}
        direction = min(distances, key=distances.get)

        if direction == 'top':
            arrow_end = (cx, top)
            text_position = (cx - 10, top - 10)
        elif direction == 'bottom':
            arrow_end = (cx, height - bottom)
            text_position = (cx - 10, height - 5)
        elif direction == 'left':
            arrow_end = (left, cy)
            text_position = (left - 30, cy + 5)
        else:
            arrow_end = (width - right, cy)
            text_position = (width - 30, cy + 5)

        # Draw the dashed arrow from the object center to the border
        color = (0, 0, 0)  # Black color for all arrows
        line_type = cv2.LINE_4
        cv2.arrowedLine(img, (cx, cy), arrow_end, color, 2, tipLength=0.3)

        # Draw a semi-transparent rectangle behind the text
        overlay = img.copy()
        cv2.rectangle(overlay, (text_position[0] - 5, text_position[1] - 20), (text_position[0] + 30, text_position[1] + 5), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Draw the number at the border with black text
        cv2.putText(img, str(num), text_position, font, 0.8, color, 2)

    labeled_image_path = "labeled_objects_optimized.jpg"
    cv2.imwrite(labeled_image_path, img)
    return labeled_image_path


def _run_detector(self, image_path, query_list):
        """
        Low-level routine to run the detection model on `query_list`.
        Returns: (detected_objects_final, labeled_image_path)
        Where `detected_objects_final` = [(num, label, [x1,y1,x2,y2]), ...].
        """
        from PIL import ImageFont
        
        # Format queries for the model
        if INV_MODEL_TYPES[self.model_id] == "owlvit":
            formatted_queries = [f"An image of {q}" for q in query_list]
        elif INV_MODEL_TYPES[self.model_id] == "grounding_dino":
            formatted_queries = " ".join([f"{q}." for q in list(set(query_list))])
        else:
            raise NotImplementedError("Model not supported")

        # Load image
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=formatted_queries, 
            images=img, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process bounding boxes
        if INV_MODEL_TYPES[self.model_id] == "grounding_dino":
            results = self.processor.post_process_grounded_object_detection(
                outputs, 
                inputs.input_ids, 
                box_threshold=0.4, 
                text_threshold=0.3, 
                target_sizes=[img.size[::-1]]
            )
            boxes = results[0]["boxes"]
            scores = results[0]["scores"]
            labels = results[0]["labels"]
        elif INV_MODEL_TYPES[self.model_id] == "owlvit":  # OWL-ViT
            logits = torch.max(outputs["logits"][0], dim=-1)
            scores = torch.sigmoid(logits.values).cpu().numpy()
            labels = logits.indices.cpu().numpy()
            boxes = outputs["pred_boxes"][0].cpu().numpy()
        else:
            raise NotImplementedError("Model not supported")            

        detected_objects_final = []
        idx = 1
        for score, box, label_idx in zip(scores, boxes, labels):
            if score < self.confidence_threshold:
                continue
            detected_objects_final.append((idx, label_idx, box.tolist()))
            idx += 1

        # Draw numbers
        labeled_image_path = draw_arrows_and_numbers(image_path, detected_objects_final)
        return detected_objects_final, labeled_image_path

def _critique_and_refine_query(self, user_request, original_concepts, labeled_image_path, objects_detected, model="o1"):
        """
        Asks the VLM/LLM: "We tried to detect <objects_detected> for the user request, 
        but maybe we need a refined set of objects. 
        Return a new list of objects or concepts to detect."
        """
        base64_labeled_image = encode_image(labeled_image_path)

        # For clarity, let's pass the original user request and 
        # the currently detected object list to the LLM. 
        # The LLM can propose a refined set of objects to detect.
        refine_messages = [
    {
    "role": "system",
    "content": """
        You are an AI system that refines detection queries. 
        You are provided with the outputs from an object detection model, along with the user's request and the objects from the user's request that were extracted and provided to the object detection model.
        Your task is to analyze whether the object detector has extracted the results properly to the user's request and, if not, refine the queries by generalizing concepts where possible.
        
        Important guidelines:
        1. If the detection results are already good, no need to refine. 
           - In that case, provide reasoning indicating no refinement was necessary and return the same list.
        2. If the detection results are poor or null, propose synonyms or more generic categories and explain why. Wherever possible, retain the singular version of the concept.
        3. Return your final answer as a JSON object with exactly two fields: "reasoning" and "refined_list".
           - "reasoning" is a short explanation of why you refined or didn't refine.
           - "refined_list" is a comma-separated list of object names that should be re-tried in detection.
        4. Output ONLY the JSON, and no other text.

        Below are some examples:

        EXAMPLE 1
        User's Request: Detect the teacup poodle
        Original concept: "Teacup poodle"

        Final output:
        {
          "reasoning": "The provided image does not have any detections for the concept of teacup poodle. The concept "teacup poodle" might be a very specific concept for the model to detect. This could  be refined to a more higher-level and generic concept like 'Dog',
          "refined_list": "dog"
        }

        EXAMPLE 2
        User's Request: Detect the sparkly stiletto shoe
        Original concept: "Sparkly stiletto shoe"

        Final output:
        {
          "reasoning": "The provided image does not specific detections that correspond for 'Sparkly stiletto shoe'. 'Sparkly stiletto shoe' might be too specific for the model. Refining to 'shoe', a more generic concept might increase the likelihood of detection.",
          "refined_list": "shoe"
        }

        EXAMPLE 3
        User's Request: Detect the hydrangea
        Original concept: "Hydrangea"

        Final output:
        {
          "reasoning": "No detections found for 'hydrangea'. The model might struggle with specific flower types. Refining to the more general concept 'flower' could yield better results.",
          "refined_list": "flower"
        }

        EXAMPLE 4
        User's Request: Detect the gourmet cheeseburger
        Original concept: "Gourmet cheeseburger"

        Final output:
        {
          "reasoning": "No detections observed for 'Gourmet cheeseburger'. 'Gourmet cheeseburger' might be too specific. Refining to 'hamburger' as it aligns with the detected object.",
          "refined_list": "hamburger"
        }

        EXAMPLE 5
        User's Request: Detect the red sports car
        Original concept: "Red sports car"

        Final output:
        {
          "reasoning": "The provided image does not have any reliable detections for 'red sports car'. Color-based detection might be challenging. Refining to the more general concept 'car' could improve detection.",
          "refined_list": "car"
        }

        Remember: 
        • If no refinement is needed (the concept is recognized well), explain that in the reasoning and return the same concept. 
        • When refinement is necessary, prioritize more generic or abstract categories that may be more reliably detected by the model.
        • Provide only the JSON. 
        • No extra commentary.
        """
        },

        {"role": "user", "content": [
            {"type": "text", "text": f"User's request: {user_request}\n Original Concepts for Detection: {original_concepts}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_labeled_image}", "detail": "high"}}
        ]}
        ]
        refine_response = self.vlm_tool.chat_completion(refine_messages, model=model, response_format={"type": "json_object"})
        
        refined_response_objects = json.loads(refine_response)["refined_list"].split(",")

        if not refined_response_objects:
            return []
        
        refined_list = [r.strip().lower() for r in refined_response_objects if r.strip()]
        return refined_list

def _validate_bboxes_with_llm(self, user_request, labeled_image_path, model="o1"):
        """
        Pass the labeled image to the LLM to filter bounding boxes 
        based on user request. Returns 'valid_numbers' list.
        """
        base64_labeled_image = encode_image(labeled_image_path)
        
        messages = [
            {"role": "system", "content": "You are an AI reviewing an object detection output.\n"
                                          "All detected objects have been marked with an arrow mapping to a corresponding number.\n"
                                          "The image contains arrows labeled with numbers pointing to specific objects.\n"
                                          "Your task is to identify the objects indicated by these arrows and determine whether each detected object is relevant to the user's query.\n"
                                          "For each numbered arrow:\n"
                                          "1. Identify the object being pointed to.\n"
                                          "2. Provide a brief description of the object (e.g., 'top-left cup with blue leaves', 'bottom-right cup with watermelon pattern', or 'background birdcage').\n"
                                          "3. Analyze whether the object is valid based on the context and the user's instructions.\n"
                                          "4. Provide a clear, step-by-step explanation for each object's validity decision.\n"                                          
                                          "Return a JSON object with the reasoning and list of valid numbers matching the user's request.\n"
                                          "Example output:\n"
                                          "{ \"reasoning\": <reasoning> , \"valid_numbers\": {object_num :\"object_name\"} }"
            },
            {"role": "user", "content": [
                {"type": "text", "text": f"The user's original request was: {user_request}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_labeled_image}", "detail": "high"}}
            ]
            }
        ]
        valid_numbers_json = self.vlm_tool.chat_completion(
            messages, 
            model=model,
            response_format={"type": "json_object"}
        )

        try:
            valid_numbers_data = json.loads(valid_numbers_json)
            return valid_numbers_data.get("valid_numbers", {})
        except json.JSONDecodeError:
            return []

  def run(self, image_path, user_request, do_critique=True):
        """
        Full pipeline:
        1. Extract objects from user request (LLM).
        2. Detect bounding boxes with that query.
        3. LLM-based validation step => filter bounding boxes.
        4. (Optional) Critique Step => refine the query if needed.
        5. Re-run detection with refined queries.
        6. Final LLM validation => final bounding boxes and annotation.
        """
        # ---------------------------------------------------
        # Step 1: initial user queries from request
        # ---------------------------------------------------
        objects_to_detect = self.vlm_tool.extract_objects_from_request(image_path, user_request, model=self.concept_detection_model)
        if not objects_to_detect:
            return None, "⚠️ No objects to detect or invalid request."

        # ---------------------------------------------------
        # Step 2: run detection with the initial user queries
        # ---------------------------------------------------
        detected_objects_final, labeled_image_path = self._run_detector(image_path, objects_to_detect)
        
        # ------------------------------------------------------
        # Step 3: Initial Critique and Object Concept Refinement
        # ------------------------------------------------------
        if do_critique:
            current_labels = ",".join(set([str(lbl) for _, lbl, _ in detected_objects_final]))

            refined_query_list = self._critique_and_refine_query(
                user_request=user_request,
                original_concepts=current_labels,
                labeled_image_path=labeled_image_path,
                objects_detected=current_labels,
                model=self.initial_critique_model
            )
            
            # If the refined list is empty or identical, we might skip re-running
            # But let's suppose we only re-run if we actually get a new set.
            if refined_query_list and set(refined_query_list) != set(objects_to_detect):
                # Re-run detection with refined query
                detected_objects_final, labeled_image_path = self._run_detector(image_path, refined_query_list)
                if not detected_objects_final:
                    return None, "No objects found for the initial query."
        
        # ---------------------------------------------------
        # Step 4: LLM-based critique
        # ---------------------------------------------------
        valid_numbers = self._validate_bboxes_with_llm(user_request, labeled_image_path,  model=self.final_critique_model)

        # filter bounding boxes
        if valid_numbers:
            filtered_objects = [(n, valid_numbers[str(n)], box) for (n, lbl, box) in detected_objects_final if str(n) in valid_numbers]
        else:
            filtered_objects = detected_objects_final

        # store them
        self.last_detection_bboxes = [x[-1] for x in filtered_objects]
        self.last_filtered_objects = filtered_objects


        # ---------------------------------------------------
        # Step 5: Produce final annotated image
        # ---------------------------------------------------
        final_img = draw_bounding_boxes(image_path, filtered_objects)

        final_text = (
            f"🔍 Validated objects: {', '.join(set(str(lbl) for _, lbl, _ in filtered_objects))}"
        )
        return final_img, final_text
