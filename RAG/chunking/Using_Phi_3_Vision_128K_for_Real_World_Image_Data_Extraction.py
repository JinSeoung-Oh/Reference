### From https://ai.gopubby.com/using-phi-3-vision-128k-for-real-world-image-data-extraction-from-invoices-to-landmarks-e372303f2922

####### Extract data from the invoice
from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor

class Phi3VisionModel:
    def __init__(self, model_id="microsoft/Phi-3-vision-128k-instruct", device="cuda"):
        self.model_id = model_id
        self.device = device
        self.model = self.load_model()
        self.processor = self.load_processor()
    
    def load_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            device_map="auto", 
            torch_dtype="auto", 
            trust_remote_code=True, 
            _attn_implementation='flash_attention_2'
        ).to(self.device)
    
    def load_processor(self):
        return AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
    
    def extract_data(self, image_path, prompt):
        image = Image.open(image_path)
        formatted_prompt = f"<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n"
        inputs = self.processor(formatted_prompt, [image], return_tensors="pt").to(self.device)
        
        generation_args = {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "do_sample": False
        }
        
        output_ids = self.model.generate(**inputs, **generation_args)
        output_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return response


# Initialize the model
phi_model = Phi3VisionModel()

######### Extract data from the invoice
invoice_image_path = "invoice.png"
invoice_prompt = "Extract the bill number, date, items with their quantities and prices, total amount, and tax details from this invoice."
invoice_response = phi_model.extract_data(invoice_image_path, invoice_prompt)

print("Invoice Extraction Result:", invoice_response)

####### Parsing Tabular Data
# Extract tabular data
table_image_path = "tabular_data.png"
table_prompt = "Extract the data from this table into a structured format like CSV or JSON, including fields for age, job, marital status, education, balance, and housing."
table_response = phi_model.extract_data(table_image_path, table_prompt)
print("Table Data Extraction Result:", table_response)

######### Understanding Graphs
graph_image_path = "graph_cycle.png"
graph_prompt = "Describe the key elements of this graph, including the labels on the axes, the data points, and any annotations and what is this graph about, and explain.."
graph_response = phi_model.extract_data(graph_image_path, graph_prompt)
print("Graph Understanding Result:", graph_response)

######### Analyzing a Traffic Scene
# Analyze traffic scene
traffic_image_path = "traffic_scene.png"
traffic_prompt = "Describe the traffic situation, including the number of vehicles, types of vehicles, and any visible traffic signs or landmarks."
traffic_response = phi_model.extract_data(traffic_image_path, traffic_prompt)
print("Traffic Scene Analysis Result:", traffic_response)

######### Identifying a Landmark from an Image
# Identify location from landmark
landmark_image_path = "tower_image.png"
landmark_prompt = "Identify the landmark in this image and provide details about its location."
landmark_response = phi_model.extract_data(landmark_image_path, landmark_prompt)
print("Landmark Identification Result:", landmark_response)


