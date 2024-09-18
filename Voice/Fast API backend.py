## https://ai.gopubby.com/from-prescription-to-voice-a-python-solution-to-help-service-elderly-and-visually-impaired-5bb6b108b00d

python -m venv .venv
.venv\Scripts\activate

pytesseract==0.3.10
opencv_python==4.7.0.68
Pillow[all]
fastapi==0.103.2
google-cloud-texttospeech[all]
uvicorn

# Enter the code below in your outils.py

import cv2
import numpy as np
import os
import google.cloud
from google.cloud import texttospeech

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_credentials.json"


def preprocess_image(img):
    # Preprocessing image for better visibility
    gray = cv2.cvtColor(np.array(img), 
                        cv2.COLOR_BGR2GRAY)   
    resized = cv2.resize(gray, 
                         None, 
                         fx=2, 
                         fy=2, 
                         interpolation=cv2.INTER_LINEAR) 
    processed_img = cv2.adaptiveThreshold(resized, 
                                          255, 
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 
                                          61, 
                                          11) 
    return processed_img


def text2speech(message_text):
    #converting text to speech
    client = texttospeech.TextToSpeechClient()                                         #instantiate the client object
    synthesis_input = texttospeech.SynthesisInput(text = message_text)                  #set the input text to be synthesized
    voice = texttospeech.VoiceSelectionParams(language_code='en-US',                    #Build voice &
                                        name='en-US-wavenet-C',
                                        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)  #specify audio file type to return
    
    response = client.synthesize_speech(input=synthesis_input,                               #perform the text to speech request
                                 voice=voice,
                                 audio_config=audio_config)
    
    with open('output.mp3', 'wb') as out:                                                  #write the response to an output file
        out.write(response.audio_content)
    print('Audio content written to output file "output.mp3"')


#### Prescription parser
# Enter the code below in your parser_prescription.py

import re
import src.outils

class PrescriptionParser():
    def __init__(self, text):
        self.text = text
    
    # Converting to speech    
    def parse(self):
        prescription_name = self.get_field('prescription_name')
        dosage = (self.get_field('dosage').replace("GIVE", "TAKE")).lower().replace('\n\n', '')
        refills = self.get_field('refills')
        expirydate = self.get_field('expirydate')
        
        message_text = f'Hello, as prescription for the drug {prescription_name}, {dosage}. It can be refilled {refills} times, on or before {expirydate}.'
        
        speech = src.outil.text2speech(message_text)
        
        return speech
                       
    # Getting the fields
    def get_field(self, field_name):
        pattern = ''
        flags = 0
        
        pattern_dict = {
            'prescription_name' : {'pattern': '(.*)Drug', 'flags' : 0},
            'dosage' : {'pattern': 'Netcare[^\n]*(.*)All', 'flags' : re.DOTALL},
            'refills' : {'pattern': 'Refilis:(.*)', 'flags' : 0},
            'expirydate' : {'pattern': 'Refills.Expire:(.*)', 'flags' : 0}
        }
        
        pattern_object = pattern_dict.get(field_name)
        if pattern_object:
            matches = re.findall(pattern_object['pattern'], self.text, flags=pattern_object['flags'])
            if len(matches) > 0:
                return matches[0].strip()



###  Defining the consolidated extract function
# Enter the code below in your extractor.py

from PIL import Image
import pytesseract
import src.outils

from src.loggings import logger
from src.parser_prescription import PrescriptionParser

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract(file_path):
    # extracting text from the image
    if file_path.endswith('.jpg'):
        img = Image.open(file_path)
        processed_image = src.outil.preprocess_image(img)
        text = pytesseract.image_to_string(processed_image, lang='eng')

        logger.info("Text message created")

    # extracting fields from the text and converting to speech  
        output_voice = PrescriptionParser(text).parse()

        logger.info("Voice message created and saved to file")

    else:
        raise Exception(f"Invalid file format")

    return output_voice


### Logging
# Enter this code into the __init__.py file of your loggings directory

import os
import sys
import logging

logging_str = "[%(asctime)s]: %(levelname)s: %(module)s: %(message)s]"
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("labelreaderLogger")


### Writing the Fast API server
# Enter thse in your main.py file

from src.loggings import logger
from fastapi import FastAPI, Form, UploadFile, File
import uvicorn
from src.extractor import extract
import uuid
import os
import sys
from src.exception import CustomException
from PIL import Image

app = FastAPI()


@app.post("/speech_from_doc")
def speech_from_doc(file: UploadFile):     # UploadFile (specific to FastAPI) is the data type of the file. (...) means not passing any value

    contents = file.file.read()   # read the content of the file from Postman

    # save the temporary image file from Postman to a temporary location (uploads) for the extractor to use
    file_path = "uploads/" + str(uuid.uuid4()) + ".jpg"  #use uuid module to attach a unique string to the file name. You dont want to overwrite the test file

    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        data = extract(file_path)
    except Exception as e:
        raise CustomException(e, sys)

    # delete the temporary image file after each run, that is, once the data is extracted
    if os.path.exists(file_path):
        os.remove(file_path)

    logger.info("Voice message created and saved to file")

    return data


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


### Custom exception
# Enter this into exceptions.py

import sys

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()   #error_detail.exc_info() has 3 outputs but we are interested in the 3rd one with details of the error
    file_name = exc_tb.tb_frame.f_code.co_filename  #we get the file name from exc_tb
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message


