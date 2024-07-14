## https://medium.com/pythoneers/17-mindblowing-python-automation-scripts-i-use-everyday-523fb1eb9284

## 1. Speakify
import PyPDF2
import pyttsx3

# Open the PDF file (Enter Path To Your PDF)
file = open('story.pdf', 'rb')
readpdf = PyPDF2.PdfReader(file)

# Initialize text-to-speech engine
speaker = pyttsx3.init()
rate = speaker.getProperty('rate')   # Get current speaking rate
speaker.setProperty('rate', 200)

volume = speaker.getProperty('volume')
speaker.setProperty('volume', 1)  # Set volume level (0.0 to 1.0)

# Get and set a different voice
voices = speaker.getProperty('voices')
for voice in voices:
    if "english" in voice.name.lower() and "us" in voice.name.lower():
        speaker.setProperty('voice', voice.id)
        break
# Iterate over each page in the PDF
for pagenumber in range(len(readpdf.pages)):
    # Extract text from the page
    page = readpdf.pages[pagenumber]
    text = page.extract_text()
    
    # Use the speaker to read the text
    # speaker.say(text)
    # speaker.runAndWait()

# Save the last extracted text to an audio file (if needed)
speaker.save_to_file(text, 'story.mp3')
speaker.runAndWait()

# Stop the speaker
speaker.stop()

# Close the PDF file
file.close()

-----------------------------------------------------------------------------------------------------------------
## 2. TabTornado
import webbrowser
with open('links.txt') as file:
    links = file.readlines()
    for link in links: 
        webbrowser.open('link')

-----------------------------------------------------------------------------------------------------------------
## 3. PicFetcher
# Importing the necessary module and function
from simple_image_download import simple_image_download as simp 

# Creating a response object
response = simp.simple_image_download

## Keyword
keyword = "Dog"

# Downloading images
try:
    response().download(keyword, 20)
    print("Images downloaded successfully.")
except Exception as e:
    print("An error occurred:", e)

-----------------------------------------------------------------------------------------------------------------
## 4. PyInspector

import os
import subprocess

def analyze_code(directory):
    # List Python files in the directory
    python_files = [file for file in os.listdir(directory) if file.endswith('.py')]

    if not python_files:
        print("No Python files found in the specified directory.")
        return

    # Analyze each Python file using pylint and flake8
    for file in python_files:
        print(f"Analyzing file: {file}")
        file_path = os.path.join(directory, file)

        # Run pylint
        print("\nRunning pylint...")
        pylint_command = f"pylint {file_path}"
        subprocess.run(pylint_command, shell=True)

        # Run flake8
        print("\nRunning flake8...")
        flake8_command = f"flake8 {file_path}"
        subprocess.run(flake8_command, shell=True)

if __name__ == "__main__":
    directory = r"C:\Users\abhay\OneDrive\Desktop\Part7"
    analyze_code(directory)

-----------------------------------------------------------------------------------------------------------------
## 5.DataDummy
import pandas as pd
from faker import Faker
import random

fake = Faker()

def generate_fake_data(num_entries=10):
    data = []

    for _ in range(num_entries):
        entry = {
            "Name": fake.name(),
            "Address": fake.address(),
            "Email": fake.email(),
            "Phone Number": fake.phone_number(),
            "Date of Birth": fake.date_of_birth(minimum_age=18, maximum_age=65).strftime("%Y-%m-%d"),
            "Random Number": random.randint(1, 100),
            "Job Title": fake.job(),
            "Company": fake.company(),
            "Lorem Ipsum Text": fake.text(),
        }
        data.append(entry)

    return pd.DataFrame(data)

if __name__ == "__main__":
    num_entries = 10  # You can adjust the number of entries you want to generate
    fake_data_df = generate_fake_data(num_entries)

    

## Dataframe with Fake Data
fake_data_df

-----------------------------------------------------------------------------------------------------------------
## 6. BgBuster
from rembg import remove 
from PIL import Image

## Path for input and output image
input_img = 'monkey.jpg'
output_img = 'monkey_rmbg.png'

## loading and removing background
inp = Image.open(input_img)
output = remove(inp)

## Saving background removed image to same location as input image
output.save(output_img)

-----------------------------------------------------------------------------------------------------------------
## 7. MemoryMate
from win10toast import ToastNotifier
import time

toaster = ToastNotifier()

def set_reminder():
    reminder_header = input("What would you like me to remember?\n")
    related_message = input("Related Message:\n")
    time_minutes = float(input("In how many minutes?\n"))

    time_seconds = time_minutes * 60

    print("Setting up reminder...")
    time.sleep(2)
    print("All set!")

    time.sleep(time_seconds)

    toaster.show_toast(
        title=f"{reminder_header}",
        msg=f"{related_message}",
        duration=10,
        threaded=True
    )

    while toaster.notification_active():
        time.sleep(0.005)

if __name__ == "__main__":
    set_reminder()

-----------------------------------------------------------------------------------------------------------------
## 8. MonitorMax 
import psutil
import time
from win10toast import ToastNotifier

# Initialize the ToastNotifier object
toaster = ToastNotifier()

# Set the threshold values for CPU usage, memory usage, GPU usage, and battery level
cpu_threshold = 40  # Percentage
memory_threshold = 40  # Percentage
gpu_threshold = 40  # Percentage
battery_threshold = 100  # Percentage

# Infinite loop to continuously monitor system resources
while True:
    try:
        # Get system resource information
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        gpu_usage = psutil.virtual_memory().percent
        battery = psutil.sensors_battery()

        # Check CPU usage
        if cpu_usage >= cpu_threshold:
            message = f"CPU usage is high: {cpu_usage}%"
            toaster.show_toast("Resource Alert", message, duration=10)

        # Check memory usage
        if memory_usage >= memory_threshold:
            message = f"Memory usage is high: {memory_usage}%"
            toaster.show_toast("Resource Alert", message, duration=10)

        # Check GPU usage
        if gpu_usage >= gpu_threshold:
            message = f"GPU usage is high: {gpu_usage}%"
            toaster.show_toast("Resource Alert", message, duration=10)

        # Check battery level
        if battery is not None and battery.percent <= battery_threshold and not battery.power_plugged:
            message = f"Battery level is low: {battery.percent}%"
            toaster.show_toast("Battery Alert", message, duration=10)

        # Wait for 5 minutes before checking the resources again
        time.sleep(300)

    except Exception as e:
        print("An error occurred:", str(e))
        break

-----------------------------------------------------------------------------------------------------------------
## 9. EmailBlitz
import smtplib 
from email.message import EmailMessage
import pandas as pd

def send_email(remail, rsubject, rcontent):
    email = EmailMessage()                          ## Creating a object for EmailMessage
    email['from'] = 'The Pythoneer Here'            ## Person who is sending
    email['to'] = remail                            ## Whom we are sending
    email['subject'] = rsubject                     ## Subject of email
    email.set_content(rcontent)                     ## content of email
    with smtplib.SMTP(host='smtp.gmail.com',port=587)as smtp:     
        smtp.ehlo()                                 ## server object
        smtp.starttls()                             ## used to send data between server and client
        smtp.login(SENDER_EMAIL,SENDER_PSWRD)       ## login id and password of gmail
        smtp.send_message(email)                    ## Sending email
        print("email send to ",remail)              ## Printing success message



if __name__ == '__main__':
    df = pd.read_excel('list.xlsx')
    length = len(df)+1

    for index, item in df.iterrows():
        email = item[0]
        subject = item[1]
        content = item[2]

        send_email(email,subject,content)

-----------------------------------------------------------------------------------------------------------------
## 10. ClipSaver
import tkinter as tk
from tkinter import ttk
import pyperclip

def update_listbox():
    new_item = pyperclip.paste()
    if new_item not in X:
        X.append(new_item)
        listbox.insert(tk.END, new_item)
        listbox.insert(tk.END, "----------------------")
    listbox.yview(tk.END)
    root.after(1000, update_listbox)

def copy_to_clipboard(event):
    selected_item = listbox.get(listbox.curselection())
    if selected_item:
        pyperclip.copy(selected_item)

X = []

root = tk.Tk()
root.title("Clipboard Manager")
root.geometry("500x500")
root.configure(bg="#f0f0f0")

frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(padx=10, pady=10)

label = tk.Label(frame, text="Clipboard Contents:", bg="#f0f0f0")
label.grid(row=0, column=0)

scrollbar = tk.Scrollbar(root)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

listbox = tk.Listbox(root, width=150, height=150, yscrollcommand=scrollbar.set)
listbox.pack(pady=10)
scrollbar.config(command=listbox.yview)

update_listbox()

listbox.bind("<Double-Button-1>", copy_to_clipboard)

root.mainloop()

-----------------------------------------------------------------------------------------------------------------
## 11. BriefBot
from transformers import BartForConditionalGeneration, BartTokenizer
import requests
from bs4 import BeautifulSoup

## Function to summarize article
def summarize_article(article_text, max_length=150):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer.encode("summarize: " + article_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

## Function to scrape content of the aricle 
def scrape_webpage(url):
    try:
        headers = { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"} ##
        response = requests.get(url, headers=headers)        
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        all_text = soup.get_text(separator='\n', strip=True)
        return all_text
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    webpage_url = "https://thehackernews.com/2024/07/new-ransomware-group-exploiting-veeam.html" ## Sample URL for Testing The Script
    webpage_text = scrape_webpage(webpage_url)

    if webpage_text:
        summary = summarize_article(webpage_text)
        print("\nSummarized Article:")
        print(summary)
    else:
        print("Webpage scraping failed.")

-----------------------------------------------------------------------------------------------------------------
## 12. SpellGuard
## Installing Library
!pip install lmproof

## Importing Library
import lmproof as lm

## Function for proofreading 
def Proofread(text):
    proof = lm.load("en")
    error_free_text = proof.proofread(text)
    return error_free_text

## Sample Text
TEXT = '' ## Place sample Text here

## Function Call 
Print(Proofread(TEXT))

-----------------------------------------------------------------------------------------------------------------
## 13. LinkStatus
import csv
import requests
import pprint

def get_status(website):
    try:
        status = requests.get(website).status_code
        return "Working" if status == 200 else "Error 404"
    except:
        return "Connection Failed!!"

def main():
    with open("sites.txt", "r") as fr:
        websites = [line.strip() for line in fr]
    
    web_status_dict = {website: get_status(website) for website in websites}
    
    pprint.pprint(web_status_dict)
    
    # Write results to CSV file
    with open("web_status.csv", "w", newline='') as csvfile:
        fieldnames = ["Website", "Status"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for website, status in web_status_dict.items():
            writer.writerow({"Website": website, "Status": status})

        print("Data Uploaded to CSV File!!")

if __name__ == "__main__":
    main()

-----------------------------------------------------------------------------------------------------------------
## 14. DailyDigest
import pyttsx3 
import requests

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def trndnews(): 
    url = " http://newsapi.org/v2/top-headlines?country=us&apiKey=GET_YOUR_OWN"
    page = requests.get(url).json() 
    article = page["articles"] 
    results = [] 
    for ar in article: 
        results.append(ar["title"]) 
    for i in range(len(results)): 
        print(i + 1, results[i]) 
    speak(results)
    
trndnews() 
 
''' Run The Script To Get The Top Headlines From USA'''

-----------------------------------------------------------------------------------------------------------------
## 15. QRGenie
import qrcode

def generate_qr_code(link, filename):
    """Generates a QR code for the given link and saves it as filename."""

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(link)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save("profile.png")

if __name__ == "__main__":
    generate_qr_code("https://abhayparashar31.medium.com/", "Profile.png")
    print(f"QR code saved successfully!!")

-----------------------------------------------------------------------------------------------------------------
## 16. ShrinkLink
import pyshorteners
def generate_short_url(long_url):
  s = pyshorteners.Shortener()
  return s.tinyurl.short(long_url)

long_url = input('Paste Long URl \n')
short_url = generate_short_url(long_url)
print(short_url)

-----------------------------------------------------------------------------------------------------------------
## 17. CaptureIt
import cv2
import numpy as np
import pyautogui

SCREEN_SIZE = tuple(pyautogui.size())
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
fps = 12.0
record_seconds = 20
out = cv2.VideoWriter("video.mp4", fourcc, fps, SCREEN_SIZE)

for _ in range(int(record_seconds * fps)):
    img = pyautogui.screenshot(region=(0, 0, 500, 900))
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out.write(frame)
    cv2.imshow("video frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
out.release()

-----------------------------------------------------------------------------------------------------------------
## 18. H2OReminder
import time
from plyer import notification

if __name__ == "__main__":
    while True:
        notification.notify(
            title="Please Drink Water",
            message="The U.S. National Academies of Sciences, Engineering, and Medicine determined that an adequate daily fluid intake is: About 15.5 cups (3.7 liters) of fluids a day for men. About 11.5 cups (2.7 liters) of fluids a day for women. Keep Drinking Water !!!",
            app_icon="./Desktop-drinkWater-Notification/icon.ico",
            timeout=12
        )
        time.sleep(1800)  ## Change it according to your water breaks!!!
