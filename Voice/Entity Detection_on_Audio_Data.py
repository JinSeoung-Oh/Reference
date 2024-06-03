## From https://medium.datadriveninvestor.com/performing-named-entity-recognition-on-audio-data-73f45c1b9739

pip install requests

API_key = "  "
endpoint = "https://api.assemblyai.com/v2/transcript"

json = {
    "audio_url": upload_url,
    "entity_detection": True,
    "speaker_labels": True
}

headers = {
    "authorization": API_key,
    "content-type": "application/json"
}

response = requests.post(endpoint, json=json, headers=headers)

response.json()

response_id = response.json()['id']

endpoint = f"https://api.assemblyai.com/v2/transcript/{response_id}"

headers = {
    "authorization": API_key,
}
response = requests.get(endpoint, headers=headers)

response.json()

current_status = "queued"
response_id = response.json()['id']
endpoint = f"https://api.assemblyai.com/v2/transcript/{response_id}"
headers = {
    "authorization": API_key,
}

while current_status not in ("completed", "error"):
    
    response = requests.get(endpoint, headers=headers)
    current_status = response.json()['status']
    
    if current_status in ("completed", "error"):
        print(response)
    else:
        sleep(10)
        
current_status
response.json()


