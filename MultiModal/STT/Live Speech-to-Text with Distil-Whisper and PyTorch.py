### Have to check what is Distil-whisper :  https://medium.com/analytics-vidhya/live-speech-to-text-with-distil-whisper-and-pytorch-4f5c1e494667

git clone https://github.com/yourusername/distil-whisper-live.git
cd distil-whisper-live

python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

[microphone]
name = "Your Microphone Name"
sample_rate = 16000  # Adjust as needed
chunk_size = 1024

docker run -p 6379:6379 redis

python main.py
