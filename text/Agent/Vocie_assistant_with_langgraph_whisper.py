### From https://ai.plainenglish.io/from-no-code-to-full-control-how-i-rebuilt-elevenlabs-ai-agent-with-langgraph-and-whisper-from-fd8fe1a112ee

### 1. Speech-to-Text: Whisper
openai_client.audio.transcriptions.create(
   model="whisper-1", 
   file=audio_bytes,
)

-------------------------------------------------------------------
### Conversational AI Agent: LangGraph
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

-------------------------------------------------------------------
### Implementing Memory Functionality
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

-------------------------------------------------------------------
### Defining the Tool: Get Date and Time
from langchain_core.tools import tool
import requests

@tool
def get_date_and_time() -> dict:
    """
    Call tool to fetch the current date and time from an API.
    """
    try:
        response = requests.get("https://timeapi.io/api/Time/current/zone?timeZone=Europe/Brussels")
        response.raise_for_status()
        data = response.json()
        return {"date_time": data["dateTime"], "timezone": data["timeZone"]}
    except requests.RequestException as e:
        return {"error": str(e)}

-------------------------------------------------------------------
### Setting Up the Language Model
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools([get_date_and_time])

-------------------------------------------------------------------
### Building the Conversation Flow
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableConfig

def chatbot(state: State, config: RunnableConfig):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# initiate the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge("chatbot", "chatbot")
graph = graph_builder.compile(checkpointer=memory)

-------------------------------------------------------------------
### Text-to-Speech: ElevenLabs API
import os
from elevenlabs import play, VoiceSettings
from elevenlabs.client import ElevenLabs
from langgraph.graph import  MessagesState

# Initialize ElevenLabs client
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))

def play_audio(state: MessagesState):
    """Plays the audio response from the remote graph with ElevenLabs."""
    
    # Response from the agent
    response = state['messages'][-1]
    
    # Prepare text by replacing ** with empty strings
    cleaned_text = response.content.replace("**", "")
    
    # Call text_to_speech API with turbo model for low latency
    response = elevenlabs_client.text_to_speech.convert(
        voice_id="YUdpWWny7k5yb4QCeweX",  # Adam pre-made voice
        output_format="mp3_22050_32",
        text=cleaned_text,
        model_id="eleven_turbo_v2_5", 
        language_code="nl",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )
    
    # Play the audio back
    play(response)

#### Building the AI Voicebot Step by Step
"""
## Step 1: Building a Basic Conversational Chatbot
The first step is to create a basic chatbot that can remember messages and fetch the current date and time. The code above for state management, memory, tool definition, language model setup, and conversation flow comprises this initial phase.

Step 2: Enhancing the Chatbot with Speech Capabilities
The next step is to add audio input and output nodes to the chatbot workflow.
"""

-------------------------------------------------------------------
### Capturing Audio Input with Whisper
import io
import threading
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

from openai import OpenAI

from langgraph.graph import  MessagesState, HumanMessage

# Initialize OpenAI client
openai_client = OpenAI()


# Audio settings
SAMPLE_RATE = 16000  # Adequate for human voice frequency
THRESHOLD = 500  # Silence detection threshold (adjust if needed)
SILENCE_DURATION = 1.5  # Duration (seconds) of silence before stopping
CHUNK_SIZE = 1024  # Number of frames per audio chunk

def record_audio_until_silence(state: MessagesState):
    """Waits for the user to start speaking, records the audio, and stops after detecting silence."""

    audio_data = []  # List to store audio chunks
    silent_chunks = 0  # Counter for silent chunks
    started_recording = False  # Flag to track if recording has started

    def record_audio():
        """Continuously records audio, waiting for the user to start speaking."""
        nonlocal silent_chunks, audio_data, started_recording

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
            print("Waiting for you to start speaking...")

            # Keep waiting indefinitely for the user to start talking
            while not started_recording:
                audio_chunk, _ = stream.read(CHUNK_SIZE)
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)

                # Check if there is voice input
                if np.abs(audio_array).max() > THRESHOLD:
                    started_recording = True
                    print("Voice detected. Recording started.")
                    audio_data.append(audio_chunk)
                    break

            # Start recording once voice is detected
            while True:
                audio_chunk, _ = stream.read(CHUNK_SIZE)
                audio_data.append(audio_chunk)
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)

                # Detect silence after user has finished speaking
                if np.abs(audio_array).max() < THRESHOLD:
                    silent_chunks += 1
                else:
                    silent_chunks = 0  # Reset if sound is detected

                # Stop if silence is detected for the specified duration
                if silent_chunks > (SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE):
                    print("Silence detected. Stopping recording.")
                    break

    # Start recording in a separate thread
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()
    recording_thread.join()

    # Stack all audio chunks into a single NumPy array and write to file
    audio_data = np.concatenate(audio_data, axis=0)
    
    # Convert to WAV format in-memory
    audio_bytes = io.BytesIO()
    write(audio_bytes, SAMPLE_RATE, audio_data)  # Use scipy's write function to save to BytesIO
    audio_bytes.seek(0)  # Go to the start of the BytesIO buffer
    audio_bytes.name = "audio.wav"  # Set a filename for the in-memory file

    # Transcribe via Whisper
    transcription = openai_client.audio.transcriptions.create(
       model="whisper-1", 
       file=audio_bytes,
       language='nl'
    )

    # Print the transcription
    print("Here is the transcription:", transcription.text)

    # Write to messages
    return {"messages": [HumanMessage(content=transcription.text)]}

-------------------------------------------------------------------
from langgraph.graph import StateGraph, MessagesState, END, START

# Define parent graph
builder = StateGraph(MessagesState)

# Add remote graph directly as a node
builder.add_node("audio_input", record_audio_until_silence)
builder.add_node("agent", graph)
builder.add_node("audio_output", play_audio)
builder.add_edge(START, "audio_input")
builder.add_edge("audio_input", "agent")
builder.add_edge("agent", "audio_output")
builder.add_edge("audio_output", "audio_input")
audio_graph = builder.compile(checkpointer=memory)

-------------------------------------------------------------------
## Step 3: Testing the Voice Assistant
from langchain_core.messages import convert_to_messages
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "1"}}

for chunk in audio_graph.stream({"messages":HumanMessage(content="Follow the user's instructions:")}, stream_mode="values", config=config):
    chunk["messages"][-1].pretty_print()

-------------------------------------------------------------------
""" 
Conclusion
The article concludes by reflecting on the rewarding experience of building an AI voicebot from scratch. By integrating:

Whisper for speech-to-text conversion,
LangGraph for creating a conversational agent with memory and tool integration, and
ElevenLabs’ API for converting text responses into natural, lifelike speech,
the author demonstrates how building a custom-coded solution provides deeper insights, enhanced control, and endless possibilities for customization.

This summary includes all the code excerpts exactly as given in the original text, ensuring that the reader sees the complete implementation details without any modifications or additional code.
"""
