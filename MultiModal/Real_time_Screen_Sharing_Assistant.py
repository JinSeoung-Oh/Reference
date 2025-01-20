### From https://levelup.gitconnected.com/build-a-real-time-screen-sharing-assistant-with-gemini-2-0-multimodal-live-api-2624fa9b0c43

## pip install --upgrade google-genai==0.3.0##
import asyncio
import json
import os
import websockets
from google import genai
import base64

# Load API key from environment
os.environ['GOOGLE_API_KEY'] = ''
MODEL = "gemini-2.0-flash-exp"  # use your model ID

client = genai.Client(
  http_options={
    'api_version': 'v1alpha',
  }
)

async def main() -> None:
    async with websockets.serve(gemini_session_handler, "localhost", 9083):
        print("Running websocket server localhost:9083...")
        await asyncio.Future()  # Keep the server running indefinitely


if __name__ == "__main__":
    asyncio.run(main())

---------------------------------------------------------------------------------------------
async def gemini_session_handler(client_websocket: websockets.WebSocketServerProtocol):
    """Handles the interaction with Gemini API within a websocket session.

    Args:
        client_websocket: The websocket connection to the client.
    """
    try:
        config_message = await client_websocket.recv()
        config_data = json.loads(config_message)
        config = config_data.get("setup", {})
        config["system_instruction"] = """You are a helpful assistant for screen sharing sessions. Your role is to: 
                                        1) Analyze and describe the content being shared on screen 
                                        2) Answer questions about the shared content 
                                        3) Provide relevant information and context about what's being shown 
                                        4) Assist with technical issues related to screen sharing 
                                        5) Maintain a professional and helpful tone. Focus on being concise and clear in your responses."""     

        async with client.aio.live.connect(model=MODEL, config=config) as session:
            print("Connected to Gemini API")

            async def send_to_gemini():
                """Sends messages from the client websocket to the Gemini API."""
                try:
                  async for message in client_websocket:
                      try:
                          data = json.loads(message)
                          if "realtime_input" in data:
                              for chunk in data["realtime_input"]["media_chunks"]:
                                  if chunk["mime_type"] == "audio/pcm":
                                      await session.send({"mime_type": "audio/pcm", "data": chunk["data"]})
                                      
                                  elif chunk["mime_type"] == "image/jpeg":
                                      await session.send({"mime_type": "image/jpeg", "data": chunk["data"]})
                                      
                      except Exception as e:
                          print(f"Error sending to Gemini: {e}")
                  print("Client connection closed (send)")
                except Exception as e:
                     print(f"Error sending to Gemini: {e}")
                finally:
                   print("send_to_gemini closed")



            async def receive_from_gemini():
                """Receives responses from the Gemini API and forwards them to the client, looping until turn is complete."""
                try:
                    while True:
                        try:
                            print("receiving from gemini")
                            async for response in session.receive():
                                if response.server_content is None:
                                    print(f'Unhandled server message! - {response}')
                                    continue

                                model_turn = response.server_content.model_turn
                                if model_turn:
                                    for part in model_turn.parts:
                                        if hasattr(part, 'text') and part.text is not None:
                                            await client_websocket.send(json.dumps({"text": part.text}))
                                        elif hasattr(part, 'inline_data') and part.inline_data is not None:
                                            print("audio mime_type:", part.inline_data.mime_type)
                                            base64_audio = base64.b64encode(part.inline_data.data).decode('utf-8')
                                            await client_websocket.send(json.dumps({
                                                "audio": base64_audio,
                                            }))
                                            print("audio received")

                                if response.server_content.turn_complete:
                                    print('\n<Turn complete>')
                        except websockets.exceptions.ConnectionClosedOK:
                            print("Client connection closed normally (receive)")
                            break  # Exit the loop if the connection is closed
                        except Exception as e:
                            print(f"Error receiving from Gemini: {e}")
                            break 

                except Exception as e:
                      print(f"Error receiving from Gemini: {e}")
                finally:
                      print("Gemini connection closed (receive)")


            # Start send loop
            send_task = asyncio.create_task(send_to_gemini())
            # Launch receive loop as a background task
            receive_task = asyncio.create_task(receive_from_gemini())
            await asyncio.gather(send_task, receive_task)


    except Exception as e:
        print(f"Error in Gemini session: {e}")
    finally:
        print("Gemini session closed.")
---------------------------------------------------------------------------------------------

async function startScreenShare() {
            try {
                stream = await navigator.mediaDevices.getDisplayMedia({
                    video: {
                        width: { max: 640 },
                        height: { max: 480 },
                    },
                });

                video.srcObject = stream;
                await new Promise(resolve => {
                    video.onloadedmetadata = () => {
                        console.log("video loaded metadata");
                        resolve();
                    }
                });

            } catch (err) {
                console.error("Error accessing the screen: ", err);
            }
        }

---------------------------------------------------------------------------------------------

function captureImage() {
            if (stream && video.videoWidth > 0 && video.videoHeight > 0 && context) {
                canvas.width = 640;
                canvas.height = 480;
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL("image/jpeg").split(",")[1].trim();
                currentFrameB64 = imageData;
            }
            else {
                console.log("no stream or video metadata not loaded");
            }
        }
---------------------------------------------------------------------------------------------

window.addEventListener("load", async () => {
  await startScreenShare();
  setInterval(captureImage, 3000);
  connect();
});
---------------------------------------------------------------------------------------------

async function initializeAudioContext() {
            if (initialized) return;

            audioInputContext = new (window.AudioContext ||
                window.webkitAudioContext)({
                sampleRate: 24000
            });
            await audioInputContext.audioWorklet.addModule("pcm-processor.js");
            workletNode = new AudioWorkletNode(audioInputContext, "pcm-processor");
            workletNode.connect(audioInputContext.destination);
            initialized = true;
        }
---------------------------------------------------------------------------------------------

 function sendVoiceMessage(b64PCM) {
            if (webSocket == null) {
                console.log("websocket not initialized");
                return;
            }

            payload = {
                realtime_input: {
                    media_chunks: [{
                        mime_type: "audio/pcm",
                        data: b64PCM,
                    },
                    {
                        mime_type: "image/jpeg",
                        data: currentFrameB64,
                    },
                    ],
                },
            };

            webSocket.send(JSON.stringify(payload));
            console.log("sent: ", payload);
        }
---------------------------------------------------------------------------------------------

function receiveMessage(event) {
            const messageData = JSON.parse(event.data);
            const response = new Response(messageData);

            if (response.text) {
                displayMessage("GEMINI: " + response.text);
            }
            if (response.audioData) {
                injestAudioChuckToPlay(response.audioData);
            }
        }
---------------------------------------------------------------------------------------------

function sendInitialSetupMessage() {

            console.log("sending setup message");
            setup_client_message = {
                setup: {
                    generation_config: { response_modalities: ["AUDIO"] },
                },
            };

            webSocket.send(JSON.stringify(setup_client_message));
        }
