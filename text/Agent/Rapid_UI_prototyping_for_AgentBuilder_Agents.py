### From https://medium.com/google-cloud/rapid-ui-prototyping-for-agentbuilder-agents-with-google-mesop-and-cloud-run-6d260f15ac6d
from google.cloud import dialogflowcx_v3beta1 as dialogflow
from google.cloud.dialogflowcx_v3beta1 import types
import mesop as me
import mesop.labs as mel
import uuid

agent_id = "11272e0f-348b-XXXXXXXXXXXXXXXXXX"
project_id = "genai-app-builder"
location = "europe-west2"

session_id = uuid.uuid4()

session_path = f"projects/{project_id}/locations/{location}/agents/{agent_id}/sessions/{session_id}"
api_endpoint = f"{location}-dialogflow.googleapis.com"

client_options = {"api_endpoint": api_endpoint}
client = dialogflow.services.sessions.SessionsClient(client_options=client_options)

 def send_message(in_message: str, in_language: str = "en_US"):
        msg = in_message
        text_input = types.TextInput(text=msg)
        query_input = types.QueryInput(text=text_input, language_code=in_language)
        
        request = types.DetectIntentRequest(
            session=session_path, query_input=query_input
        )
        response = client.detect_intent(request=request)

        print(response.query_result)
        return response.query_result.response_messages[0].text.text[0]
   
me.colab_run()

@me.page(path="/chat")
def chat():
  mel.chat(transform)

def transform(prompt: str, history: list[mel.ChatMessage]) -> str:
  return send_message(prompt,agent_language)

me.colab_show(path="/chat")

----------------------------------------------------------------------------------------------------
%%writefile deployment/main.py
import mesop as me
import mesop.labs as mel
import requests
import os
from google.cloud import dialogflowcx_v3beta1 as dialogflow
import uuid
from google.cloud.dialogflowcx_v3beta1 import types
from dataclasses import field

AGENT_PROJECT= os.environ.get('AGENT_PROJECT')
AGENT_LOCATION = os.environ.get('AGENT_LOCATION')
AGENT_ID = os.environ.get('AGENT_ID')
AGENT_LANGUAGE = os.environ.get('AGENT_LANGUAGE')

project_id = AGENT_PROJECT
location = AGENT_LOCATION
agent_id = AGENT_ID

api_endpoint = f"{location}-dialogflow.googleapis.com"
client_options = {"api_endpoint": api_endpoint}
client = dialogflow.services.sessions.SessionsClient(client_options=client_options)


@me.stateclass
class State:
  session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@me.page(path="/")
def chat():
  state = me.state(State)
    ## send message to agentbuilder
  def send_message(in_message: str, state: State, in_language: str = "en_US"):
          session_id = state.session_id
          session_path = f"projects/{project_id}/locations/{location}/agents/{agent_id}/sessions/{session_id}"
          msg = in_message
          language_code = in_language
          text_input = types.TextInput()
          text_input = types.TextInput(text=msg)
          query_input = types.QueryInput(text=text_input, language_code=in_language)
          request = types.DetectIntentRequest(
              session=session_path, query_input=query_input
          )
          response = client.detect_intent(request=request)

          print(response.query_result)
          return response.query_result.response_messages[0].text.text[0]


  ## prepare message handler for mesop
  def transform(prompt: str, history: list[mel.ChatMessage]) -> str:
    return send_message(prompt, state, AGENT_LANGUAGE)

  mel.chat(transform)
