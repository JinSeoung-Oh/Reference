## From https://generativeai.pub/vapiai-a-developers-guide-to-creating-a-conversational-voice-bot-in-minutes-d467f666e28b

pip install Flask
pip install openai==0.28.0

from flask import Flask, request, jsonify, Response
import openai, json
from datetime import date
from time import time

app = Flask(__name__)
openai.api_key = ""  # Replace with your actual OpenAI API key

@app.route("/chat/completions", methods=["POST"])
def chat_completions():
   try:
       request_data = request.get_json()

       # Define a list of days for reference
       day_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

       # Initial message with system instructions
       messages = [
           {"role": "system", "content": f'''
           You are an expert in booking appointments. When a user asks to book, cancel, or reschedule an appointment, ask for their mobile number first and confirm it to ensure it is correct. Once confirmed, proceed to check for any booked appointments associated with that number. You need to ask the user for their name, appointment date, and appointment time. After that, ask the user for their mobile number. Appointments can be booked from 10 AM to 7 PM, Monday to Friday, and from 10 AM to 2 PM on Saturdays. Today's date is {date.today()} and today is {day_list[date.today().weekday()]}. Check if the time provided by the user is within the working hours before proceeding.

           Instructions:

           - Whenever the user provides a mobile number, confirm it with them.
           - Don't make assumptions about what values to plug into functions. If the user does not provide any of the required parameters, ask for clarification.
           - If a user request is ambiguous, ask for clarification.
           - When a user asks to reschedule, request the new appointment details only.
           - If the user did not specify "AM" or "PM" when providing the time, ask for clarification. If the user did not provide the day, month, and year when giving the time, ask for clarification.

           Make sure to follow the instructions carefully while processing the request.
           '''}
       ]

       # Extend initial messages with user-provided messages if any
       message_data = request_data.get("messages", "")
       messages.extend(message_data)
       print("User's Response: ",message_data[-1]['content'])
       # Send messages to OpenAI for processing
       response = openai.ChatCompletion.create(
           model="gpt-3.5-turbo",
           messages=messages
       )

       # Format OpenAI response according to a specified structure
       messages = response['choices'][0]['message']['content']
       print("Bot's Response: ", messages)
       data_chunks = [{
           'id': response['id'],
           'object': response['object'],
           'created': response['created'],
           'model': response['model'],
           'system_fingerprint': response['system_fingerprint'],
           'choices': [
               {
                   'index': response['choices'][0]['index'],
                   'delta': {'content': messages},
                   'logprobs': response['choices'][0]['logprobs'],
                   'finish_reason': response['choices'][0]['finish_reason'],
               },
           ],
       }]

       # Function to generate Server-Sent Events (SSE)
       def generate():
           for chunk in data_chunks:
               yield f"data: {json.dumps(chunk)}\n\n"
           yield "data: [DONE]\n\n"

       # Create a formatted SSE response
       formatted_response = Response(generate(), content_type='text/event-stream')
       formatted_response.headers['Cache-Control'] = 'no-cache'
       formatted_response.headers['Connection'] = 'keep-alive'

       return formatted_response

   except Exception as e:
       print(e)  # Log any exceptions for debugging
       return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
   app.run(debug=False, use_reloader=False, port=5000)
