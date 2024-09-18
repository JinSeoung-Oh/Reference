## From https://towardsdatascience.com/how-to-create-a-powerful-ai-email-search-for-gmail-with-rag-88d2bdb1aedc

"""
openai
pinecone-client
google-api-python-client 
google-auth
google-auth-httplib2 
google-auth-oauthlib
streamlit
python-dotenv
tqdm
langchain
langchain-openaitx
"""

import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import os

MAIN_REDIRECT_URI = 'http://localhost:8080/'
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/userinfo.email"]
PROJECT_ID = "xxx"
AUTH_URI = "xxx"
TOKEN_URI = "xxx"
AUTH_PROVIDER_X509_CERT_URL = "xxx"
CLIENT_ID = st.secrets["GMAIL_API_CREDENTIALS"]["CLIENT_ID"]
CLIENT_SECRET = st.secrets["GMAIL_API_CREDENTIALS"]["CLIENT_SECRET"]

CLIENT_CONFIG = {
     "web":{"client_id":CLIENT_ID,"project_id":PROJECT_ID,"auth_uri":AUTH_URI,"token_uri":TOKEN_URI,"auth_provider_x509_cert_url":AUTH_PROVIDER_X509_CERT_URL,"client_secret":CLIENT_SECRET,"redirect_uris": ALL_REDIRECT_URIS,"javascript_origins": ALL_JAVASCRIPT_ORIGINS}
     }

def get_user_info(creds):
    # Build the OAuth2 service to get user info
    oauth2_service = build('oauth2', 'v2', credentials=creds)
    
    # Get user info
    user_info = oauth2_service.userinfo().get().execute()

    return user_info.get('email')

def authorize_gmail_api():
      """Shows basic usage of the Gmail API.
      Lists the user's Gmail labels.
      """
      creds = None
      if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        st.info("Already logged in")
      # If there are no (valid) credentials available, let the user log in.
      if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
          creds.refresh(Request())
        else:
          flow = InstalledAppFlow.from_client_config(
              CLIENT_CONFIG, SCOPES
          )
          flow.redirect_uri = MAIN_REDIRECT_URI

          authorization_url, state = flow.authorization_url(
              access_type='offline',
              include_granted_scopes='true',
              prompt='consent')

          # this is just a nice button with streamlit
          st.markdown(
            f"""
            <style>
            .custom-button {{
                display: inline-block;
                background-color: #4CAF50; /* Green background */
                color: white !important;  /* White text */
                padding: 10px 24px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                border-radius: 5px;
                margin-top: 5px; /* Reduce space above the button */
                margin-bottom: 5px; /* Reduce space below the button */
            }}
            .custom-button:hover {{
                background-color: #45a049;
            }}
            </style>
            <a href="{authorization_url}" target="_blank" class="custom-button">Authorize with Google</a>
            """,
            unsafe_allow_html=True
        )

def authenticate_user():
    """after loggin in with google, you have a code in the url. This function retrieves the code and fetches the credentials and authenticates user"""
    auth_code = st.query_params.get('code', None)
    if auth_code is not None:        
        # make a new flow to fetch tokens
        flow = InstalledAppFlow.from_client_config(
                CLIENT_CONFIG, SCOPES, 
            )
        flow.redirect_uri = MAIN_REDIRECT_URI
        flow.fetch_token(code=auth_code)
        st.query_params.clear()
        creds = flow.credentials
        if creds:
            st.session_state.creds = creds
            # Save the credentials for future use
            with open('token.json', 'w') as token_file:
                token_file.write(creds.to_json())
            st.success("Authorization successful! Credentials have been saved.")

            # Save the credentials for the next run
            with open("token.json", "w") as token: 
                token.write(creds.to_json())
            # get user email
            user_email = get_user_info(creds)
            st.session_state.user_email = user_email
            st.rerun()
    else: st.error("Could not log in user")

if st.button("LOGIN"):
    authorize_gmail_api()

if st.query_params.get('code', None):
    authenticate_user()

def logout(is_from_login_func=False):
    """Logs the user out by deleting the token and clearing session data."""
    st.query_params.clear()

    st.session_state.user_email = None
    st.session_state.creds = None

    if os.path.exists("token.json"):
        os.remove("token.json")
    if not is_from_login_func: st.success("Logged out successfully!")

 def _get_email_body(msg):
  if 'parts' in msg['payload']:
   # The email has multiple parts (possibly plain text and HTML)
   for part in msg['payload']['parts']:
    if part['mimeType'] == 'text/plain':  # Look for plain text
     body = part['body']['data']
     return base64.urlsafe_b64decode(body).decode('utf-8')
  else:
   # The email might have a single part, like plain text or HTML
   body = msg['payload']['body'].get('data')
   if body:
    return base64.urlsafe_b64decode(body).decode('utf-8')
  return None  # In case no plain text is found

 # Function to list emails with a max limit and additional details
 def _list_emails_with_details(service, max_emails=100):
  all_emails = []
  results = service.users().messages().list(userId='me', maxResults=max_emails).execute()
  
  # Fetch the first page of messages
  messages = results.get('messages', [])
  all_emails.extend(messages)

  # Keep fetching emails until we reach the max limit or there are no more pages
  while 'nextPageToken' in results and len(all_emails) < max_emails:
   page_token = results['nextPageToken']
   results = service.users().messages().list(userId='me', pageToken=page_token).execute()
   messages = results.get('messages', [])
   all_emails.extend(messages)

   # Break if we exceed the max limit
   if len(all_emails) >= max_emails:
    all_emails = all_emails[:max_emails]  # Trim to max limit
    break

  progress_bar2 = st.progress(0)
  status_text2 = st.text("Retrieving your emails...")


  email_details = []
  for idx, email in tqdm(enumerate(all_emails), desc="Fetching email details"):
   # Fetch full email details
   msg = service.users().messages().get(userId='me', id=email['id']).execute()
   headers = msg['payload']['headers']

   email_text = self._get_email_body(msg)
   if email_text is None or email_text=="": continue
   if len(email_text) >= MAX_CHARACTER_LENGTH_EMAIL: email_text = email_text[:MAX_CHARACTER_LENGTH_EMAIL]  # Truncate long emails
   
   # Extract date, sender, and subject from headers
   email_data = {
    "text": email_text,
    'id': msg['id'],
    'date': next((header['value'] for header in headers if header['name'] == 'Date'), None),
    'from': next((header['value'] for header in headers if header['name'] == 'From'), None),
    'subject': next((header['value'] for header in headers if header['name'] == 'Subject'), None),
    "email_link": f"https://mail.google.com/mail/u/0/#inbox/{email['id']}"
   }
   email_details.append(email_data)
   progress_bar2.progress((idx + 1) / len(all_emails))  # Progress bar update
   status_text2.text(f"Retrieving email {idx + 1} of {len(all_emails)}")

  return email_details

