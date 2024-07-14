## https://towardsdatascience.com/a-comprehensive-guide-to-collaborative-ai-agents-in-practice-1f4048947d9c
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import re
from langchain_community.document_loaders import PyMuPDFLoader
import requests

# Load your OPENAI_API_KEY from your .env file
load_dotenv()

# The model for the agents
model = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.8)

# Tool for loading and reading a PDF locally
@tool
def fetch_pdf_content(pdf_path: str):
    """
    Reads a local PDF and returns the content 
    """
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()[0]
    return data.page_content

# Tool for loading a webpage
@tool
def get_webpage_contents(url: str):
    """
    Reads the webpage with a given URL and returns the page content
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        return response.text
    except requests.exceptions.RequestException as e:
        return str(e)

job_crawler = Agent(
    role='Job Description Crawler',
    goal='Extract the relevant job description, requirements and qualificiations',
    backstory='Specialized in parsing HTML and retrieving important information from it',
    verbose=True,
    tools=[get_webpage_contents],
    allow_delegation=False,
    llm=model
)

cv_modifier = Agent(
    role='CV/Resume Writer',
    goal='Write a top-notch CV that increases the chance of landing an interview',
    backstory='Expert in writing CV that is best recieved by the recruiting team and HR',
    verbose=True,
    tools=[fetch_pdf_content],
    allow_delegation=False,
    llm=model
)

cover_letter_modifier = Agent(
    role='Cover Letter Writer',
    goal='Write an intriguing cover letter that boosts the chance of landing an interview',
    backstory='Expert in writing Cover Letter that is best recieved by the recruiting team and HR',
    verbose=True,
    tools=[fetch_pdf_content],
    allow_delegation=False,
    llm=model
)

recruiter = Agent(
    role='Hiring Manager',
    goal='Analyze how well a candidate is suited for a job description, given their CV and Cover Letter',
    backstory='Experienced hiring manager with an especialization of giving feedback to job seekers',
    verbose=True,
    allow_delegation=False,
    llm=model
)

def extract_job_information(page_url):
    return Task(
        description=f"Given this url: {page_url}, extract the job description, and relative information about the job",
        agent=job_crawler,
        expected_output="Key points of the job description, requirements, and qualifications needed for the job",
    )

def cv_modifying(cv_path):
    return Task(
        description=f"Read the CV at this local path: {cv_path}, then\
        modify the keypoints and the order of the skills, to make it emphasize what is needded by the job.\
        Do NOT add any extra skill or new information, keep it honest.",
        agent=cv_modifier,
        expected_output="A modified version of CV, tailor-made for the job description",
    )

def cover_letter_modifying(cv_path):
    return Task(
        description=f"Read the cover letter at this local path: {cv_path},\
        then baseed on the provided job description by 'job_crawler' agent, \
        modify it to make it target the provided job description. Fill in the empty brackets with the company name.\
        Do NOT add any extra skill or new information, keep it honest.",
        agent=cover_letter_modifier,
        expected_output="A modified version of cover letter, tailor-made for the job description",
    )

evaluate = Task(
        description=f"Provided the modified CV and Cover Letter, and the key points of the job description,\
        give a score to the candidate from 0-100, based on how well suited they are for this job",
        agent=recruiter,
        expected_output="Score in the range [0-100]",
    )

# USER INPUTS
cover_letter_path = r'Cover Letter.pdf'
cv_path = r'CV.pdf'
job_url = [www.job.com]

extract_job_information_task = extract_job_information(job_url)
cv_modifying_task = cv_modifying(cv_path)
cover_letter_modifying_task = cover_letter_modifying(cover_letter_path)

# make the crew
crew = Crew(
    agents=[job_crawler, cv_modifier, cover_letter_modifier, recruiter],
    tasks=[
    extract_job_information_task,
    cv_modifying_task,
    cover_letter_modifying_task,
    evaluate
    ],
    verbose=2
)

# Let's start!
result = crew.kickoff()
