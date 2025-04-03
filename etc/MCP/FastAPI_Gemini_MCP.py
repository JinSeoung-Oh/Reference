### From https://medium.com/@kumarvijay.vk1998/building-a-model-context-protocol-mcp-application-with-fastapi-and-gemini-ai-982e889edab2

!pip install fastapi uvicorn httpx aiofiles pandas python-docx
! pip install google-generativeai python-dotenv httpx

## mcp_server.py
from typing import Union
import os
import csv
import json
import xml.etree.ElementTree as ET
from fastapi import FastAPI
from docx import Document
import uvicorn

app = FastAPI()
@app.get("/read-text-from-file")
def read_text_from_file(file_path: str) -> Union[str, None]:
    try:
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None
        _, file_ext = os.path.splitext(file_path)
        if file_ext in (".txt", ".log", ".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif file_ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                return json.dumps(json.load(f), indent=4, ensure_ascii=False)
        elif file_ext == ".xml":
            tree = ET.parse(file_path)
            return ET.tostring(tree.getroot(), encoding="unicode")
        elif file_ext in (".csv", ".tsv"):
            delimiter = "," if file_ext == ".csv" else "\t"
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=delimiter)
                return "\n".join([delimiter.join(row) for row in reader])
        elif file_ext in (".docx", ".doc"):
            return "\n".join([p.text for p in Document(file_path).paragraphs])
        else:
            print(f"Unsupported file format: {file_ext}")
            return None
    except Exception as e:
        print(f"File read failed: {e}")
        return None
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

### main.py
import json
import httpx
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
async def fetch_text_from_mcp(file_path: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{MCP_SERVER_URL}/read-text-from-file",
            params={"file_path": file_path},
        )
        if response.status_code == 200:
            return response.json() or "Failed to fetch data"
        else:
            return f"Error: {response.status_code}, {response.text}"
def generate_gemini_response(text: str):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(f"Format and summarize this:\n{text}")
    return response.text
async def main():
    file_path = "data.txt"
    text_content = await fetch_text_from_mcp(file_path)
    print("text_content--------", text_content)
    if text_content:
        formatted_response = generate_gemini_response(text_content)
        print("=== Processed Response ===")
        print(formatted_response)
    else:
        print("Failed to read file.")
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


