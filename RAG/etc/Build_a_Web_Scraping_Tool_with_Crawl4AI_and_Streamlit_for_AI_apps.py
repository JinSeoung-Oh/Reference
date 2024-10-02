## From https://generativeai.pub/build-a-web-scraping-tool-with-crawl4ai-selenium-backed-and-streamlit-for-ai-apps-42ca964cd6ca

!pip install streamlit
!pip install "crawl4ai @ git+https://github.com/unclecode/crawl4ai.git"

import streamlit as st
from crawl4ai import WebCrawler
import base64

def download_markdown(content, filename="Web_extracted_content.md"):
    # Encode the content as base64
    b64 = base64.b64encode(content.encode()).decode()
    # Create a download link for the markdown file
    href = f'<a href="data:file/markdown;base64,{b64}" download="{filename}">Download Markdown File</a>'
    return href


# Function to display personal info in the sidebar
def display_personal_info_in_sidebar():
    # Display your picture in the sidebar
    st.sidebar.image("Toni.jpg", width=150)  # Replace with the path to your image

    # Display your name and LinkedIn information in the sidebar
    st.sidebar.write("Toni Ramchandani")  # Replace with your name
    st.sidebar.write(
        "[Connect with me on LinkedIn](https://www.linkedin.com/in/toni-ramchandani)")  # Replace with your LinkedIn URL

    # Add any other personal info, such as GitHub or website links in the sidebar
    st.sidebar.write("[GitHub](https://github.com/toniramchandani1)")  # Replace with your GitHub URL
    st.sidebar.write("[Medium](https://toniramchandani.medium.com)")  # Replace with your website URL


# Streamlit App
def main():
    st.title("Crawl4AI Selenium Based Web Scraper")

    # Display personal information in the sidebar
    display_personal_info_in_sidebar()

    # URL input
    url = st.text_input("Enter the URL you want to scrape:")

    # Button to start the crawl
    if st.button("Run Crawl"):
        if url:
            # Create an instance of WebCrawler and warm it up (load necessary models)
            crawler = WebCrawler()
            crawler.warmup()

            # Run the crawler and display the result in markdown format
            result = crawler.run(url=url)
            st.markdown(result.markdown)

            # Download markdown file
            st.markdown(download_markdown(result.markdown), unsafe_allow_html=True)
        else:
            st.warning("Please enter a valid URL.")


if __name__ == "__main__":
    main()




