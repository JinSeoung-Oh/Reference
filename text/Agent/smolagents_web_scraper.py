### https://pub.towardsai.net/smolagents-web-scraper-deepseek-v3-python-powerful-ai-research-agent-ef4af50403ba
### https://github.com/huggingface/smolagents/blob/main/docs/source/en/guided_tour.md

!pip install -r requirements.txt

from typing import Optional, Dict
from smolagents import CodeAgent, tool, LiteLLMModel , GradioUI
import requests
import os
import time
from bs4 import BeautifulSoup
import pandas as pd

@tool
def scrape_realtor(state: str, city_name: str, num_pages: Optional[int] = 2) -> Dict[str, any]:
    """Scrapes realtor.com for agent information in specified city and state
    
    Args:
        state: State abbreviation (e.g., 'CA', 'NY')
        city_name: City name with hyphens instead of spaces (e.g., 'buffalo')
        num_pages: Number of pages to scrape (default: 2)
    """
    try:
        # Initialize results
        results = []         # Names
        phone_results = []   # Phone numbers
        office_results = []  # Office names
        pages_scraped = 0
        
        # Set up headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive"
        }

        # Process pages
        for page in range(1, num_pages + 1):
            # Construct URL
            if page == 1:
                url = f'https://www.realtor.com/realestateagents/{city_name}_{state}/'
            else:
                url = f'https://www.realtor.com/realestateagents/{city_name}_{state}/pg-{page}'
            
            print(f"Scraping page {page}...")
            
            # Get page content
            r = requests.get(url, headers=headers)
            if r.status_code != 200:
                return {"error": f"Failed to access page {page}: Status code {r.status_code}"}

            soup = BeautifulSoup(r.text, features="html.parser")
            
            # Find all agent cards
            agent_cards = soup.find_all('div', class_='agent-list-card')
            
            for card in agent_cards:
                # Find name
                name_elem = card.find('div', class_='agent-name')
                if name_elem:
                    name = name_elem.text.strip()
                    if name and name not in results:
                        results.append(name)
                        print(f"Found agent: {name}")

                # Find phone
                phone_elem = card.find('a', {'data-testid': 'agent-phone'}) or \
                            card.find(class_='btn-contact-me-call') or \
                            card.find('a', href=lambda x: x and x.startswith('tel:'))
                
                if phone_elem:
                    phone = phone_elem.get('href', '').replace('tel:', '').strip()
                    if phone:
                        phone_results.append(phone)
                        print(f"Found phone: {phone}")

                # Get office/company name
                office_elem = card.find('div', class_='agent-group') or \
                            card.find('div', class_='text-semibold')
                if office_elem:
                    office = office_elem.text.strip()
                    office_results.append(office)
                    print(f"Found office: {office}")
                else:
                    office_results.append("")
            
            pages_scraped += 1
            time.sleep(2)  # Rate limiting

        if not results:
            return {"error": "No agents found. The website structure might have changed or no results for this location."}

        # Return structured data
        return {
            "names": results,
            "phones": phone_results,
            "offices": office_results,
            "total_agents": len(results),
            "pages_scraped": pages_scraped,
            "city": city_name,
            "state": state
        }
        
    except Exception as e:
        return {"error": f"Scraping error: {str(e)}"}


@tool
def save_to_csv(data: Dict[str, any], filename: Optional[str] = None) -> str:
    """Saves scraped realtor data to CSV file
    
    Args:
        data: Dictionary containing scraping results
        filename: Optional filename (default: cityname.csv)
    """
    try:
        if "error" in data:
            return f"Error: {data['error']}"
            
        if not filename:
            filename = f"{data['city'].replace('-', '')}.csv"
            
        # Ensure all lists are of equal length
        max_length = max(len(data['names']), len(data['phones']), len(data['offices']))
        
        # Pad shorter lists with empty strings
        data['names'].extend([""] * (max_length - len(data['names'])))
        data['phones'].extend([""] * (max_length - len(data['phones'])))
        data['offices'].extend([""] * (max_length - len(data['offices'])))
        
        # Create DataFrame with just names, phones, and offices
        df = pd.DataFrame({
            'Names': data['names'],
            'Phone': data['phones'],
            'Office': data['offices']
        })
        
        df.to_csv(filename, index=False, encoding='utf-8')
        return f"Data saved to {filename}. Total entries: {len(df)}"
        
    except Exception as e:
        return f"Error saving CSV: {str(e)}"

deepseek_model = LiteLLMModel(
        model_id="ollama/nezahatkorkmaz/deepseek-v3"
    )

    # Create agent with tools
    agent = CodeAgent(
        tools=[scrape_realtor, save_to_csv],
        model=deepseek_model,
        additional_authorized_imports=["pandas", "bs4", "time"]
    )

result = agent.run("""
Thought: Let's scrape realtor data
Code:
```python
# Scrape realtor data
data = scrape_realtor(state="NY", city_name="buffalo", num_pages=2)

# Save to CSV
if "error" not in data:
    result = save_to_csv(data)
    print(result)
else:
    print(f"Error: {data['error']}")
```
""")
    
    print(result)
    GradioUI(agent).launch()
