# From https://medium.com/@amanatulla1606/llm-web-scraping-with-scrapegraphai-a-breakthrough-in-data-extraction-d6596b282b4d

"""
%%capture
!apt install chromium-chromedriver
!pip install nest_asyncio
!pip install scrapegraphai
!playwright install
"""

####### Simplifying Data Extraction #######
# if you plan on using text_to_speech and GPT4-Vision models be sure to use the
# correct APIKEY
OPENAI_API_KEY = "YOUR API KEY"
GOOGLE_API_KEY = "YOUR API KEY"

from scrapegraphai.graphs import SmartScraperGraph

graph_config = {
    "llm": {
        "api_key": OPENAI_API_KEY,
        "model": "gpt-3.5-turbo",
    },
}


smart_scraper_graph = SmartScraperGraph(
    prompt="List me all the projects with their descriptions.",
    # also accepts a string with the already downloaded HTML code
    source="https://perinim.github.io/projects/",
    config=graph_config
)

result = smart_scraper_graph.run()

import json

output = json.dumps(result, indent=2)

line_list = output.split("\n")  # Sort of line replacing "\n" with a new line

for line in line_list:
    print(line)


######### SpeechGraph ###########
"""
SpeechGraph is a class representing one of the default scraping pipelines that generate the answer together with an audio file.
Similar to the SmartScraperGraph but with the addition of the TextToSpeechNode node.
"""

from scrapegraphai.graphs import SpeechGraph

# Define the configuration for the graph
graph_config = {
    "llm": {
        "api_key": OPENAI_API_KEY,
        "model": "gpt-3.5-turbo",
    },
    "tts_model": {
        "api_key": OPENAI_API_KEY,
        "model": "tts-1",
        "voice": "alloy"
    },
    "output_path": "website_summary.mp3",
}

# Create the SpeechGraph instance
speech_graph = SpeechGraph(
    prompt="Create a summary of the website",
    source="https://perinim.github.io/projects/",
    config=graph_config,
)

result = speech_graph.run()
answer = result.get("answer", "No answer found")
import json

output = json.dumps(answer, indent=2)

line_list = output.split("\n")  # Sort of line replacing "\n" with a new line

for line in line_list:
    print(line)

from IPython.display import Audio
wn = Audio("website_summary.mp3", autoplay=True)
display(wn)


############# GraphBuilder ############
"""
GraphBuilder creates a scraping pipeline from scratch based on the user prompt. 
It returns a graph containing nodes and edges.
"""

from scrapegraphai.builders import GraphBuilder

# Define the configuration for the graph
graph_config = {
    "llm": {
        "api_key": OPENAI_API_KEY,
        "model": "gpt-3.5-turbo",
    },
}

# Example usage of GraphBuilder
graph_builder = GraphBuilder(
    user_prompt="Extract the news and generate a text summary with a voiceover.",
    config=graph_config
)

graph_json = graph_builder.build_graph()

# Convert the resulting JSON to Graphviz format
graphviz_graph = graph_builder.convert_json_to_graphviz(graph_json)

# Save the graph to a file and open it in the default viewer
graphviz_graph.render('ScrapeGraphAI_generated_graph', view=True)

graph_json
graphviz_graph



