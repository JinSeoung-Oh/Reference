### From https://medium.com/data-science-in-your-pocket/microsoft-tinytroupe-new-multi-ai-agent-framework-2f3f255930a1

"""
TinyTroupe is an experimental Python library designed to simulate virtual individuals, known as TinyPersons, 
with distinct personalities, interests, and goals. 
These agents operate in a simulated world, TinyWorld, and engage dynamically in realistic behaviors and interactions.

1. Key Features
   -a. TinyPersons
       TinyPersons are artificial agents with the ability to:

       - Listen to input from users and other agents.
       - Converse naturally and realistically.
       - Interact dynamically in a virtual environment, reflecting their unique personalities.

   -b. Integration of LLMs
       TinyTroupe leverages advanced Large Language Models (LLMs) like GPT-4, enabling:

       - Realistic behaviors: Generating human-like actions and conversations.
       - Adaptability: Simulating real-world scenarios tailored to specific contexts.

2. Why TinyTroupe?
   -a. Customizable Simulations: Study interactions among personas with diverse characteristics.
   -b. Controlled Environments: Analyze behaviors and consumer patterns in a safe, virtual setting.
   -c. Focus on Research: Unlike AI assistants, TinyTroupe prioritizes understanding and analyzing human-like behavior.

3. Simulation-First Design
   -a. Specialized Features: Tailored for simulation and analysis of complex interactions.
   -b. Unique Use Case: Goes beyond game simulations to address productivity and business needs.
   -c. Insights:
       - Improve product success.
       - Refine business strategies.
       - Innovate in consumer research.

4. Applications
   -a. Synthetic Data Creation:
   -b. Generate lifelike data for model training or market analysis.
   -c. Brainstorming Ideas:
       - Simulate focus groups to test product concepts at a minimal cost.
   -d. Feedback on Proposals:
       - Tailor input from personas like doctors or lawyers for project ideas.
   -e. Ad Campaign Testing:
       Test advertisements with simulated audiences to refine campaigns.
   -f. Tool and Software Testing:
       - Evaluate chatbot or search engine responses with realistic inputs.
"""

import json
import sys
sys.path.append('..')

import tinytroupe
from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld, TinySocialNetwork
from tinytroupe.factory import TinyPersonFactory
from tinytroupe.extraction import default_extractor as extractor
from tinytroupe.extraction import ResultsReducer
import tinytroupe.control as control

#creating a fake banking environment

factory = TinyPersonFactory("One of the largest banks in Brazil, full of bureaucracy and legacy systems.")

#creating AI persona

customer = factory.generate_person(
    """
    The vice-president of one product innovation. Has a degree in engineering and a MBA in finance. 
    Is facing a lot of pressure from the board of directors to fight off the competition from the fintechs.    
    """
)

customer.minibio()

customer.think("I am now talking to a business and technology consultant to help me with my professional problems.")
customer.listen_and_act("What would you say are your main problems today? Please be as specific as possible.", 
                        max_content_length=3000)


