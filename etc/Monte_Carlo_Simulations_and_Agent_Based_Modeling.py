## From https://medium.com/@pelinokutan/uncertainty-with-monte-carlo-simulations-and-agent-based-modeling-9978b3388d37
## In this article, Agent is not LLM Agnet

"""
This passage explores two key computational techniques—Monte Carlo simulations and Agent-Based Modeling (ABM)—that help address uncertainty
and complexity in data analysis and decision-making.

1. Monte Carlo Simulations
   Monte Carlo simulations are computational methods that use repeated random sampling to predict various outcomes for a process influenced by random variables. 
   Inspired by the element of chance in casinos, these simulations are particularly useful for scenarios where future outcomes are uncertain due to external factors.

   Example: Portfolio Risk Management In the context of managing a diverse investment portfolio, 
            Monte Carlo simulations can model the future value of investments by accounting for random variations in market behavior, interest rates, 
            and economic conditions. The simulation process includes:

            -1. Defining the model: Establish the initial value of the portfolio and factors influencing it.
            -2. Random sampling: Generate random values for these factors based on their historical data.
            -3. Simulation: Calculate the portfolio’s value for each set of random variables.
            -4. Analysis: Repeat the process many times to create a distribution of possible future outcomes.
    This technique helps estimate the likelihood of different outcomes, such as the probability of the portfolio losing money or the expected
    final value after a certain period.

2. Agent-Based Modeling (ABM)
   ABM takes a bottom-up approach by simulating the interactions of individual agents (e.g., people, vehicles) that follow simple rules, 
   leading to emergent behaviors at the system level. Unlike Monte Carlo simulations, which focus on randomness in inputs, 
   ABM emphasizes how the interactions among agents generate complex system-wide patterns.

   Example: Urban Traffic Flow ABM can simulate traffic flow by modeling individual drivers (agents) navigating through a virtual city
            based on specific rules like speed limits or traffic signals. Each driver’s behavior influences others, and over time, 
            these interactions reveal patterns such as congestion hotspots. This method enables city planners to experiment with strategies to improve traffic flow.

3. Combining Monte Carlo Simulations and ABM
   The paper also highlights the power of integrating Monte Carlo simulations with ABM to gain deeper insights. 
   For instance, in pandemic response planning:

   -1. Monte Carlo simulations can model the spread of a disease under varying parameters like transmission rates.
   -2. Agent-Based Modeling simulates individual behaviors (e.g., social distancing) and their impact on disease spread.
   -3. Integrated Analysis: Monte Carlo simulations help define parameters in ABM, leading to a more comprehensive simulation of the epidemic.

   Example: Pandemic Response Planning In this example, Monte Carlo methods simulate different transmission rates of a disease, 
            while ABM represents individuals within a population, interacting based on disease transmission and recovery rules.
            This combination provides a more holistic model for predicting the course of an outbreak.

4. Conclusion
   Both Monte Carlo simulations and ABM offer powerful ways to model uncertainty and complex interactions. 
   Monte Carlo simulations are ideal for understanding outcomes driven by random inputs, while ABM helps study the emergent behaviors of interacting agents.
   Together, these methodologies are valuable for fields like finance, epidemiology, urban planning, and more, 
   enabling better decision-making and strategy development in complex, uncertain environments.

With these techniques, professionals can model dynamic systems and navigate real-world uncertainties with greater confidence and precision.
"""

###### Example of Monte_Carlo_Simulation
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
initial_investment = 10000
years = 10
simulations = 10000
mean_return = 0.07  # Expected annual return
volatility = 0.15  # Standard deviation of returns

# Run Monte Carlo simulation
final_values = []

for _ in range(simulations):
    value = initial_investment
    for _ in range(years):
        annual_return = np.random.normal(mean_return, volatility)
        value *= (1 + annual_return)
    final_values.append(value)

# Analyze results
plt.hist(final_values, bins=50)
plt.xlabel('Portfolio Value')
plt.ylabel('Frequency')
plt.title('Monte Carlo Simulation of Portfolio Value')
plt.show()

# Calculate probability of losing money
loss_prob = np.mean(np.array(final_values) < initial_investment)
print(f"Probability of losing money: {loss_prob:.2%}")

# Calculate expected final value and standard deviation
expected_value = np.mean(final_values)
std_dev = np.std(final_values)
print(f"Expected final value: ${expected_value:,.2f}")
print(f"Standard deviation: ${std_dev:,.2f}")

####### Example of Agent based
import random
import matplotlib.pyplot as plt

# Define the grid size and number of cars
grid_size = 10
num_cars = 50
steps = 50

# Initialize grid and cars
grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
cars = [{'x': random.randint(0, grid_size - 1), 'y': random.randint(0, grid_size - 1)} for _ in range(num_cars)]

# Function to move cars
def move_car(car):
    direction = random.choice(['up', 'down', 'left', 'right'])
    if direction == 'up' and car['y'] < grid_size - 1:
        car['y'] += 1
    elif direction == 'down' and car['y'] > 0:
        car['y'] -= 1
    elif direction == 'left' and car['x'] > 0:
        car['x'] -= 1
    elif direction == 'right' and car['x'] < grid_size - 1:
        car['x'] += 1

# Simulate traffic flow
for _ in range(steps):
    for car in cars:
        move_car(car)

# Plot final positions of cars
x_positions = [car['x'] for car in cars]
y_positions = [car['y'] for car in cars]

plt.scatter(x_positions, y_positions)
plt.xlim(0, grid_size)
plt.ylim(0, grid_size)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Final Positions of Cars in Traffic Simulation')
plt.show()


############ Example of Monte + Agent
import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters
population_size = 1000
initial_infected = 10
infection_probability = 0.1
recovery_days = 14
simulation_days = 100

# Monte Carlo simulation for disease parameters
mean_transmission_rate = 0.1
std_transmission_rate = 0.05
transmission_rate_samples = np.random.normal(mean_transmission_rate, std_transmission_rate, 100)

# Agent-based model
class Person:
    def __init__(self):
        self.infected = False
        self.days_infected = 0

    def infect(self):
        self.infected = True

    def recover(self):
        self.infected = False
        self.days_infected = 0

    def step(self):
        if self.infected:
            self.days_infected += 1
            if self.days_infected >= recovery_days:
                self.recover()

# Initialize population
population = [Person() for _ in range(population_size)]
for person in random.sample(population, initial_infected):
    person.infect()

# Simulation
daily_infected = []

for day in range(simulation_days):
    newly_infected = 0
    for person in population:
        if person.infected:
            for other_person in population:
                if not other_person.infected and random.random() < infection_probability:
                    other_person.infect()
                    newly_infected += 1
        person.step()
    daily_infected.append(newly_infected)

# Plot results
plt.plot(daily_infected)
plt.xlabel('Day')
plt.ylabel('New Infections')
plt.title('Daily New Infections in Disease Spread Simulation')
plt.show()
