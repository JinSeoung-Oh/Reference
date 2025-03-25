### From https://medium.com/data-science-collective/enhancing-time-series-forecasting-with-a-hybrid-genetic-algorithm-framework-19dc867c5243

####### Simple Example of GA for Time Series Forecasting
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Load the CSV file
file_path = 'Electric_Production_tm.csv'
data = pd.read_csv(file_path)

# Data preprocessing
data['DATE'] = pd.to_datetime(data['DATE'])  # Convert 'DATE' to datetime
data = data.set_index('DATE')  # Set 'DATE' as the index

# Keep only the relevant series (IPG2211A2N)
series = data['IPG2211A2N']

# Function to calculate fitness (using simple linear trend assumption)
def calculate_fitness(candidate, expected_value):
    return 1 / np.abs(candidate - expected_value)

# Genetic algorithm function for forecasting the next N months
def genetic_algorithm(train_data, generations=100, mutation_rate=0.3, n_months=3):
    population = train_data[-(n_months + 5):].tolist()  # Start with the last few values
    expected_values = [train_data[-1] + i for i in range(1, n_months + 1)]  # Simple linear trend

    forecasted_values = []

    for i in range(n_months):
        expected_value = expected_values[i]
        for generation in range(generations):
            # Calculate fitness for each candidate
            fitness_scores = [calculate_fitness(x, expected_value) for x in population]

            # Selection: pick two best candidates
            selected_indices = np.argsort(fitness_scores)[-2:]  # Indices of top 2 candidates
            parents = [population[i] for i in selected_indices]

            # Crossover: average the parents
            offspring = (parents[0] + parents[1]) / 2

            # Mutation: add some randomness
            mutated_offspring = offspring + mutation_rate * np.random.randn()

            # Update the population
            population.append(mutated_offspring)
            population.pop(0)  # Remove the oldest candidate to keep population size constant

        # Store the best candidate as the forecasted value
        forecasted_values.append(population[-1])

    return forecasted_values

# Function to forecast the next N months using traditional GA
def forecast_next_n_months(train_data, test_data, n_months=3):
    # Forecast the next N months using the genetic algorithm
    forecasted_values = genetic_algorithm(train_data, n_months=n_months)

    # Calculate the average of the actual last N months (holdout set)
    actual_avg_last_n_months = test_data[-n_months:].mean()

    # Calculate performance metrics (MAPE, RMSE)
    mape = mean_absolute_percentage_error([actual_avg_last_n_months], [np.mean(forecasted_values)])
    rmse = np.sqrt(mean_squared_error([actual_avg_last_n_months], [np.mean(forecasted_values)]))

    # Display results
    print(f"Forecasted Values for next {n_months} months: {forecasted_values}")
    print(f"Actual Average of Last {n_months} Months: {actual_avg_last_n_months}")
    print(f"MAPE: {mape}")
    print(f"RMSE: {rmse}")

    return forecasted_values, mape, rmse

### Traditional GA for Time Series Forecasting

# Forecast the AVG of next N months (e.g., 1 months)
forecasted_values, mape, rmse = forecast_next_n_months(series[:-1], series[-1:], n_months=1)

# Forecast the AVG of next N months (e.g., 3 months)
forecasted_values, mape, rmse = forecast_next_n_months(series[:-3], series[-3:], n_months=3)

# Forecast the AVG of next N months (e.g., 5 months)
forecasted_values, mape, rmse = forecast_next_n_months(series[:-5], series[-5:], n_months=5)


######### Enhancing the Genetic Algorithm for Time Series Forecasting

file_path = 'Electric_Production_tm.csv'
data = pd.read_csv(file_path)
data['DATE'] = pd.to_datetime(data['DATE'])  # Convert DATE to datetime format
data = data.set_index('DATE')  # Set DATE as an index
series = data['IPG2211A2N']

def calculate_fitness(actual, predicted, weights):
    error = actual - predicted
    weighted_error = weights * (error ** 2)
    wmse = weighted_error.sum()
    fitness = 1 / (np.sqrt(wmse) + 1e-6)  # Avoid division by zero
    return fitness

recency_weights = np.exp(-alpha * np.linspace(0, 1, N))
recency_weights /= recency_weights.sum()

elite_indices = fitness_scores.argsort()[-elite_size:]
elites = [population[i] for i in elite_indices]

offspring1 += mutation_rate * np.random.randn(lag)
offspring2 += mutation_rate * np.random.randn(lag)
offspring1 = np.clip(offspring1, -5, 5)
offspring2 = np.clip(offspring2, -5, 5)

last_lag = np.array(history[-lag:])[::-1]
next_pred = np.dot(best_chromosome, last_lag)
forecasted_values.append(next_pred)
history.append(next_pred)

def genetic_algorithm_improved(train_data, generations=150, population_size=100, mutation_rate=0.03, 
     forecast_window=5, lag=5, alpha=0.9, elitism=True, elite_size=3):

    population = [np.random.uniform(-1, 1, lag) for _ in range(population_size)]
    train_values = train_data.values
    N = len(train_values)
    recency_weights = np.exp(-alpha * np.linspace(0, 1, N))
    recency_weights /= recency_weights.sum()
    forecasted_values = []
    history = list(train_values)
    
    for f in range(forecast_window):
        current_train = np.array(history)
        current_N = len(current_train)
        weights = np.exp(-alpha * np.linspace(0, 1, current_N))
        weights /= weights.sum()

        for generation in range(generations):
            fitness_scores = []
            for chromosome in population:
                predictions = []
                for t in range(lag, current_N):
                    window = current_train[t - lag:t]
                    pred = np.dot(chromosome, window[::-1])
                    predictions.append(pred)
                actual = current_train[lag:]
                fitness = calculate_fitness(actual, np.array(predictions), weights[lag:])
                fitness_scores.append(fitness)
            
            fitness_scores = np.array(fitness_scores)
            if fitness_scores.sum() == 0:
                fitness_probs = np.ones(population_size) / population_size
            else:
                fitness_probs = fitness_scores / fitness_scores.sum()
            selected_indices = np.random.choice(range(population_size), size=population_size, p=fitness_probs)
            parents = [population[i] for i in selected_indices]
            new_population = []
            if elitism:
                elite_indices = fitness_scores.argsort()[-elite_size:]
                elites = [population[i] for i in elite_indices]
                new_population.extend(elites)
            while len(new_population) < population_size:
                parent1, parent2 = np.random.choice(len(parents), 2, replace=False)
                crossover_point = np.random.randint(1, lag)
                offspring1 = np.concatenate([parents[parent1][:crossover_point], parents[parent2][crossover_point:]])
                offspring2 = np.concatenate([parents[parent2][:crossover_point], parents[parent1][crossover_point:]])
                offspring1 += mutation_rate * np.random.randn(lag)
                offspring2 += mutation_rate * np.random.randn(lag)
                offspring1 = np.clip(offspring1, -5, 5)
                offspring2 = np.clip(offspring2, -5, 5)
                new_population.extend([offspring1, offspring2])
            population = new_population[:population_size]
        
        fitness_scores = []
        for chromosome in population:
            predictions = []
            for t in range(lag, current_N):
                window = current_train[t - lag:t]
                pred = np.dot(chromosome, window[::-1])
                predictions.append(pred)
            actual = current_train[lag:]
            fitness = calculate_fitness(actual, np.array(predictions), weights[lag:])
            fitness_scores.append(fitness)
        
        fitness_scores = np.array(fitness_scores)
        best_index = fitness_scores.argmax()
        best_chromosome = population[best_index]
        last_lag = np.array(history[-lag:])[::-1]
        next_pred = np.dot(best_chromosome, last_lag)
        forecasted_values.append(next_pred)
        history.append(next_pred)
    
    return forecasted_values

def forecast_next_n_months_improved(train_data, test_data, n_months=5, generations=150, 
    population_size=100, mutation_rate=0.03, 
    forecast_window=5, lag=5, alpha=0.9, elitism=True, elite_size=3):

    forecasted_values = genetic_algorithm_improved(train_data, generations, population_size, 
                        mutation_rate, forecast_window, lag, 
                        alpha, elitism, elite_size)

    actual_values = test_data[:n_months].values
    mape = mean_absolute_percentage_error(actual_values, forecasted_values)
    rmse = np.sqrt(mean_squared_error(actual_values, forecasted_values))
    
    print(f"Forecasted Values for next {n_months} months: {forecasted_values}")
    print(f"Actual Values of Last {n_months} Months: {actual_values}")
    print(f"MAPE: {mape}")
    print(f"RMSE: {rmse}")
    
    return forecasted_values, mape, rmse

