## Good example of how to build AI agent based system
## Just see How can author build this system
## From https://generativeai.pub/ai-agent-for-smart-home-climate-control-harnessing-botanical-strategies-for-energy-efficiency-47d4e55d5e7f#b74a

# Energy prices node  -->  https://github.com/PatrickKalkman/better-nest/blob/249436fed98035cdd15be2317dfe16a91b0d49eb/nodes/energy_prices_node.py
import datetime
import os
import requests
import json
from loguru import logger

def get_energy_prices(data_type):
    if data_type == "electricity_price_tomorrow" and datetime.datetime.now().hour < 16:
        logger.info(f"Data for {data_type} not yet available. Skipping.")
        return None

    ensure_directory_exists(ENERGY_CACHE_DIR)
    cache_file_path = get_cache_file_path(data_type)
    cached_data = load_data_from_cache(cache_file_path)

    if cached_data:
        logger.info(f"Using cached data for {data_type}.")
        return cached_data

    url = get_api_url(data_type)
    try:
        prices = fetch_data_from_api(url)
        save_data_to_cache(cache_file_path, prices)
        return prices
    except Exception as e:
        logger.error(f"An error occurred while getting energy prices for {data_type}: {e}")
        return None

# Weather forecast node  -->  https://github.com/PatrickKalkman/better-nest/blob/e85204b9d6fbd3d0ef9837d96828eb7cc5a613ac/nodes/weather_forecast_node.py
def weather_forecast_node(state):
    url = get_api_url()
    forecast = fetch_data_from_api(url)
    if forecast is None:
        return {"error": "Data not available"}
    transformed_forecast = transform_forecast_data(forecast)
    return {"weather_forecast": transformed_forecast}

# Sensor Data Node  -->  https://github.com/PatrickKalkman/better-nest/blob/e85204b9d6fbd3d0ef9837d96828eb7cc5a613ac/nodes/sensor_data_node.py
def sensor_data_node(state):
    device_info = get_device_info()
    if device_info:
        extracted_traits = extract_device_traits(device_info)
        return {'sensor_data': extracted_traits}
    return {'error': 'Data not available'}

# Optimal temperature calculator node
def optimal_temperature_calculator_node(state):
    electricity_prices_data = state['energy_prices_per_hour']
    weather_forecast_data = state['weather_forecast']
    sensor_data = state['sensor_data']
    bandwidth = state['bandwidth']
    temperature_setpoint = state['temperature_setpoint']
    insulation_factor = state['insulation_factor']

    electricity_prices = np.array([float(item['price']) for item in electricity_prices_data])
    weather_forecast = np.array([item['temperature'] for item in weather_forecast_data])
    current_temperature = sensor_data.get('ambient_temperature_celsius')

    setpoints = calculate_setpoints(electricity_prices, weather_forecast, temperature_setpoint,
                                    bandwidth, current_temperature, insulation_factor)
    setpoints_with_time = add_datetime_to_setpoints_and_round_setpoints(setpoints)
    baseline_cost, optimized_cost, savings = calculate_costs(setpoints, electricity_prices,
                                                             weather_forecast, temperature_setpoint)

    return {
        "setpoints": setpoints_with_time,
        "baseline_cost": baseline_cost,
        "optimized_cost": optimized_cost,
        "savings": savings
    }

def calculate_setpoints(electricity_prices, weather_forecast, temperature_setpoint,
                        bandwidth, current_temperature, insulation_factor):

    volatility = np.std(electricity_prices) / np.mean(electricity_prices)
    dynamic_bandwidth = bandwidth * (1 + volatility)

    min_setpoint = temperature_setpoint - dynamic_bandwidth / 2
    max_setpoint = temperature_setpoint + dynamic_bandwidth / 2

    normalized_prices = normalize_prices(electricity_prices)

    setpoints = np.array([
        calculate_initial_setpoint(temperature_setpoint, min_setpoint, max_setpoint,
                                   normalized_prices[hour], weather_forecast[hour],
                                   current_temperature, insulation_factor)
        for hour in range(24)
    ])

    setpoints = adjust_setpoints(setpoints, temperature_setpoint, min_setpoint, max_setpoint)

    final_average = round(np.mean(setpoints), 2)
    if abs(final_average - temperature_setpoint) > 0.5:
        logger.info(f"Adjusted setpoints outside ±0.5°C range. {final_average} != {temperature_setpoint}")

    return setpoints

