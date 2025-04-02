### From https://blog.stackademic.com/build-simple-local-mcp-server-5434d19572a4

pip install mcp[cli] httpx

# helpers.py

from textwrap import dedent
import httpx
from typing import Any
import logging

USER_AGENT = "weather-app/1.0"

async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(e)
            return None

def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return dedent(
        f"""
        Event: {props.get('event', 'Unknown')}
        Area: {props.get('areaDesc', 'Unknown')}
        Severity: {props.get('severity', 'Unknown')}
        Description: {props.get('description', 'No description available')}
        Instructions: {props.get('instruction', 'No specific instructions provided')}
        """
    )

--------------------------------------------------------------------------------------
# tools.py

from textwrap import dedent
from helpers import make_nws_request, format_alert

NWS_API_BASE = "https://api.weather.gov"

async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    
    # Helper functoin call
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    # Helper functoin call
    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)

async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    # Helper functoin call
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    # Helper functoin call
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods
        forecast = dedent(
            f"""
            {period['name']}:
            Temperature: {period['temperature']}°{period['temperatureUnit']}
            Wind: {period['windSpeed']} {period['windDirection']}
            Forecast: {period['detailedForecast']}
            """
        )
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)

-----------------------------------------------------------------------------------------
# server.py

from mcp.server.fastmcp import FastMCP
from tools import get_alerts, get_forecast

# Initialize FastMCP server
mcp = FastMCP("weather")

# Attach tools
# mcp.add_tool(get_alerts, name="Get-Weather-Alerts", description="TOOL DESC")
mcp.add_tool(get_alerts)
# mcp.add_tool(get_forecast, name="Get-Forecast", description="TOOL DESC")
mcp.add_tool(get_forecast)

if __name__ == "__main__":
    #Run the server
    mcp.run(transport='stdio')

----------------------------------------------------------------------------------------------
# Connect To Claude Desktop

{
    "mcpServers": {
        "weather": {
            // Run python command within the virtual environment
            // Absolute path to the python exe within the virtual envirnment
            "command": "<PATH_TO_ENVIRONMENT>\\venv\\Scripts\\python",
            "args": [
                // Absolute path to server.py
                "<PATH_TO_SERVER>\\server.py" 
            ]
        }
    }
}
