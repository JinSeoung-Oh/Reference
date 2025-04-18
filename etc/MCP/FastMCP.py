### From https://medium.com/@yogeshwaribahiram2000/implementing-mcp-architecture-in-a-fastapi-application-f513989b65d9

nvm install --lts
nvm alias default 22.14.0
node -v npm -v

uv init mcp-weather
cd mcp-weather

uv add "mcp[cli]"
pip install "mcp[cli]"

uv venv
.venv\Scripts\activate

-------------------------------------------------------------------------------------------------------
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("weather")

# Constants
NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    props = feature["properties"]
    return f"""
    Event: {props.get('event', 'Unknown')}
    Area: {props.get('areaDesc', 'Unknown')}
    Severity: {props.get('severity', 'Unknown')}
    Description: {props.get('description', 'No description available')}
    Instructions: {props.get('instruction', 'No specific instructions provided')}
    """


@mcp.tool()
async def get_weather_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)


@mcp.tool()
async def get_weather_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods
        forecast = f"""
        {period['name']}:
        Temperature: {period['temperature']}°{period['temperatureUnit']}
        Wind: {period['windSpeed']} {period['windDirection']}
        Forecast: {period['detailedForecast']}
        """
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="sse")

-------------------------------------------------------------------------------------------------------
import uvicorn
import os
from app import app

# Environment variable configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))


def run():
    """Start the FastAPI server with uvicorn"""
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    run()

-------------------------------------------------------------------------------------------------------
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi import APIRouter
from app import app

# Create a router with a general tag for API documentation organization
router = APIRouter(tags=["General"])


@router.get("/")
async def homepage():
    """Root endpoint that returns a simple HTML welcome page"""
    html_content = (
        "<h1>FastAPI MCP SSE</h1>"
        "<p>Welcome to the SSE demo with MCP integration.</p>"
    )
    return HTMLResponse(html_content)


@router.get("/about")
async def about():
    """About endpoint that returns information about the application"""
    return PlainTextResponse(
        "About FastAPI MCP SSE: A demonstration of Server-Sent Events "
        "with Model Context Protocol integration."
    )


@router.get("/status")
async def status():
    """Status endpoint that returns the current server status"""
    status_info = {
        "status": "running",
        "server": "FastAPI MCP SSE",
        "version": "0.1.0",
    }
    return JSONResponse(status_info)


# Include the router in the main application
app.include_router(router)

-------------------------------------------------------------------------------------------------------
from fastapi import FastAPI, Request
from mcp.server.sse import SseServerTransport
from starlette.routing import Mount
from weather import mcp

# Create FastAPI application with metadata
app = FastAPI(
    title="FastAPI MCP SSE",
    description="A demonstration of Server-Sent Events with Model Context "
    "Protocol integration",
    version="0.1.0",
)

# Create SSE transport instance for handling server-sent events
sse = SseServerTransport("/messages/")

# Mount the /messages path to handle SSE message posting
app.router.routes.append(Mount("/messages", app=sse.handle_post_message))


# Add documentation for the /messages endpoint
@app.get("/messages", tags=["MCP"], include_in_schema=True)
def messages_docs():
    """
    Messages endpoint for SSE communication

    This endpoint is used for posting messages to SSE clients.
    Note: This route is for documentation purposes only.
    The actual implementation is handled by the SSE transport.
    """
    pass  # This is just for documentation, the actual handler is mounted above


@app.get("/sse", tags=["MCP"])
async def handle_sse(request: Request):
    """
    SSE endpoint that connects to the MCP server

    This endpoint establishes a Server-Sent Events connection with the client
    and forwards communication to the Model Context Protocol server.
    """
    # Use sse.connect_sse to establish an SSE connection with the MCP server
    async with sse.connect_sse(request.scope, request.receive, request._send) as (
        read_stream,
        write_stream,
    ):
        # Run the MCP server with the established streams
        await mcp._mcp_server.run(
            read_stream,
            write_stream,
            mcp._mcp_server.create_initialization_options(),
        )


# Import routes at the end to avoid circular imports
# This ensures all routes are registered to the app
import routes  # noqa

-------------------------------------------------------------------------------------------------------
## pyproject.toml
[project]
name = "test-mcp"
version = "0.1.0"
description = "A working example to create a FastAPI server with SSE-based MCP support"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.11",
    "httpx>=0.28.1",
    "mcp[cli]>=1.6.0",
    "unicorn>=2.1.3",
]

[project.scripts]
start = "server:run"

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
package-dir = {"" = "src"}

-----------------------------------------------------------------------------------------------------
uv pip install -r pyproject.toml
python src/server.py
uv run start
