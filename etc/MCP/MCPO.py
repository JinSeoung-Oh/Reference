### From https://mychen76.medium.com/mcpo-supercharge-open-webui-with-mcp-tools-4ee55024c371

$ python -m venv .venv
$ source .venv/bin/activate
$ pip install mcpo 

# 1.time mcp server
$ pip install mcp-server-time

# 2.memory mcp server
$ npm install @modelcontextprotocol/server-memory

# 3.fetch mcp server
$ pip install mcp-server-fetch

---------------------------------------------------------------------------
❯ cat config.json 
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "time": {
      "command": "uvx",
      "args": ["mcp-server-time", "--local-timezone=America/New_York"]
    },
  "fetch": {
    "command": "uvx",
    "args": ["mcp-server-fetch"]
  }
  }
}

---------------------------------------------------------------------------
$ uvx mcpo --config config.json --port 8001
