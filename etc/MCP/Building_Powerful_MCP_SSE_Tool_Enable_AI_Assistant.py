### From https://mychen76.medium.com/building-powerful-mcp-sse-tool-enable-ai-assistant-f94f401d2178
### From https://github.com/minyang-chen/customer-order-mcp-ai.git

"""
1. Overview
   MCP-SSE’s Enable AI-Assistant is a next-generation enterprise solution that combines generative AI, a microservices architecture,
   and MCP tools. It is designed to build advanced AI assistants that leverage natural language processing (NLP) to translate
   customer requests into API calls and provide real-time interactive updates on backend data.

2. Core Components of the Solution
   -a. Solution Architecture
       -1. Illustration (Figure-1):
           A diagram (Figure-1) shows a storefront AI assistant enabled by MCP tools, depicting how various microservices 
           and AI components interact.
       -2. Microservices Integration:
           The system is built on a microservices framework, where each service is small and independent, 
           communicating through clear APIs. This approach accelerates development, deployment, and scaling.
   -b. Core Technology Stack
       -1. Large Language Models (LLMs):
           -1) Serve as the “brain” of the AI assistant, enabling natural language understanding, generation, and interaction.
           -2) Popular providers include OpenAI, Google’s Gemini, Groq, Mistral AI, and Alibaba Cloud’s Qwen. 
               In this context, Groq is used for the Qwen 2.5 model.
       -2. Tool and Function Calling:
           -1) LLMs are enhanced to generate structured outputs (e.g., JSON) that detail exactly which functions to call and 
               what arguments to use.
           -2) This allows the assistant to interact with external tools and APIs, extending its capabilities beyond simple text 
               generation.
       -3. Model Context Protocol (MCP):
           -1) An open standard developed by Anthropic that acts as a universal connector.
           -2) Enables LLMs to seamlessly interact with external tools such as APIs, databases, and business applications.
           -3) MCP uses a client-server architecture where AI applications (MCP clients) connect to servers exposing data 
               and tools (MCP servers).

3. Communication Mechanisms
   -a. Protocols Supported:
       -1. STDIO (Standard Input/Output): Used for local integrations.
       -2. SSE (Server-Sent Events): Used for network-based communications.
   -b. Message Formatting:
       Both transport methods use JSON-RPC 2.0 to format messages.

4. Use Cases and Applications
   -a. General Use Cases
       -1. Accessing Databases:
           LLMs can query and retrieve data directly.
       -2. Using APIs:
           They can trigger actions (e.g., checking inventory, placing orders) by calling external APIs.
       -3. Interacting with Tools:
           LLMs leverage external tools to extend functionality and improve real-time responses.
   -b. AI SDK by Vercel
       -1. A TypeScript toolkit that aids developers in creating AI-powered applications using frameworks such as React,
           Next.js, and Node.js.
       -2. In this solution, Next.js is used to build both server-side service APIs and client-side applications for AI assistants.
    -c. What Is an AI Assistant with MCP Tool Enabled?
        -1. Enhanced Conversational AI:
            These assistants go beyond traditional chatbots by integrating tools, memory capabilities, and external data sources
            to understand user intents accurately.
        -2. Practical Capabilities:
            They can handle tasks like product inquiries, checking inventory, making purchases, and providing real-time updates 
            on order status.

5. Practical Use Case: Storefront AI/Assistant
   -a. Features:
       -1. Customer Service:
           Provides real-time access to product information and inventory levels, enabling custom orders.
       -2. Product Recommendations: 
           Suggests products based on customer preferences and current stock.
       -3. Real-Time Microservices Interaction:
           Uses the MCP tool server to interface with various microservices, ensuring dynamic updates.
       -4. Order Processing:
           Supports product purchases (using product IDs and quantities), updates inventory in real time, 
           and offers ad-hoc analytics on order transactions via natural language queries.
   -b. Components in the Storefront AI Assistant:
       -1. LLM Provider and Model with Tools Support:
           The core AI engine that understands and generates natural language.
       -2. Product API Server:
           A microservice exposing a /products API for retrieving product information.
       -3. Order Fulfillment API Server:
           A microservice with APIs for /order, /inventory, and /purchase to handle transactions.
       -4. MCP SSE Server:
           Exposes the order fulfillment APIs as tools to the LLM using the SSE protocol.
       -5. AI Assistant with MCP Tools Enabled (MCP Client):
           Captures user interactions, processes requests, and coordinates with the above services.

6. Key Takeaways
   -a. Integration Simplification:
       MCP significantly eases the process of connecting large language models with external tools and data sources.
   -b. Real-Time Interactivity:
       The solution leverages microservices and MCP's communication protocols to provide up-to-date, interactive responses.
   -c. Enterprise Focus:
       Designed for enterprise use, this architecture enables intelligent, context-aware AI assistants that enhance customer
       engagement and streamline operations.
"""

## Step 1. Create Order Fulfillment Microservices API
import express from "express";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.json());

.....................................

app.get("/inventory", async (req, res) => {
  const { inventory, orders, printers } = await getState();
  const inventoryWithDetails = inventory.map((item) => {
    const printer = printers.find((g: Printer) => g.id === item.printerId);
    return {
      ...item,
      printer,
    };
  });
  res.json(inventoryWithDetails);
});

app.get("/orders", async (req, res) => {
  const { orders } = await getState();
  res.json(
    [...orders]
      .sort(
        (a, b) =>
          new Date(a.orderDate).getTime() - new Date(b.orderDate).getTime()
      )
      .reverse()
  );
});

// @ts-ignore
app.post("/purchase", async (req, res) => {
  const { inventory, orders, printers } = await getState();

  const { customerName, items } = req.body as {
    customerName: string;
    items: Array<{ printerId: number; quantity: number }>;
  };

  if (!customerName || !items || items.length === 0) {
    return res.status(400).json({ error: "Invalid request body" });
  }

  let totalAmount = 0;
  for (const item of items) {
    const inventoryItem = inventory.find((i) => i.printerId === item.printerId);
    const printer = printers.find((g: Printer) => g.id === item.printerId);

    if (!inventoryItem || !printer) {
      return res
        .status(404)
        .json({ error: `Printer with id ${item.printerId} not found` });
    }

    if (inventoryItem.quantity < item.quantity) {
      return res.status(400).json({
        error: `Insufficient inventory for printer ${printer.name}. Available: ${inventoryItem.quantity}`,
      });
    }

    totalAmount += printer.price * item.quantity;
  }

  // Create order
  const order: Order = {
    id: orders.length + 1,
    customerName,
    items,
    totalAmount,
    orderDate: new Date().toISOString(),
  };

  // Update inventory
  items.forEach((item) => {
    const inventoryItem = inventory.find((i) => i.printerId === item.printerId)!;
    inventoryItem.quantity -= item.quantity;
  });

  orders.push(order);

  res.json(order);
});

const PORT = process.env.PORT || 8081;
app.listen(PORT, () => {
  console.log(
    `Printer Store Fulfillment API Server is running on port http://localhost:${PORT}`
  );
  console.log(`/inventory`);
  console.log(`/orders`);
  console.log(`/purchase`);    
});

----------------------------------------------------------------------------------------
## Step 2. Creating an MCP Server
//## order-server.js

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

export const server = new McpServer({
  name: "Printer Order Fulfillment MCP Server",
  version: "1.0.0",
});

server.tool('find-product', 'Find a product', {}, async () => {
  return {
    content: [
      {
        type: 'text',
        text: 'The Product is available in stock',
      },
    ],
  };
});

server.tool("getOrders", "Get product orders", async () => {
  console.error("Fetching orders");
  const res = await fetch("http://localhost:8081/orders");
  const orders = await res.json();

  return { content: [{ type: "text", text: JSON.stringify(orders) }] };
});

server.tool("getInventory", "Get product inventory", async () => {
  console.error("Fetching inventory");
  const res = await fetch("http://localhost:8081/inventory");
  const inventory = await res.json();

  return { content: [{ type: "text", text: JSON.stringify(inventory) }] };
});

server.tool(
  "buy",
  "buy a printer",
  {
    items: z
      .array(
        z.object({
          printerId: z.number().describe("ID of the printer to buy"),
          quantity: z.number().describe("Quantity of printer to buy"),
        })
      )
      .describe("list of printer to buy"),
    customerName: z.string().describe("Name of the customer"),
  },
  async ({ items, customerName }) => {
    console.error("Purchasing", { items, customerName });
    const res = await fetch("http://localhost:8081/purchase", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        items,
        customerName,
      }),
    });
    const order = await res.json();

    return { content: [{ type: "text", text: JSON.stringify(order) }] };
  }
);

----------------------------------------------------------------------------------------
## Implement the MCP SSEServerTransport
/## order-sse-server.js
import express from "express";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";

import { server } from "./order-server.js";

const app = express();

let transport;
app.get("/sse", async (req, res) => {
  transport = new SSEServerTransport("/messages", res);
  await server.connect(transport);
});

app.post("/messages", async (req, res) => {
  await transport.handlePostMessage(req, res);
});

const port = process.env.PORT || 8083;
app.listen(port, () => {
  console.log(`Printer Store MCP SSE Server is running on http://localhost:${port}/sse`);
  console.log("/sse")
  console.log("/messages")
});

----------------------------------------------------------------------------------------
## Step 3. Creating an MCP Client
  let mcpClient;
  try {
    mcpClient = await experimental_createMCPClient({
      transport: {
        type: "sse",
        url: "http://localhost:8083/sse",
        headers: {
          example: "header",
        },
      },
    });

    const toolset = await mcpClient.tools(); 

    // Generate Answer
    const { text: answer } = await generateText({
      model: groq("qwen-2.5-32b"), 
      tools: toolset,
      maxSteps: 10,
      onStepFinish: async ({ toolResults }) => {
        console.log(`\nSTEP RESULTS: ${JSON.stringify(toolResults, null, 2)}`);
      },
      system: SYSTEM_PROMPT,
      prompt: question,
      //temperature: 1.5,
    });
      console.log("FINAL ANSWER:", answer);
    }
