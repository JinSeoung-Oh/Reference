### From https://www.marktechpost.com/2025/07/26/building-a-multi-node-graph-based-ai-agent-framework-for-complex-task-automation/?amp

!pip install -q google-generativeai networkx matplotlib

import google.generativeai as genai
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Callable
import json
import asyncio
from dataclasses import dataclass
from enum import Enum

API_KEY = "use your API key here"
genai.configure(api_key=API_KEY)

class NodeType(Enum):
    INPUT = "input"
    PROCESS = "process"
    DECISION = "decision"
    OUTPUT = "output"

@dataclass
class AgentNode:
    id: str
    type: NodeType
    prompt: str
    function: Callable = None
    dependencies: List[str] = None

def create_research_agent():
    agent = GraphAgent()

    # Input node
    agent.add_node(AgentNode(
        id="topic_input",
        type=NodeType.INPUT,
        prompt="Research topic input"
    ))

    agent.add_node(AgentNode(
        id="research_plan",
        type=NodeType.PROCESS,
        prompt="Create a comprehensive research plan for the topic. Include 3-5 key research questions and methodology.",
        dependencies=["topic_input"]
    ))

    agent.add_node(AgentNode(
        id="literature_review",
        type=NodeType.PROCESS,
        prompt="Conduct a thorough literature review. Identify key papers, theories, and current gaps in knowledge.",
        dependencies=["research_plan"]
    ))

    agent.add_node(AgentNode(
        id="analysis",
        type=NodeType.PROCESS,
        prompt="Analyze the research findings. Identify patterns, contradictions, and novel insights.",
        dependencies=["literature_review"]
    ))

    agent.add_node(AgentNode(
        id="quality_check",
        type=NodeType.DECISION,
        prompt="Evaluate research quality. Is the analysis comprehensive? Are there missing perspectives? Return 'APPROVED' or 'NEEDS_REVISION' with reasons.",
        dependencies=["analysis"]
    ))

    agent.add_node(AgentNode(
        id="final_report",
        type=NodeType.OUTPUT,
        prompt="Generate a comprehensive research report with executive summary, key findings, and recommendations.",
        dependencies=["quality_check"]
    ))

    return agent

def create_problem_solver():
    agent = GraphAgent()

    agent.add_node(AgentNode(
        id="problem_input",
        type=NodeType.INPUT,
        prompt="Problem statement"
    ))

    agent.add_node(AgentNode(
        id="problem_analysis",
        type=NodeType.PROCESS,
        prompt="Break down the problem into components. Identify constraints and requirements.",
        dependencies=["problem_input"]
    ))

    agent.add_node(AgentNode(
        id="solution_generation",
        type=NodeType.PROCESS,
        prompt="Generate 3 different solution approaches. For each, explain the methodology and expected outcomes.",
        dependencies=["problem_analysis"]
    ))

    agent.add_node(AgentNode(
        id="solution_evaluation",
        type=NodeType.DECISION,
        prompt="Evaluate each solution for feasibility, cost, and effectiveness. Rank them and select the best approach.",
        dependencies=["solution_generation"]
    ))

    agent.add_node(AgentNode(
        id="implementation_plan",
        type=NodeType.OUTPUT,
        prompt="Create a detailed implementation plan with timeline, resources, and success metrics.",
        dependencies=["solution_evaluation"]
    ))

    return agent

def run_research_demo():
    """Run the research agent demo"""
    print("🚀 Advanced Graph Agent Framework Demo")
    print("=" * 50)

    research_agent = create_research_agent()
    print("\n📊 Research Agent Graph Structure:")
    research_agent.visualize()

    print("\n🔍 Executing Research Task...")

    research_agent.results["topic_input"] = "Artificial Intelligence in Healthcare"

    execution_order = list(nx.topological_sort(research_agent.graph))

    for node_id in execution_order:
        if node_id == "topic_input":
            continue

        context = {}
        node = research_agent.nodes[node_id]

        if node.dependencies:
            for dep in node.dependencies:
                context[dep] = research_agent.results.get(dep, "")

        prompt = node.prompt
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            prompt = f"Context:\n{context_str}\n\nTask: {prompt}"

        try:
            response = research_agent.model.generate_content(prompt)
            result = response.text.strip()
            research_agent.results[node_id] = result
            print(f"✓ {node_id}: {result[:100]}...")
        except Exception as e:
            research_agent.results[node_id] = f"Error: {str(e)}"
            print(f"✗ {node_id}: Error - {str(e)}")

    print("\n📋 Research Results:")
    for node_id, result in research_agent.results.items():
        print(f"\n{node_id.upper()}:")
        print("-" * 30)
        print(result)

    return research_agent.results

def run_problem_solver_demo():
    """Run the problem solver demo"""
    print("\n" + "=" * 50)
    problem_solver = create_problem_solver()
    print("\n🛠️ Problem Solver Graph Structure:")
    problem_solver.visualize()

    print("\n⚙️ Executing Problem Solving...")

    problem_solver.results["problem_input"] = "How to reduce carbon emissions in urban transportation"

    execution_order = list(nx.topological_sort(problem_solver.graph))

    for node_id in execution_order:
        if node_id == "problem_input":
            continue

        context = {}
        node = problem_solver.nodes[node_id]

        if node.dependencies:
            for dep in node.dependencies:
                context[dep] = problem_solver.results.get(dep, "")

        prompt = node.prompt
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            prompt = f"Context:\n{context_str}\n\nTask: {prompt}"

        try:
            response = problem_solver.model.generate_content(prompt)
            result = response.text.strip()
            problem_solver.results[node_id] = result
            print(f"✓ {node_id}: {result[:100]}...")
        except Exception as e:
            problem_solver.results[node_id] = f"Error: {str(e)}"
            print(f"✗ {node_id}: Error - {str(e)}")

    print("\n📋 Problem Solving Results:")
    for node_id, result in problem_solver.results.items():
        print(f"\n{node_id.upper()}:")
        print("-" * 30)
        print(result)

    return problem_solver.results

print("🎯 Running Research Agent Demo:")
research_results = run_research_demo()

print("\n🎯 Running Problem Solver Demo:")
problem_results = run_problem_solver_demo()

print("\n✅ All demos completed successfully!")
