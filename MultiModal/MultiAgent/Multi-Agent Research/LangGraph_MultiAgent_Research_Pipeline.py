### From https://www.marktechpost.com/2025/08/07/a-coding-implementation-to-advanced-langgraph-multi-agent-research-pipeline-for-automated-insights-generation/

!pip install -q langgraph langchain-google-genai langchain-core


import os
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
import json

os.environ["GOOGLE_API_KEY"] = "Use Your Own API Key"

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

class AgentState(TypedDict):
   messages: Annotated[List[BaseMessage], operator.add]
   current_agent: str
   research_data: dict
   analysis_complete: bool
   final_report: str
  
def simulate_web_search(query: str) -> str:
   """Simulated web search - replace with real API in production"""
   return f"Search results for '{query}': Found relevant information about {query} including recent developments, expert opinions, and statistical data."

def simulate_data_analysis(data: str) -> str:
   """Simulated data analysis tool"""
   return f"Analysis complete: Key insights from the data include emerging trends, statistical patterns, and actionable recommendations."

def research_agent(state: AgentState) -> AgentState:
   """Agent that researches a given topic"""
   messages = state["messages"]
   last_message = messages[-1].content
  
   search_results = simulate_web_search(last_message)
  
   prompt = f"""You are a research agent. Based on the query: '{last_message}'
  
   Here are the search results: {search_results}
  
   Conduct thorough research and gather relevant information. Provide structured findings with:
   1. Key facts and data points
   2. Current trends and developments 
   3. Expert opinions and insights
   4. Relevant statistics
  
   Be comprehensive and analytical in your research summary."""
  
   response = llm.invoke([HumanMessage(content=prompt)])
  
   research_data = {
       "topic": last_message,
       "findings": response.content,
       "search_results": search_results,
       "sources": ["academic_papers", "industry_reports", "expert_analyses"],
       "confidence": 0.88,
       "timestamp": "2024-research-session"
   }
  
   return {
       "messages": state["messages"] + [AIMessage(content=f"Research completed on '{last_message}': {response.content}")],
       "current_agent": "analysis",
       "research_data": research_data,
       "analysis_complete": False,
       "final_report": ""
   }
  
def analysis_agent(state: AgentState) -> AgentState:
   """Agent that analyzes research data and extracts insights"""
   research_data = state["research_data"]
  
   analysis_results = simulate_data_analysis(research_data.get('findings', ''))
  
   prompt = f"""You are an analysis agent. Analyze this research data in depth:
  
   Topic: {research_data.get('topic', 'Unknown')}
   Research Findings: {research_data.get('findings', 'No findings')}
   Analysis Results: {analysis_results}
  
   Provide deep insights including:
   1. Pattern identification and trend analysis
   2. Comparative analysis with industry standards
   3. Risk assessment and opportunities 
   4. Strategic implications
   5. Actionable recommendations with priority levels
  
   Be analytical and provide evidence-based insights."""
  
   response = llm.invoke([HumanMessage(content=prompt)])
  
   return {
       "messages": state["messages"] + [AIMessage(content=f"Analysis completed: {response.content}")],
       "current_agent": "report",
       "research_data": state["research_data"],
       "analysis_complete": True,
       "final_report": ""
   }

def report_agent(state: AgentState) -> AgentState:
   """Agent that generates final comprehensive reports"""
   research_data = state["research_data"]
  
   analysis_message = None
   for msg in reversed(state["messages"]):
       if isinstance(msg, AIMessage) and "Analysis completed:" in msg.content:
           analysis_message = msg.content.replace("Analysis completed: ", "")
           break
  
   prompt = f"""You are a professional report generation agent. Create a comprehensive executive report based on:
  
    Research Topic: {research_data.get('topic')}
    Research Findings: {research_data.get('findings')}
    Analysis Results: {analysis_message or 'Analysis pending'}
  
   Generate a well-structured, professional report with these sections:
  
   ## EXECUTIVE SUMMARY  
   ## KEY RESEARCH FINDINGS 
   [Detail the most important discoveries and data points]
  
   ## ANALYTICAL INSIGHTS
   [Present deep analysis, patterns, and trends identified]
  
   ## STRATEGIC RECOMMENDATIONS
   [Provide actionable recommendations with priority levels]
  
   ## RISK ASSESSMENT & OPPORTUNITIES
   [Identify potential risks and opportunities]
  
   ## CONCLUSION & NEXT STEPS
   [Summarize and suggest follow-up actions]
  
   Make the report professional, data-driven, and actionable."""
  
   response = llm.invoke([HumanMessage(content=prompt)])
  
   return {
       "messages": state["messages"] + [AIMessage(content=f" FINAL REPORT GENERATED:\n\n{response.content}")],
       "current_agent": "complete",
       "research_data": state["research_data"],
       "analysis_complete": True,
       "final_report": response.content
   }

def should_continue(state: AgentState) -> str:
   """Determine which agent should run next based on current state"""
   current_agent = state.get("current_agent", "research")
  
   if current_agent == "research":
       return "analysis"
   elif current_agent == "analysis":
       return "report"
   elif current_agent == "report":
       return END
   else:
       return END

workflow = StateGraph(AgentState)
workflow.add_node("research", research_agent)
workflow.add_node("analysis", analysis_agent)
workflow.add_node("report", report_agent)
workflow.add_conditional_edges(
   "research",
   should_continue,
   {"analysis": "analysis", END: END}
)
workflow.add_conditional_edges(
   "analysis",
   should_continue,
   {"report": "report", END: END}
)
workflow.add_conditional_edges(
   "report",
   should_continue,
   {END: END}
)
workflow.set_entry_point("research")
app = workflow.compile()


def run_research_assistant(query: str):
   """Run the complete research workflow"""
   initial_state = {
       "messages": [HumanMessage(content=query)],
       "current_agent": "research",
       "research_data": {},
       "analysis_complete": False,
       "final_report": ""
   }
  
   print(f" Starting Multi-Agent Research on: '{query}'")
   print("=" * 60)
  
   current_state = initial_state
  
   print(" Research Agent: Gathering information...")
   current_state = research_agent(current_state)
   print(" Research phase completed!\n")
  
   print(" Analysis Agent: Analyzing findings...")
   current_state = analysis_agent(current_state)
   print(" Analysis phase completed!\n")
  
   print(" Report Agent: Generating comprehensive report...")
   final_state = report_agent(current_state)
   print(" Report generation completed!\n")
  
   print("=" * 60)
   print(" MULTI-AGENT WORKFLOW COMPLETED SUCCESSFULLY!")
   print("=" * 60)
  
   final_report = final_state['final_report']
   print(f"\n COMPREHENSIVE RESEARCH REPORT:\n")
   print(final_report)
  
   return final_state

if __name__ == "__main__":
   print(" Advanced LangGraph Multi-Agent System Ready!")
   print(" Remember to set your GOOGLE_API_KEY!")
  
   example_queries = [
       "Impact of renewable energy on global markets",
       "Future of remote work post-pandemic"
   ]
  
   print(f"\n Example queries you can try:")
   for i, query in enumerate(example_queries, 1):
       print(f"  {i}. {query}")
  
   print(f"\n Usage: run_research_assistant('Your research question here')")
  
   result = run_research_assistant("What are emerging trends in sustainable technology?")

