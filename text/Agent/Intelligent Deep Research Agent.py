### From https://medium.com/the-ai-forum/building-an-intelligent-deep-research-agent-from-query-to-insights-with-ai-powered-analysis-using-29f6fc47abad

#### 1. LLM Integration Layer (Groq + OpenAI Client)
def initialize_groq_model():
 """Initialize Groq client using OpenAI library with Groq endpoint."""
 client = OpenAI(
 base_url="https://api.groq.com/openai/v1",
 api_key=config.GROQ_API_KEY
 )
 return client

#### 2. Research Tools Layer (Tavily Integration)
class ResearchTools:
 def research_query(self, query: str, iterations: int = None):
 """Conduct comprehensive research with multiple iterations."""
 # Multi-iteration search with quality assessment
 # Real-time source validation
 # Content drift detection

#### 3. Agentic Decision Layer
class AgenticDecisionMaker:
 def make_decision(self, context: DecisionContext) -> Decision:
 """Make autonomous decisions about research direction."""
 # Analyze current results quality
 # Determine if more research is needed
 # Suggest search refinements

#### 4. Quality Assessment Layer
class RealWorldHandler:
 def process_sources(self, sources: List[Dict]) -> List[Dict]:
 """Assess and enhance source quality."""
 # URL validation and accessibility
 # Content drift detection
 # Disinformation risk assessment

------------------------------------------------------------------------------
#### The Deep Research Workflow: Step by Step
------------------------------------------------------------------------------
#### Phase 1: Query Intelligence & Planning
#### 1. Query Complexity Assessment
@dataclass
class QuerySpecification:
  complexity: QueryComplexity # SIMPLE, MODERATE, CHALLENGING
  query_type: QueryType # FACTUAL, ANALYTICAL, COMPARATIVE, SYNTHETIC
  determinacy: float # How specific is the query?
  difficulty: float # How hard to research?
  diversity: float # How many perspectives needed?

#### 2. Strategic Planning
def construct_challenging_query(base_query: str, complexity: QueryComplexity):
 """Create a comprehensive research specification."""
   return QuerySpecification(
   base_query=base_query,
   max_iterations=determine_iterations(complexity),
   expected_sources=calculate_source_needs(complexity),
   quality_threshold=set_quality_bar(complexity)
   )
------------------------------------------------------------------------------
#### Phase 2: Iterative Research Execution
#### 1. The Agentic Research Loop
while current_iteration < max_iterations:
   # 1. Execute search iteration
   results = self.tools.research_query(refined_query, 1)
   
   # 2. Assess quality and completeness
   quality_metrics = assess_results_quality(results)
   
   # 3. Make autonomous decision
   decision = self.decision_maker.make_decision(context)
   
   # 4. Execute decision
   if decision.decision_type == "termination":
     break
   elif decision.decision_type == "search_refinement":
     refined_query = refine_search_strategy(query, decision.parameters)
   elif decision.decision_type == "analysis_deepening":
     # Continue with enhanced focus areas

#### 2.Decision Context Example
@dataclass
class DecisionContext:
 current_results: Dict[str, Any]
 iteration_count: int
 sources_found: int
 expected_sources: int
 analysis_quality: float
 search_quality: float
 time_elapsed: float
------------------------------------------------------------------------------
#### Phase 3: Real-World Quality Assessment
#### 1. Multi-Dimensional Source Validation
def assess_source_quality(source: Dict) -> float:
   """Comprehensive source quality assessment."""
   
   # 1. URL Validation
   url_quality = validate_url_accessibility(source['url'])
   
   # 2. Content Drift Detection
   drift_score = check_content_drift(source)
   
   # 3. Disinformation Risk Assessment
   disinfo_risk = assess_disinformation_risk(source)
   
   # 4. Domain Authority
   domain_score = evaluate_domain_authority(source['url'])
   
   return calculate_composite_quality_score(
   url_quality, drift_score, disinfo_risk, domain_score
   )
------------------------------------------------------------------------------
#### Phase 4: Comprehensive Analysis Generation
#### 1. Multi-Stage Analysis Process
def _generate_analysis(self, query: str, research_results: Dict) -> str:
   """Generate comprehensive analysis using research findings."""
   
   # 1. Context Preparation
   findings_text = synthesize_key_findings(research_results)
   sources_text = format_source_attribution(research_results)
   
   # 2. Structured Analysis Prompt
   analysis_prompt = f"""
   Research Question: {query}
   
   Research Findings: {findings_text}
   Sources: {sources_text}
   Provide comprehensive analysis that:
   1. Directly addresses the research question
   2. Synthesizes key findings from sources
   3. Provides clear insights and conclusions
   4. Acknowledges limitations or gaps
   5. Is well-structured and accessible
   """
 
   # 3. LLM Generation with Research Context
   return self.generate_response(analysis_prompt)
------------------------------------------------------------------------------
#### Phase 5: Benchmarking & Performance Assessment
#### 1. Comprehensive Evaluation Framework
@dataclass
class BenchmarkMetrics:
 accuracy: float # Information accuracy (0–1)
 completeness: float # Coverage completeness (0–1)
 relevance: float # Query relevance (0–1)
 source_quality: float # Average source quality (0–1)
 response_time: float # Processing time (seconds)
 source_diversity: float # Source type diversity (0–1)
 reasoning_quality: float # Analysis coherence (0–1)

-----------------------------------------------------------------------------------------------------------------
#### Phase 1: Query Processing & Strategy Planning
#### src/api.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from src.agent import research_agent
from src.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Deep Research Agent API",
    description="A deep research agent powered by Groq Llama 3.3 Versatile",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pydantic models
class ResearchRequest(BaseModel):
    query: str
    iterations: Optional[int] = None
    research_type: Optional[str] = "deep"  # "deep" or "quick"
    complexity: Optional[str] = "moderate"  # "simple", "moderate", "challenging"
    query_type: Optional[str] = "analytical"  # "factual", "analytical", "comparative", "synthetic"

class ResearchResponse(BaseModel):
    success: bool
    query: str
    analysis: Optional[str] = None
    answer: Optional[str] = None
    sources: Optional[list] = None
    total_sources: Optional[int] = None
    iterations: Optional[int] = None
    error: Optional[str] = None
    benchmark_metrics: Optional[dict] = None
    agentic_decisions: Optional[dict] = None
    real_world_quality: Optional[float] = None
    query_specification: Optional[dict] = None

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api")
async def api_info():
    """API endpoint with API information."""
    return {
        "message": "Deep Research Agent API with Agentic Capabilities",
        "version": "2.0.0",
        "model": config.GROQ_MODEL,
        "endpoints": {
            "/research": "POST - Perform agentic deep research",
            "/quick-search": "POST - Perform quick search",
            "/health": "GET - Health check",
            "/config": "GET - Get configuration",
            "/benchmark/report": "GET - Get benchmark report",
            "/analytics/decisions": "GET - Get decision analytics",
            "/analytics/reset": "POST - Reset analytics"
        },
        "features": {
            "agentic_research": "Autonomous decision-making during research",
            "benchmarking": "Comprehensive evaluation metrics",
            "real_world_handling": "Content drift, URL decay, and disinformation detection",
            "query_complexity": "Support for simple, moderate, and challenging queries",
            "query_types": "Factual, analytical, comparative, and synthetic queries"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Basic health check
        return {
            "status": "healthy",
            "model": config.GROQ_MODEL,
            "api_keys_configured": bool(config.GROQ_API_KEY and config.TAVILY_API_KEY)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research", response_model=ResearchResponse)
async def perform_research(request: ResearchRequest):
    """
    Perform deep research on a given query with agentic capabilities.
    
    Args:
        request: Research request with query and optional parameters
        
    Returns:
        Research results with analysis, sources, and benchmark metrics
    """
    try:
        logger.info(f"Received research request: {request.query}")
        
        if request.research_type == "quick":
            results = research_agent.quick_search(request.query)
        else:
            # Convert string parameters to enums
            from src.benchmark_framework import QueryComplexity, QueryType
            
            complexity_map = {
                "simple": QueryComplexity.SIMPLE,
                "moderate": QueryComplexity.MODERATE,
                "challenging": QueryComplexity.CHALLENGING
            }
            
            query_type_map = {
                "factual": QueryType.FACTUAL,
                "analytical": QueryType.ANALYTICAL,
                "comparative": QueryType.COMPARATIVE,
                "synthetic": QueryType.SYNTHETIC
            }
            
            complexity = complexity_map.get(request.complexity, QueryComplexity.MODERATE)
            query_type = query_type_map.get(request.query_type, QueryType.ANALYTICAL)
            
            results = research_agent.research(
                query=request.query,
                iterations=request.iterations,
                complexity=complexity,
                query_type=query_type
            )
        
        return ResearchResponse(**results)
        
    except Exception as e:
        logger.error(f"Research failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quick-search", response_model=ResearchResponse)
async def quick_search(request: ResearchRequest):
    """
    Perform a quick web search for simple queries.
    
    Args:
        request: Search request with query
        
    Returns:
        Quick search results
    """
    try:
        logger.info(f"Received quick search request: {request.query}")
        
        results = research_agent.quick_search(request.query)
        return ResearchResponse(**results)
        
    except Exception as e:
        logger.error(f"Quick search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    """Get current configuration (without sensitive data)."""
    return {
        "model": config.GROQ_MODEL,
        "temperature": config.TEMPERATURE,
        "max_tokens": config.MAX_TOKENS,
        "max_research_iterations": config.MAX_RESEARCH_ITERATIONS,
        "max_search_results": config.MAX_SEARCH_RESULTS,
        "allow_clarification": config.ALLOW_CLARIFICATION
    }

@app.get("/benchmark/report")
async def get_benchmark_report():
    """Get comprehensive benchmark report."""
    try:
        report = research_agent.get_benchmark_report()
        return report
    except Exception as e:
        logger.error(f"Failed to get benchmark report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/decisions")
async def get_decision_analytics():
    """Get agentic decision analytics."""
    try:
        analytics = research_agent.decision_maker.get_decision_analytics()
        return analytics
    except Exception as e:
        logger.error(f"Failed to get decision analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analytics/reset")
async def reset_analytics():
    """Reset all analytics and decision history."""
    try:
        research_agent.reset_analytics()
        return {"message": "Analytics reset successfully"}
    except Exception as e:
        logger.error(f"Failed to reset analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host=config.HOST,
        port=config.PORT,
        reload=True
    ) 
------------------------------------------------------------------------------
#### src/benchmark_framework.py
"""
Benchmark Framework for evaluating research agent performance.
Inspired by the InfoDeepSeek paper methodology.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    CHALLENGING = "challenging"

class QueryType(Enum):
    """Types of research queries."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    SYNTHETIC = "synthetic"

@dataclass
class QuerySpecification:
    """Specification for a research query."""
    base_query: str
    complexity: QueryComplexity
    query_type: QueryType
    max_iterations: int
    expected_sources: int
    determinacy: float  # How well-defined the query is (0-1)
    difficulty: float   # How difficult to answer (0-1)
    diversity: float    # How diverse the sources should be (0-1)

@dataclass
class BenchmarkMetrics:
    """Benchmark evaluation metrics."""
    accuracy: float
    completeness: float
    relevance: float
    source_quality: float
    response_time: float
    source_diversity: float
    reasoning_quality: float

class BenchmarkFramework:
    """Framework for benchmarking research performance."""
    
    def __init__(self):
        """Initialize the benchmark framework."""
        self.evaluation_history = []
        self.performance_metrics = {
            "total_queries": 0,
            "average_accuracy": 0.0,
            "average_completeness": 0.0,
            "average_relevance": 0.0,
            "average_response_time": 0.0
        }
    
    def construct_challenging_query(self, base_query: str, 
                                  complexity: QueryComplexity,
                                  query_type: QueryType) -> QuerySpecification:
        """
        Construct a challenging query specification based on complexity and type.
        
        Args:
            base_query: The original query
            complexity: Query complexity level
            query_type: Type of query
            
        Returns:
            QuerySpecification with appropriate parameters
        """
        try:
            # Set parameters based on complexity
            complexity_params = {
                QueryComplexity.SIMPLE: {
                    "max_iterations": 2,
                    "expected_sources": 3,
                    "determinacy": 0.8,
                    "difficulty": 0.3,
                    "diversity": 0.5
                },
                QueryComplexity.MODERATE: {
                    "max_iterations": 3,
                    "expected_sources": 5,
                    "determinacy": 0.6,
                    "difficulty": 0.6,
                    "diversity": 0.7
                },
                QueryComplexity.CHALLENGING: {
                    "max_iterations": 5,
                    "expected_sources": 8,
                    "determinacy": 0.4,
                    "difficulty": 0.9,
                    "diversity": 0.9
                }
            }
            
            params = complexity_params.get(complexity, complexity_params[QueryComplexity.MODERATE])
            
            # Adjust parameters based on query type
            if query_type == QueryType.FACTUAL:
                params["determinacy"] += 0.2
                params["difficulty"] -= 0.1
            elif query_type == QueryType.SYNTHETIC:
                params["max_iterations"] += 1
                params["expected_sources"] += 2
                params["diversity"] += 0.1
            
            # Ensure values are within bounds
            for key in ["determinacy", "difficulty", "diversity"]:
                params[key] = max(0.0, min(1.0, params[key]))
            
            query_spec = QuerySpecification(
                base_query=base_query,
                complexity=complexity,
                query_type=query_type,
                max_iterations=params["max_iterations"],
                expected_sources=params["expected_sources"],
                determinacy=params["determinacy"],
                difficulty=params["difficulty"],
                diversity=params["diversity"]
            )
            
            logger.info(f"Constructed query specification: {complexity.value} {query_type.value}")
            return query_spec
            
        except Exception as e:
            logger.error(f"Error constructing query specification: {e}")
            # Return default specification
            return QuerySpecification(
                base_query=base_query,
                complexity=QueryComplexity.MODERATE,
                query_type=QueryType.ANALYTICAL,
                max_iterations=3,
                expected_sources=5,
                determinacy=0.6,
                difficulty=0.6,
                diversity=0.7
            )
    
    def evaluate_query(self, query_spec: QuerySpecification, 
                      results: Dict[str, Any]) -> BenchmarkMetrics:
        """
        Evaluate research results against query specification.
        
        Args:
            query_spec: Original query specification
            results: Research results to evaluate
            
        Returns:
            BenchmarkMetrics with evaluation scores
        """
        try:
            start_time = time.time()
            
            # Calculate accuracy (based on source quality and analysis quality)
            source_quality = self._evaluate_source_quality(results.get("sources", []))
            analysis_quality = self._evaluate_analysis_quality(results.get("analysis", ""))
            accuracy = (source_quality + analysis_quality) / 2
            
            # Calculate completeness (based on expected vs actual sources)
            expected_sources = query_spec.expected_sources
            actual_sources = results.get("total_sources", 0)
            completeness = min(1.0, actual_sources / expected_sources) if expected_sources > 0 else 0.0
            
            # Calculate relevance (based on query type and content alignment)
            relevance = self._evaluate_relevance(query_spec, results)
            
            # Calculate source diversity
            source_diversity = self._evaluate_source_diversity(results.get("sources", []))
            
            # Calculate reasoning quality
            reasoning_quality = self._evaluate_reasoning_quality(results.get("analysis", ""))
            
            # Response time (if available)
            response_time = results.get("response_time", time.time() - start_time)
            
            metrics = BenchmarkMetrics(
                accuracy=accuracy,
                completeness=completeness,
                relevance=relevance,
                source_quality=source_quality,
                response_time=response_time,
                source_diversity=source_diversity,
                reasoning_quality=reasoning_quality
            )
            
            # Store evaluation
            self.evaluation_history.append({
                "query_spec": query_spec,
                "results": results,
                "metrics": metrics,
                "timestamp": time.time()
            })
            
            # Update performance metrics
            self._update_performance_metrics(metrics)
            
            logger.info(f"Query evaluation completed. Accuracy: {accuracy:.2f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating query: {e}")
            # Return default metrics
            return BenchmarkMetrics(
                accuracy=0.5,
                completeness=0.5,
                relevance=0.5,
                source_quality=0.5,
                response_time=0.0,
                source_diversity=0.5,
                reasoning_quality=0.5
            )
    
    def _evaluate_source_quality(self, sources: List[Dict[str, Any]]) -> float:
        """Evaluate the quality of sources."""
        if not sources:
            return 0.0
        
        quality_scores = []
        for source in sources:
            score = 0.0
            
            # Check for URL presence
            if source.get("url"):
                score += 0.3
            
            # Check for title
            if source.get("title"):
                score += 0.2
            
            # Check for content length
            content = source.get("content", "")
            if len(content) > 100:
                score += 0.3
            
            # Check for relevance score
            if source.get("score", 0) > 0.5:
                score += 0.2
            
            quality_scores.append(score)
        
        return sum(quality_scores) / len(quality_scores)
    
    def _evaluate_analysis_quality(self, analysis: str) -> float:
        """Evaluate the quality of analysis."""
        if not analysis:
            return 0.0
        
        score = 0.0
        
        # Length-based scoring
        if len(analysis) > 500:
            score += 0.3
        elif len(analysis) > 200:
            score += 0.2
        elif len(analysis) > 100:
            score += 0.1
        
        # Structure-based scoring
        if "conclusion" in analysis.lower() or "summary" in analysis.lower():
            score += 0.2
        
        # Evidence-based scoring
        if "according to" in analysis.lower() or "based on" in analysis.lower():
            score += 0.2
        
        # Analysis depth
        if "however" in analysis.lower() or "furthermore" in analysis.lower():
            score += 0.3
        
        return min(1.0, score)
    
    def _evaluate_relevance(self, query_spec: QuerySpecification, 
                          results: Dict[str, Any]) -> float:
        """Evaluate relevance of results to query."""
        # Simple keyword-based relevance
        query_words = set(query_spec.base_query.lower().split())
        analysis = results.get("analysis", "").lower()
        
        if not analysis:
            return 0.0
        
        analysis_words = set(analysis.split())
        overlap = len(query_words.intersection(analysis_words))
        relevance = overlap / len(query_words) if query_words else 0.0
        
        return min(1.0, relevance * 2)  # Scale up for better scoring
    
    def _evaluate_source_diversity(self, sources: List[Dict[str, Any]]) -> float:
        """Evaluate diversity of sources."""
        if not sources:
            return 0.0
        
        # Simple domain-based diversity
        domains = set()
        for source in sources:
            url = source.get("url", "")
            if url:
                try:
                    domain = url.split("//")[1].split("/")[0]
                    domains.add(domain)
                except:
                    pass
        
        # Diversity score based on unique domains
        diversity = len(domains) / len(sources) if sources else 0.0
        return min(1.0, diversity * 2)  # Scale up
    
    def _evaluate_reasoning_quality(self, analysis: str) -> float:
        """Evaluate the quality of reasoning in analysis."""
        if not analysis:
            return 0.0
        
        score = 0.0
        analysis_lower = analysis.lower()
        
        # Logical connectors
        connectors = ["therefore", "because", "since", "thus", "consequently", "as a result"]
        for connector in connectors:
            if connector in analysis_lower:
                score += 0.1
        
        # Evidence presentation
        evidence_markers = ["research shows", "studies indicate", "data suggests", "findings reveal"]
        for marker in evidence_markers:
            if marker in analysis_lower:
                score += 0.1
        
        # Balanced analysis
        if "on the other hand" in analysis_lower or "alternatively" in analysis_lower:
            score += 0.2
        
        return min(1.0, score)
    
    def _update_performance_metrics(self, metrics: BenchmarkMetrics):
        """Update overall performance metrics."""
        self.performance_metrics["total_queries"] += 1
        total = self.performance_metrics["total_queries"]
        
        # Running average calculation
        self.performance_metrics["average_accuracy"] = (
            (self.performance_metrics["average_accuracy"] * (total - 1) + metrics.accuracy) / total
        )
        self.performance_metrics["average_completeness"] = (
            (self.performance_metrics["average_completeness"] * (total - 1) + metrics.completeness) / total
        )
        self.performance_metrics["average_relevance"] = (
            (self.performance_metrics["average_relevance"] * (total - 1) + metrics.relevance) / total
        )
        self.performance_metrics["average_response_time"] = (
            (self.performance_metrics["average_response_time"] * (total - 1) + metrics.response_time) / total
        )
    
    def get_benchmark_report(self) -> Dict[str, Any]:
        """Get comprehensive benchmark report."""
        try:
            if not self.evaluation_history:
                return {
                    "status": "No evaluations performed yet",
                    "performance_summary": self.performance_metrics,
                    "recent_evaluations": []
                }
            
            # Recent evaluations (last 10)
            recent_evaluations = []
            for eval_data in self.evaluation_history[-10:]:
                recent_evaluations.append({
                    "query": eval_data["query_spec"].base_query,
                    "complexity": eval_data["query_spec"].complexity.value,
                    "accuracy": eval_data["metrics"].accuracy,
                    "completeness": eval_data["metrics"].completeness,
                    "relevance": eval_data["metrics"].relevance
                })
            
            # Calculate overall score
            overall_score = (
                self.performance_metrics["average_accuracy"] * 0.3 +
                self.performance_metrics["average_completeness"] * 0.25 +
                self.performance_metrics["average_relevance"] * 0.25 +
                (1.0 - min(1.0, self.performance_metrics["average_response_time"] / 60.0)) * 0.2
            )
            
            self.performance_metrics["overall_score"] = overall_score
            
            return {
                "status": "success",
                "performance_summary": self.performance_metrics,
                "recent_evaluations": recent_evaluations,
                "total_evaluations": len(self.evaluation_history)
            }
            
        except Exception as e:
            logger.error(f"Error generating benchmark report: {e}")
            return {
                "status": "error",
                "error": str(e),
                "performance_summary": self.performance_metrics
            }

# Global benchmark framework instance
benchmark_framework = BenchmarkFramework() 
------------------------------------------------------------------------------
#### src/agent.py
from src.models import groq_model, generate_response
from src.research_tools import research_tools
from src.config import config
from src.benchmark_framework import benchmark_framework, QueryComplexity, QueryType
from src.agentic_decision_maker import agentic_decision_maker, DecisionContext
from src.real_world_handler import real_world_handler
import logging
from typing import Dict, Any, List
import time

logger = logging.getLogger(__name__)

class DeepResearchAgent:
    """Deep Research Agent using Groq with agentic capabilities."""
    
    def __init__(self):
        """Initialize the research agent."""
        self.client = groq_model
        self.tools = research_tools
        self.benchmark_framework = benchmark_framework
        self.decision_maker = agentic_decision_maker
        self.real_world_handler = real_world_handler
        logger.info("Deep Research Agent initialized successfully")
    
    def research(self, query: str, iterations: int = None, 
                complexity: QueryComplexity = QueryComplexity.MODERATE,
                query_type: QueryType = QueryType.ANALYTICAL) -> Dict[str, Any]:
        """
        Perform deep research on a given query with agentic capabilities.
        
        Args:
            query: Research question
            iterations: Number of research iterations
            complexity: Query complexity level
            query_type: Type of query
            
        Returns:
            Research results with findings and analysis
        """
        try:
            logger.info(f"Starting agentic research on: {query}")
            start_time = time.time()
            
            # Construct challenging query specification
            query_spec = self.benchmark_framework.construct_challenging_query(
                base_query=query,
                complexity=complexity,
                query_type=query_type
            )
            
            # Initialize research state
            current_iteration = 0
            all_sources = []
            all_findings = []
            analysis_quality = 0.0
            search_quality = 0.0
            
            # Agentic research loop
            while current_iteration < query_spec.max_iterations:
                current_iteration += 1
                logger.info(f"Research iteration {current_iteration}/{query_spec.max_iterations}")
                
                # Perform research iteration
                iteration_results = self.tools.research_query(query_spec.base_query, 1)
                
                if "error" not in iteration_results:
                    all_sources.extend(iteration_results.get("sources", []))
                    all_findings.extend(iteration_results.get("findings", []))
                    
                    # Update quality metrics
                    search_quality = min(1.0, len(all_sources) / query_spec.expected_sources)
                
                # Create decision context
                decision_context = DecisionContext(
                    current_query=query_spec.base_query,
                    current_results={
                        "sources": all_sources,
                        "findings": all_findings,
                        "analysis_quality": analysis_quality,
                        "search_quality": search_quality
                    },
                    iteration_count=current_iteration,
                    max_iterations=query_spec.max_iterations,
                    sources_found=len(all_sources),
                    expected_sources=query_spec.expected_sources,
                    analysis_quality=analysis_quality,
                    search_quality=search_quality,
                    time_elapsed=time.time() - start_time
                )
                
                # Make agentic decision
                decision = self.decision_maker.make_decision(decision_context)
                logger.info(f"Agentic decision: {decision.decision_type.value} with {decision.confidence.value} confidence")
                
                # Execute decision
                if decision.decision_type.value == "termination":
                    break
                elif decision.decision_type.value == "search_refinement":
                    # Refine search strategy
                    query_spec.base_query = self._refine_query(query_spec.base_query, decision.parameters)
                
                # Add delay between iterations
                if current_iteration < query_spec.max_iterations:
                    time.sleep(1)
            
            # Process sources through real-world handlers
            processed_sources = self.real_world_handler.process_sources(all_sources)
            
            # Generate comprehensive analysis
            analysis = self._generate_analysis(query, {
                "sources": processed_sources,
                "findings": all_findings
            })
            
            # Update analysis quality
            analysis_quality = self._assess_analysis_quality(analysis, processed_sources)
            
            # Evaluate results using benchmark framework
            final_results = {
                "success": True,
                "query": query,
                "analysis": analysis,
                "sources": processed_sources,
                "findings": all_findings,
                "total_sources": len(processed_sources),
                "iterations": current_iteration,
                "query_specification": {
                    "complexity": complexity.value,
                    "query_type": query_type.value,
                    "determinacy": query_spec.determinacy,
                    "difficulty": query_spec.difficulty,
                    "diversity": query_spec.diversity
                },
                "agentic_decisions": self.decision_maker.get_decision_analytics(),
                "real_world_quality": self._calculate_overall_quality(processed_sources)
            }
            
            # Benchmark evaluation
            benchmark_metrics = self.benchmark_framework.evaluate_query(query_spec, final_results)
            final_results["benchmark_metrics"] = {
                "accuracy": benchmark_metrics.accuracy,
                "completeness": benchmark_metrics.completeness,
                "relevance": benchmark_metrics.relevance,
                "source_quality": benchmark_metrics.source_quality,
                "response_time": benchmark_metrics.response_time,
                "source_diversity": benchmark_metrics.source_diversity,
                "reasoning_quality": benchmark_metrics.reasoning_quality
            }
            
            logger.info(f"Agentic research completed. Found {final_results['total_sources']} sources.")
            return final_results
            
        except Exception as e:
            logger.error(f"Agentic research failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def _generate_analysis(self, query: str, research_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive analysis using Groq model.
        
        Args:
            query: Original research question
            research_results: Research findings and sources
            
        Returns:
            Generated analysis
        """
        try:
            # Prepare context from research findings
            findings_text = "\n\n".join(research_results.get("findings", [])[:10])
            sources_text = "\n".join([
                f"- {source.get('title', 'No title')}: {source.get('url', 'No URL')}"
                for source in research_results.get("sources", [])[:5]
            ])
            
            analysis_prompt = f"""
            Research Question: {query}
            
            Research Findings:
            {findings_text}
            
            Sources:
            {sources_text}
            
            Please provide a comprehensive analysis that:
            1. Directly addresses the research question
            2. Synthesizes the key findings from the sources
            3. Provides clear insights and conclusions
            4. Acknowledges any limitations or gaps in the research
            5. Is well-structured and easy to understand
            
            Format your response with clear sections and bullet points where appropriate.
            """
            
            analysis = generate_response(
                self.client,
                [{"role": "user", "content": analysis_prompt}],
                system_prompt="You are a professional research analyst. Provide thorough, well-reasoned analysis based on the provided research findings."
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to generate analysis: {e}")
            return f"Error generating analysis: {str(e)}"
    
    def quick_search(self, query: str) -> Dict[str, Any]:
        """
        Perform a quick web search for simple queries.
        
        Args:
            query: Search query
            
        Returns:
            Quick search results
        """
        try:
            logger.info(f"Performing quick search for: {query}")
            
            search_results = self.tools.web_search(query)
            
            if "error" in search_results:
                return {
                    "success": False,
                    "error": search_results["error"],
                    "query": query
                }
            
            # Generate a brief response
            brief_analysis = self._generate_brief_analysis(query, search_results)
            
            return {
                "success": True,
                "query": query,
                "answer": search_results.get("answer", ""),
                "analysis": brief_analysis,
                "sources": search_results.get("results", [])
            }
            
        except Exception as e:
            logger.error(f"Quick search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def _generate_brief_analysis(self, query: str, search_results: Dict[str, Any]) -> str:
        """
        Generate a brief analysis for quick searches.
        
        Args:
            query: Search query
            search_results: Search results
            
        Returns:
            Brief analysis
        """
        try:
            results_text = "\n\n".join([
                f"- {result.get('title', 'No title')}: {result.get('content', '')[:200]}..."
                for result in search_results.get("results", [])[:3]
            ])
            
            brief_prompt = f"""
            Query: {query}
            
            Search Results:
            {results_text}
            
            Please provide a brief, informative response that directly answers the query based on the search results.
            Keep it concise but comprehensive.
            """
            
            brief_analysis = generate_response(
                self.client,
                [{"role": "user", "content": brief_prompt}],
                system_prompt="You are a helpful assistant. Provide clear, concise answers based on the provided information."
            )
            
            return brief_analysis
            
        except Exception as e:
            logger.error(f"Failed to generate brief analysis: {e}")
            return f"Error generating brief analysis: {str(e)}"
    
    def _refine_query(self, current_query: str, parameters: Dict[str, Any]) -> str:
        """Refine query based on decision parameters."""
        try:
            refinement_type = parameters.get("refinement_type", "semantic_expansion")
            focus_areas = parameters.get("focus_areas", [])
            
            if refinement_type == "semantic_expansion" and focus_areas:
                # Add focus areas to query
                focus_text = " ".join(focus_areas)
                refined_query = f"{current_query} {focus_text}"
            else:
                # Simple query expansion
                refined_query = f"{current_query} detailed analysis comprehensive research"
            
            return refined_query
            
        except Exception as e:
            logger.error(f"Error refining query: {e}")
            return current_query
    
    def _assess_analysis_quality(self, analysis: str, sources: List[Dict[str, Any]]) -> float:
        """Assess the quality of generated analysis."""
        try:
            # Simple quality assessment based on length and source usage
            analysis_length = len(analysis)
            source_count = len(sources)
            
            # Normalize quality score
            length_score = min(1.0, analysis_length / 1000)  # Target 1000+ characters
            source_score = min(1.0, source_count / 5)  # Target 5+ sources
            
            quality_score = (length_score + source_score) / 2
            return quality_score
            
        except Exception as e:
            logger.error(f"Error assessing analysis quality: {e}")
            return 0.5
    
    def _calculate_overall_quality(self, sources: List[Dict[str, Any]]) -> float:
        """Calculate overall quality score for sources."""
        try:
            if not sources:
                return 0.0
            
            quality_scores = [source.get("real_world_quality", 1.0) for source in sources]
            return sum(quality_scores) / len(quality_scores)
            
        except Exception as e:
            logger.error(f"Error calculating overall quality: {e}")
            return 0.5
    
    def get_benchmark_report(self) -> Dict[str, Any]:
        """Get comprehensive benchmark report."""
        try:
            return self.benchmark_framework.get_benchmark_report()
        except Exception as e:
            logger.error(f"Error getting benchmark report: {e}")
            return {"error": str(e)}
    
    def reset_analytics(self):
        """Reset all analytics and decision history."""
        try:
            self.decision_maker.reset_analytics()
            logger.info("Analytics reset successfully")
        except Exception as e:
            logger.error(f"Error resetting analytics: {e}")

# Global agent instance
research_agent = DeepResearchAgent() 
------------------------------------------------------------------------------
#### Phase 2: Agentic Research Loop Initialization
#### src/research_tools.py
from tavily import TavilyClient
from src.config import config
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Any
import time

logger = logging.getLogger(__name__)

class ResearchTools:
    """Collection of research tools for the Deep Research Agent."""
    
    def __init__(self):
        """Initialize research tools."""
        try:
            self.tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)
            logger.info("Tavily client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tavily client: {e}")
            raise
    
    def web_search(self, query: str, max_results: int = None) -> Dict[str, Any]:
        """
        Perform web search using Tavily.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing search results
        """
        try:
            max_results = max_results or config.MAX_SEARCH_RESULTS
            
            search_result = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True,
                include_raw_content=True
            )
            
            logger.info(f"Web search completed for query: {query}")
            return search_result
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {
                "error": str(e),
                "results": [],
                "answer": "Search failed"
            }
    
    def research_query(self, query: str, iterations: int = None) -> Dict[str, Any]:
        """
        Perform comprehensive research on a query.
        
        Args:
            query: Research question
            iterations: Number of research iterations
            
        Returns:
            Research results with findings and sources
        """
        iterations = iterations or config.MAX_RESEARCH_ITERATIONS
        findings = []
        sources = []
        
        try:
            for i in range(iterations):
                logger.info(f"Research iteration {i + 1}/{iterations}")
                
                # Perform web search
                search_results = self.web_search(query)
                
                if "error" in search_results:
                    continue
                
                # Process results
                for result in search_results.get("results", []):
                    source_info = {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "score": result.get("score", 0)
                    }
                    
                    sources.append(source_info)
                    findings.append(result.get("content", ""))
                
                # Add delay between iterations
                if i < iterations - 1:
                    time.sleep(1)
            
            # Compile research summary
            research_summary = {
                "query": query,
                "iterations": iterations,
                "findings": findings,
                "sources": sources,
                "answer": search_results.get("answer", ""),
                "total_sources": len(sources)
            }
            
            logger.info(f"Research completed. Found {len(sources)} sources.")
            return research_summary
            
        except Exception as e:
            logger.error(f"Research failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "findings": [],
                "sources": [],
                "answer": "Research failed"
            }

# Global research tools instance
research_tools = ResearchTools() 

------------------------------------------------------------------------------
#### Phase 3: Web Search & Source Collection
#### src/config.py
import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Deep Research Agent."""
    
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    
    # Model Configuration
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
    
    # Research Configuration
    MAX_RESEARCH_ITERATIONS: int = int(os.getenv("MAX_RESEARCH_ITERATIONS", "3"))
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    ALLOW_CLARIFICATION: bool = os.getenv("ALLOW_CLARIFICATION", "true").lower() == "true"
    
    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required API keys are present."""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required. Please set it in your .env file.")
        if not cls.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is required. Please set it in your .env file.")
        return True

# Global config instance
config = Config() 

------------------------------------------------------------------------------
#### Phase 4: Real-World Quality Assessment
#### src/real_world_handler.py
"""
Real-world scenario handler for content drift, URL decay, and disinformation detection.
Inspired by the InfoDeepSeek paper's real-world challenges.
"""

import requests
import time
import hashlib
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

class RealWorldHandler:
    """Handles real-world scenarios like content drift, URL decay, and disinformation."""
    
    def __init__(self):
        """Initialize the real-world handler."""
        self.content_cache = {}
        self.url_validation_cache = {}
        self.disinformation_indicators = [
            "fake news", "conspiracy", "unverified", "rumor", "hoax",
            "clickbait", "misleading", "false claim", "debunked"
        ]
        self.quality_indicators = [
            "research", "study", "academic", "peer-reviewed", "scientific",
            "official", "government", "institution", "university", "journal"
        ]
    
    def process_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process sources through real-world handlers.
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            Processed sources with quality assessments
        """
        try:
            processed_sources = []
            
            for source in sources:
                processed_source = source.copy()
                
                # Validate URL
                url_status = self._validate_url(source.get("url", ""))
                processed_source["url_status"] = url_status
                
                # Check for content drift
                content_drift = self._check_content_drift(source)
                processed_source["content_drift"] = content_drift
                
                # Assess disinformation risk
                disinfo_risk = self._assess_disinformation_risk(source)
                processed_source["disinformation_risk"] = disinfo_risk
                
                # Calculate overall quality
                quality_score = self._calculate_quality_score(processed_source)
                processed_source["real_world_quality"] = quality_score
                
                processed_sources.append(processed_source)
            
            logger.info(f"Processed {len(processed_sources)} sources through real-world handlers")
            return processed_sources
            
        except Exception as e:
            logger.error(f"Error processing sources: {e}")
            # Return original sources if processing fails
            return sources
    
    def _validate_url(self, url: str) -> Dict[str, Any]:
        """
        Validate URL accessibility and status.
        
        Args:
            url: URL to validate
            
        Returns:
            URL validation status
        """
        try:
            if not url or not url.startswith(("http://", "https://")):
                return {
                    "valid": False,
                    "status_code": None,
                    "error": "Invalid URL format",
                    "accessible": False
                }
            
            # Check cache first
            if url in self.url_validation_cache:
                cached_result = self.url_validation_cache[url]
                # Use cached result if less than 1 hour old
                if time.time() - cached_result["timestamp"] < 3600:
                    return cached_result["status"]
            
            # Validate URL
            try:
                response = requests.head(url, timeout=10, allow_redirects=True)
                status = {
                    "valid": True,
                    "status_code": response.status_code,
                    "accessible": response.status_code < 400,
                    "redirected": len(response.history) > 0,
                    "final_url": response.url if response.url != url else None
                }
                
                # Cache result
                self.url_validation_cache[url] = {
                    "status": status,
                    "timestamp": time.time()
                }
                
                return status
                
            except requests.exceptions.RequestException as e:
                status = {
                    "valid": False,
                    "status_code": None,
                    "error": str(e),
                    "accessible": False
                }
                
                # Cache negative result for shorter time
                self.url_validation_cache[url] = {
                    "status": status,
                    "timestamp": time.time()
                }
                
                return status
                
        except Exception as e:
            logger.error(f"Error validating URL {url}: {e}")
            return {
                "valid": False,
                "status_code": None,
                "error": f"Validation error: {str(e)}",
                "accessible": False
            }
    
    def _check_content_drift(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for content drift in source.
        
        Args:
            source: Source dictionary
            
        Returns:
            Content drift assessment
        """
        try:
            url = source.get("url", "")
            content = source.get("content", "")
            
            if not url or not content:
                return {
                    "drift_detected": False,
                    "confidence": 0.0,
                    "reason": "Insufficient data for drift detection"
                }
            
            # Create content hash
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Check if we have previous content for this URL
            if url in self.content_cache:
                cached_hash = self.content_cache[url]["hash"]
                cached_time = self.content_cache[url]["timestamp"]
                
                # If content has changed significantly
                if cached_hash != content_hash:
                    time_diff = time.time() - cached_time
                    
                    # Significant change in short time indicates drift
                    if time_diff < 3600:  # Less than 1 hour
                        drift_confidence = min(1.0, (3600 - time_diff) / 3600)
                        
                        self.content_cache[url] = {
                            "hash": content_hash,
                            "timestamp": time.time()
                        }
                        
                        return {
                            "drift_detected": True,
                            "confidence": drift_confidence,
                            "reason": f"Content changed within {time_diff/60:.1f} minutes"
                        }
            
            # Update cache
            self.content_cache[url] = {
                "hash": content_hash,
                "timestamp": time.time()
            }
            
            return {
                "drift_detected": False,
                "confidence": 0.0,
                "reason": "No drift detected or first observation"
            }
            
        except Exception as e:
            logger.error(f"Error checking content drift: {e}")
            return {
                "drift_detected": False,
                "confidence": 0.0,
                "reason": f"Error in drift detection: {str(e)}"
            }
    
    def _assess_disinformation_risk(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess disinformation risk in source.
        
        Args:
            source: Source dictionary
            
        Returns:
            Disinformation risk assessment
        """
        try:
            content = source.get("content", "").lower()
            title = source.get("title", "").lower()
            url = source.get("url", "").lower()
            
            combined_text = f"{title} {content}"
            
            # Check for disinformation indicators
            disinfo_score = 0.0
            disinfo_indicators_found = []
            
            for indicator in self.disinformation_indicators:
                if indicator in combined_text:
                    disinfo_score += 0.1
                    disinfo_indicators_found.append(indicator)
            
            # Check for quality indicators (reduce risk)
            quality_score = 0.0
            quality_indicators_found = []
            
            for indicator in self.quality_indicators:
                if indicator in combined_text:
                    quality_score += 0.1
                    quality_indicators_found.append(indicator)
            
            # Domain-based assessment
            domain_risk = 0.0
            if url:
                try:
                    domain = urlparse(url).netloc.lower()
                    
                    # High-risk domains
                    if any(risk_term in domain for risk_term in ["fake", "conspiracy", "hoax"]):
                        domain_risk += 0.3
                    
                    # Low-risk domains
                    if any(trust_term in domain for trust_term in [".edu", ".gov", ".org"]):
                        domain_risk -= 0.2
                        
                except:
                    pass
            
            # Calculate overall risk
            overall_risk = max(0.0, min(1.0, disinfo_score + domain_risk - quality_score))
            
            # Risk categories
            if overall_risk < 0.3:
                risk_level = "low"
            elif overall_risk < 0.6:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            return {
                "risk_score": overall_risk,
                "risk_level": risk_level,
                "disinformation_indicators": disinfo_indicators_found,
                "quality_indicators": quality_indicators_found,
                "domain_assessment": {
                    "domain_risk": domain_risk,
                    "domain": urlparse(url).netloc if url else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error assessing disinformation risk: {e}")
            return {
                "risk_score": 0.5,
                "risk_level": "unknown",
                "error": str(e)
            }
    
    def _calculate_quality_score(self, source: Dict[str, Any]) -> float:
        """
        Calculate overall quality score for source.
        
        Args:
            source: Processed source dictionary
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        try:
            quality_components = []
            
            # URL accessibility (25% weight)
            url_status = source.get("url_status", {})
            if url_status.get("accessible", False):
                quality_components.append(("url_accessibility", 1.0, 0.25))
            else:
                quality_components.append(("url_accessibility", 0.0, 0.25))
            
            # Content drift (20% weight) - lower drift is better
            content_drift = source.get("content_drift", {})
            drift_detected = content_drift.get("drift_detected", False)
            if drift_detected:
                drift_penalty = content_drift.get("confidence", 0.0)
                quality_components.append(("content_stability", 1.0 - drift_penalty, 0.20))
            else:
                quality_components.append(("content_stability", 1.0, 0.20))
            
            # Disinformation risk (30% weight) - lower risk is better
            disinfo_risk = source.get("disinformation_risk", {})
            risk_score = disinfo_risk.get("risk_score", 0.5)
            quality_components.append(("information_reliability", 1.0 - risk_score, 0.30))
            
            # Source metadata quality (25% weight)
            metadata_score = 0.0
            if source.get("title"):
                metadata_score += 0.4
            if source.get("content") and len(source["content"]) > 100:
                metadata_score += 0.4
            if source.get("score", 0) > 0.5:
                metadata_score += 0.2
            
            quality_components.append(("metadata_quality", metadata_score, 0.25))
            
            # Calculate weighted average
            total_score = sum(score * weight for _, score, weight in quality_components)
            
            # Ensure score is between 0 and 1
            final_score = max(0.0, min(1.0, total_score))
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 0.5  # Default neutral score
    
    def get_quality_report(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate quality report for processed sources.
        
        Args:
            sources: List of processed sources
            
        Returns:
            Quality report
        """
        try:
            if not sources:
                return {"error": "No sources to analyze"}
            
            # Overall statistics
            total_sources = len(sources)
            accessible_urls = sum(1 for s in sources if s.get("url_status", {}).get("accessible", False))
            high_quality = sum(1 for s in sources if s.get("real_world_quality", 0) > 0.7)
            medium_quality = sum(1 for s in sources if 0.4 <= s.get("real_world_quality", 0) <= 0.7)
            low_quality = sum(1 for s in sources if s.get("real_world_quality", 0) < 0.4)
            
            # Risk assessment
            high_risk = sum(1 for s in sources if s.get("disinformation_risk", {}).get("risk_level") == "high")
            medium_risk = sum(1 for s in sources if s.get("disinformation_risk", {}).get("risk_level") == "medium")
            low_risk = sum(1 for s in sources if s.get("disinformation_risk", {}).get("risk_level") == "low")
            
            # Content drift
            drift_detected = sum(1 for s in sources if s.get("content_drift", {}).get("drift_detected", False))
            
            # Average quality score
            quality_scores = [s.get("real_world_quality", 0) for s in sources]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            return {
                "total_sources": total_sources,
                "accessibility": {
                    "accessible": accessible_urls,
                    "inaccessible": total_sources - accessible_urls,
                    "accessibility_rate": accessible_urls / total_sources if total_sources > 0 else 0
                },
                "quality_distribution": {
                    "high_quality": high_quality,
                    "medium_quality": medium_quality,
                    "low_quality": low_quality,
                    "average_quality": avg_quality
                },
                "risk_assessment": {
                    "high_risk": high_risk,
                    "medium_risk": medium_risk,
                    "low_risk": low_risk,
                    "risk_free_rate": low_risk / total_sources if total_sources > 0 else 0
                },
                "content_stability": {
                    "drift_detected": drift_detected,
                    "stable_sources": total_sources - drift_detected,
                    "stability_rate": (total_sources - drift_detected) / total_sources if total_sources > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            return {"error": str(e)}

# Global real-world handler instance
real_world_handler = RealWorldHandler() 

------------------------------------------------------------------------------
#### Phase 5: Agentic Decision Making
#### src/agentic_decision_maker.py
"""
Agentic Decision Maker for autonomous research decisions.
Inspired by the InfoDeepSeek paper's agentic RAG approach.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time
from src.models import groq_model
from src.benchmark_framework import QueryComplexity, QueryType, QuerySpecification

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of agentic decisions."""
    CONTINUATION = "continuation"
    TERMINATION = "termination"
    SEARCH_REFINEMENT = "search_refinement"
    ANALYSIS_DEEPENING = "analysis_deepening"
    SOURCE_VALIDATION = "source_validation"

class ConfidenceLevel(Enum):
    """Confidence levels for decisions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class DecisionContext:
    """Context information for making decisions."""
    current_query: str
    current_results: Dict[str, Any]
    iteration_count: int
    max_iterations: int
    sources_found: int
    expected_sources: int
    analysis_quality: float
    search_quality: float
    time_elapsed: float

@dataclass
class AgenticDecision:
    """Represents an agentic decision."""
    decision_type: DecisionType
    confidence: ConfidenceLevel
    reasoning: str
    parameters: Dict[str, Any]
    timestamp: float

class AgenticDecisionMaker:
    """Makes autonomous decisions during research process."""
    
    def __init__(self):
        """Initialize the decision maker."""
        self.decision_history = []
        self.decision_analytics = {
            "total_decisions": 0,
            "decision_types": {},
            "average_confidence": 0.0,
            "success_rate": 0.0
        }
    
    def make_decision(self, context: DecisionContext) -> AgenticDecision:
        """
        Make an agentic decision based on current context.
        
        Args:
            context: Current research context
            
        Returns:
            AgenticDecision with decision type and parameters
        """
        try:
            logger.info(f"Making agentic decision for iteration {context.iteration_count}")
            
            # Evaluate current state
            state_evaluation = self._evaluate_current_state(context)
            
            # Make decision based on evaluation
            decision = self._make_strategic_decision(context, state_evaluation)
            
            # Store decision
            self.decision_history.append(decision)
            self._update_analytics(decision)
            
            logger.info(f"Decision made: {decision.decision_type.value} with {decision.confidence.value} confidence")
            return decision
            
        except Exception as e:
            logger.error(f"Error making agentic decision: {e}")
            # Return safe default decision
            return AgenticDecision(
                decision_type=DecisionType.CONTINUATION,
                confidence=ConfidenceLevel.LOW,
                reasoning="Error in decision making, defaulting to continuation",
                parameters={},
                timestamp=time.time()
            )
    
    def _evaluate_current_state(self, context: DecisionContext) -> Dict[str, float]:
        """
        Evaluate the current research state.
        
        Args:
            context: Current research context
            
        Returns:
            Dictionary with evaluation scores
        """
        try:
            evaluation = {}
            
            # Source adequacy
            source_ratio = context.sources_found / context.expected_sources if context.expected_sources > 0 else 0
            evaluation["source_adequacy"] = min(1.0, source_ratio)
            
            # Quality assessment
            evaluation["analysis_quality"] = context.analysis_quality
            evaluation["search_quality"] = context.search_quality
            
            # Progress assessment
            iteration_progress = context.iteration_count / context.max_iterations
            evaluation["iteration_progress"] = iteration_progress
            
            # Time efficiency
            expected_time = context.max_iterations * 30  # 30 seconds per iteration
            time_efficiency = max(0.0, 1.0 - (context.time_elapsed / expected_time))
            evaluation["time_efficiency"] = time_efficiency
            
            # Overall satisfaction
            overall_satisfaction = (
                evaluation["source_adequacy"] * 0.3 +
                evaluation["analysis_quality"] * 0.3 +
                evaluation["search_quality"] * 0.2 +
                evaluation["time_efficiency"] * 0.2
            )
            evaluation["overall_satisfaction"] = overall_satisfaction
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating current state: {e}")
            return {
                "source_adequacy": 0.5,
                "analysis_quality": 0.5,
                "search_quality": 0.5,
                "iteration_progress": 0.5,
                "time_efficiency": 0.5,
                "overall_satisfaction": 0.5
            }
    
    def _make_strategic_decision(self, context: DecisionContext, 
                               evaluation: Dict[str, float]) -> AgenticDecision:
        """
        Make strategic decision based on context and evaluation.
        
        Args:
            context: Research context
            evaluation: State evaluation scores
            
        Returns:
            Strategic decision
        """
        try:
            # Decision logic based on evaluation
            overall_satisfaction = evaluation["overall_satisfaction"]
            iteration_progress = evaluation["iteration_progress"]
            source_adequacy = evaluation["source_adequacy"]
            
            # Termination conditions
            if overall_satisfaction >= 0.8 and source_adequacy >= 0.7:
                return AgenticDecision(
                    decision_type=DecisionType.TERMINATION,
                    confidence=ConfidenceLevel.HIGH,
                    reasoning="High satisfaction and adequate sources achieved",
                    parameters={"termination_reason": "satisfaction_threshold"},
                    timestamp=time.time()
                )
            
            if iteration_progress >= 0.9 and overall_satisfaction >= 0.6:
                return AgenticDecision(
                    decision_type=DecisionType.TERMINATION,
                    confidence=ConfidenceLevel.MEDIUM,
                    reasoning="Near max iterations with acceptable satisfaction",
                    parameters={"termination_reason": "iteration_limit"},
                    timestamp=time.time()
                )
            
            # Search refinement conditions
            if source_adequacy < 0.5 and iteration_progress < 0.7:
                return AgenticDecision(
                    decision_type=DecisionType.SEARCH_REFINEMENT,
                    confidence=ConfidenceLevel.HIGH,
                    reasoning="Insufficient sources found, refining search strategy",
                    parameters={
                        "refinement_type": "semantic_expansion",
                        "focus_areas": self._identify_focus_areas(context)
                    },
                    timestamp=time.time()
                )
            
            # Analysis deepening conditions
            if evaluation["analysis_quality"] < 0.6 and source_adequacy >= 0.6:
                return AgenticDecision(
                    decision_type=DecisionType.ANALYSIS_DEEPENING,
                    confidence=ConfidenceLevel.MEDIUM,
                    reasoning="Adequate sources but analysis needs improvement",
                    parameters={"analysis_depth": "comprehensive"},
                    timestamp=time.time()
                )
            
            # Source validation conditions
            if evaluation["search_quality"] < 0.5:
                return AgenticDecision(
                    decision_type=DecisionType.SOURCE_VALIDATION,
                    confidence=ConfidenceLevel.MEDIUM,
                    reasoning="Search quality low, validating sources",
                    parameters={"validation_type": "quality_check"},
                    timestamp=time.time()
                )
            
            # Default continuation
            return AgenticDecision(
                decision_type=DecisionType.CONTINUATION,
                confidence=ConfidenceLevel.MEDIUM,
                reasoning="Continuing research with current strategy",
                parameters={},
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error making strategic decision: {e}")
            return AgenticDecision(
                decision_type=DecisionType.CONTINUATION,
                confidence=ConfidenceLevel.LOW,
                reasoning="Error in decision logic, continuing safely",
                parameters={},
                timestamp=time.time()
            )
    
    def _identify_focus_areas(self, context: DecisionContext) -> List[str]:
        """
        Identify focus areas for search refinement.
        
        Args:
            context: Research context
            
        Returns:
            List of focus areas
        """
        try:
            focus_areas = []
            query_words = context.current_query.lower().split()
            
            # Add contextual terms based on query
            if "technology" in query_words or "tech" in query_words:
                focus_areas.extend(["innovation", "development", "trends"])
            
            if "environment" in query_words or "climate" in query_words:
                focus_areas.extend(["sustainability", "impact", "policy"])
            
            if "economy" in query_words or "economic" in query_words:
                focus_areas.extend(["market", "financial", "growth"])
            
            if "health" in query_words or "medical" in query_words:
                focus_areas.extend(["research", "treatment", "outcomes"])
            
            # Default focus areas
            if not focus_areas:
                focus_areas = ["research", "analysis", "current", "recent"]
            
            return focus_areas[:3]  # Limit to top 3
            
        except Exception as e:
            logger.error(f"Error identifying focus areas: {e}")
            return ["research", "analysis", "current"]
    
    def _update_analytics(self, decision: AgenticDecision):
        """
        Update decision analytics.
        
        Args:
            decision: Recent decision made
        """
        try:
            self.decision_analytics["total_decisions"] += 1
            
            # Update decision type counts
            decision_type = decision.decision_type.value
            if decision_type not in self.decision_analytics["decision_types"]:
                self.decision_analytics["decision_types"][decision_type] = 0
            self.decision_analytics["decision_types"][decision_type] += 1
            
            # Update average confidence
            confidence_values = {"low": 0.33, "medium": 0.66, "high": 1.0}
            confidence_value = confidence_values[decision.confidence.value]
            
            total_decisions = self.decision_analytics["total_decisions"]
            current_avg = self.decision_analytics["average_confidence"]
            
            self.decision_analytics["average_confidence"] = (
                (current_avg * (total_decisions - 1) + confidence_value) / total_decisions
            )
            
            # Simple success rate calculation (high confidence decisions are considered successful)
            high_confidence_decisions = sum(
                1 for d in self.decision_history if d.confidence == ConfidenceLevel.HIGH
            )
            self.decision_analytics["success_rate"] = (
                high_confidence_decisions / total_decisions if total_decisions > 0 else 0.0
            )
            
        except Exception as e:
            logger.error(f"Error updating analytics: {e}")
    
    def get_decision_analytics(self) -> Dict[str, Any]:
        """
        Get decision analytics and insights.
        
        Returns:
            Dictionary with decision analytics
        """
        try:
            if not self.decision_history:
                return {
                    "status": "No decisions made yet",
                    "analytics": self.decision_analytics
                }
            
            # Recent decisions (last 5)
            recent_decisions = []
            for decision in self.decision_history[-5:]:
                recent_decisions.append({
                    "decision_type": decision.decision_type.value,
                    "confidence": decision.confidence.value,
                    "reasoning": decision.reasoning,
                    "timestamp": decision.timestamp
                })
            
            # Decision patterns
            decision_patterns = self._analyze_decision_patterns()
            
            return {
                "status": "success",
                "analytics": self.decision_analytics,
                "recent_decisions": recent_decisions,
                "decision_patterns": decision_patterns,
                "total_history": len(self.decision_history)
            }
            
        except Exception as e:
            logger.error(f"Error getting decision analytics: {e}")
            return {
                "status": "error",
                "error": str(e),
                "analytics": self.decision_analytics
            }
    
    def _analyze_decision_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in decision history."""
        try:
            if len(self.decision_history) < 2:
                return {"insufficient_data": True}
            
            # Most common decision type
            decision_counts = {}
            for decision in self.decision_history:
                dt = decision.decision_type.value
                decision_counts[dt] = decision_counts.get(dt, 0) + 1
            
            most_common = max(decision_counts.items(), key=lambda x: x[1])
            
            # Average confidence by decision type
            confidence_by_type = {}
            for decision in self.decision_history:
                dt = decision.decision_type.value
                conf_val = {"low": 0.33, "medium": 0.66, "high": 1.0}[decision.confidence.value]
                
                if dt not in confidence_by_type:
                    confidence_by_type[dt] = []
                confidence_by_type[dt].append(conf_val)
            
            avg_confidence_by_type = {
                dt: sum(vals) / len(vals) 
                for dt, vals in confidence_by_type.items()
            }
            
            return {
                "most_common_decision": {
                    "type": most_common[0],
                    "count": most_common[1],
                    "percentage": most_common[1] / len(self.decision_history) * 100
                },
                "average_confidence_by_type": avg_confidence_by_type,
                "decision_distribution": decision_counts
            }
            
        except Exception as e:
            logger.error(f"Error analyzing decision patterns: {e}")
            return {"error": str(e)}
    
    def reset_analytics(self):
        """Reset decision history and analytics."""
        try:
            self.decision_history.clear()
            self.decision_analytics = {
                "total_decisions": 0,
                "decision_types": {},
                "average_confidence": 0.0,
                "success_rate": 0.0
            }
            logger.info("Decision analytics reset successfully")
            
        except Exception as e:
            logger.error(f"Error resetting analytics: {e}")

# Global agentic decision maker instance
agentic_decision_maker = AgenticDecisionMaker() 

------------------------------------------------------------------------------
#### Phase 6: Analysis Generation & LLM Synthesis
#### src/models.py
from openai import OpenAI
from src.config import config
import logging
import os

logger = logging.getLogger(__name__)

# Initialize Groq client using OpenAI library
def initialize_groq_model():
    """Initialize Groq client using OpenAI library with Groq endpoint."""
    try:
        config.validate()
        
        # Initialize OpenAI client with Groq endpoint
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=config.GROQ_API_KEY
        )
        logger.info(f"Groq client initialized successfully using OpenAI library")
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        raise

def generate_response(client, messages: list, system_prompt: str = None) -> str:
    """
    Generate a response using the OpenAI client (connected to Groq).
    
    Args:
        client: OpenAI client instance (connected to Groq)
        messages: List of message dictionaries
        system_prompt: Optional system prompt
        
    Returns:
        Generated response string
    """
    try:
        # Prepare messages for OpenAI API format
        api_messages = []
        
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        
        for message in messages:
            if message.get("role") in ["user", "assistant", "system"]:
                api_messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })
        
        # Generate response using OpenAI client (connected to Groq)
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=api_messages,
            temperature=0,
            max_tokens=config.MAX_TOKENS
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error generating response: {str(e)}"

def research_prompt(client, query: str, context: str = "") -> str:
    """
    Generate a research-focused prompt.
    
    Args:
        client: OpenAI client instance (connected to Groq)
        query: The research question
        context: Additional context or previous findings
        
    Returns:
        Formatted research prompt
    """
    system_prompt = """You are a professional research assistant. Your task is to:
1. Analyze the given research question thoroughly
2. Use available tools to gather relevant information
3. Synthesize findings into a comprehensive, well-structured response
4. Provide citations and sources when possible
5. Be objective, accurate, and thorough in your analysis

Focus on providing detailed, factual information with clear reasoning."""
    
    if context:
        prompt = f"Research Question: {query}\n\nPrevious Context: {context}\n\nPlease provide a comprehensive analysis."
    else:
        prompt = f"Research Question: {query}\n\nPlease provide a comprehensive analysis."
    
    return generate_response(
        client,
        [{"role": "user", "content": prompt}],
        system_prompt
    )

# Global Groq client instance
groq_model = initialize_groq_model() 
------------------------------------------------------------------------------
#### Phase 7: Multi-Dimensional Evaluation
#### src/benchmark_framework.py
"""
Benchmark Framework for evaluating research agent performance.
Inspired by the InfoDeepSeek paper methodology.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    CHALLENGING = "challenging"

class QueryType(Enum):
    """Types of research queries."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    SYNTHETIC = "synthetic"

@dataclass
class QuerySpecification:
    """Specification for a research query."""
    base_query: str
    complexity: QueryComplexity
    query_type: QueryType
    max_iterations: int
    expected_sources: int
    determinacy: float  # How well-defined the query is (0-1)
    difficulty: float   # How difficult to answer (0-1)
    diversity: float    # How diverse the sources should be (0-1)

@dataclass
class BenchmarkMetrics:
    """Benchmark evaluation metrics."""
    accuracy: float
    completeness: float
    relevance: float
    source_quality: float
    response_time: float
    source_diversity: float
    reasoning_quality: float

class BenchmarkFramework:
    """Framework for benchmarking research performance."""
    
    def __init__(self):
        """Initialize the benchmark framework."""
        self.evaluation_history = []
        self.performance_metrics = {
            "total_queries": 0,
            "average_accuracy": 0.0,
            "average_completeness": 0.0,
            "average_relevance": 0.0,
            "average_response_time": 0.0
        }
    
    def construct_challenging_query(self, base_query: str, 
                                  complexity: QueryComplexity,
                                  query_type: QueryType) -> QuerySpecification:
        """
        Construct a challenging query specification based on complexity and type.
        
        Args:
            base_query: The original query
            complexity: Query complexity level
            query_type: Type of query
            
        Returns:
            QuerySpecification with appropriate parameters
        """
        try:
            # Set parameters based on complexity
            complexity_params = {
                QueryComplexity.SIMPLE: {
                    "max_iterations": 2,
                    "expected_sources": 3,
                    "determinacy": 0.8,
                    "difficulty": 0.3,
                    "diversity": 0.5
                },
                QueryComplexity.MODERATE: {
                    "max_iterations": 3,
                    "expected_sources": 5,
                    "determinacy": 0.6,
                    "difficulty": 0.6,
                    "diversity": 0.7
                },
                QueryComplexity.CHALLENGING: {
                    "max_iterations": 5,
                    "expected_sources": 8,
                    "determinacy": 0.4,
                    "difficulty": 0.9,
                    "diversity": 0.9
                }
            }
            
            params = complexity_params.get(complexity, complexity_params[QueryComplexity.MODERATE])
            
            # Adjust parameters based on query type
            if query_type == QueryType.FACTUAL:
                params["determinacy"] += 0.2
                params["difficulty"] -= 0.1
            elif query_type == QueryType.SYNTHETIC:
                params["max_iterations"] += 1
                params["expected_sources"] += 2
                params["diversity"] += 0.1
            
            # Ensure values are within bounds
            for key in ["determinacy", "difficulty", "diversity"]:
                params[key] = max(0.0, min(1.0, params[key]))
            
            query_spec = QuerySpecification(
                base_query=base_query,
                complexity=complexity,
                query_type=query_type,
                max_iterations=params["max_iterations"],
                expected_sources=params["expected_sources"],
                determinacy=params["determinacy"],
                difficulty=params["difficulty"],
                diversity=params["diversity"]
            )
            
            logger.info(f"Constructed query specification: {complexity.value} {query_type.value}")
            return query_spec
            
        except Exception as e:
            logger.error(f"Error constructing query specification: {e}")
            # Return default specification
            return QuerySpecification(
                base_query=base_query,
                complexity=QueryComplexity.MODERATE,
                query_type=QueryType.ANALYTICAL,
                max_iterations=3,
                expected_sources=5,
                determinacy=0.6,
                difficulty=0.6,
                diversity=0.7
            )
    
    def evaluate_query(self, query_spec: QuerySpecification, 
                      results: Dict[str, Any]) -> BenchmarkMetrics:
        """
        Evaluate research results against query specification.
        
        Args:
            query_spec: Original query specification
            results: Research results to evaluate
            
        Returns:
            BenchmarkMetrics with evaluation scores
        """
        try:
            start_time = time.time()
            
            # Calculate accuracy (based on source quality and analysis quality)
            source_quality = self._evaluate_source_quality(results.get("sources", []))
            analysis_quality = self._evaluate_analysis_quality(results.get("analysis", ""))
            accuracy = (source_quality + analysis_quality) / 2
            
            # Calculate completeness (based on expected vs actual sources)
            expected_sources = query_spec.expected_sources
            actual_sources = results.get("total_sources", 0)
            completeness = min(1.0, actual_sources / expected_sources) if expected_sources > 0 else 0.0
            
            # Calculate relevance (based on query type and content alignment)
            relevance = self._evaluate_relevance(query_spec, results)
            
            # Calculate source diversity
            source_diversity = self._evaluate_source_diversity(results.get("sources", []))
            
            # Calculate reasoning quality
            reasoning_quality = self._evaluate_reasoning_quality(results.get("analysis", ""))
            
            # Response time (if available)
            response_time = results.get("response_time", time.time() - start_time)
            
            metrics = BenchmarkMetrics(
                accuracy=accuracy,
                completeness=completeness,
                relevance=relevance,
                source_quality=source_quality,
                response_time=response_time,
                source_diversity=source_diversity,
                reasoning_quality=reasoning_quality
            )
            
            # Store evaluation
            self.evaluation_history.append({
                "query_spec": query_spec,
                "results": results,
                "metrics": metrics,
                "timestamp": time.time()
            })
            
            # Update performance metrics
            self._update_performance_metrics(metrics)
            
            logger.info(f"Query evaluation completed. Accuracy: {accuracy:.2f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating query: {e}")
            # Return default metrics
            return BenchmarkMetrics(
                accuracy=0.5,
                completeness=0.5,
                relevance=0.5,
                source_quality=0.5,
                response_time=0.0,
                source_diversity=0.5,
                reasoning_quality=0.5
            )
    
    def _evaluate_source_quality(self, sources: List[Dict[str, Any]]) -> float:
        """Evaluate the quality of sources."""
        if not sources:
            return 0.0
        
        quality_scores = []
        for source in sources:
            score = 0.0
            
            # Check for URL presence
            if source.get("url"):
                score += 0.3
            
            # Check for title
            if source.get("title"):
                score += 0.2
            
            # Check for content length
            content = source.get("content", "")
            if len(content) > 100:
                score += 0.3
            
            # Check for relevance score
            if source.get("score", 0) > 0.5:
                score += 0.2
            
            quality_scores.append(score)
        
        return sum(quality_scores) / len(quality_scores)
    
    def _evaluate_analysis_quality(self, analysis: str) -> float:
        """Evaluate the quality of analysis."""
        if not analysis:
            return 0.0
        
        score = 0.0
        
        # Length-based scoring
        if len(analysis) > 500:
            score += 0.3
        elif len(analysis) > 200:
            score += 0.2
        elif len(analysis) > 100:
            score += 0.1
        
        # Structure-based scoring
        if "conclusion" in analysis.lower() or "summary" in analysis.lower():
            score += 0.2
        
        # Evidence-based scoring
        if "according to" in analysis.lower() or "based on" in analysis.lower():
            score += 0.2
        
        # Analysis depth
        if "however" in analysis.lower() or "furthermore" in analysis.lower():
            score += 0.3
        
        return min(1.0, score)
    
    def _evaluate_relevance(self, query_spec: QuerySpecification, 
                          results: Dict[str, Any]) -> float:
        """Evaluate relevance of results to query."""
        # Simple keyword-based relevance
        query_words = set(query_spec.base_query.lower().split())
        analysis = results.get("analysis", "").lower()
        
        if not analysis:
            return 0.0
        
        analysis_words = set(analysis.split())
        overlap = len(query_words.intersection(analysis_words))
        relevance = overlap / len(query_words) if query_words else 0.0
        
        return min(1.0, relevance * 2)  # Scale up for better scoring
    
    def _evaluate_source_diversity(self, sources: List[Dict[str, Any]]) -> float:
        """Evaluate diversity of sources."""
        if not sources:
            return 0.0
        
        # Simple domain-based diversity
        domains = set()
        for source in sources:
            url = source.get("url", "")
            if url:
                try:
                    domain = url.split("//")[1].split("/")[0]
                    domains.add(domain)
                except:
                    pass
        
        # Diversity score based on unique domains
        diversity = len(domains) / len(sources) if sources else 0.0
        return min(1.0, diversity * 2)  # Scale up
    
    def _evaluate_reasoning_quality(self, analysis: str) -> float:
        """Evaluate the quality of reasoning in analysis."""
        if not analysis:
            return 0.0
        
        score = 0.0
        analysis_lower = analysis.lower()
        
        # Logical connectors
        connectors = ["therefore", "because", "since", "thus", "consequently", "as a result"]
        for connector in connectors:
            if connector in analysis_lower:
                score += 0.1
        
        # Evidence presentation
        evidence_markers = ["research shows", "studies indicate", "data suggests", "findings reveal"]
        for marker in evidence_markers:
            if marker in analysis_lower:
                score += 0.1
        
        # Balanced analysis
        if "on the other hand" in analysis_lower or "alternatively" in analysis_lower:
            score += 0.2
        
        return min(1.0, score)
    
    def _update_performance_metrics(self, metrics: BenchmarkMetrics):
        """Update overall performance metrics."""
        self.performance_metrics["total_queries"] += 1
        total = self.performance_metrics["total_queries"]
        
        # Running average calculation
        self.performance_metrics["average_accuracy"] = (
            (self.performance_metrics["average_accuracy"] * (total - 1) + metrics.accuracy) / total
        )
        self.performance_metrics["average_completeness"] = (
            (self.performance_metrics["average_completeness"] * (total - 1) + metrics.completeness) / total
        )
        self.performance_metrics["average_relevance"] = (
            (self.performance_metrics["average_relevance"] * (total - 1) + metrics.relevance) / total
        )
        self.performance_metrics["average_response_time"] = (
            (self.performance_metrics["average_response_time"] * (total - 1) + metrics.response_time) / total
        )
    
    def get_benchmark_report(self) -> Dict[str, Any]:
        """Get comprehensive benchmark report."""
        try:
            if not self.evaluation_history:
                return {
                    "status": "No evaluations performed yet",
                    "performance_summary": self.performance_metrics,
                    "recent_evaluations": []
                }
            
            # Recent evaluations (last 10)
            recent_evaluations = []
            for eval_data in self.evaluation_history[-10:]:
                recent_evaluations.append({
                    "query": eval_data["query_spec"].base_query,
                    "complexity": eval_data["query_spec"].complexity.value,
                    "accuracy": eval_data["metrics"].accuracy,
                    "completeness": eval_data["metrics"].completeness,
                    "relevance": eval_data["metrics"].relevance
                })
            
            # Calculate overall score
            overall_score = (
                self.performance_metrics["average_accuracy"] * 0.3 +
                self.performance_metrics["average_completeness"] * 0.25 +
                self.performance_metrics["average_relevance"] * 0.25 +
                (1.0 - min(1.0, self.performance_metrics["average_response_time"] / 60.0)) * 0.2
            )
            
            self.performance_metrics["overall_score"] = overall_score
            
            return {
                "status": "success",
                "performance_summary": self.performance_metrics,
                "recent_evaluations": recent_evaluations,
                "total_evaluations": len(self.evaluation_history)
            }
            
        except Exception as e:
            logger.error(f"Error generating benchmark report: {e}")
            return {
                "status": "error",
                "error": str(e),
                "performance_summary": self.performance_metrics
            }

# Global benchmark framework instance
benchmark_framework = BenchmarkFramework() 
------------------------------------------------------------------------------
#### Phase 8: Results Packaging & Analytics
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔍 Deep Research Agent</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="/static/css/style.css" rel="stylesheet">
</head>
<body>
    <div class="container-fluid">
        <!-- Header -->
        <header class="header-section">
            <div class="container">
                <div class="row align-items-center py-4">
                    <div class="col-md-6">
                        <h1 class="main-title">
                            <i class="fas fa-search-plus text-primary"></i>
                            Deep Research Agent
                        </h1>
                        <p class="subtitle">Powered by Groq Llama 3 & Advanced AI</p>
                    </div>
                    <div class="col-md-6 text-md-end">
                        <div class="status-badge">
                            <i class="fas fa-circle text-success"></i>
                            <span>AI Agent Active</span>
                        </div>
                    </div>
                </div>
            </div>
        </header>

        <!-- Main Content -->
        <main class="main-content">
            <div class="container">
                <div class="row">
                    <!-- Search Panel -->
                    <div class="col-lg-4">
                        <div class="search-panel">
                            <div class="card shadow-lg border-0">
                                <div class="card-header bg-gradient-primary text-white">
                                    <h5 class="mb-0">
                                        <i class="fas fa-rocket me-2"></i>
                                        Research Query
                                    </h5>
                                </div>
                                <div class="card-body p-4">
                                    <form id="researchForm">
                                        <div class="mb-3">
                                            <label for="query" class="form-label fw-semibold">
                                                <i class="fas fa-question-circle me-1"></i>
                                                Research Question
                                            </label>
                                            <textarea 
                                                class="form-control form-control-lg" 
                                                id="query" 
                                                name="query" 
                                                rows="3" 
                                                placeholder="What would you like to research? Ask anything..."
                                                required></textarea>
                                        </div>

                                        <div class="row mb-3">
                                            <div class="col-6">
                                                <label for="research_type" class="form-label fw-semibold">
                                                    <i class="fas fa-cogs me-1"></i>
                                                    Type
                                                </label>
                                                <select class="form-select" id="research_type" name="research_type">
                                                    <option value="deep">🔬 Deep Research</option>
                                                    <option value="quick">⚡ Quick Search</option>
                                                </select>
                                            </div>
                                            <div class="col-6">
                                                <label for="complexity" class="form-label fw-semibold">
                                                    <i class="fas fa-layer-group me-1"></i>
                                                    Complexity
                                                </label>
                                                <select class="form-select" id="complexity" name="complexity">
                                                    <option value="simple">🟢 Simple</option>
                                                    <option value="moderate" selected>🟡 Moderate</option>
                                                    <option value="challenging">🔴 Challenging</option>
                                                </select>
                                            </div>
                                        </div>

                                        <div class="mb-3">
                                            <label for="query_type" class="form-label fw-semibold">
                                                <i class="fas fa-brain me-1"></i>
                                                Analysis Type
                                            </label>
                                            <select class="form-select" id="query_type" name="query_type">
                                                <option value="factual">📊 Factual</option>
                                                <option value="analytical" selected>🧠 Analytical</option>
                                                <option value="comparative">⚖️ Comparative</option>
                                                <option value="synthetic">🔗 Synthetic</option>
                                            </select>
                                        </div>

                                        <div class="mb-4" id="iterationsDiv">
                                            <label for="iterations" class="form-label fw-semibold">
                                                <i class="fas fa-repeat me-1"></i>
                                                Research Iterations: <span id="iterationsValue">3</span>
                                            </label>
                                            <input type="range" class="form-range" id="iterations" name="iterations" min="1" max="5" value="3">
                                        </div>

                                        <button type="submit" class="btn btn-primary btn-lg w-100" id="submitBtn">
                                            <i class="fas fa-search me-2"></i>
                                            Start Research
                                        </button>
                                    </form>
                                </div>
                            </div>

                            <!-- Quick Stats -->
                            <div class="card shadow-sm border-0 mt-4">
                                <div class="card-body p-3">
                                    <div class="row text-center">
                                        <div class="col-4">
                                            <div class="stat-item">
                                                <i class="fas fa-database text-primary"></i>
                                                <div class="stat-number" id="totalQueries">0</div>
                                                <div class="stat-label">Queries</div>
                                            </div>
                                        </div>
                                        <div class="col-4">
                                            <div class="stat-item">
                                                <i class="fas fa-link text-success"></i>
                                                <div class="stat-number" id="totalSources">0</div>
                                                <div class="stat-label">Sources</div>
                                            </div>
                                        </div>
                                        <div class="col-4">
                                            <div class="stat-item">
                                                <i class="fas fa-chart-line text-warning"></i>
                                                <div class="stat-number" id="avgQuality">0%</div>
                                                <div class="stat-label">Quality</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Results Panel -->
                    <div class="col-lg-8">
                        <div class="results-panel">
                            <!-- Welcome Message -->
                            <div class="welcome-card" id="welcomeCard">
                                <div class="card shadow-lg border-0">
                                    <div class="card-body text-center p-5">
                                        <div class="welcome-icon mb-4">
                                            <i class="fas fa-robot"></i>
                                        </div>
                                        <h3 class="welcome-title">Welcome to Deep Research Agent</h3>
                                        <p class="welcome-text">
                                            Ask me anything! I'll conduct comprehensive research using advanced AI 
                                            and provide you with detailed analysis backed by reliable sources.
                                        </p>
                                        <div class="feature-highlights mt-4">
                                            <div class="row">
                                                <div class="col-md-4">
                                                    <div class="feature-item">
                                                        <i class="fas fa-brain text-primary"></i>
                                                        <h6>AI-Powered</h6>
                                                        <small>Advanced reasoning with Groq Llama</small>
                                                    </div>
                                                </div>
                                                <div class="col-md-4">
                                                    <div class="feature-item">
                                                        <i class="fas fa-shield-alt text-success"></i>
                                                        <h6>Reliable Sources</h6>
                                                        <small>Verified information from trusted sources</small>
                                                    </div>
                                                </div>
                                                <div class="col-md-4">
                                                    <div class="feature-item">
                                                        <i class="fas fa-lightning-bolt text-warning"></i>
                                                        <h6>Fast Results</h6>
                                                        <small>Quick analysis with comprehensive insights</small>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Loading State -->
                            <div class="loading-card d-none" id="loadingCard">
                                <div class="card shadow-lg border-0">
                                    <div class="card-body text-center p-5">
                                        <div class="loading-spinner mb-4">
                                            <div class="spinner-border text-primary" role="status">
                                                <span class="visually-hidden">Loading...</span>
                                            </div>
                                        </div>
                                        <h4 class="loading-title">Researching...</h4>
                                        <p class="loading-text" id="loadingText">Starting research process...</p>
                                        <div class="progress mt-3">
                                            <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                                 role="progressbar" style="width: 0%" id="progressBar"></div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- Results Display -->
                            <div class="results-display d-none" id="resultsDisplay">
                                <!-- Analysis Section -->
                                <div class="card shadow-lg border-0 mb-4">
                                    <div class="card-header bg-gradient-success text-white">
                                        <h5 class="mb-0">
                                            <i class="fas fa-microscope me-2"></i>
                                            Research Analysis
                                        </h5>
                                    </div>
                                    <div class="card-body p-4">
                                        <div class="analysis-content" id="analysisContent">
                                            <!-- Analysis will be inserted here -->
                                        </div>
                                    </div>
                                </div>

                                <!-- Metrics Section -->
                                <div class="card shadow-lg border-0 mb-4">
                                    <div class="card-header bg-gradient-info text-white">
                                        <h5 class="mb-0">
                                            <i class="fas fa-chart-bar me-2"></i>
                                            Research Metrics
                                        </h5>
                                    </div>
                                    <div class="card-body p-4">
                                        <div class="row" id="metricsContent">
                                            <!-- Metrics will be inserted here -->
                                        </div>
                                    </div>
                                </div>

                                <!-- Sources Section -->
                                <div class="card shadow-lg border-0">
                                    <div class="card-header bg-gradient-warning text-white">
                                        <h5 class="mb-0">
                                            <i class="fas fa-link me-2"></i>
                                            Sources & References
                                        </h5>
                                    </div>
                                    <div class="card-body p-4">
                                        <div class="sources-content" id="sourcesContent">
                                            <!-- Sources will be inserted here -->
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <!-- Custom JS -->
    <script src="/static/js/app.js"></script>
</body>
</html> 







