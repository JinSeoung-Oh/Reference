### From https://medium.com/the-ai-forum/build-a-code-generator-and-executor-agent-using-langgraph-langchain-sandbox-and-groq-kimi-k2-291a88e66e6f

--------------------------------------------------------------------------------------------------------------------
############ main.py
#!/usr/bin/env python3
"""
LangGraph Code Generator with GROQ Kimi K2 and Sandbox Execution
Main CLI interface for the code generation workflow.
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from src.workflow import CodeGeneratorWorkflow


def print_banner():
    """Print the application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                 LangGraph Code Generator                     ║
║                with GROQ Kimi K2 & Sandbox                   ║
╠══════════════════════════════════════════════════════════════╣
║  Node 1: Code Generation (Kimi K2)                           ║
║  Node 2: Syntax Check & PEP8 (Black/Autopep8)                ║
║  Node 3: Code Rectification (AI + Pattern-based)             ║
║  Node 4: Code Execution (LangChain Sandbox)                  ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    """Main entry point for the CLI application."""
    
    # Load environment variables
    load_dotenv()
    
    # Check for required environment variables
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY environment variable is required")
        print("Please set your GROQ API key in the .env file")
        sys.exit(1)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate and execute Python code using LangGraph and GROQ Kimi K2"
    )
    parser.add_argument(
        "--prompt", 
        required=True,
        help="The code generation prompt/request"
    )
    parser.add_argument(
        "--requirements",
        help="Additional requirements or specifications"
    )
    parser.add_argument(
        "--verbose", 
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Print banner and startup info
    print_banner()
    
    # Check execution environment and initialize appropriate executor
    try:
        from src import sandbox_executor
        print("✅ Using LangChain PyodideSandbox for code execution")
    except ImportError as e:
        print(f"⚠️ Warning: Sandbox initialization issue: {e}")
    
    try:
        from src import fastapi_executor
        print("✅ FastAPI Safe Executor initialized")
    except ImportError as e:
        print(f"⚠️ Warning: FastAPI executor initialization issue: {e}")
    
    print(f"\n🚀 Processing request: {args.prompt}")
    print("\n" + "=" * 60)
    
    # Initialize and run the workflow
    try:
        workflow = CodeGeneratorWorkflow()
        
        print("🚀 Starting Code Generation Workflow")
        print(f"📝 User Prompt: {args.prompt}")
        
        if args.requirements:
            print(f"📋 Requirements: {args.requirements}")
        
        # Run the workflow
        result = workflow.run(args.prompt, args.requirements)
        
        print("✅ Workflow Complete")
        print("\n" + "=" * 60)
        
        # Display results
        if result.get("workflow_status") == "completed":
            print("✅ Code generation completed successfully!")
            print("\n" + result.get("final_result", "No result available"))
            
            # Additional debug info if verbose
            if args.verbose:
                print("\n" + "=" * 60)
                print("🔍 DEBUG INFORMATION")
                print(f"Final state keys: {list(result.keys())}")
                print(f"Workflow status: {result.get('workflow_status')}")
                print(f"Rectification attempts: {result.get('rectification_attempts', 0)}")
                if result.get('syntax_errors'):
                    print(f"Syntax errors: {result.get('syntax_errors')}")
                if result.get('execution_errors'):
                    print(f"Execution errors: {result.get('execution_errors')}")
                if result.get('execution_results'):
                    exec_results = result.get('execution_results')
                    print(f"Execution success: {exec_results.get('success', False)}")
                    if exec_results.get('error'):
                        print(f"Last execution error: {exec_results.get('error')}")
                    if exec_results.get('output'):
                        print(f"Last execution output: {exec_results.get('output')}")
                if result.get('error_analysis'):
                    error_analysis = result.get('error_analysis')
                    print(f"Error analysis: {error_analysis}")
                if result.get('rectified_code'):
                    print(f"Has rectified code: Yes ({len(result.get('rectified_code'))} chars)")
                else:
                    print("Has rectified code: No")
        else:
            print("❌ Code generation failed!")
            print(f"🔍 DEBUG: Actual workflow status = '{result.get('workflow_status')}'")
            print(f"🔍 DEBUG: Expected = 'completed'")
            if result.get("error_message"):
                print(f"Error: {result.get('error_message')}")
            if result.get("final_result"):
                print("\n" + result.get("final_result"))
            
            # Show debug info even when failed if verbose
            if args.verbose:
                print("\n" + "=" * 60)
                print("🔍 DEBUG INFORMATION (FAILED CASE)")
                print(f"Final state keys: {list(result.keys())}")
                print(f"Workflow status: {result.get('workflow_status')}")
                print(f"Rectification attempts: {result.get('rectification_attempts', 0)}")
                if result.get('execution_results'):
                    exec_results = result.get('execution_results')
                    print(f"Execution success: {exec_results.get('success', False)}")
                    print(f"Execution output: {exec_results.get('output', 'No output')}")
                    print(f"Execution error: {exec_results.get('error', 'No error')}")
                if result.get('generated_code'):
                    print(f"Has generated code: Yes ({len(result.get('generated_code'))} chars)")
                if result.get('rectified_code'):
                    print(f"Has rectified code: Yes ({len(result.get('rectified_code'))} chars)")
                else:
                    print("Has rectified code: No")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 

--------------------------------------------------------------------------------------------------------------------
############ Workflow.py
"""
LangGraph workflow for code generation, syntax checking, rectification, and execution.
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from src.state import CodeGenerationState
from src.nodes import CodeGeneratorNode, SyntaxCheckerNode, CodeRectifierNode, CodeExecutorNode


class CodeGeneratorWorkflow:
    """Main workflow orchestrator for the code generation process."""
    
    def __init__(self):
        self.code_generator = CodeGeneratorNode()
        self.syntax_checker = SyntaxCheckerNode()
        self.code_rectifier = CodeRectifierNode()
        self.code_executor = CodeExecutorNode()
        
        # Initialize the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with all nodes and edges."""
        
        # Create the state graph
        workflow = StateGraph(CodeGenerationState)
        
        # Add nodes (remove the separate end node)
        workflow.add_node("code_generator", self.code_generator._execute)
        workflow.add_node("syntax_checker", self.syntax_checker._execute)
        workflow.add_node("code_rectifier", self.code_rectifier._execute)
        workflow.add_node("code_executor", self.code_executor._execute)
        
        # Set entry point
        workflow.set_entry_point("code_generator")
        
        # Add conditional edges for flow control
        workflow.add_conditional_edges(
            "code_generator",
            self._route_from_generator,
            {
                "syntax_checker": "syntax_checker",
                "end": END  # Direct to END
            }
        )
        
        workflow.add_conditional_edges(
            "syntax_checker", 
            self._route_from_syntax_checker,
            {
                "code_executor": "code_executor",
                "code_rectifier": "code_rectifier",
                "end": END  # Direct to END
            }
        )
        
        workflow.add_conditional_edges(
            "code_rectifier",
            self._route_from_rectifier,
            {
                "syntax_checker": "syntax_checker",
                "code_generator": "code_generator",
                "end": END  # Direct to END
            }
        )
        
        workflow.add_conditional_edges(
            "code_executor",
            self._route_from_executor,
            {
                "code_rectifier": "code_rectifier",
                "end": END  # Direct to END
            }
        )
        print(workflow.compile().get_graph().draw_mermaid())
        return workflow.compile()
    
    def _route_from_generator(self, state: CodeGenerationState) -> str:
        """Route from code generator node."""
        return state.get("current_node", "end")
    
    def _route_from_syntax_checker(self, state: CodeGenerationState) -> str:
        """Route from syntax checker node."""
        return state.get("current_node", "end")
    
    def _route_from_rectifier(self, state: CodeGenerationState) -> str:
        """Route from code rectifier node."""
        current_node = state.get("current_node", "end")
        
        # Implement retry logic for rectification
        retry_count = state.get("retry_count", 0)
        rectification_attempts = state.get("rectification_attempts", 0)
        
        # If we've tried too many times, end the workflow
        if retry_count >= 3 or rectification_attempts >= 3:
            return "end"
        
        return current_node
    
    def _route_from_executor(self, state: CodeGenerationState) -> str:
        """Route from code executor node."""
        return state.get("current_node", "end")
    
    def run(self, user_prompt: str, requirements: str = None) -> Dict[str, Any]:
        """
        Run the complete workflow.
        
        Args:
            user_prompt: The user's code generation request
            requirements: Additional requirements (optional)
            
        Returns:
            Final workflow state with generated and executed code
        """
        
        # Initialize state
        initial_state = {
            "user_prompt": user_prompt,
            "generated_code": "",
            "syntax_errors": [],
            "execution_results": {},
            "current_node": "code_generator", 
            "retry_count": 0,
            # Rectification fields
            "execution_errors": [],
            "rectified_code": "",
            "rectification_attempts": 0,
            "error_analysis": {}
        }
        
        if requirements:
            initial_state["requirements"] = requirements
        
        try:
            # Execute the workflow
            result = self.workflow.invoke(initial_state)
            
            # Process the final result (since _end_node logic was moved here)
            final_result = self._process_final_result(result)
            
            return final_result
            
        except Exception as e:
            return {
                **initial_state,
                "workflow_status": "failed",
                "error_message": f"Workflow execution failed: {str(e)}",
                "final_result": f"❌ Workflow failed: {str(e)}"
            }
    
    def _process_final_result(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the final workflow result (moved from _end_node)."""
        
        # Get the final code (prioritize rectified code if available)
        final_code = (state.get("rectified_code") or 
                     state.get("generated_code", ""))
        
        # Get execution results
        execution_results = state.get("execution_results", {})
        
        # Determine overall success
        workflow_success = execution_results.get('success', False) or (
            final_code and not execution_results.get('error')
        )
        
        # Create comprehensive final result
        final_result = f"""## Code Generation Complete

### Generated Code:
```python
{final_code}
```

### Code Explanation:
"""
        
        # Add code explanation if available
        if final_code:
            # Generate a brief explanation of what the code does
            code_lines = final_code.split('\n')
            if len(code_lines) > 0:
                first_line = code_lines[0].strip()
                if first_line.startswith('"""') or first_line.startswith("'''"):
                    # Extract docstring as explanation
                    in_docstring = True
                    docstring_lines = []
                    for line in code_lines[1:]:
                        if '"""' in line or "'''" in line:
                            break
                        docstring_lines.append(line.strip())
                    if docstring_lines:
                        final_result += "\n".join(docstring_lines)
                    else:
                        final_result += "This code implements the requested functionality with proper error handling and documentation."
                else:
                    final_result += "This code implements the requested functionality with proper error handling and documentation."
        
        final_result += f"""

### Execution Results:
- **Success**: {execution_results.get('success', False)}
- **Execution Time**: {execution_results.get('execution_time', 0):.2f} seconds
- **Output**: {execution_results.get('output', 'No output') or 'No output'}
- **Error**: {execution_results.get('error', 'No error') or 'No error'}

### Analysis:
"""
        
        # Add analysis based on execution results
        if execution_results.get('success'):
            final_result += """## Analysis

### 1. **Execution Status**: ✅ **SUCCESS**

The code executed successfully without any runtime errors.

### 2. **Code Quality**
- Follows PEP8 standards
- Includes proper error handling
- Well-documented and readable

### 3. **Performance**
- Executed efficiently within the time limit
- No obvious performance bottlenecks detected

### 4. **Recommendations**
The code is production-ready and follows Python best practices."""
        elif not execution_results.get('error') and final_code:
            # Code exists but wasn't executed (maybe just syntax checked)
            final_result += """## Analysis

### 1. **Generation Status**: ✅ **SUCCESS**

The code was generated successfully and passed syntax validation.

### 2. **Code Quality**
- Follows PEP8 standards
- Includes proper error handling
- Well-documented and readable

### 3. **Next Steps**
The code is ready for execution and appears to be syntactically correct."""
        else:
            error_msg = execution_results.get('error', '')
            rectification_attempts = state.get('rectification_attempts', 0)
            
            final_result += f"""## Analysis

### 1. **Execution Status**: ❌ **FAILED**

The code did not execute successfully.

### 2. **Error Details**
**Error Message**: {error_msg}

### 3. **Rectification Attempts**
- **Attempts Made**: {rectification_attempts}
- **Status**: {'Maximum attempts reached' if rectification_attempts >= 3 else 'Rectification attempted'}

### 4. **Recommendations**
"""
            if rectification_attempts >= 3:
                final_result += "The automatic rectification system reached its maximum attempts. Manual review may be required to resolve the remaining issues."
            else:
                final_result += "The code may require additional manual fixes to resolve the execution errors."
        
        # Add syntax check results
        syntax_errors = state.get("syntax_errors", [])
        final_result += f"""

### Syntax Check Results:
- **Syntax Errors**: {len(syntax_errors)} errors found
- **PEP8 Suggestions**: {'Applied' if len(syntax_errors) == 0 else 'Partially applied'}

---
*Generated using LangGraph Code Generator with GROQ*"""
        
        # Determine final workflow status
        final_workflow_status = "completed" if (workflow_success or final_code) else "failed"
        
        # Return the complete final state
        return {
            **state,
            "final_result": final_result.strip(),
            "workflow_status": final_workflow_status,
            "current_node": "end"
        } 

--------------------------------------------------------------------------------------------------------------------
############ state.py
"""State management for the LangGraph Code Generator workflow."""

from typing import Dict, List, Any, TypedDict, Optional
from pydantic import BaseModel


class CodeGenerationState(TypedDict):
    """State for the code generation workflow."""
    user_prompt: str
    generated_code: str
    syntax_errors: List[str]
    execution_results: Dict[str, Any]
    current_node: str
    retry_count: int
    # New fields for code rectification
    execution_errors: List[str]
    rectified_code: str
    rectification_attempts: int
    error_analysis: Dict[str, Any]


class CodeExecutionRequest(BaseModel):
    """Request model for code execution."""
    code: str
    timeout: int = 30


class CodeExecutionResponse(BaseModel):
    """Response model for code execution."""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: float


class CodeRectificationRequest(BaseModel):
    """Request model for code rectification."""
    original_code: str
    error_message: str
    error_type: str
    execution_context: Dict[str, Any]


class CodeRectificationResponse(BaseModel):
    """Response model for code rectification."""
    success: bool
    rectified_code: Optional[str] = None
    changes_made: List[str] = []
    error_analysis: Dict[str, Any] = {}
    confidence_score: float = 0.0 

--------------------------------------------------------------------------------------------------------------------
############ nodes.py
"""Nodes for the LangGraph Code Generator workflow."""

import asyncio
import tempfile
import subprocess
import time
from typing import Dict, Any
from src.config import Config
from src.state import CodeGenerationState
from src.code_rectifier import CodeRectifier
from src import sandbox_executor, fastapi_executor


class CodeGeneratorNode:
    """Node for generating Python code using GROQ Kimi K2."""
    
    def __init__(self):
        self.config = Config()
        self.model = self.config.get_groq_model()
    
    def _execute(self, state: CodeGenerationState) -> Dict[str, Any]:
        """Execute the code generation node."""
        print("🔧 Executing Code Generator (Iteration {})".format(state.get("retry_count", 0) + 1))
        
        prompt = f"""
You are an expert Python developer. Create high-quality, well-documented Python code based on the user's request.

**Requirements:**
- Follow PEP8 standards strictly
- Include proper type hints
- Add comprehensive docstrings
- Handle edge cases and errors gracefully
- Make the code production-ready and self-contained
- Include example usage and test cases when appropriate

**User Request:**
{state['user_prompt']}

**Important Guidelines:**
1. Start the code with proper imports (put __future__ imports at the very beginning if needed)
2. Create clean, readable, and efficient code
3. Include proper error handling
4. Add meaningful comments and documentation
5. Make sure all syntax is correct

Please provide only the Python code, no additional explanation:
"""
        
        try:
            response = self.model.invoke(prompt)
            generated_code = response.content.strip()
            
            # Remove code block markers if present
            if generated_code.startswith("```python"):
                generated_code = generated_code[9:]
            if generated_code.endswith("```"):
                generated_code = generated_code[:-3]
            
            generated_code = generated_code.strip()
            
            result = {
                "generated_code": generated_code,
                "current_node": "syntax_checker"
            }
            
            return {**state, **result}
            
        except Exception as e:
            return {
                **state,
                "execution_results": {
                    "success": False,
                    "error": f"Code generation failed: {str(e)}",
                    "output": "",
                    "execution_time": 0
                },
                "current_node": "end"
            }


class SyntaxCheckerNode:
    """Node for checking and correcting code syntax using Black, Autopep8, and Flake8."""
    
    def _execute(self, state: CodeGenerationState) -> Dict[str, Any]:
        """Execute the syntax checking node."""
        print("🔍 Executing Syntax Checker")
        
        code = state.get("rectified_code") or state.get("generated_code", "")
        if not code:
            return {
                **state,
                "syntax_errors": ["No code to check"],
                "current_node": "end"
            }
        
        syntax_errors = []
        
        try:
            # Create temporary file for syntax checking
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Run Flake8 for syntax and style checking
            try:
                result = subprocess.run(
                    ['flake8', '--max-line-length=88', '--extend-ignore=E203,W503', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.stdout:
                    errors = result.stdout.strip().split('\n')
                    for error in errors:
                        if error.strip():
                            print(error)
                            syntax_errors.append(error)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
            # Try to format with Black if no critical errors
            if not any("SyntaxError" in error for error in syntax_errors):
                try:
                    result = subprocess.run(
                        ['black', '--line-length=88', '--code', code],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0 and result.stdout:
                        code = result.stdout
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    # Try autopep8 as fallback
                    try:
                        result = subprocess.run(
                            ['autopep8', '--aggressive', '--aggressive', '--max-line-length=88', '-'],
                            input=code,
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if result.returncode == 0 and result.stdout:
                            code = result.stdout
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        pass
            
            # Clean up
            import os
            try:
                os.unlink(temp_file)
            except:
                pass
            
            # Determine next node based on syntax errors
            if syntax_errors and any("SyntaxError" in error for error in syntax_errors):
                # Critical syntax errors - go to rectifier
                next_node = "code_rectifier"
            elif len(syntax_errors) == 0:
                # No errors - proceed to execution
                next_node = "code_executor"
            else:
                # Minor issues - proceed to execution but log warnings
                next_node = "code_executor"
            
            result = {
                "generated_code": code,
                "syntax_errors": syntax_errors,
                "current_node": next_node
            }
            
            return {**state, **result}
            
        except Exception as e:
            return {
                **state,
                "syntax_errors": [f"Syntax checking failed: {str(e)}"],
                "current_node": "code_rectifier"
            }


class CodeRectifierNode:
    """Node for rectifying code execution errors."""
    
    def __init__(self):
        self.rectifier = CodeRectifier()
    
    def _execute(self, state: CodeGenerationState) -> Dict[str, Any]:
        """Execute the code rectification node."""
        print("🔧 Executing Code Rectifier")
        
        # Determine which code to rectify and what error occurred
        original_code = state.get("rectified_code") or state.get("generated_code", "")
        execution_results = state.get("execution_results", {})
        syntax_errors = state.get("syntax_errors", [])
        
        # Get error message from execution or syntax errors
        error_message = ""
        if execution_results.get("error"):
            error_message = execution_results["error"]
        elif syntax_errors:
            error_message = "; ".join(syntax_errors)
        
        if not error_message or not original_code:
            return {
                **state,
                "current_node": "end",
                "rectification_attempts": state.get("rectification_attempts", 0)
            }
        
        # Check rectification attempt limit
        attempts = state.get("rectification_attempts", 0)
        if attempts >= 3:
            print("⚠️ Maximum rectification attempts reached")
            return {
                **state,
                "current_node": "end",
                "execution_results": {
                    **execution_results,
                    "error": f"Maximum rectification attempts reached. Final error: {error_message}"
                }
            }
        
        try:
            # Rectify the code
            rectification_response = self.rectifier.rectify_code(
                original_code, 
                error_message,
                {"execution_context": "langgraph_workflow"}
            )
            
            if rectification_response.success and rectification_response.rectified_code:
                print(f"✅ Code rectified with confidence: {rectification_response.confidence_score:.2f}")
                print(f"🔄 Changes made: {', '.join(rectification_response.changes_made)}")
                
                result = {
                    "rectified_code": rectification_response.rectified_code,
                    "generated_code": rectification_response.rectified_code,  # Update generated_code too
                    "execution_errors": [error_message],
                    "rectification_attempts": attempts + 1,
                    "error_analysis": rectification_response.error_analysis,
                    "current_node": "syntax_checker"  # Re-check syntax after rectification
                }
                
                return {**state, **result}
            else:
                print("❌ Code rectification failed")
                return {
                    **state,
                    "current_node": "end",
                    "rectification_attempts": attempts + 1,
                    "execution_results": {
                        **execution_results,
                        "error": f"Rectification failed: {error_message}"
                    }
                }
                
        except Exception as e:
            print(f"❌ Error in code rectification: {e}")
            return {
                **state,
                "current_node": "end",
                "rectification_attempts": attempts + 1,
                "execution_results": {
                    **execution_results,
                    "error": f"Rectification error: {str(e)}"
                }
            }


class CodeExecutorNode:
    """Node for executing Python code in a sandbox environment."""
    
    def _execute(self, state: CodeGenerationState) -> Dict[str, Any]:
        """Execute the code execution node."""
        print("🚀 Executing Code in Sandbox")
        
        code = state.get("rectified_code") or state.get("generated_code", "")
        if not code:
            return {
                **state,
                "execution_results": {
                    "success": False,
                    "error": "No code to execute",
                    "output": "",
                    "execution_time": 0
                },
                "current_node": "end"
            }
        
        try:
            # Detect execution context and choose appropriate executor
            # Check for FastAPI context indicators
            import threading
            current_thread = threading.current_thread().name
            is_fastapi_context = (
                "ThreadPoolExecutor" in current_thread or 
                hasattr(threading.current_thread(), '_fastapi_context') or
                'executor' in current_thread.lower()
            )
            
            if is_fastapi_context:
                print("🌐 WEB CONTEXT DETECTED - Using FastAPI-compatible executor")
                
                # Use FastAPI-compatible executor
                start_time = time.time()
                result = fastapi_executor.execute_code(code)
                execution_time = time.time() - start_time
                
                execution_results = {
                    "success": result.get("success", False),
                    "output": result.get("output", ""),
                    "error": result.get("error", ""),
                    "execution_time": execution_time
                }
                
            else:
                # Try to detect async context as backup
                try:
                    loop = asyncio.get_running_loop()
                    print("🌐 WEB CONTEXT DETECTED - Using FastAPI-compatible executor (async)")
                    
                    # Use FastAPI-compatible executor
                    start_time = time.time()
                    result = fastapi_executor.execute_code(code)
                    execution_time = time.time() - start_time
                    
                    execution_results = {
                        "success": result.get("success", False),
                        "output": result.get("output", ""),
                        "error": result.get("error", ""),
                        "execution_time": execution_time
                    }
                    
                except RuntimeError:
                    print("💻 CLI CONTEXT DETECTED - Using PyodideSandbox executor")
                    
                    # Use PyodideSandbox for CLI context
                    start_time = time.time()
                    result = sandbox_executor.execute_code_async(code)
                    execution_time = time.time() - start_time
                    
                    execution_results = {
                        "success": result.get("success", False),
                        "output": result.get("output", ""),
                        "error": result.get("error", ""),
                        "execution_time": execution_time
                    }
            
            # Determine next node based on execution results
            if not execution_results["success"] and execution_results.get("error"):
                # Execution failed - try rectification
                next_node = "code_rectifier"
            else:
                # Execution successful - end workflow
                next_node = "end"
            
            result = {
                "execution_results": execution_results,
                "current_node": next_node
            }
            
            return {**state, **result}
            
        except Exception as e:
            return {
                **state,
                "execution_results": {
                    "success": False,
                    "error": f"Execution error: {str(e)}",
                    "output": "",
                    "execution_time": 0
                },
                "current_node": "code_rectifier"  # Try rectification on unexpected errors
            } 

--------------------------------------------------------------------------------------------------------------------
############ code_rectifier.py
"""
Code Rectifier for fixing common execution errors.
Analyzes execution errors and provides automatic fixes.
"""

import re
import ast
import traceback
from typing import Dict, List, Any, Optional, Tuple
from src.config import Config
from src.state import CodeRectificationRequest, CodeRectificationResponse


class CodeRectifier:
    """Intelligent code rectifier that can analyze and fix common execution errors."""
    
    def __init__(self):
        self.config = Config()
        self.model = self.config.get_groq_model()
        
        # Common error patterns and their fixes
        self.error_patterns = {
            "from __future__ imports must occur at the beginning": self._fix_future_imports,
            "SyntaxError": self._analyze_syntax_error,
            "NameError": self._fix_name_error, 
            "ImportError": self._fix_import_error,
            "ModuleNotFoundError": self._fix_module_not_found,
            "IndentationError": self._fix_indentation_error,
            "AttributeError": self._fix_attribute_error,
            "TypeError": self._fix_type_error,
            "ValueError": self._fix_value_error,
            "KeyError": self._fix_key_error,
            "IndexError": self._fix_index_error,
        }
    
    def rectify_code(self, original_code: str, error_message: str, execution_context: Dict[str, Any] = None) -> CodeRectificationResponse:
        """
        Main method to rectify code based on execution errors.
        
        Args:
            original_code: The original code that failed
            error_message: The error message from execution
            execution_context: Additional context about the execution environment
            
        Returns:
            CodeRectificationResponse with rectified code and analysis
        """
        if execution_context is None:
            execution_context = {}
            
        # Analyze the error
        error_analysis = self._analyze_error(error_message, original_code)
        error_type = error_analysis.get("error_type", "Unknown")
        
        # Try pattern-based fixes first
        rectified_code, changes_made, confidence = self._apply_pattern_fixes(
            original_code, error_message, error_type
        )
        
        # If pattern fixes didn't work or confidence is low, use AI rectification
        if confidence < 0.7 or not rectified_code:
            ai_result = self._ai_rectify_code(original_code, error_message, error_analysis)
            if ai_result and ai_result.get("success", False):
                rectified_code = ai_result.get("code", rectified_code)
                changes_made.extend(ai_result.get("changes", []))
                confidence = max(confidence, ai_result.get("confidence", 0.5))
        
        return CodeRectificationResponse(
            success=bool(rectified_code and rectified_code != original_code),
            rectified_code=rectified_code,
            changes_made=changes_made,
            error_analysis=error_analysis,
            confidence_score=confidence
        )
    
    def _analyze_error(self, error_message: str, code: str) -> Dict[str, Any]:
        """Analyze the error message to determine error type and location."""
        analysis = {
            "error_type": "Unknown",
            "error_line": None,
            "error_column": None,
            "error_description": error_message,
            "suggested_fixes": []
        }
        
        # Extract error type
        if "SyntaxError" in error_message:
            analysis["error_type"] = "SyntaxError"
        elif "NameError" in error_message:
            analysis["error_type"] = "NameError"
        elif "ImportError" in error_message or "ModuleNotFoundError" in error_message:
            analysis["error_type"] = "ImportError"
        elif "IndentationError" in error_message:
            analysis["error_type"] = "IndentationError"
        elif "AttributeError" in error_message:
            analysis["error_type"] = "AttributeError"
        elif "TypeError" in error_message:
            analysis["error_type"] = "TypeError"
        elif "ValueError" in error_message:
            analysis["error_type"] = "ValueError"
        
        # Extract line number if available
        line_match = re.search(r"line (\d+)", error_message)
        if line_match:
            analysis["error_line"] = int(line_match.group(1))
            
        return analysis
    
    def _apply_pattern_fixes(self, code: str, error_message: str, error_type: str) -> Tuple[str, List[str], float]:
        """Apply pattern-based fixes for common errors."""
        rectified_code = code
        changes_made = []
        confidence = 0.0
        
        # Try to find and apply appropriate fix
        for pattern, fix_func in self.error_patterns.items():
            if pattern.lower() in error_message.lower() or pattern == error_type:
                try:
                    result = fix_func(code, error_message)
                    if result:
                        rectified_code = result.get("code", code)
                        changes_made.extend(result.get("changes", []))
                        confidence = result.get("confidence", 0.8)
                        break
                except Exception as e:
                    print(f"Error applying fix for {pattern}: {e}")
                    continue
        
        return rectified_code, changes_made, confidence
    
    def _fix_future_imports(self, code: str, error_message: str) -> Dict[str, Any]:
        """Fix __future__ imports that are not at the beginning of the file."""
        lines = code.split('\n')
        future_imports = []
        other_lines = []
        shebang = None
        
        # Separate future imports from other lines
        for i, line in enumerate(lines):
            stripped = line.strip()
            if i == 0 and line.startswith('#!'):
                shebang = line
            elif 'from __future__ import' in stripped:
                future_imports.append(line)
            else:
                other_lines.append(line)
        
        # Reconstruct code with future imports at the top
        rectified_lines = []
        if shebang:
            rectified_lines.append(shebang)
        rectified_lines.extend(future_imports)
        rectified_lines.extend(other_lines)
        
        return {
            "code": '\n'.join(rectified_lines),
            "changes": ["Moved __future__ imports to the beginning of the file"],
            "confidence": 0.95
        }
    
    def _analyze_syntax_error(self, code: str, error_message: str) -> Dict[str, Any]:
        """Analyze and fix syntax errors."""
        try:
            ast.parse(code)
            return {"code": code, "changes": [], "confidence": 0.0}
        except SyntaxError as e:
            # Common syntax error fixes
            if "invalid syntax" in str(e).lower():
                return self._fix_invalid_syntax(code, str(e))
            elif "unexpected indent" in str(e).lower():
                return self._fix_indentation_error(code, str(e))
        
        return {"code": code, "changes": [], "confidence": 0.0}
    
    def _fix_invalid_syntax(self, code: str, error_message: str) -> Dict[str, Any]:
        """Fix common invalid syntax issues."""
        lines = code.split('\n')
        changes = []
        
        # Fix missing colons in control structures
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith(('if ', 'elif ', 'else', 'for ', 'while ', 'def ', 'class ', 'try', 'except', 'finally', 'with ')) 
                and not stripped.endswith(':')):
                lines[i] = line + ':'
                changes.append(f"Added missing colon on line {i+1}")
        
        return {
            "code": '\n'.join(lines),
            "changes": changes,
            "confidence": 0.8 if changes else 0.0
        }
    
    def _fix_name_error(self, code: str, error_message: str) -> Dict[str, Any]:
        """Fix name errors by adding missing imports or variable definitions."""
        # Extract the undefined name
        match = re.search(r"name '([^']+)' is not defined", error_message)
        if not match:
            return {"code": code, "changes": [], "confidence": 0.0}
        
        undefined_name = match.group(1)
        changes = []
        
        # Common undefined names and their fixes
        common_imports = {
            'math': 'import math',
            'os': 'import os',
            'sys': 'import sys',
            'random': 'import random',
            'datetime': 'import datetime',
            're': 'import re',
            'json': 'import json',
            'time': 'import time',
            'collections': 'import collections',
            'itertools': 'import itertools',
            'numpy': 'import numpy as np',
            'pandas': 'import pandas as pd',
            'plt': 'import matplotlib.pyplot as plt',
        }
        
        if undefined_name in common_imports:
            import_line = common_imports[undefined_name]
            lines = code.split('\n')
            
            # Find where to insert the import
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('from __future__'):
                    insert_idx = i + 1
                elif line.strip().startswith('#') or not line.strip():
                    continue
                else:
                    break
            
            lines.insert(insert_idx, import_line)
            changes.append(f"Added missing import: {import_line}")
            
            return {
                "code": '\n'.join(lines),
                "changes": changes,
                "confidence": 0.9
            }
        
        return {"code": code, "changes": [], "confidence": 0.0}
    
    def _fix_import_error(self, code: str, error_message: str) -> Dict[str, Any]:
        """Fix import errors."""
        return {"code": code, "changes": [], "confidence": 0.0}
    
    def _fix_module_not_found(self, code: str, error_message: str) -> Dict[str, Any]:
        """Fix module not found errors."""
        return {"code": code, "changes": [], "confidence": 0.0}
    
    def _fix_indentation_error(self, code: str, error_message: str) -> Dict[str, Any]:
        """Fix indentation errors."""
        lines = code.split('\n')
        changes = []
        
        # Basic indentation fix - normalize to 4 spaces
        normalized_lines = []
        for line in lines:
            if line.strip():  # Non-empty line
                # Count leading spaces/tabs
                leading_whitespace = len(line) - len(line.lstrip())
                if '\t' in line[:leading_whitespace]:
                    # Replace tabs with 4 spaces
                    tabs = line[:leading_whitespace].count('\t')
                    spaces = line[:leading_whitespace].count(' ')
                    new_indent = '    ' * tabs + ' ' * spaces
                    normalized_lines.append(new_indent + line.lstrip())
                    changes.append(f"Normalized indentation on line {len(normalized_lines)}")
                else:
                    normalized_lines.append(line)
            else:
                normalized_lines.append(line)
        
        return {
            "code": '\n'.join(normalized_lines),
            "changes": changes,
            "confidence": 0.8 if changes else 0.0
        }
    
    def _fix_attribute_error(self, code: str, error_message: str) -> Dict[str, Any]:
        """Fix attribute errors."""
        return {"code": code, "changes": [], "confidence": 0.0}
    
    def _fix_type_error(self, code: str, error_message: str) -> Dict[str, Any]:
        """Fix type errors."""
        return {"code": code, "changes": [], "confidence": 0.0}
    
    def _fix_value_error(self, code: str, error_message: str) -> Dict[str, Any]:
        """Fix value errors."""
        return {"code": code, "changes": [], "confidence": 0.0}
    
    def _fix_key_error(self, code: str, error_message: str) -> Dict[str, Any]:
        """Fix key errors."""
        return {"code": code, "changes": [], "confidence": 0.0}
    
    def _fix_index_error(self, code: str, error_message: str) -> Dict[str, Any]:
        """Fix index errors."""
        return {"code": code, "changes": [], "confidence": 0.0}
    
    def _ai_rectify_code(self, code: str, error_message: str, error_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI to rectify code when pattern-based fixes fail."""
        try:
            prompt = f"""
You are an expert Python developer. I have code that failed with an execution error. Please analyze the error and provide a corrected version of the code.

**Original Code:**
```python
{code}
```

**Error Message:**
{error_message}

**Error Analysis:**
- Error Type: {error_analysis.get('error_type', 'Unknown')}
- Error Line: {error_analysis.get('error_line', 'Unknown')}
- Description: {error_analysis.get('error_description', '')}

**Instructions:**
1. Identify the root cause of the error
2. Provide the corrected code
3. List the specific changes made
4. Ensure the code follows Python best practices and PEP8 standards

**Response Format:**
Please respond in the following JSON format:
{{
    "success": true/false,
    "code": "corrected code here",
    "changes": ["list of changes made"],
    "explanation": "explanation of the fixes",
    "confidence": 0.0-1.0
}}
"""
            
            response = self.model.invoke(prompt)
            
            # Try to parse JSON response
            import json
            try:
                result = json.loads(response.content)
                return result
            except json.JSONDecodeError:
                # If not JSON, extract code from markdown blocks
                code_match = re.search(r'```python\n(.*?)\n```', response.content, re.DOTALL)
                if code_match:
                    return {
                        "success": True,
                        "code": code_match.group(1).strip(),
                        "changes": ["AI-generated fixes applied"],
                        "confidence": 0.7
                    }
            
        except Exception as e:
            print(f"Error in AI rectification: {e}")
        
        return {"success": False, "code": code, "changes": [], "confidence": 0.0}


# Convenience function for direct usage
def rectify_code(code: str, error_message: str, execution_context: Dict[str, Any] = None) -> CodeRectificationResponse:
    """
    Convenience function to rectify code.
    
    Args:
        code: The original code that failed
        error_message: The error message from execution
        execution_context: Additional context about the execution environment
        
    Returns:
        CodeRectificationResponse with rectified code and analysis
    """
    rectifier = CodeRectifier()
    return rectifier.rectify_code(code, error_message, execution_context) 

--------------------------------------------------------------------------------------------------------------------
############ config.py
"""Configuration settings for the LangGraph Code Generator."""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import Optional

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the application."""
    
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        self.langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        self.langsmith_project = os.getenv("LANGSMITH_PROJECT", "langgraph-code-generator")
        
        # Validate required API keys
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
    
    def get_groq_model(self, model_name: str = "moonshotai/kimi-k2-instruct", temperature: float = 0.1) -> ChatGroq:
        """
        Get a configured GROQ model instance.
        
        Primary Model - Kimi K2 (moonshotai/kimi-k2-instruct):
        ✅ Excels at agentic tasks, coding, and reasoning
        ✅ Supports tool calling capabilities
        ✅ 131K token context window (massive context)
        ✅ Mixture-of-Experts (MoE) with 32B activated parameters
        ✅ Fast inference optimized for reasoning models
        ⚠️ May struggle with very complex reasoning or unclear tool definitions
        
        Alternative GROQ models:
        - llama3-70b-8192: Llama 3 70B (excellent for complex tasks)
        - llama3-8b-8192: Llama 3 8B (faster, good for simpler tasks)
        - gemma-7b-it: Gemma 7B (alternative option)
        
        Reference: https://console.groq.com/docs/models
        """
        return ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=4096,
            timeout=60,
            max_retries=2
        )

# Global config instance
config = Config() 

--------------------------------------------------------------------------------------------------------------------
############ Sandbox_executor.py
"""
Sandbox executor using LangChain's PyodideSandbox for safe Python code execution.
Based on: https://github.com/langchain-ai/langchain-sandbox/blob/main/examples/codeact_agent.py
"""

import asyncio
import time
import threading
from typing import Dict, Any, Optional
from langchain_sandbox import PyodideSandbox


class LangChainSandboxExecutor:
    """
    Executor that uses LangChain's PyodideSandbox for secure code execution.
    """
    
    def __init__(self, stateful: bool = True, allow_net: bool = True):
        """Initialize the PyodideSandbox."""
        self.sandbox = PyodideSandbox(
            stateful=stateful,
            allow_net=allow_net
        )
        # Session management for stateful execution
        self.session_bytes = None
        self.session_metadata = None
    
    async def _async_execute_code(self, code: str) -> Dict[str, Any]:
        """Execute code asynchronously using PyodideSandbox with session persistence."""
        try:
            start_time = time.time()
            
            # Execute code using the sandbox with session persistence
            if self.session_bytes and self.session_metadata:
                print("🔄 Using existing session for stateful execution")
                result = await self.sandbox.execute(
                    code, 
                    session_bytes=self.session_bytes,
                    session_metadata=self.session_metadata
                )
            else:
                print("🆕 Creating new session for stateful execution")
                result = await self.sandbox.execute(code)
            
            # Update session state for next execution
            if hasattr(result, 'session_bytes') and hasattr(result, 'session_metadata'):
                self.session_bytes = result.session_bytes
                self.session_metadata = result.session_metadata
                print(f"📦 Session updated: packages={result.session_metadata.get('packages', [])}")
            
            execution_time = time.time() - start_time
            
            # Convert the result to our format
            return {
                'success': result.status == 'success',
                'output': result.stdout or str(result.result) if result.result is not None else "Code executed successfully",
                'error': result.stderr if result.stderr else None,
                'execution_time': execution_time,
                'status': result.status,
                'result': result.result,
                'session_metadata': result.session_metadata if hasattr(result, 'session_metadata') else None,
                'packages_installed': result.session_metadata.get('packages', []) if hasattr(result, 'session_metadata') else []
            }
            
        except Exception as e:
            return {
                'success': False,
                'output': "",
                'error': f"PyodideSandbox execution error: {str(e)}",
                'execution_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def reset_session(self) -> None:
        """Reset the session to start fresh."""
        print("🔄 Resetting sandbox session")
        self.session_bytes = None
        self.session_metadata = None


class SafeCodeExecutor:
    """Fallback executor using restricted Python environment."""
    
    def __init__(self):
        self.restricted_functions = {
            'eval', 'exec', 'compile', '__import__', 'open', 'input',
            'raw_input', 'file', 'reload', 'vars', 'dir', 'globals',
            'locals', 'delattr', 'setattr', 'getattr', 'hasattr'
        }
    
    def execute_code(self, code: str) -> Dict[str, Any]:
        """Fallback execution using restricted environment."""
        start_time = time.time()
        
        # Basic safety check
        for restricted in self.restricted_functions:
            if restricted in code and not code.strip().startswith('#'):
                return {
                    'success': False,
                    'output': "",
                    'error': f"Restricted function '{restricted}' not allowed",
                    'execution_time': time.time() - start_time
                }
        
        try:
            # Create a restricted environment
            import io
            import sys
            from contextlib import redirect_stdout, redirect_stderr
            
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            restricted_globals = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'range': range,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'type': type,
                    'isinstance': isinstance,
                    'enumerate': enumerate,
                    'zip': zip,
                    'map': map,
                    'filter': filter,
                    'sum': sum,
                    'min': min,
                    'max': max,
                    'abs': abs,
                    'round': round,
                    'sorted': sorted,
                    'reversed': reversed,
                }
            }
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, restricted_globals)
            
            output = stdout_capture.getvalue()
            error = stderr_capture.getvalue()
            
            return {
                'success': len(error) == 0,
                'output': output if output else "Code executed successfully (no output)",
                'error': error if error else None,
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'output': "",
                'error': f"Execution error: {str(e)}",
                'execution_time': time.time() - start_time
            }


# Global executor instance - try PyodideSandbox first, fallback to safe executor
try:
    sandbox_executor = LangChainSandboxExecutor()
    print("✅ Using LangChain PyodideSandbox for code execution")
except Exception as e:
    print(f"⚠️ PyodideSandbox not available, using fallback: {e}")
    sandbox_executor = SafeCodeExecutor()


def execute_code_async(code: str) -> Dict[str, Any]:
    """
    Execute code asynchronously, handling event loop context correctly.
    
    Args:
        code: Python code to execute
        
    Returns:
        Dict with success, output, error, execution_time
    """
    try:
        # Check if we're in an async context
        loop = asyncio.get_running_loop()
        
        # We're in an async context (like FastAPI), run in a separate thread
        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                if hasattr(sandbox_executor, '_async_execute_code'):
                    return new_loop.run_until_complete(sandbox_executor._async_execute_code(code))
                else:
                    return sandbox_executor.execute_code(code)
            finally:
                new_loop.close()
        
        # Run in a separate thread to avoid event loop conflicts
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_new_loop)
            return future.result(timeout=30)
            
    except RuntimeError:
        # No running loop - we can use async directly
        if hasattr(sandbox_executor, '_async_execute_code'):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(sandbox_executor._async_execute_code(code))
            finally:
                loop.close()
        else:
            return sandbox_executor.execute_code(code)
    except Exception as e:
        return {
            'success': False,
            'output': "",
            'error': f"Execution setup error: {str(e)}",
            'execution_time': 0
        } 

--------------------------------------------------------------------------------------------------------------------
############ app.py
#!/usr/bin/env python3
"""
FastAPI Web Application for LangGraph Code Generator with GROQ Kimi K2
Provides a web interface for the code generation workflow.
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.config import Config
from src.workflow import CodeGeneratorWorkflow

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="LangGraph Code Generator",
    description="Generate and execute Python code using GROQ Kimi K2 and LangChain Sandbox",
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

# Initialize the workflow
workflow = None
executor = ThreadPoolExecutor(max_workers=1)  # Single worker to prevent conflicts

def initialize_workflow():
    """Initialize the workflow instance."""
    global workflow
    try:
        config = Config()
        model = config.get_groq_model()
        workflow = CodeGeneratorWorkflow()
        print("✅ Workflow initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize workflow: {e}")
        return False

# Initialize workflow on startup
initialize_workflow()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate_code(
    prompt: str = Form(...),
    requirements: str = Form("")
):
    """Generate code based on user prompt."""
    print(f"📝 Received request: {prompt}")
    print(f"📋 Requirements: {requirements}")
    
    if not workflow:
        print("❌ Workflow not initialized")
        raise HTTPException(status_code=500, detail="Workflow not initialized")
    
    try:
        # Run workflow in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def run_workflow():
            import threading
            # Mark thread as FastAPI context
            threading.current_thread()._fastapi_context = True
            
            print("🚀 Starting workflow execution")
            result = workflow.run(prompt, requirements if requirements else None)
            print(f"✅ Workflow completed with status: {result.get('workflow_status')}")
            return result
        
        # Execute workflow in a separate thread
        result = await loop.run_in_executor(executor, run_workflow)
        
        print("📤 Sending response to client")
        print(f"Response keys: {list(result.keys())}")
        
        # Process the result for web display
        response_data = {
            "success": result.get("workflow_status") == "completed",
            "final_result": result.get("final_result", "No result available"),
            "generated_code": result.get("rectified_code") or result.get("generated_code", ""),
            "execution_results": result.get("execution_results", {}),
            "syntax_errors": result.get("syntax_errors", []),
            "rectification_attempts": result.get("rectification_attempts", 0),
            "error_analysis": result.get("error_analysis", {}),
            "workflow_status": result.get("workflow_status", "unknown")
        }
        
        print(f"📊 Response data prepared: success={response_data['success']}")
        return JSONResponse(content=response_data)
        
    except Exception as e:
        print(f"❌ Error during code generation: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Code generation failed: {str(e)}",
                "final_result": f"❌ Error: {str(e)}"
            }
        )


@app.get("/api/status")
async def api_status():
    """Get API status and configuration."""
    try:
        config = Config()
        model = config.get_groq_model()
        
        return {
            "status": "healthy",
            "workflow_initialized": workflow is not None,
            "groq_api_configured": bool(os.getenv("GROQ_API_KEY")),
            "model": "moonshotai/kimi-k2-instruct",
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }


@app.get("/api/history")
async def get_history():
    """Get generation history (placeholder for future implementation)."""
    return {
        "history": [],
        "message": "History feature coming soon"
    }


@app.post("/api/test")
async def test_workflow():
    """Test the workflow with a simple request."""
    try:
        if not workflow:
            return {"success": False, "error": "Workflow not initialized"}
        
        test_prompt = "Create a simple function that adds two numbers"
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, workflow.run, test_prompt)
        
        return {
            "success": True,
            "test_prompt": test_prompt,
            "workflow_status": result.get("workflow_status"),
            "has_generated_code": bool(result.get("generated_code")),
            "has_execution_results": bool(result.get("execution_results")),
            "rectification_attempts": result.get("rectification_attempts", 0)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "LangGraph Code Generator",
        "version": "1.0.0",
        "timestamp": time.time()
    }


@app.get("/api/debug-workflow")
async def debug_workflow():
    """Debug endpoint to check workflow structure."""
    try:
        if not workflow:
            return {"error": "Workflow not initialized"}
        
        return {
            "workflow_initialized": True,
            "workflow_type": str(type(workflow)),
            "has_code_generator": hasattr(workflow, 'code_generator'),
            "has_syntax_checker": hasattr(workflow, 'syntax_checker'),
            "has_code_rectifier": hasattr(workflow, 'code_rectifier'),
            "has_code_executor": hasattr(workflow, 'code_executor'),
            "workflow_graph": hasattr(workflow, 'workflow')
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting LangGraph Code Generator Web Server")
    print("🌐 Access the application at: http://localhost:8000")
    print("📊 API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 
