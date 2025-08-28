### From https://www.marktechpost.com/2025/08/23/a-full-code-implementation-to-design-a-graph-structured-ai-agent-with-gemini-for-task-planning-retrieval-computation-and-self-critique/

import os, json, time, ast, math, getpass
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any
import google.generativeai as genai


try:
   import networkx as nx
except ImportError:
   nx = None

def make_model(api_key: str, model_name: str = "gemini-1.5-flash"):
   genai.configure(api_key=api_key)
   return genai.GenerativeModel(model_name, system_instruction=(
       "You are GraphAgent, a principled planner-executor. "
       "Prefer structured, concise outputs; use provided tools when asked."
   ))


def call_llm(model, prompt: str, temperature=0.2) -> str:
   r = model.generate_content(prompt, generation_config={"temperature": temperature})
   return (r.text or "").strip()

def safe_eval_math(expr: str) -> str:
   node = ast.parse(expr, mode="eval")
   allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
              ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
              ast.USub, ast.UAdd, ast.FloorDiv, ast.AST)
   def check(n):
       if not isinstance(n, allowed): raise ValueError("Unsafe expression")
       for c in ast.iter_child_nodes(n): check(c)
   check(node)
   return str(eval(compile(node, "<math>", "eval"), {"__builtins__": {}}, {}))


DOCS = [
   "Solar panels convert sunlight to electricity; capacity factor ~20%.",
   "Wind turbines harvest kinetic energy; onshore capacity factor ~35%.",
   "RAG = retrieval-augmented generation joins search with prompting.",
   "LangGraph enables cyclic graphs of agents; good for tool orchestration.",
]

def search_docs(q: str, k: int = 3) -> List[str]:
   ql = q.lower()
   scored = sorted(DOCS, key=lambda d: -sum(w in d.lower() for w in ql.split()))
   return scored[:k]

@dataclass
class State:
   task: str
   plan: str = ""
   scratch: List[str] = field(default_factory=list)
   evidence: List[str] = field(default_factory=list)
   result: str = ""
   step: int = 0
   done: bool = False


def node_plan(state: State, model) -> str:
   prompt = f"""Plan step-by-step to solve the user task.
Task: {state.task}
Return JSON: {{"subtasks": ["..."], "tools": {{"search": true/false, "math": true/false}}, "success_criteria": ["..."]}}"""
   js = call_llm(model, prompt)
   try:
       plan = json.loads(js[js.find("{"): js.rfind("}")+1])
   except Exception:
       plan = {"subtasks": ["Research", "Synthesize"], "tools": {"search": True, "math": False}, "success_criteria": ["clear answer"]}
   state.plan = json.dumps(plan, indent=2)
   state.scratch.append("PLAN:\n"+state.plan)
   return "route"


def node_route(state: State, model) -> str:
   prompt = f"""You are a router. Decide next node.
Context scratch:\n{chr(10).join(state.scratch[-5:])}
If math needed -> 'math', if research needed -> 'research', if ready -> 'write'.
Return one token from [research, math, write]. Task: {state.task}"""
   choice = call_llm(model, prompt).lower()
   if "math" in choice and any(ch.isdigit() for ch in state.task):
       return "math"
   if "research" in choice or not state.evidence:
       return "research"
   return "write"


def node_research(state: State, model) -> str:
   prompt = f"""Generate 3 focused search queries for:
Task: {state.task}
Return as JSON list of strings."""
   qjson = call_llm(model, prompt)
   try:
       queries = json.loads(qjson[qjson.find("["): qjson.rfind("]")+1])[:3]
   except Exception:
       queries = [state.task, "background "+state.task, "pros cons "+state.task]
   hits = []
   for q in queries:
       hits.extend(search_docs(q, k=2))
   state.evidence.extend(list(dict.fromkeys(hits)))
   state.scratch.append("EVIDENCE:\n- " + "\n- ".join(hits))
   return "route"


def node_math(state: State, model) -> str:
   prompt = "Extract a single arithmetic expression from this task:\n"+state.task
   expr = call_llm(model, prompt)
   expr = "".join(ch for ch in expr if ch in "0123456789+-*/().%^ ")
   try:
       val = safe_eval_math(expr)
       state.scratch.append(f"MATH: {expr} = {val}")
   except Exception as e:
       state.scratch.append(f"MATH-ERROR: {expr} ({e})")
   return "route"


def node_write(state: State, model) -> str:
   prompt = f"""Write the final answer.
Task: {state.task}
Use the evidence and any math results below, cite inline like [1],[2].
Evidence:\n{chr(10).join(f'[{i+1}] '+e for i,e in enumerate(state.evidence))}
Notes:\n{chr(10).join(state.scratch[-5:])}
Return a concise, structured answer."""
   draft = call_llm(model, prompt, temperature=0.3)
   state.result = draft
   state.scratch.append("DRAFT:\n"+draft)
   return "critic"


def node_critic(state: State, model) -> str:
   prompt = f"""Critique and improve the answer for factuality, missing steps, and clarity.
If fix needed, return improved answer. Else return 'OK'.
Answer:\n{state.result}\nCriteria:\n{state.plan}"""
   crit = call_llm(model, prompt)
   if crit.strip().upper() != "OK" and len(crit) > 30:
       state.result = crit.strip()
       state.scratch.append("REVISED")
   state.done = True
   return "end"


NODES: Dict[str, Callable[[State, Any], str]] = {
   "plan": node_plan, "route": node_route, "research": node_research,
   "math": node_math, "write": node_write, "critic": node_critic
}


def run_graph(task: str, api_key: str) -> State:
   model = make_model(api_key)
   state = State(task=task)
   cur = "plan"
   max_steps = 12
   while not state.done and state.step < max_steps:
       state.step += 1
       nxt = NODES[cur](state, model)
       if nxt == "end": break
       cur = nxt
   return state


def ascii_graph():
   return """
START -> plan -> route -> (research <-> route) & (math <-> route) -> write -> critic -> END
"""

if __name__ == "__main__":
   key = os.getenv("GEMINI_API_KEY") or getpass.getpass(" Enter GEMINI_API_KEY: ")
   task = input(" Enter your task: ").strip() or "Compare solar vs wind for reliability; compute 5*7."
   t0 = time.time()
   state = run_graph(task, key)
   dt = time.time() - t0
   print("\n=== GRAPH ===", ascii_graph())
   print(f"\n Result in {dt:.2f}s:\n{state.result}\n")
   print("---- Evidence ----")
   print("\n".join(state.evidence))
   print("\n---- Scratch (last 5) ----")
   print("\n".join(state.scratch[-5:]))

