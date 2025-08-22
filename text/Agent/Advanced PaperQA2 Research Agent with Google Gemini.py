### From https://www.marktechpost.com/2025/08/09/building-an-advanced-paperqa2-research-agent-with-google-gemini-for-scientific-literature-analysis/?amp

!pip install paper-qa>=5 google-generativeai requests pypdf2 -q


import os
import asyncio
import tempfile
import requests
from pathlib import Path
from paperqa import Settings, ask, agent_query
from paperqa.settings import AgentSettings
import google.generativeai as genai


GEMINI_API_KEY = "Use Your Own API Key Here"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


genai.configure(api_key=GEMINI_API_KEY)
print("✅ Gemini API key configured successfully!")

def download_sample_papers():
   """Download sample AI/ML research papers for demonstration"""
   papers = {
       "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
       "bert_paper.pdf": "https://arxiv.org/pdf/1810.04805.pdf",
       "gpt3_paper.pdf": "https://arxiv.org/pdf/2005.14165.pdf"
   }
  
   papers_dir = Path("sample_papers")
   papers_dir.mkdir(exist_ok=True)
  
   print("📥 Downloading sample research papers...")
   for filename, url in papers.items():
       filepath = papers_dir / filename
       if not filepath.exists():
           try:
               response = requests.get(url, stream=True, timeout=30)
               response.raise_for_status()
               with open(filepath, 'wb') as f:
                   for chunk in response.iter_content(chunk_size=8192):
                       f.write(chunk)
               print(f"✅ Downloaded: {filename}")
           except Exception as e:
               print(f"❌ Failed to download {filename}: {e}")
       else:
           print(f"📄 Already exists: {filename}")
  
   return str(papers_dir)


papers_directory = download_sample_papers()


def create_gemini_settings(paper_dir: str, temperature: float = 0.1):
   """Create optimized settings for PaperQA2 with Gemini models"""
  
   return Settings(
       llm="gemini/gemini-1.5-flash",
       summary_llm="gemini/gemini-1.5-flash",
      
       agent=AgentSettings(
           agent_llm="gemini/gemini-1.5-flash",
           search_count=6, 
           timeout=300.0, 
       ),
      
       embedding="gemini/text-embedding-004",
      
       temperature=temperature,
       paper_directory=paper_dir,
      
       answer=dict(
           evidence_k=8,            
           answer_max_sources=4,      
           evidence_summary_length="about 80 words",
           answer_length="about 150 words, but can be longer",
           max_concurrent_requests=2,
       ),
      
       parsing=dict(
           chunk_size=4000,
           overlap=200,
       ),
      
       verbosity=1,
   )

class PaperQAAgent:
   """Advanced AI Agent for scientific literature analysis using PaperQA2"""
  
   def __init__(self, papers_directory: str, temperature: float = 0.1):
       self.settings = create_gemini_settings(papers_directory, temperature)
       self.papers_dir = papers_directory
       print(f"🤖 PaperQA Agent initialized with papers from: {papers_directory}")
      
   async def ask_question(self, question: str, use_agent: bool = True):
       """Ask a question about the research papers"""
       print(f"\n❓ Question: {question}")
       print("🔍 Searching through research papers...")
      
       try:
           if use_agent:
               response = await agent_query(query=question, settings=self.settings)
           else:
               response = ask(question, settings=self.settings)
              
           return response
          
       except Exception as e:
           print(f"❌ Error processing question: {e}")
           return None
  
   def display_answer(self, response):
       """Display the answer with formatting"""
       if response is None:
           print("❌ No response received")
           return
          
       print("\n" + "="*60)
       print("📋 ANSWER:")
       print("="*60)
      
       answer_text = getattr(response, 'answer', str(response))
       print(f"\n{answer_text}")
      
       contexts = getattr(response, 'contexts', getattr(response, 'context', []))
       if contexts:
           print("\n" + "-"*40)
           print("📚 SOURCES USED:")
           print("-"*40)
           for i, context in enumerate(contexts[:3], 1):
               context_name = getattr(context, 'name', getattr(context, 'doc', f'Source {i}'))
               context_text = getattr(context, 'text', getattr(context, 'content', str(context)))
               print(f"\n{i}. {context_name}")
               print(f"   Text preview: {context_text[:150]}...")
  
   async def multi_question_analysis(self, questions: list):
       """Analyze multiple questions in sequence"""
       results = {}
       for i, question in enumerate(questions, 1):
           print(f"\n🔄 Processing question {i}/{len(questions)}")
           response = await self.ask_question(question)
           results = response
          
           if response:
               print(f"✅ Completed: {question[:50]}...")
           else:
               print(f"❌ Failed: {question[:50]}...")
              
       return results
  
   async def comparative_analysis(self, topic: str):
       """Perform comparative analysis across papers"""
       questions = [
           f"What are the key innovations in {topic}?",
           f"What are the limitations of current {topic} approaches?",
           f"What future research directions are suggested for {topic}?",
       ]
      
       print(f"\n🔬 Starting comparative analysis on: {topic}")
       return await self.multi_question_analysis(questions)


async def basic_demo():
   """Demonstrate basic PaperQA functionality"""
   agent = PaperQAAgent(papers_directory)
  
   question = "What is the transformer architecture and why is it important?"
   response = await agent.ask_question(question)
   agent.display_answer(response)


print("🚀 Running basic demonstration...")
await basic_demo()


async def advanced_demo():
   """Demonstrate advanced multi-question analysis"""
   agent = PaperQAAgent(papers_directory, temperature=0.2)
  
   questions = [
       "How do attention mechanisms work in transformers?",
       "What are the computational challenges of large language models?",
       "How has pre-training evolved in natural language processing?"
   ]
  
   print("🧠 Running advanced multi-question analysis...")
   results = await agent.multi_question_analysis(questions)
  
   for question, response in results.items():
       print(f"\n{'='*80}")
       print(f"Q: {question}")
       print('='*80)
       if response:
           answer_text = getattr(response, 'answer', str(response))
           display_text = answer_text[:300] + "..." if len(answer_text) > 300 else answer_text
           print(display_text)
       else:
           print("❌ No answer available")


print("\n🚀 Running advanced demonstration...")
await advanced_demo()


async def research_comparison_demo():
   """Demonstrate comparative research analysis"""
   agent = PaperQAAgent(papers_directory)
  
   results = await agent.comparative_analysis("attention mechanisms in neural networks")
  
   print("\n" + "="*80)
   print("📊 COMPARATIVE ANALYSIS RESULTS")
   print("="*80)
  
   for question, response in results.items():
       print(f"\n🔍 {question}")
       print("-" * 50)
       if response:
           answer_text = getattr(response, 'answer', str(response))
           print(answer_text)
       else:
           print("❌ Analysis unavailable")
       print()


print("🚀 Running comparative research analysis...")
await research_comparison_demo()

def create_interactive_agent():
   """Create an interactive agent for custom queries"""
   agent = PaperQAAgent(papers_directory)
  
   async def query(question: str, show_sources: bool = True):
       """Interactive query function"""
       response = await agent.ask_question(question)
      
       if response:
           answer_text = getattr(response, 'answer', str(response))
           print(f"\n🤖 Answer:\n{answer_text}")
          
           if show_sources:
               contexts = getattr(response, 'contexts', getattr(response, 'context', []))
               if contexts:
                   print(f"\n📚 Based on {len(contexts)} sources:")
                   for i, ctx in enumerate(contexts[:3], 1):
                       ctx_name = getattr(ctx, 'name', getattr(ctx, 'doc', f'Source {i}'))
                       print(f"  {i}. {ctx_name}")
       else:
           print("❌ Sorry, I couldn't find an answer to that question.")
          
       return response
  
   return query


interactive_query = create_interactive_agent()


print("\n🎯 Interactive agent ready! You can now ask custom questions:")
print("Example: await interactive_query('How do transformers handle long sequences?')")


def print_usage_tips():
   """Print helpful usage tips"""
   tips = """
   🎯 USAGE TIPS FOR PAPERQA2 WITH GEMINI:
  
   1. 📝 Question Formulation:
      - Be specific about what you want to know
      - Ask about comparisons, mechanisms, or implications
      - Use domain-specific terminology
  
   2. 🔧 Model Configuration:
      - Gemini 1.5 Flash is free and reliable
      - Adjust temperature (0.0-1.0) for creativity vs precision
      - Use smaller chunk_size for better processing
  
   3. 📚 Document Management:
      - Add PDFs to the papers directory
      - Use meaningful filenames
      - Mix different types of papers for better coverage
  
   4. ⚡ Performance Optimization:
      - Limit concurrent requests for free tier
      - Use smaller evidence_k values for faster responses
      - Cache results by saving the agent state
  
   5. 🧠 Advanced Usage:
      - Chain multiple questions for deeper analysis
      - Use comparative analysis for research reviews
      - Combine with other tools for complete workflows
  
   📖 Example Questions to Try:
   - "Compare the attention mechanisms in BERT vs GPT models"
   - "What are the computational bottlenecks in transformer training?"
   - "How has pre-training evolved from word2vec to modern LLMs?"
   - "What are the key innovations that made transformers successful?"
   """
   print(tips)


print_usage_tips()


def save_analysis_results(results: dict, filename: str = "paperqa_analysis.txt"):
   """Save analysis results to a file"""
   with open(filename, 'w', encoding='utf-8') as f:
       f.write("PaperQA2 Analysis Results\n")
       f.write("=" * 50 + "\n\n")
      
       for question, response in results.items():
           f.write(f"Question: {question}\n")
           f.write("-" * 30 + "\n")
           if response:
               answer_text = getattr(response, 'answer', str(response))
               f.write(f"Answer: {answer_text}\n")
              
               contexts = getattr(response, 'contexts', getattr(response, 'context', []))
               if contexts:
                   f.write(f"\nSources ({len(contexts)}):\n")
                   for i, ctx in enumerate(contexts, 1):
                       ctx_name = getattr(ctx, 'name', getattr(ctx, 'doc', f'Source {i}'))
                       f.write(f"  {i}. {ctx_name}\n")
           else:
               f.write("Answer: No response available\n")
           f.write("\n" + "="*50 + "\n\n")
  
   print(f"💾 Results saved to: {filename}")


print("✅ Tutorial complete! You now have a fully functional PaperQA2 AI Agent with Gemini.")

