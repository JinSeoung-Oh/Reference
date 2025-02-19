## From https://medium.com/@romanbessouat/use-reasoning-models-like-deepseek-r1-to-transform-your-rag-into-rat-for-smarter-ai-7e81a4af6944

"""
1. Core Concept of RAT
   Retrieval-Augmented Thinking (RAT) is an extension of Retrieval-Augmented Generation (RAG).

   -a. In RAG, a language model first retrieves relevant documents from a knowledge base, then generates an answer in one pass.
   -b. RAT goes further by adding an iterative reasoning loop, allowing the model to refine its thought process and retrieved information 
       over multiple cycles before generating a final, polished answer.

2. Key Differences Between RAG and RAT
   -a. Single Pass vs. Multiple Passes
       -1. RAG: Retrieve documents → Generate response. (One pass)
       -2. RAT: Retrieve documents → Reason, reflect, and possibly retrieve more → Generate refined response. (Multiple passes)
   -b. Refinement
       -1. RAT’s multi-pass approach helps the system catch nuances or complex details that a single-pass RAG might overlook.
   -c. Depth of Reasoning
       -1. RAT models can conduct deeper analysis, akin to a human re-checking facts or reevaluating a solution, 
           leading to more accurate and context-rich answers.

3. Core Algorithm (Iterative Reasoning)
   -a. Overview
       At the heart of RAT is a cycle of reflection—sometimes called “iterative refinement” or “reasoning loops.” 
       Here’s how it works at a conceptual level:

      -1. Initial Retrieval
          -1) The system retrieves relevant snippets or documents in response to the user’s query.
      -2. Initial Reflection
          -1) A reasoning model (which could be an LLM specialized for reasoning tasks) processes the user’s question alongside 
              the initially retrieved documents.
          -2) It generates a preliminary answer or reflection.
      -3. Feedback / Reretrieval
          -1) This initial reflection can lead to a new retrieval step if the model identifies missing or insufficient information.
          -2) Alternatively, the model might refine its own explanation by focusing on more specific details.
      -4. Subsequent Iterations
          -1) The refined reflection gets fed back into the retrieval and reasoning process as many times as specified 
              (the text calls this parameter reflection or the number of reasoning iterations).
          -2) Each iteration aims to improve clarity, correctness, or completeness of the answer.
      -5. Stopping Criterion
          -1) RAT stops either after a certain number of iterations (e.g., 1, 2, or more) or when the model’s reasoning “stabilizes,” 
              meaning no substantial improvement is found.
      -6. Final Reflection
          -1) The final stage of iterative reasoning—essentially the best refined “internal conclusion.”
      -7. Augmented Generation
          -1) The final reflection is handed to an “answering model” (which could be the same or a different LLM) that produces a concise,
              user-friendly output.
   -b. Key Features Driving RAT
       -1) Iterative Reasoning: Allows deeper analysis by re-checking the same question multiple times, akin to a “thought loop.”
       -2) Dynamic Retrieval: If new information is needed at any point, RAT can query the knowledge base again—focusing on more specific keywords 
                              or updated needs.
       -3) Contextual Precision: By repeatedly integrating new or more relevant information, RAT’s final answers tend to be more accurate and
                                 better aligned with the user’s intention.

4. How RAT Works in Practice
   Below is the step-by-step outline from the text, with a practical example:

   -a. User Input
       -1. The user poses a question, e.g., “How can I improve my productivity?”
   -b. Knowledge Retrieval
       -1. The system fetches relevant documents or chunks from a knowledge base (e.g., PDFs, GitHub repositories, articles).
       -2. For productivity tips, it might gather text about time management, prioritization, and focus techniques.
   -c. Reasoning Loop
       -1. The RAT model examines the user question and the retrieved text, forming an initial reflection 
           (e.g., “Focus on prioritizing tasks effectively”).
       -2. This reflection is then revisited: the system can refine the retrieved documents, add new insights 
           (e.g., “Break tasks into smaller chunks”), or correct any inconsistencies.
       -3. Each iteration adds clarity and depth until the system reaches a final reflection 
           (e.g., “Use the Eisenhower Matrix, break tasks down, and time-block your schedule”).
   -d. Final Reflection
       -1. RAT produces a distilled, well-thought-out internal answer summarizing the model’s iterative reasoning.
   -e. Augmented Generation
       -1. The final reflection is handed off to the “answering” LLM.
       -2. The user receives an organized, natural-language response such as:
           “To boost your productivity, use the Eisenhower Matrix to prioritize tasks based on urgency and importance. 
           Break them into smaller steps and use time-blocking for deep, focused work.”

5. Practical Implementation with RAGLight
   The text references RAGLight, a framework that streamlines building RAG or RAT pipelines. Key points:

   -a. Knowledge Base: You can define multiple data sources, such as FolderSource (local documents) and GitHubSource (online repositories).
   -b. Models: You can specify:
       -1. A generation model (for final answers), e.g., "llama3".
       -2. A reasoning model (for iterative loops), e.g., "deepseek-r1:1.5b".
   -c. Reflection: A numeric parameter controlling how many times the model re-checks or refines before producing a final answer.
"""

from raglight.rat.simple_rat_api import RATPipeline
from raglight.models.data_source_model import FolderSource, GitHubSource
from raglight.config.settings import Settings

Settings.setup_logging()

pipeline = RATPipeline(
    knowledge_base=[
        FolderSource(path="<path to your folder with pdf>/knowledge_base"),
        GitHubSource(url="https://github.com/Bessouat40/RAGLight")
    ],
    model_name="llama3",
    reasoning_model_name="deepseek-r1:1.5b",
    reflection=1
)

pipeline.build()

response = pipeline.generate(
    "How can I create an easy RAGPipeline using raglight framework ? "
    "Give me the the easier python implementation"
)
print(response)

"""
-a. pipeline.build():
    Processes the knowledge sources, creates embeddings, and initializes a vector store for retrieval.
-b. pipeline.generate(query):
    -1. For RAT:
        -1) Generates an embedding for the query,
        -2) Retrieves the most relevant chunks,
        -3) Performs the iterative reflection steps (reflection=1, or more if you want deeper refinement),
        -4) Outputs a final user-facing answer.

6. Why RAT is Useful
   -a. Complex Question Answering: Multi-layered queries benefit from iterative re-checking.
   -b. Research-Intensive Workflows: Where each iteration can surface new or more accurate data.
   -c. Domain-Specific AI Assistants: In specialized fields (legal, medical, technical), iteration helps ensure more precise final answers.

7. Final Takeaways
   -a. RAT is essentially RAG + Iterative Reasoning. The iterative loop aspect is what sets RAT apart, letting the model refine answers step by step.
   -b. Core Algorithm: Retrieve → Initial Reflection → Iterative Refinement → Final Reflection → Augmented Generation.
   -c. How It Works: RAT collects knowledge from a knowledge base, re-checks or adjusts the data as the model “thinks,” then produces a polished, contextually rich answer.
   -d. Implementation: Tools like RAGLight ease the setup by offering modular pipelines for retrieval, reasoning, and final answer generation—making RAT accessible without deeply re-engineering each step.
"""
