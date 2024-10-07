## From https://medium.com/@devmallyakarar/training-language-models-to-self-correction-via-reinforcement-learning-a-deep-dive-into-score-with-ff85421b4186
## This article provide very interest thing about self-correction.

"""
1. Self-Correction in LLMs
   Large language models (LLMs) like GPT often struggle with self-correction, which is important for tasks requiring iterative refinement,
   such as complex reasoning and programming. 
   Traditional supervised fine-tuning (SFT) methods don't equip models well for real-time corrections because they train on fixed datasets. 
   These approaches lead to a mismatch between the training environment and real-world application scenarios, where mistakes can be diverse and unpredictable.

   The main problem with SFT is that it overfits LLMs to a specific dataset and doesn’t train them to identify and correct their unique mistakes 
   during inference (real-world usage).
   As a result, models might appear to be making superficial corrections when they encounter errors, but they rarely fix them meaningfully.

2. Challenges of Supervised Fine-Tuning (SFT):
   -1. Distribution Mismatch
       SFT trains models on predefined datasets, causing a gap between training data and the model's real-world outputs.
       The errors a model generates at test time often differ from what it was trained to fix, causing poor performance in self-correction.
   -2. Minimal Edits
       SFT trains models to make minimal, "safe" corrections that may look correct but lack depth, preventing meaningful self-correction.

3. SCoRe: Self-Correction via Reinforcement Learning
   SCoRe introduces a new method to train LLMs using reinforcement learning (RL) for multi-turn corrections, rather than relying on pre-generated corrections.
   The model learns to improve its own responses iteratively based on real-time feedback it generates, eliminating the distribution mismatch. 
   This way, the model interacts with its mistakes, refining its outputs with multiple attempts.

   -1. Two-Phase Training Process:
       a. Phase 1 – Policy Initialization via RL: This stage sets up the model's initial correction behavior using RL. 
                    The model learns how to make meaningful corrections without reverting to ineffective or trivial edits.
       b. Phase 2 – Reward Bonus for Self-Correction: SCoRe encourages substantial changes and rewards corrections that truly fix errors, 
                    rather than small tweaks that don’t solve the underlying problems.

   -2. Key Results and Benchmarks:
      SCoRe's approach demonstrated significant improvements. In tests with advanced LLMs like Gemini 1.0 Pro and Gemini 1.5 Flash, 
      the self-correction performance increased by:
      - 15.6% on the MATH benchmark (for complex mathematical reasoning).
      - 9.1% on the HumanEval benchmark (for programming and code generation).
      These gains show that SCoRe effectively enhances the ability of models to detect, analyze, and correct their errors in real-time, 
      improving the overall robustness and accuracy of LLMs in complex tasks.

4. Why LLMs Struggle with Self-Correction:
   LLMs were originally designed for single-turn tasks, optimized to give a correct answer on the first pass. 
   This architecture makes them less capable of handling iterative tasks where responses need ongoing refinement, 
   such as math proofs or multi-step problem-solving in programming.

   Additionally, LLMs lack intrinsic error detection. In real-world scenarios, a model might not even recognize that it made a mistake,
   leading to situations where incorrect responses are repeated or not meaningfully corrected.

5. Why SCoRe Is Different:
   SCoRe teaches LLMs to self-correct using their own mistakes by generating real-time feedback through RL.
   It uses reward shaping to penalize trivial corrections and encourages deep, substantive revisions that truly fix the problem.
   This differs from prompt engineering (using specific prompts to coax a better response) or using separate models to verify and refine responses.

   Unlike traditional approaches, SCoRe focuses on enhancing the model's internal capability to recognize and fix its own mistakes,
   making it a promising direction for future LLM development.

6. Limitations of Current Models:
   Despite these advances, self-correction remains a challenging problem. Even with SCoRe, LLMs are still prone to vanishing gradients 
   for long-term dependencies and might struggle with tasks that require retaining information over long sequences. 
   They also face challenges in complex multi-step tasks where significant reasoning is required at each step.

####
1. SCoRe Training Overview:
   SCoRe enhances the self-correction capabilities of large language models (LLMs) by using self-generated correction traces during training, 
   avoiding the distribution mismatch problem typical of supervised fine-tuning (SFT). 
   It focuses on training the model with errors it is likely to make in real-world scenarios.

2. Phase 1: Policy Initialization via Reinforcement Learning (RL):
   The first phase of SCoRe involves initializing the model’s policy through RL, where the model is trained to focus on improving its second-attempt responses. 
   Traditional models often make shallow corrections or avoid fixing errors. 
   SCoRe's policy initialization counteracts this by guiding the model to make meaningful revisions rather than trivial adjustments.

   In this stage, regularization techniques prevent the model from over-optimizing its first attempt at the expense of the second. 
   This ensures the model doesn't fall into behaviors where it avoids corrections altogether. 
   Instead, the model is encouraged to balance the quality of the first and second attempts, 
   improving its capacity for substantial corrections while maintaining the quality of its initial responses.

3. Phase 2(Joint Optimization with Reward Shaping): Multi-Turn RL with Reward Shaping:
   In the second phase, multi-turn RL is used to maximize rewards for meaningful corrections made between the first and second attempts. 
   The model is trained to correct itself more effectively over multiple turns. 
   The reward function is crucial here, as it emphasizes improvements between attempts by penalizing unnecessary or trivial changes.

   Reward Shaping ensures that corrections are meaningful, rewarding the model for turning incorrect responses into correct ones while discouraging minor
   or superficial edits. This structured reward encourages deeper self-refinement, improving the model’s generalization to new tasks and environments.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import Adam
from torch.distributions import Categorical
from nltk.translate.bleu_score import sentence_bleu

class SelfCorrectionModel(nn.Module):
    def __init__(self, base_model_name="gpt2"):
        super(SelfCorrectionModel, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(base_model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask, return_dict=True)

    def generate_response(self, prompt, max_length=50):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        output = self.model.generate(input_ids, max_length=max_length, num_return_sequences=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

def reward_function(y1, y2, target):
    """A simple reward function based on BLEU score."""
    bleu_y1 = sentence_bleu([target.split()], y1.split())
    bleu_y2 = sentence_bleu([target.split()], y2.split())
    
    if y2 == target:
        return 1.0
    elif bleu_y2 > bleu_y1:
        return 0.5
    else:
        return -0.5

def policy_gradient_step(log_probs, rewards):
    """Applies a policy gradient update."""
    policy_loss = []
    for log_prob, reward in zip(log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    return torch.stack(policy_loss).sum()

def train(model, optimizer, prompt, target, max_iters=1000, max_response_len=50):
    model.train()
    for epoch in range(max_iters):
        # Generate first response (y1)
        y1 = model.generate_response(prompt, max_length=max_response_len)
        
        # Self-correction: Generate a second response based on the first
        y2 = model.generate_response(y1, max_length=max_response_len)
        
        # Calculate reward for the correction
        reward = reward_function(y1, y2, target)

        # Update policy using policy gradient
        input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = model(input_ids)
        log_probs = Categorical(logits=outputs.logits).log_prob(input_ids.squeeze())
        loss = policy_gradient_step(log_probs, [reward])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Reward {reward}, First Response: {y1}, Second Response: {y2}")

def evaluate(model, test_data, max_response_len=50):
    model.eval()
    total_reward = 0
    total_bleu_score = 0
    count = 0
    
    with torch.no_grad():
        for prompt, target in test_data:
            y1 = model.generate_response(prompt, max_length=max_response_len)
            y2 = model.generate_response(y1, max_length=max_response_len)
            
            reward = reward_function(y1, y2, target)
            bleu_score = sentence_bleu([target.split()], y2.split())

            total_reward += reward
            total_bleu_score += bleu_score
            count += 1
            
            print(f"Prompt: {prompt}")
            print(f"First Response: {y1}")
            print(f"Corrected Response: {y2}")
            print(f"Target: {target}")
            print(f"Reward: {reward}, BLEU Score: {bleu_score}\n")
    
    avg_reward = total_reward / count
    avg_bleu = total_bleu_score / count
    
    print(f"Average Reward: {avg_reward}")
    print(f"Average BLEU Score: {avg_bleu}")

# Example Usage
if __name__ == "__main__":
    model = SelfCorrectionModel()
    optimizer = Adam(model.parameters(), lr=1e-5)
    
    prompt = "What is the capital of France?"
    target = "The capital of France is Paris."
    
    # Train the model
    train(model, optimizer, prompt, target)
    
    # Test data for evaluation
    test_data = [
        ("What is the capital of Germany?", "The capital of Germany is Berlin."),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("What is the capital of India?", "The capital of India is New Delhi."),
    ]
    
    # Evaluate the model after training
    evaluate(model, test_data)
