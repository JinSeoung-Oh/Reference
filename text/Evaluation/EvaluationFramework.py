### From https://ai.gopubby.com/llm-evaluation-the-secret-to-better-ai-outputs-c34365b88abd

!pip install langchain langchain-community deepeval google-generativeai rouge-score langchain-google-genai fuzzywuzzy

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
import difflib

# Load models
model_name = "google/flan-t5-base"
generator = pipeline("text2text-generation", model=model_name, device=-1)
semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class EvaluationManager:
    @staticmethod
    def rouge_evaluation(expected, actual, threshold=0.5):
        similarity_ratio = difflib.SequenceMatcher(None, expected.lower(), actual.lower()).ratio()
        return {
            'score': similarity_ratio,
            'passed': similarity_ratio >= threshold,
            'details': {'similarity_ratio': similarity_ratio}
        }

    @staticmethod
    def semantic_similarity_evaluation(expected, actual, threshold=0.5):
        expected_embedding = semantic_model.encode(expected, convert_to_tensor=True)
        actual_embedding = semantic_model.encode(actual, convert_to_tensor=True)
        similarity = util.cos_sim(expected_embedding, actual_embedding).item()
        return {
            'score': similarity,
            'passed': similarity >= threshold,
            'details': {'cosine_similarity': similarity}
        }

    @staticmethod
    def fuzzy_match(expected, actual, threshold=60):
        score = fuzz.ratio(expected.lower(), actual.lower())
        return {
            'score': score / 100,
            'passed': score >= threshold,
            'details': {'fuzzy_match_score': score}
        }

def run_evaluation(test_cases):
    evaluator = EvaluationManager()
    results = []

    for case in test_cases:
        # Prompt the model
        prompt = f"Answer concisely and precisely: {case['input']}"
        generated_response = generator(prompt, max_length=50, truncation=True, num_return_sequences=1)[0]['generated_text']
        case['actual_output'] = generated_response.strip()

        print(f"\nTest Case: {case['input']}")
        print(f"Generated Output: {case['actual_output']}")
        print(f"Expected Output: {case['expected_output']}\n")

        # Rouge Evaluation
        rouge_result = evaluator.rouge_evaluation(case['expected_output'], case['actual_output'])
        print("Rouge Metric:")
        print(f"Score: {rouge_result['score']:.2f}")
        print(f"Passed: {rouge_result['passed']}")
        print(f"Details: {rouge_result['details']}\n")

        # Semantic Similarity Evaluation
        semantic_result = evaluator.semantic_similarity_evaluation(case['expected_output'], case['actual_output'])
        print("Semantic Evaluation:")
        print(f"Score: {semantic_result['score']:.2f}")
        print(f"Passed: {semantic_result['passed']}")
        print(f"Details: {semantic_result['details']}\n")

        # Fuzzy Matching
        fuzzy_result = evaluator.fuzzy_match(case['expected_output'], case['actual_output'])
        print("Fuzzy Matching:")
        print(f"Score: {fuzzy_result['score']:.2f}")
        print(f"Passed: {fuzzy_result['passed']}")
        print(f"Details: {fuzzy_result['details']}\n")

        results.append({
            'input': case['input'],
            'actual_output': case['actual_output'],
            'expected_output': case['expected_output'],
            'rouge_score': rouge_result,
            'semantic_similarity': semantic_result,
            'fuzzy_match': fuzzy_result
        })

    return results


# Test Cases
test_cases = [
    {'input': "Is Python better than R", 'expected_output': "Yes, Python is better programming language"},
    {'input': "What is the capital of India", 'expected_output': "New Delhi"},

]

# Run Evaluation
results = run_evaluation(test_cases)
