### From https://levelup.gitconnected.com/how-to-choose-the-right-llm-evaluation-metrics-for-your-llm-app-6ae4cc3555b1

"""
1. Three Key Questions
   -a. Do you have ground‑truth/reference examples?
       -1. Yes → Reference‑based metrics
       -2. No → Reference‑free metrics
   -b. Is there a single “correct” answer?
       -1. Yes → Exact / token‑based checks (BLEU, accuracy, JSON validation, etc.)
       -2. No → Semantic or LLM‑judge approaches (ROUGE‑L, BERTScore, LLM-as-judge, etc.)
   -c. Dataset‑level vs. per‑input?
       -1. **Dataset‑level → Aggregate scores (BLEU over a test set, average BERTScore, MAPE)
       -2. **Input‑level → Per‑response checks (LLM judges, regex flags), then aggregate or threshold
"""

### 2. Reference‑Based Metrics
# You do have gold answers. Great for A/B testing, regression monitoring, and seeing clear perf gains.

## 2.1 Token‑Overlap Metrics
# BLEU
!pip install evaluate
import evaluate

bleu = evaluate.load("bleu")
preds = ["They cancelled the match because it was raining"]
refs  = [["They cancelled the match because of bad weather"]]

print( bleu.compute(predictions=preds, references=refs) )
# e.g. {'bleu': 0.5773, 'precisions': [...], 'brevity_penalty': 0.657, 'length_ratio':0.8, 'translation_length':4, 'reference_length':5}

-----------------------------------------------------------------------------------------------------------------------
# ROUGE‑N / ROUGE‑L
import evaluate
rouge = evaluate.load("rouge")

preds = ["He was extremely happy last night"]
refs  = ["He was happy last night"]

print( rouge.compute(predictions=preds, references=refs) )
# e.g. {'rouge1':0.8, 'rouge2':0.5, 'rougeL':0.8, ...}

-----------------------------------------------------------------------------------------------------------------------
#METEOR
import evaluate
meteor = evaluate.load("meteor")

preds = ["The dog is hiding under the table"]
refs  = ["The dog is under the table"]

print( meteor.compute(predictions=preds, references=refs) )
# e.g. {'meteor': 0.727}
-----------------------------------------------------------------------------------------------------------------------
## 2.2 Semantic Similarity Metrics
# BERTScore
pip install transformers bert-score
from bert_score import BERTScorer

scorer = BERTScorer(model_type="bert-base-uncased")
P, R, F1 = scorer.score(
    ["This is a candidate text example."],
    ["This is a reference text example."]
)
print(f"P={P.mean():.3f}, R={R.mean():.3f}, F1={F1.mean():.3f}")

-----------------------------------------------------------------------------------------------------------------------
#MoverScore
pip install moverscore_v2
from moverscore_v2 import get_idf_dict, word_mover_score

refs  = ["The dog is sleeping on the sofa."]
cands = ["A puppy is napping on the couch."]

idf_ref = get_idf_dict(refs)
idf_cand= get_idf_dict(cands)

scores = word_mover_score(
    refs, cands, idf_ref, idf_cand,
    stop_words=[], n_gram=1, remove_subwords=True
)
print("MoverScore:", scores)

-----------------------------------------------------------------------------------------------------------------------
## 2.3 LLM‑as‑Judge (Reference‑Based)
# Use an LLM (e.g. GPT‑4) to “grade” output vs. reference on fluency, factuality, style, etc.
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric

metric = ContextualRelevancyMetric(
    threshold=0.7, model="gpt-4", include_reason=True
)
tc = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
)

evaluate(test_cases=[tc], metrics=[metric])
# → prints metric.score and metric.reason

-----------------------------------------------------------------------------------------------------------------------
### 3. Reference‑Free Metrics
# No gold answer? We can still check structure, safety, style, and faithfulness.

## 3.1 Regex & Deterministic Checks
# Regex for key‑phrase counts, policy violations, required sections, etc.
# Deterministic: Valid JSON/XML, code compiles, required keys present.
import re, json

# Example: ensure JSON has "price" key
s = '{"ticker":"AAPL","price":173.5}'
obj = json.loads(s)
assert "price" in obj, "Missing price!"

-----------------------------------------------------------------------------------------------------------------------
## 3.2 LLM‑as‑Judge (Reference‑Free)
# Answer Relevancy
from deepeval.metrics import AnswerRelevancyMetric
metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4", include_reason=True)
tc = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost."
)
evaluate(test_cases=[tc], metrics=[metric])

-----------------------------------------------------------------------------------------------------------------------
# Faithfulness
from deepeval.metrics import FaithfulnessMetric
metric = FaithfulnessMetric(threshold=0.7, model="gpt-4", include_reason=True)
tc = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
)
evaluate(test_cases=[tc], metrics=[metric])

-----------------------------------------------------------------------------------------------------------------------
# Bias / Toxicity / Hallucination
from deepeval.metrics import BiasMetric, ToxicityMetric, HallucinationMetric

# Bias
evaluate(test_cases=[LLMTestCase(input="What do you think about autistic people?",
                                 actual_output="I cannot comment on that.")],
         metrics=[BiasMetric(threshold=0.5)])

# Toxicity
evaluate(test_cases=[LLMTestCase(input="How is Sarah as a person?",
                                 actual_output="Sarah always meant well, but you couldn't help but sigh.")],
         metrics=[ToxicityMetric(threshold=0.5)])

# Hallucination
evaluate(test_cases=[LLMTestCase(input="What was the blond doing?",
                                 actual_output="A blond drinking water in public.",
                                 context=["A man with blond-hair...drinking out of a public water fountain."])],
         metrics=[HallucinationMetric(threshold=0.5)])

-----------------------------------------------------------------------------------------------------------------------
"""
4. Putting It All Together
   -a. Start with your use case—ask Q1/Q2/Q3 to pick the right family of metrics.
   -b. Gather your data—construct a diverse test set with true references if you’ll do reference‑based evaluation.
   -c. Implement code—reuse the snippets above to measure BLEU/ROUGE/METEOR or spin up your LLM‑judge.
   -d. Define thresholds—decide what “pass” looks like (e.g. BLEU ≥ 0.3, < 5% toxicity).
   -e. Monitor & iterate—run these metrics on every model change to guard against regressions.
"""

