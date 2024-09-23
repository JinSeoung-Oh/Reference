## From https://blog.knowledgator.com/llm2encoders-e7d90b9f5966

"""
The evolution of language models has followed two primary paths: generative models (e.g., GPT, Llama, and T5) and discriminative models (e.g., BERT and DeBERTa).
Generative models have dominated recent advancements in large language models (LLMs) with their ability to generate human-like text and handle a wide range of tasks. 
Discriminative models, particularly those using encoder-based architectures, excel at tasks requiring structured outputs and deep understanding, 
like named entity recognition (NER) and text classification. 
However, discriminative models have seen slower progress until works like GLiNER and UTC reignited interest.

1. GLiNER and its Limitations
   GLiNER showed that small encoder models could achieve strong zero-shot performance in NER and tasks like relation extraction, classification, 
   and open information extraction. Despite its success, GLiNER relied on older encoder architectures that
   lack modern features such as flash-attention and are constrained by smaller window sizes and limited generalization due to small training corpora.

2. Bridging Generative and Discriminative Models
   Recognizing the need to combine the strengths of both generative and discriminative models, 
   researchers adapted LLM2Vec to convert decoder architectures into bi-directional encoders. 
   This method combines the power of modern LLMs with the benefits of encoder-based models for tasks like NER and relation extraction.

3. LLM2Vec and GLiNER Enhancements
   For this project, models like Sheared Llama, Tiny Llama, and Qwen (0.5B and 1.5B versions) were trained on Masked Next Token Prediction using Wikipedia corpora. 
   By converting decoders to encoders, the pooling strategy for entity representations had to be adjusted, 
   shifting from special token aggregation (<<ENT>>) to first token pooling. 
   Fine-tuned Qwen 1.5B-based GLiNER achieved the best results on 24 datasets for zero-shot NER, 
   while the Tiny-Llama version excelled in zero-shot performance on biological datasets.

4. Comparison of GLiNER Models
   Comparing older and newer GLiNER models is difficult due to differences in datasets and training parameters, 
   but the newer bi-encoder GLiNER models (e.g., DeBERTa-based and Llama-1.3B) show clear improvements in performance. 
   These bi-encoder models are more efficient and interpretable, especially when scaled, and work well with converted decoder models,
   even without strong bi-directional attention, indicating promising scalability and efficiency for future research.
"""
!pip install gliner -U
!pip install llm2vec
!pip install transformers -U 

from gliner import GLiNER

model = GLiNER.from_pretrained("knowledgator/gliner-llama-1B-v1.0")

text = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
"""

labels = ["person", "award", "date", "competitions", "teams"]

entities = model.predict_entities(text, labels, threshold=0.5)

for entity in entities:
    print(entity["text"], "=>", entity["label"])

###################################################################
# run a large text corpus
from gliner import GLiNER

model = GLiNER.from_pretrained("knowledgator/gliner-llama-1B-v1.0")

texts = ["Your texts"]

labels = ["person", "award", "date", "competitions", "teams"]

entities = model.run(texts, labels, threshold=0.5)

for entity in entities[0:
    print(entity["text"], "=>", entity["label"])

#################################################################
# Flash Attention or increase sequence length
from gliner import GLiNER
import torch

model = GLiNER.from_pretrained("knowledgator/gliner-llama-1B-v1.0",
                                _attn_implementation = 'flash_attention_2',
                                                max_len = 2048).to('cuda:0', dtype=torch.float16)

#################################################################
# torch SDPA implementation of attention
model = GLiNER.from_pretrained("knowledgator/gliner-llama-1B-v1.0",
_attn_implementation = 'sdpa').to('cuda:0', dtype=torch.float16)

## GLiClass: efficient zero-shot text classifiers
!pip install gliclass
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

model = GLiClassModel.from_pretrained("knowledgator/gliclass-qwen-0.5B-v1.0")
tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-qwen-0.5B-v1.0")

pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type='multi-label', device='cuda:0')

text = "One day I will see the world!"
labels = ["travel", "dreams", "sport", "science", "politics"]
results = pipeline(text, labels, threshold=0.5)[0] #because we have one text

for result in results:
 print(result["label"], "=>", result["score"])









