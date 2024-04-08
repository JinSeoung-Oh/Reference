# from https://medium.com/intel-tech/four-data-cleaning-techniques-to-improve-large-language-model-llm-performance-77bee9003625

####### 1. Data Cleaning and Noise Reduction
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopword
from nltk.stem import WordNetLemmatizer

# Sample text with emojis, hashtags, and other characters
text = “I love coding! 😊 #PythonProgramming is fun! 🐍✨ Let’s clean some text 🧹”

# Tokenization
tokens = word_tokenize(text)

# Remove Noise
cleaned_tokens = [re.sub(r’[^\w\s]’, ‘’, token) for token in tokens]

# Normalization (convert to lowercase)
cleaned_tokens = [token.lower() for token in cleaned_tokens]

# Remove Stopwords
stop_words = set(stopwords.words(‘english’))
cleaned_tokens = [token for token in cleaned_tokens if token not in stop_words]

# Lemmatization
lemmatizer = WordNetLemmatizer()
cleaned_tokens = [lemmatizer.lemmatize(token) for token in cleaned_tokens]

print(cleaned_tokens)

####### 2. Text Standardization and Normalization
import re

# Sample text with spelling errors
text_with_errors = “””But ’s not  oherence  about more language  oherence . 
Other important aspect is ensuring accurte retrievel by  oherence  product name spellings. 
Additionally, refning descriptions  oherenc the  oherence of the contnt.”””

# Function to correct spelling errors
def correct_spelling_errors(text):
    # Define dictionary of common spelling mistakes and their corrections
    spelling_corrections = {
        “ oherence ”: “everything”,
        “ oherence ”: “refinement”,
        “accurte”: “accurate”,
        “retrievel”: “retrieval”,
        “ oherence ”: “correcting”,
        “refning”: “refining”,
        “ oherenc”: “enhances”,
        “ oherence”: “coherence”,
        “contnt”: “content”,
    }

    # Iterate over each key-value pair in the dictionary and replace the
    # misspelled words with their correct versions
    for mistake, correction in spelling_corrections.items():
        text = re.sub(mistake, correction, text)

    return text

# Correct spelling errors in the sample text
cleaned_text = correct_spelling_errors(text_with_errors)

print(cleaned_text)

###### 3. Metadata Handling
Import spacy
import json

# Load English language model
nlp = spacy.load(“en_core_web_sm”)

# Sample text with meta data candidates
text = “””In a blog post titled ‘The Top 10 Tech Trends of 2024,’ 
John Doe discusses the rise of artificial intelligence and machine learning 
in various industries. The article mentions companies like Google and Microsoft 
as pioneers in AI research. Additionally, it highlights emerging technologies 
such as natural language processing and computer vision.”””

# Process the text with spaCy
doc = nlp(text)

# Extract named entities and their labels
meta_data = [{“text”: ent.text, “label”: ent.label_} for ent in doc.ents]

# Convert meta data to JSON format
meta_data_json = json.dumps(meta_data)

print(meta_data_json)

# output
“””
[
    {“text”: “2024”, “label”: “DATE”},
    {“text”: “John Doe”, “label”: “PERSON”},
    {“text”: “Google”, “label”: “ORG”},
    {“text”: “Microsoft”, “label”: “ORG”},
    {“text”: “AI”, “label”: “ORG”},
    {“text”: “natural language processing”, “label”: “ORG”},
    {“text”: “computer vision”, “label”: “ORG”}
]
“””

##### 4. Contextual Information Handling
from googletrans import Translator

# Original text
text = “Hello, how are you?”

# Translate text
translator = Translator()
translated_text = translator.translate(text, src=’en’, dest=’es’).text

print(“Original Text:”, text)
print(“Translated Text:”, translated_text)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Sample documents
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing involves analyzing and understanding human languages.",
    "Deep learning algorithms mimic the structure and function of the human brain.",
    "Sentiment analysis aims to determine the emotional tone of a text."
]

# Convert text into numerical feature vectors
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Apply Latent Dirichlet Allocation (LDA) for topic modeling
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

# Display topics
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([vectorizer.get_feature_names()[i] for i in topic.argsort()[:-5 - 1:-1]]))




####### example
synthetic_text = """
Sarah (S): Technology Enthusiast
Mark (M): AI Expert
S: Hey Mark! How's it going? Heard about the latest advancements in Generative AI (GA)?
M: Hey Sarah! Yes, I've been diving deep into the realm of GA lately. It's fascinating how it's shaping the future of technology!
S: Absolutely! I mean, GA has been making waves across various industries. What do you think is driving its significance?
M: Well, GA, especially Retrieval Augmented Generative (RAG), is revolutionizing content generation. It's not just about regurgitating information anymore; it's about creating contextually relevant and engaging content.
S: Right! And with Machine Learning (ML) becoming more sophisticated, the possibilities seem endless.
M: Exactly! With advancements in ML algorithms like GPT (Generative Pre-trained Transformer), we're seeing unprecedented levels of creativity in AI-generated content.
S: But what about concerns regarding bias and ethics in GA?
M: Ah, the age-old question! While it's true that GA can inadvertently perpetuate biases present in the training data, there are techniques like Adversarial Training (AT) that aim to mitigate such issues.
S: Interesting! So, where do you see GA headed in the next few years?
M: Well, I believe we'll witness a surge in applications leveraging GA for personalized experiences. From virtual assistants to content creation tools, GA will become ubiquitous in our daily lives.
S: That's exciting! Imagine AI-powered virtual companions tailored to our preferences.
M: Indeed! And with advancements in Natural Language Processing (NLP) and computer vision, these virtual companions will be more intuitive and lifelike than ever before.
S: I can't wait to see what the future holds!
M: Agreed! It's an exciting time to be in the field of AI.
S: Absolutely! Thanks for sharing your insights, Mark.
M: Anytime, Sarah. Let's keep pushing the boundaries of Generative AI together!
S: Definitely! Catch you later, Mark!
M: Take care, Sarah!
"""

# Sample text with emojis, hashtags, and unicode characters

# Tokenization
tokens = word_tokenize(synthetic_text)

# Remove Noise
cleaned_tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]

# Normalization (convert to lowercase)
cleaned_tokens = [token.lower() for token in cleaned_tokens]

# Remove Stopwords
stop_words = set(stopwords.words('english'))
cleaned_tokens = [token for token in cleaned_tokens if token not in stop_words]

# Lemmatization
lemmatizer = WordNetLemmatizer()
cleaned_tokens = [lemmatizer.lemmatize(token) for token in cleaned_tokens]


MESSAGE_SYSTEM_CONTENT = "You are a customer service agent that helps 
a customer with answering questions. Please answer the question based on the
provided context below. 
Make sure not to make any changes to the context if possible,
when prepare answers so as to provide accurate responses. If the answer 
cannot be found in context, just politely say that you do not know, 
do not try to make up an answer."

def response_test(question:str, context:str, model:str = "gpt-4"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": MESSAGE_SYSTEM_CONTENT,
            },
            {"role": "user", "content": question},
            {"role": "assistant", "content": context},
        ],
    )
    
    return response.choices[0].message.content

response = response_test(question1, new_content_string)
print(response)
