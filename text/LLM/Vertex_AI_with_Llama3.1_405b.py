## From https://medium.com/google-cloud/the-chronicles-of-llama-the-new-llama-405b-on-vertex-ai-876e2bbfc51f

import { vertexAI, llama3 } from '@genkit-ai/vertexai';

configureGenkit({
  plugins: [
    // Configure Vertex AI plugin
    vertexAI({
      location: 'us-central1',
      modelGarden: {
        models: [llama3],
      },
      evaluation: {
        metrics: [
          VertexAIEvaluationMetricType.SAFETY,
          VertexAIEvaluationMetricType.FLUENCY,
        ],
      },
    }),
  ],
  logLevel: 'debug',
  enableTracingAndMetrics: true,
});

// Generate content from Llama3
const llmResponse = await generate({
  model: llama3,
  prompt: 'What is Vertex AI?',
});


### Llama 3.1 405B on MaaS
# Import libraries
import openai
from google.auth import default, transport

# Set some parameters
temperature = 1.0  
max_tokens = 500
top_p = 1.0  

# Get credentials
credentials, _ = default()
auth_request = transport.requests.Request()
credentials.refresh(auth_request)

# Initialize the OpenAI client
client = openai.OpenAI(
    base_url = f'https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi/chat/completions?',
    api_key = credentials.token)

# Submit a model request
response = client.chat.completions.create(
  model='meta/llama3-405b-instruct-maas',
  messages=[
      {"role": "user", "content": "What is Vertex AI?"},
      {"role": "assistant", "content": "Sure, Vertex AI is:"}
  ],
  temperature=temperature,
  max_tokens=max_tokens,
  top_p=top_p,
)

# Get the response
print(response.choices[0].message.content)


######
question = "What about llama spitting?"

context = " ".join([context.text for context in rag.retrieval_query(
    rag_resources=[
        rag.RagResource(
            rag_corpus=rag_corpus.name,
        )
    ],
    text=question,
    similarity_top_k=1,
    vector_distance_threshold=0.5,
).contexts.contexts])

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[{'role': 'system', 'content': '''You are an AI assistant. Your goal is to answer questions using the pieces of context. If you don't know the answer, say that you don't know.'''},
              {'role': 'user', 'content': question},
              {'role': 'assistant', 'content': context}])

print(response.choices[0].message.content)
