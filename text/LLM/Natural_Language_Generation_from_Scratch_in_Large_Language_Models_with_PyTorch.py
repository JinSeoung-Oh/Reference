## From https://medium.com/@pashashaik/natural-language-generation-from-scratch-in-large-language-models-with-pytorch-4d9379635316

# step 1 - Tokenization

# Define the model name to be used from Hugging Face
model_name = "HuggingFaceH4/zephyr-7b-beta"

# Initialize tokenizer for converting text to model-understandable format
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the causal language model with auto device mapping and custom settings
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',  # Automatically assign model to best available device
    trust_remote_code=True,  # Allow use of custom layers from Hugging Face Hub
    torch_dtype=torch.float16,  # Set model parameters to float16 for efficiency
)

# Set the model to evaluation mode
model.eval()

# Define the prompt text to generate language from
prompt = "write a story about a cat"
tokens = tokenizer.tokenize(prompt)
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
output = model.generate(input_ids, max_length=50)
generated_text = tokenizer.decode(output[0])

# step 2 - Forward Pass
output = model(input_ids)
print(output.keys())

output.logits.shape

## Important - step 3. Decoding Strategies
###################################################################
""" 1. Greedy Decoding
Greedy decoding is probably the easiest decoding strategy.At each time step, when we input our sequence into the model,
we receive a probability distribution across the entire vocabulary(30K tokens in our case). 
Here, we consistently select the token with the highest probability at every step.Greedy decoding is probably the easiest decoding strategy.
At each time step, when we input our sequence into the model, we receive a probability distribution across the entire vocabulary(30K tokens in our case).
Here, we consistently select the token with the highest probability at every step.

Why Greedy Decoding is Not best choice ?
-1. Not the Best Choice: Greedy decoding often doesn’t produce the best overall text. It only picks the most likely next word each time, which might not make the best sentence when you look at it as a whole.
-2. Repetitive and Predictable: It tends to create text that’s not very varied or interesting. Since it always goes for the most obvious next word, the results can be quite generic.
-3. Can’t Fix Mistakes: Once greedy decoding makes a mistake, it can’t go back and fix it. This means that one wrong word choice can lead to more errors down the line.
-4. Limited Exploration: Greedy decoding doesn’t consider many different options or paths. It just sticks to a narrow set of choices, which means it might miss out on better or more suitable ways to complete the text.
"""
def greedy_decoding(input_ids, max_tokens=300):
    with torch.inference_mode():
        for _ in range(max_tokens):
            # input_ids shape: [1, current_sequence_length]
            outputs = model(input_ids)

            # outputs.logits shape: [1, current_sequence_length, vocab_size]
            next_token_logits = outputs.logits[:, -1, :]
            # next_token_logits shape: [1, vocab_size]

            next_token = torch.argmax(next_token_logits, dim=-1)
            # next_token shape: [1] (the most probable next token ID)
            
            # stop generation if the model produces the end of sentence </s> token 
            if next_token == tokenizer.eos_token_id:
                break

            # rearrange(next_token, 'c -> 1 c'): changes shape to [1, 1] for concatenation
            input_ids = torch.cat([input_ids, rearrange(next_token, 'c -> 1 c')], dim=-1)
            # input_ids shape after concatenation: [1, current_sequence_length + 1]

        generated_text = tokenizer.decode(input_ids[0])
        # input_ids[0] shape for decoding: [current_sequence_length]

    return generated_text

# Call the function with initial input_ids
print(greedy_decoding(input_ids))

###################################################################
"""2. Beam Search
The fundamental idea of beam search is exploring multiple paths instead of just single one.: Unlike greedy decoding, 
which only keeps the single best path (the most probable next word) at each step, beam search keeps track of the k most probable paths at each step. 
This means at every step in the sequence, it doesn’t just consider the single most likely next word, but the k most likely next words.
"""
from einops import rearrange
import torch.nn.functional as F

def beam_search(input_ids, max_tokens=100, beam_size=2):
    # Initialize the scores for each beam with zeros. Shape: [beam_size]
    beam_scores = torch.zeros(beam_size).to(device)
    
    # Duplicate the initial sequence for each beam. Shape: [beam_size, seq_length]
    beam_sequences = input_ids.clone()
    
    # Create a boolean mask to keep track of active beams. Shape: [beam_size]
    active_beams = torch.ones(beam_size, dtype=torch.bool)
   
    
    for step in range(max_tokens):
        # Generate model outputs for the current sequences. 
        # The model is expected to handle batched input, hence the shape of beam_sequences is [beam_size, current_seq_length].
        outputs = model(beam_sequences)
        
        # Extract the last logits from the output to get the probabilities for the next token. Shape: [beam_size, vocab_size]
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        
        # Calculate the score for each beam and token by flattening the probabilities and selecting the top ones.
        # The flattened shape is [beam_size * vocab_size], from which we select the top beam_size scores.
        top_scores, top_indices = torch.topk(probs.flatten(), k=beam_size, sorted=False)
       
        # Map flat indices back to beam and token indices.
        # beam_indices is the index in the beam, shape: [beam_size]
        # token_indices is the index of the token in the vocabulary, shape: [beam_size]
        beam_indices = top_indices // probs.shape[-1]
        token_indices = top_indices % probs.shape[-1]
        
        # Update the sequences with the new tokens at the end. Shape after update: [beam_size, current_seq_length + 1]
        # This concatenates the best token for each beam to the end of the sequences.
        beam_sequences = torch.cat([
            beam_sequences[beam_indices],
            token_indices.unsqueeze(-1)
        ], dim=-1)

        # Update the beam scores with the top scores. Shape: [beam_size]
        beam_scores = top_scores
        
        # Check for the end-of-sequence tokens and update the active beams.
        # If a beam produces an EOS token, it is marked as inactive.
        active_beams = ~(token_indices == tokenizer.eos_token_id)
        
        # If all beams are inactive, exit the loop.
        if not active_beams.any():
            print("no active beams")
            break
    
    # Select the beam with the highest score as the best sequence. Shape: [best_seq_length]
    best_beam = beam_scores.argmax()
    best_sequence = beam_sequences[best_beam]
    
    # Decode the best sequence to generate the final text.
    generated_text = tokenizer.decode(best_sequence)
    
    return generated_text

input_ids = torch.tensor([[1, 2, 3, 0], [1, 2, 3, 2]])
beam_size = 2
vocab = [0, 1, 2, 3]
probs = torch.tensor([
    [0.2, 0.3, 0.5, 0.1], [0.2, 0.2, 0.2, 0.2]
])

top_scores, top_indices = torch.topk(probs.flatten(), k=beam_size, sorted=False)
beam_indices = top_indices // len(vocab)
token_indices = top_indices % len(vocab)

beam_sequences = torch.cat([
    input_ids[beam_indices],
    token_indices.unsqueeze(-1)
], dim=-1)

input_ids = torch.tensor([[1, 2, 3, 0], [1,2, 3, 2]])
beam_size = 2
vocab = [0, 1 , 2, 3]
probs = torch.tensor([
    [0.2, 0.3, 0.5, 0.1], [0.2, 0.2, 0.2, 0.2]
])
top_scores, top_indices = torch.topk(probs.flatten(), k=beam_size, sorted=False)
print(top_scores, top_indices) # tensor([0.5000, 0.3000]) tensor([2, 1])
beam_indices = top_indices // len(vocab)
token_indices = top_indices % len(vocab)
print(beam_indices,token_indices) # tensor([0, 0]) tensor([2, 1])
beam_sequences = torch.cat([
    input_ids[beam_indices],
    token_indices.unsqueeze(-1)
], dim=-1)
print(beam_sequences) # tensor([[1, 2, 3, 0, 2], [1, 2, 3, 0, 1]])

###################################################################
""" 3. Top-K Sampling
The concept of Top-K sampling involves sampling only from the top K tokens in the probability distribution.
Let’s say we set K to 50; this means that when we predict the next token, we sort the entire 32,000-word vocabulary according to the softmax scores of our logits.
Then, we retain only the top 50 options from the vocabulary and remove the rest of the items. Subsequently, we randomly select a token from the previously filtered list.

-1) Dynamic Probability Distributions: When we use Top-K sampling, the list of possible next words (tokens) and their chances of being picked can change a lot each time we choose a new word. Sometimes, many words have a similar chance of being chosen, and other times, one or a few words are much more likely than the others.
-2) When the List is Flat (Many Words Have Similar Chances): Imagine we have a big basket of fruits, and you’re almost equally likely to pick any fruit because you like them all pretty much the same. If you’re only allowed to choose from the top 5 fruits in the basket, you might miss out on a lot of other fruits you also like just as much. This is like when the probability distribution is flat, and limiting ourselves to the top K options (like the top 5 fruits) means we ignore many other good choices.
-3) When the List is Peaky (One or a Few Words are Much More Likely): Imagine you’re at a buffet with one dish you absolutely love and many others that are just okay. If you’re allowed to choose from a very large selection, including all the okay dishes alongside your favorite, the presence of so many less appealing options could lead you to pick something less satisfying by chance. In a peaky distribution, where one or a few words are much more likely to be the next correct choice, setting a high K allows many less probable (less appealing) options to be considered. This increases the chances of selecting a word that isn’t the best fit, simply because it’s among the many options available, diluting the likelihood of choosing the most appropriate or expected word based on the context.
"""
def top_k_sampling(input_ids, max_tokens=100, top_k=50, temperature=1.0):
   for _ in range(max_tokens):
        # Temporarily disables gradient calculation to improve performance and reduce memory usage
        with torch.inference_mode():
            # Obtain the model's outputs based on the input IDs
            outputs = model(input_ids)
            
            # Extract logits of the next token
            next_token_logits = outputs.logits[:, -1, :]
            
            # Select the top K tokens from the probability distribution
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            
            # Apply softmax to convert logits to probabilities, with optional temperature scaling
            top_k_probs = F.softmax(top_k_logits / temperature, dim=-1)
            
            # Sample from the top K tokens to determine the next token
            next_token_index = torch.multinomial(top_k_probs, num_samples=1)
            
            # Map the sampled token back to its original index in the logits tensor
            next_token = top_k_indices.gather(-1, next_token_index)
            
            # Concatenate the new token to the sequence of input IDs
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
    # Decode the tensor of input IDs to a string of text
    generated_text = tokenizer.decode(input_ids[0])
    
    return generated_text

###################################################################
""" 4. Top-P (Nucleus) Sampling:
Instead of selecting a fixed number of top tokens (like the top 50 in Top-K sampling), Top-P sampling chooses tokens based on their cumulative probability. It looks at the list of all possible next tokens, ranked by their probability, and adds them up from the most likely to the least likely until the total probability reaches a threshold P (like 0.9 or 90%).

-1) Cumulative Probability Mass: Imagine you have a pie chart representing the probability of each token being the next word, with the whole pie being 100%. Top-P sampling slices this pie from the most significant piece (most probable token) onwards, stopping when the combined slice reaches the P threshold. This cumulative slice represents the “mass” where the probabilities are concentrated.
-2) Adapting to the Distribution (Varies K): The key advantage of Top-P sampling is its flexibility. The number of tokens selected (equivalent to K in Top-K sampling) automatically adjusts based on how spread out or concentrated the probabilities are. If a few tokens hold most of the probability mass (a “peaky” distribution), only those will be considered, resulting in a smaller “K”. If the probability is more evenly distributed (“flatter”), more tokens are included to reach the P threshold, resulting in a larger “K”.
-3) Uniformity of Pt: “Pt” refers to the probability distribution at a given time step (t) for selecting the next token. Top-P sampling dynamically adjusts the number of options considered by directly responding to the shape of this distribution. When Pt is uniform (flatter), indicating many plausible next tokens, more tokens are included to reach the cumulative probability threshold P. When Pt is less uniform (peakier), fewer tokens are needed.
"""
def top_p_sampling(input_ids, max_tokens=100, top_p=0.95):
    # Temporarily disables gradient calculation to improve performance and reduce memory usage
    with torch.inference_mode():
        for _ in range(max_tokens):
                # Obtain the model's outputs based on the input IDs
                outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Sort the logits in descending order and apply softmax to get probabilities
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                sorted_probabilities = F.softmax(sorted_logits, dim=-1) # Apply temperature
                
                # Calculate cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probabilities, dim=-1)
                
                # Identify and remove tokens with cumulative probability above the threshold (top_p)
                # Ensuring the first token is always selected by setting the first position to False
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = False  # Keep the most probable token
                
                # Get the actual indices to remove from the original logits tensor
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                
                # Set the logits of removed tokens to -infinity to exclude them from sampling
                next_token_logits.scatter_(-1, indices_to_remove[None, :], float('-inf'))
                
                # Re-calculate probabilities after filtering and sample from this distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append the sampled token to the input IDs for the next iteration
                input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Decode the tensor of input IDs to a string of text
        generated_text = tokenizer.decode(input_ids[0])
    
    return generated_text

