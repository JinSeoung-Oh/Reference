### From https://medium.com/@hirok4/understanding-transformer-sinusoidal-position-embedding-7cbaaf3b9f6a

"""
1. Importance of Timestep Information in Diffusion Models
   -a. Diffusion Process:
       In diffusion models, noise is added progressively in the forward process and removed in the reverse process. 
       Timestep information is crucial because it guides how noise is introduced and reduced over time. 
       Embedding this time information into the network is essential for maintaining model accuracy and generalization across different timesteps.

2. Challenges with Naive Timestep Embedding
   -a. Simple Concatenation Method:
       A straightforward approach is to append the timestep to input features (e.g., [inputFeature, timestepT]). 
       However, this method has major drawbacks:

       -1) Generalization Issues:
           If the model is trained with a certain timestep (e.g., 50) and then used with a different timestep (e.g., 10000) during inference, 
           accuracy may drop due to lack of consistent understanding of varying timesteps.
       -2) Normalization of Timestep Range:
           Normalizing the timestep to a 0∼1 range might seem like a solution, but if the total number of timesteps changes, 
           the relative meaning of a normalized value changes, losing consistency across different training and inference regimes.
   -b. Ideal Requirements for Timestep Embedding:
       -1) Preserve relationships between different timesteps.
       -2) Maintain consistent meaning for each timestep, even if the total number changes.
       -3) Provide unique representations for each timestep.

3. Transformer Sinusoidal Position Embedding (PE) for Timestep Encoding
   -a. Solution Proposed:
       The diffusion model paper adopts Transformer sinusoidal position embedding from the Attention paper to address the challenges. 
       This method:

       -1) Preserves continuity information between successive timesteps using sine and cosine functions.
       -2) Ensures that position (timestep) information remains consistent even if the total number of timesteps changes during inference.
       -3) Shares parameters across time, embedding the timestep into the network in a consistent and unique way.
   -b. Positional Encoding Equation:
       PE(𝑝𝑜𝑠,2𝑖)=sin(𝑝𝑜𝑠/(10000^(2𝑖/𝑑))), PE(𝑝𝑜𝑠,2𝑖+1)=cos(𝑝𝑜𝑠/(10000^(2𝑖/𝑑)))
       
       - pos is the timestep position.
       - i is the dimension index.
       - d is the total embedding dimension.
       - The denominator uses 10000^(2𝑖/𝑑) to set different frequencies.
       """
       # Python Implementation of Sinusoidal Positional Embedding:

       class SinusoidalPositionalEmbedding(nn.Module):
           def __init__(self, embedding_dim):
               super().__init__()
               self.embedding_dim = embedding_dim

           def forward(self, timesteps):
               positions = np.arange(timesteps)[:, np.newaxis]  # Shape: (timesteps, 1)
               dimensions = np.arange(self.embedding_dim)[np.newaxis, :]  # Shape: (1, embedding_dim)

               # Compute angles using sine for even indices and cosine for odd indices
               angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / self.embedding_dim)
               angle_rads = positions * angle_rates

               pos_encoding = np.zeros_like(angle_rads)
               pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
               pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
               return pos_encoding
"""
   -c. Explanation:
       -1) Positions Array: Creates an array of timestep positions.
       -2) Dimensions Array:Creates an array of dimension indices for embedding.
       -3) Angle Rates & Radians:Calculates the rate at which sine and cosine functions oscillate for each dimension.
       -4) Sine and Cosine Application: Applies sine to even-indexed dimensions and cosine to odd-indexed ones, constructing the positional encoding.

   -d. Advantages of Using Sinusoidal PE:
       -1) Continuity: Maintains continuous variation between timesteps via sine and cosine functions.
       -2) Generalization Across Different Timesteps:Provides consistent representations even if inference timesteps differ from training timesteps.
       -3) Unique Representation:Each timestep has a unique embedding, preserving relationships and meaning.

4. Why the Number 10000 in Sinusoidal Functions?
   -a. Role of 10000:
       In the positional encoding formula, the constant 10000 is used to scale the input to the sine and cosine functions, 
       determining the wavelengths. While the exact reason for choosing 10000 isn’t detailed in the original paper, it serves to:
       -1) Set a long enough wavelength to cover a wide range of positions.
       -2) Provide a balanced spread of sine and cosine values across dimensions.
   -b. Effect of Changing 10000:
       -1) Altering this constant changes the wavelength. For example, using 100 instead of 10000 produces longer wavelengths, 
           affecting the embedding’s frequency components.
       -2) The chosen constant ensures a good trade-off between representing both short-range and long-range dependencies consistently.

5. Summary
   -1) Embedding timestep information is essential in diffusion models for accurate noise scheduling.
   -2) Naive methods of embedding timesteps directly into input features have limitations in generalization and consistency.
   -3) Transformer sinusoidal position embedding effectively incorporates timestep information by using sine and cosine functions, 
       preserving continuity and unique representations for each timestep.
   -4) The constant 10000 in the equation sets the frequency scale, balancing representation across various timesteps.
   -5) This method allows the model to generalize across different inference timesteps and maintain robust performance 
       without retraining for each specific timestep scenario.
