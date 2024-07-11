## From https://medium.com/@jrosseruk/demystifying-gcns-a-step-by-step-guide-to-building-a-graph-convolutional-network-layer-in-pytorch-09bf2e788a51
"""
# Introduction to Graph Neural Networks (GNNs)
  Graph Neural Networks (GNNs) are a specialized class of neural networks designed to effectively process 
  and analyze graph-structured data. This type of data is prevalent in numerous fields such as social networks, 
  molecular biology, and transportation systems. Unlike traditional neural networks, which treat data points as independent entities,
  GNNs leverage the relationships and connections between data points (nodes) to enhance predictive accuracy and
  extract richer insights.

# Why Use GNNs? A Party Planning Example
  -1. Traditional Neural Networks
      Imagine you're planning a big party and need to arrange seating for your friends. 
      If you use a traditional neural network, each friend's seating would be considered independently, 
      potentially ignoring their relationships and preferences. This could result in suboptimal seating arrangements,
      separating friends who would enjoy each other's company.

   -2. Graph Neural Networks
       Enter GNNs! They can account for the entire network of friendships, understanding who knows whom and who shares interests. 
       For instance, a GNN would know to seat Donna next to Martha because they both know John Smith and have worked at UNIT. 
       This relational understanding ensures everyone has a great time, illustrating the power of GNNs 
       in leveraging complex relational data.

# Graph Convolutional Networks (GCNs)
  GCNs are a fundamental layer in GNNs, analogous to convolutional layers in Convolutional Neural Networks (CNNs).
  While CNNs aggregate information from surrounding pixels to create a condensed representation,
  GCNs aggregate information from neighboring nodes in a graph.

  -1. Basic Concepts
     - Node Features Matrix (X): This matrix contains the feature vectors of all nodes in the graph.
     - Adjacency Matrix (A): This matrix represents the connections between nodes, with 1s indicating connections and 0s otherwise.

# Graph Convolution Process
  -1. Aggregation of Neighboring Features:
      - The adjacency matrix 𝐴 is multiplied by the node features matrix 𝑋, effectively summing the features of neighboring nodes.
      - To include a node's own features, an identity matrix 𝐼 is added to the adjacency matrix, resulting in 𝐴+𝐼.
  -2. Normalization
      - Nodes may have varying numbers of neighbors. To normalize the aggregated features, we use the degree matrix 𝐷, 
        where the diagonal elements represent the number of neighbors for each node.
     - Row normalization involves dividing by the number of neighbors, but symmetric normalization is often preferred. 
       Symmetric normalization is achieved by 𝐷^−(1/2)(𝐴+𝐼)𝐷^−(1/2), which accounts for the degree of both the current node 
       and its neighbors.

# Incorporating Learnable Parameters
  - A weight matrix 𝑊 is introduced, similar to weights in linear regression, to be learned during training.
  - A non-linearity (e.g., ReLU) is applied to capture complex relationships.

# Final GCN Layer Equation:
  The final equation for a GCN layer is:
  𝐻^(𝑙+1)=𝜎(𝐷^(−1/2)(𝐴^)~𝐷^(−1/2)𝐻^(𝑙)𝑊^(𝑙))
  Where:
  - 𝐴^~=𝐴+𝐼(Adjacency matrix with added self-loops)
  - 𝐷 is the degree matrix of 𝐴^~
  - 𝐻^(𝑙) is the feature matrix at layer 𝑙(initially 𝐻^(0)=𝑋
  - 𝑊^(𝑙) is the weight matrix at layer 𝑙
  - 𝜎 is the activation function (e.g., ReLU)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """
        GCN layer

        Args:
            input_dim (int): Dimension of the input
            output_dim (int): Dimension of the output (a softmax distribution)
            A (torch.Tensor): 2D adjacency matrix
    """

    def __init__(self, input_dim: int, output_dim: int, A: torch.Tensor):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.A = A

        # A_hat = A + I
        self.A_hat = self.A + torch.eye(self.A.size(0))

        # Create diagonal degree matrix D
        self.ones = torch.ones(input_dim, input_dim)
        self.D = torch.matmul(self.A.float(), self.ones.float())

        # Extract the diagonal elements
        self.D = torch.diag(self.D)

        # Create a new tensor with the diagonal elements and zeros elsewhere
        self.D = torch.diag_embed(self.D)
        
        # Create D^{-1/2}
        self.D_neg_sqrt = torch.diag_embed(torch.diag(torch.pow(self.D, -0.5)))
        
        # Initialise the weight matrix as a parameter
        self.W = nn.Parameter(torch.rand(input_dim, output_dim))

    def forward(self, X: torch.Tensor):

        # D^-1/2 * (A_hat * D^-1/2)
        support_1 = torch.matmul(self.D_neg_sqrt, torch.matmul(self.A_hat, self.D_neg_sqrt))
        
        # (D^-1/2 * A_hat * D^-1/2) * (X * W)
        support_2 = torch.matmul(support_1, torch.matmul(X, self.W))
        
        # ReLU(D^-1/2 * A_hat * D^-1/2 * X * W)
        H = F.relu(support_2)

        return H

if __name__ == "__main__":

    # Example Usage
    input_dim = 3  # Assuming the input dimension is 3
    output_dim = 2  # Assuming the output dimension is 2

    # Example adjacency matrix
    A = torch.tensor([[1., 0., 0.],
                      [0., 1., 1.],
                      [0., 1., 1.]])  

    # Create the GCN Layer
    gcn_layer = GCNLayer(input_dim, output_dim, A)

    # Example input feature matrix
    X = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])

    # Forward pass
    output = gcn_layer(X)
    
    print(output)
    # tensor([[ 6.3438,  5.8004],
    #         [13.3558, 13.7459],
    #         [15.5052, 16.0948]], grad_fn=<ReluBackward0>)
