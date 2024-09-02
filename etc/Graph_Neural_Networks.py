## From https://medium.com/@jrosseruk/demystifying-gcns-a-step-by-step-guide-to-building-a-graph-convolutional-network-layer-in-pytorch-09bf2e788a51
## From https://towardsdatascience.com/structure-and-relationships-graph-neural-networks-and-a-pytorch-implementation-c9d83b71c041
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

# Mathematical Description of GNNs
  A graph 𝐺 is defined as 𝐺 = (𝑉,𝐸), where 𝑉 is the set of nodes and 𝐸 represents the edges between them. This graph is often represented by an adjacency matrix 
  𝐴, where 𝐴_𝑖𝑗 is 1 if there's an edge between nodes 𝑖 and 𝑗, and 0 otherwise. If the graph has 𝑛 nodes, 𝐴 has dimensions (𝑛×𝑛). Nodes have features, and if each node has 
  𝑓 features, the feature matrix 𝑋 has dimensions (𝑛×𝑓)

  Single Node Calculations: GNNs learn the interdependence between nodes by considering the features of their neighbors. For a node 𝑗 with 𝑁_𝑗
  neighbors, GNNs transform the neighbors' features, aggregate them, and update node 𝑗's feature space. 
  Transformation can be done through methods like MLP or linear transformation. Aggregation can be done by summation, averaging, or pooling. The updated feature of node 
  𝑗 is then computed, possibly using a learnable weight matrix and a non-linear activation function.

  Graph Level Calculation: For a graph with 𝑛 nodes, the features can be concatenated into a single matrix 
  𝑋. Neighbor feature transformation and aggregation can then be written using the adjacency matrix 𝐴 and identity matrix 𝐼. Normalization is often done using the degree matrix 
  𝐷. The Graph Convolution Network (GCN) approach normalizes features and enables learning of node relationships but assumes equal importance for all neighbors.

  Graph Attention Networks (GATs): To address the limitation of GCNs, GATs use attention mechanisms to compute the importance of a neighbor's features to the target node. 
  This allows for different contributions from different neighbors. The attention coefficients are calculated and normalized using a SoftMax function,
  and feature aggregation is performed accordingly.

  Multilayer GNN Models: Multiple layers can be stacked to increase the model's complexity, capturing more global features and complex relationships.
  However, this increases the risk of overfitting, necessitating regularization techniques. Finally, the network outputs a feature matrix 𝐻
  that can be used for tasks like node or graph classification. This concludes the mathematical description of GCNs and GATs.
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


###########################################
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from torch_geometric.data import DataLoader

# load and scale the dataset
df = pd.read_csv('SensorDataSynthetic.csv').dropna()
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

nodes_order = [
    'Sensor1', 'Sensor2', 'Sensor3', 'Sensor4', 
    'Sensor5', 'Sensor6', 'Sensor7', 'Sensor8'
]

# define the graph connectivity for the data
edges = torch.tensor([
    [0, 1, 2, 2, 3, 3, 6, 2],  # source nodes
    [1, 2, 3, 4, 5, 6, 2, 7]   # target nodes
], dtype=torch.long)

graphs = []

# iterate through each row of data to create a graph for each observation
# some nodes will not have any data, not the case here but created a mask to allow us to deal with any nodes that do not have data available
for _, row in df_scaled.iterrows():
    node_features = []
    node_data_mask = []
    for node in nodes_order:
        if node in df_scaled.columns:
            node_features.append([row[node]])
            node_data_mask.append(1) # mask value of to indicate present of data
        else:
            # missing nodes feature if necessary
            node_features.append(2)
            node_data_mask.append(0) # data not present
    
    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    node_data_mask_tensor = torch.tensor(node_data_mask, dtype=torch.float)

    
    # Create a Data object for this row/graph
    graph_data = Data(x=node_features_tensor, edge_index=edges.t().contiguous(), mask = node_data_mask_tensor)
    graphs.append(graph_data)


# splitting the data into train, test observation
# Split indices
observation_indices = df_scaled.index.tolist()
train_indices, test_indices = train_test_split(observation_indices, test_size=0.05, random_state=42)

# Create training and testing graphs
train_graphs = [graphs[i] for i in train_indices]
test_graphs = [graphs[i] for i in test_indices]

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph() 
for src, dst in edges.t().numpy():
    G.add_edge(nodes_order[src], nodes_order[dst])

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_weight='bold')
plt.title('Graph Visualization')
plt.show()

from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn

class GNNModel(nn.Module):
    def __init__(self, num_node_features):
        super(GNNModel, self).__init__()
        self.conv1 = GATConv(num_node_features, 16)
        self.conv2 = GATConv(16, 8)
        self.fc = nn.Linear(8, 1)  # Outputting a single value per node

    def forward(self, data, target_node_idx=None):
        x, edge_index = data.x, data.edge_index
        edge_index = edge_index.T
        x = x.clone()

        # Mask the target node's feature with a value of zero! 
        # Aim is to predict this value from the features of the neighbours
        if target_node_idx is not None:
            x[target_node_idx] = torch.zeros_like(x[target_node_idx])

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.05, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=0.05, training=self.training)
        x = self.fc(x)

        return x

model = GNNModel(num_node_features=1) 
batch_size = 8
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-6)
criterion = torch.nn.MSELoss()
num_epochs = 200  
train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True) 
model.train()

for epoch in range(num_epochs):
    accumulated_loss = 0 
    optimizer.zero_grad()
    loss = 0  
    for batch_idx, data in enumerate(train_loader):
        mask = data.mask  
        for i in range(1,data.num_nodes):
            if mask[i] == 1:  # Only train on nodes with data
                output = model(data, i)  # get predictions with the target node masked
                                         # check the feed forward part of the model
                target = data.x[i] 
                prediction = output[i].view(1) 
                loss += criterion(prediction, target)
        #Update parameters at the end of each set of batches
        if (batch_idx+1) % batch_size == 0 or (batch_idx +1 ) == len(train_loader):
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
            accumulated_loss += loss.item()
            loss = 0

    average_loss = accumulated_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Average Loss: {average_loss}')

# Testing
test_loader = DataLoader(test_graphs, batch_size=1, shuffle=True)
model.eval()

actual = []
pred = []

for data in test_loader:
    mask = data.mask
    for i in range(1,data.num_nodes):
        output = model(data, i)
        prediction = output[i].view(1)
        target = data.x[i]

        actual.append(target)
        pred.append(prediction)


import plotly.graph_objects as go
from plotly.offline import iplot

actual_values_float = [value.item() for value in actual]
pred_values_float = [value.item() for value in pred]


scatter_trace = go.Scatter(
    x=actual_values_float,
    y=pred_values_float,
    mode='markers',
    marker=dict(
        size=10,
        opacity=0.5,  
        color='rgba(255,255,255,0)',  
        line=dict(
            width=2,
            color='rgba(152, 0, 0, .8)', 
        )
    ),
    name='Actual vs Predicted'
)

line_trace = go.Scatter(
    x=[min(actual_values_float), max(actual_values_float)],
    y=[min(actual_values_float), max(actual_values_float)],
    mode='lines',
    marker=dict(color='blue'),
    name='Perfect Prediction'
)

data = [scatter_trace, line_trace]

layout = dict(
    title='Actual vs Predicted Values',
    xaxis=dict(title='Actual Values'),
    yaxis=dict(title='Predicted Values'),
    autosize=False,
    width=800,
    height=600
)

fig = dict(data=data, layout=layout)

iplot(fig)

