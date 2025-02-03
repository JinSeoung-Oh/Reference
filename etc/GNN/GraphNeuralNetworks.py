### From https://pub.towardsai.net/graph-neural-networks-unlocking-the-power-of-relationships-in-predictions-04dd74daa742

"""
1. Core Concept: Graph Neural Networks (GNNs)
   -a. What Are GNNs?
       -1. GNNs are specialized neural networks designed for graph-structured data. Unlike CNNs (good at images) or RNNs (good at sequences), 
           GNNs thrive where data is naturally represented by nodes (entities) and edges (relationships).
       -2. Examples:
           -1) Social networks: People (nodes) connected by friendships (edges).
           -2) Molecules: Atoms (nodes) linked by bonds (edges).
           -3) Economic indicators: Different factors (nodes) influencing each other (edges).
   -b. Why GNNs?
       -1. They capture both individual features (node attributes) and relational information (edges). 
           This interplay is often ignored or overly simplified by traditional models.
       -2. They can predict, classify, or forecast outcomes at the node, edge, or entire-graph level by leveraging the structure of the data.
   -c. Key Terms:
       -1. Nodes (V): The fundamental units in a graph (e.g., a city or an economic indicator).
       -2. Edges (E): Connections between nodes (e.g., roads between cities, or correlations between indicators).
       -3. Node Features (h_v): Attribute vectors describing each node (e.g., population size, interest rate).
       -4. Edge Features (w_{uv}): Weights or additional data about the connection (e.g., distance, correlation strength).
       -5. Adjacency Matrix (A): A matrix specifying which nodes are directly connected.

2. Core Algorithms
   2.1 Convolutional Graph Neural Networks (GCNs)
       -a. Concept
           -1. GCNs extend the idea of convolution from images (where data is on a 2D grid) to graphs (where data is irregular and “node-centric”).
           -2. The main operation: each node updates its representation by aggregating features from itself and its neighbors.
       -b. Layer-Wise Propagation Rule
           𝐻^(𝑙+1)=𝜎(𝐷^(~ −1/2)𝐴^~𝐷^(~ −1/2)𝐻^(𝑙)𝑊^(𝑙))
           -1. A=A+I : The adjacency matrix plus self-loops, so a node also sees its own features.
           -2. 𝐷^~ : Diagonal degree matrix of 𝐴^~
           -3. H^(l) : Node features at layer 𝑙
           -4. 𝑊^(𝑙) : Trainable weights at layer 𝑙
           -5. 𝜎 : Non-linear activation (e.g., ReLU).
       -c. Training a GCN
           -1. Input Layer: The initial node feature matrix (𝐻^(0)=𝑋)
           -2. Hidden Layers: Repeatedly apply the propagation rule to capture neighborhood information.
           -3. Output Layer: Final node embeddings can be used for node-level predictions or aggregated for graph-level tasks.
           -4. Loss Function: Typically MSE (for regression) or cross-entropy (for classification).
           -5. Backpropagation: GCN parameters (𝑊^(𝑙)) are updated by gradient descent optimizers like Adam or SGD.
       -d. Unique Advantages
           -1. Localized Filters: Aggregation stays close to each node’s neighbors.
           -2. Weight Sharing: The same convolution weights apply to all nodes, akin to convolutional kernels in image CNNs.
           -3. Spectral Foundations: In some treatments, GCNs are interpreted via the graph Laplacian’s eigen-decomposition.
       -e. Application Example: Economic Forecasting
           -1. Nodes = Economic indicators (interest rate, inflation, employment, etc.).
           -2. Edges = Interdependencies (how one indicator affects another).
           -3. Result: The GCN can predict future values of each indicator by learning from current states and relational structure
                       (e.g., “If interest rates rise, how does it affect inflation?”).

   2.2 Graph Attention Networks (GATs) for Feature Selection 
       -a. Concept
           -1. GATs build on GCNs by introducing an attention mechanism. Instead of all neighbors having equal influence, 
               GATs learn to focus on the most relevant neighbors (or, for feature selection, the most important features).
       -b. Attention Mechanism
           -1. Learns weights (attention scores) that determine how strongly one feature/node should influence another.
           -2. These scores are typically normalized with a softmax function.
       -c. Feature Selection Approach
           -1. Treat each feature as a node in a “feature graph.”
           -2. A learnable adjacency/attention matrix captures how features relate to each other (e.g., correlation).
           -3. After passing through GAT layers, feature importance can be extracted from the final activations. 
               Features with higher activation values are deemed more critical for the task (e.g., classification).
       -d. Benefits
           -1. GAT-based feature selection is interpretable: the attention scores and final activation magnitudes show which features matter most.
           -2. Helps reduce dimensionality by identifying redundant or irrelevant features.
   
   2.3 Temporal GNNs
       -a. Concept
           -1. Temporal GNNs handle time-evolving graphs: node features (and possibly edges) change at each time step.
           -2. Particularly useful for stock index forecasting or any domain with dynamic relationships.
      -b. How It Works
          -1. Each time step 𝑡 has a graph 𝐺_𝑡. Node features are updated according to both the current graph structure and the previous states.
          -2. Approaches vary:
              -1) Recurrent + GNN: Combine GNN layers with RNNs (LSTM/GRU) for temporal dependence.
              -2) Independent Steps + Aggregation: Perform a GNN pass for each time step and combine the representations 
                  (e.g., summation or another aggregator).
      -c. Application Example: Stock Market Prediction
          -1. Nodes = Financial indicators (interest rate, inflation, unemployment, stock index).
          -2. Edges = Inter-indicator relationships.
          -3. Over time, the GNN learns how trends in one indicator propagate to others.
          -4. Predicts the future stock index by considering both recent and long-term patterns in all indicators.

3. How It Works (Step-by-Step)
   -a. Build/Represent the Graph
       -1. Define nodes and edges.
       -2. Prepare an adjacency matrix or an edge list that encodes the relationships.
   -b. Assign Node (and Possibly Edge) Features
       -1. Node features: e.g., interest rate, employment rate.
       -2. Edge features (optional): e.g., correlation strength between indicators.
   -c. Construct the Model
       -1. GCN: Use graph convolution layers that aggregate info from neighbors.
       -2. GAT: Incorporate attention layers to learn dynamic, feature-to-feature (or node-to-node) weights.
       -3. Temporal GNN: Process sequences of graphs over time (with or without recurrent modules).
   -d. Training
       -1. Forward Pass: The GNN aggregates or attends over neighbors at each layer.
       -2. Loss Calculation: Compare predictions to ground truth (MSE for forecasting, cross-entropy for classification, etc.).
       -3. Backpropagation + Optimizer: Compute gradients of the GNN parameters and update them (e.g., using Adam).
   -e. Inference/Prediction
       -1. Feed new data into the trained GNN.
       -2. For tasks like forecasting (economic or stock indices), the model outputs predicted values at future steps.
       -3. For feature selection, examine final activation/attention scores to see which features matter most.
   -f. Evaluation
       -1. Metrics vary by task:
           -1) Regression: MSE, MAE, or MAPE.
           -2) Classification: Accuracy, AUC, F1.
       -2. Qualitative Analysis: Inspect attention scores (in GAT) or final predicted values (in GCN) for real-world interpretability.

4. Conclusion
   -a. Power and Flexibility: GNNs shine wherever data points (nodes) have non-trivial relationships (edges).
   -b. Multiple Flavors:
       -1. Convolutional GNNs (GCNs): Extend standard “convolutions” to graphs, ideal for tasks like economic forecasting where interconnected 
                                      indicators are crucial.
       -2. GATs: Leverage attention to highlight the most significant connections or features, offering interpretability and improved focus on 
                 what matters most.
       -3. Temporal GNNs: Capture time-varying relationships, perfect for time-sensitive tasks like predicting stock indices.

   -c. Broad Applications: Finance, healthcare, traffic optimization, drug discovery, social networks, and more.
   -d. Future Outlook: As graph-based methods mature, they will likely be integrated with other advanced models (like large language models)
                       to create Graph RAG (Retrieval-Augmented Generation) pipelines and beyond, expanding the scope of what AI can accomplish.

In essence, GNNs are a powerful tool for connected, evolving data, revealing patterns that linear or grid-based models might miss. 
Their ability to handle diverse tasks — from forecasting economic indicators to selecting critical features to predicting future trends 
in stock markets — illustrates just how transformative graph-based reasoning can be in modern AI.
"""

### A. Convolutional GNNs for Economic Forecasting
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd  # Make sure to import pandas if not already

# Read data from the CSV file
df = pd.read_csv("economic_indicators.csv")
time_series_data = df.values  # Convert DataFrame to numpy array

# Normalize data
scaler = MinMaxScaler()
time_series_data = scaler.fit_transform(time_series_data)
node_features = torch.tensor(time_series_data, dtype=torch.float)  # Node features for 6 time steps

# Define edges for a fully connected graph (each indicator influences all others)
num_nodes = node_features.size(1)
edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes)], 
                          dtype=torch.long).t()

# Create a PyG Data object
data = Data(x=node_features, edge_index=edge_index)

# Define Advanced GNN Model
class AdvancedGNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AdvancedGNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x

# Initialize model, optimizer, and loss
model = AdvancedGNNModel(input_dim=node_features.size(1),
                         hidden_dim=8,
                         output_dim=node_features.size(1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Example target (true labels) for the last time step (scaled)
# Suppose the real future values are [0.036, 0.054, 0.014, 3620]
y_true = torch.tensor(scaler.transform([[0.036, 0.054, 0.014, 3620]]),
                      dtype=torch.float)

# Train Model
for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out[-1], y_true)  # Use the last time step for prediction
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Make Predictions
model.eval()
predicted = model(data)
predicted_values = scaler.inverse_transform(predicted[-1].detach()
                                                      .numpy()
                                                      .reshape(1, -1))
print("Predicted values (last time step):", predicted_values)

### B. GAT for Feature Selection in Supervised Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd  # Make sure to import pandas if not already
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

num_features = 10   # Number of features
num_samples = 2000  # Number of samples
key_features = [0, 1, 2, 3]  # Indices of the most important features (hypothetically)

# Load data from CSV file
df = pd.read_csv("model_data.csv")
X_full = df.iloc[:, :-1].values  # Features
Y = df["Target"].values          # Binary target

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X_full, Y, test_size=0.3, random_state=42
)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

# Feature adjacency matrix (relationships between features)
adjacency_matrix = torch.rand((num_features, num_features))
adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2  # Make it symmetric
adjacency_matrix.fill_diagonal_(1.0)  # Self-loops

# Define an Attention-Based GNN for feature selection
class AttentionGNNFeatureSelector(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(AttentionGNNFeatureSelector, self).__init__()
        # Attention weights for feature-to-feature connections
        self.attention_weights = nn.Parameter(torch.Tensor(num_features, num_features))
        nn.init.xavier_uniform_(self.attention_weights)  # Initialize weights

        self.conv1 = nn.Linear(num_features, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, num_features)
        self.classifier = nn.Linear(num_features, 1)  # For binary classification

    def forward(self, X, adj):
        batch_size = X.size(0)

        # Compute attention scores
        attention = torch.matmul(self.attention_weights, adj)
        attention = F.softmax(attention, dim=-1)  # Normalize attention weights

        # Apply attention to each sample's features
        h = []
        for i in range(batch_size):
            h_sample = torch.matmul(attention, X[i])
            h.append(h_sample)
        h = torch.stack(h)

        # Graph layers
        h = F.relu(self.conv1(h))  # First graph convolution layer
        h = F.relu(self.conv2(h))  # Second graph convolution layer

        # Compute feature importance as mean activation across all samples
        feature_importance = torch.abs(h.mean(dim=0))

        # Final classification
        output = self.classifier(h)
        return output, feature_importance

# Initialize the model, optimizer, and loss
model = AttentionGNNFeatureSelector(num_features=num_features, hidden_dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()

# Train model
for epoch in range(50):
    model.train()
    optimizer.zero_grad()

    predictions, feature_importance = model(X_train, adjacency_matrix)
    loss = loss_fn(predictions, Y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_predictions, _ = model(X_test, adjacency_matrix)
            auc = roc_auc_score(Y_test.numpy(),
                                torch.sigmoid(test_predictions).numpy())
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Test AUC: {auc:.4f}")

# Output final feature importance
print("Feature Importance:", feature_importance.detach().numpy())

### C. Temporal GNN for Future Stock Index Prediction
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd  # Make sure to import pandas if not already

# Suppose the CSV has columns representing [Interest Rate, Inflation Rate, Unemployment, Stock Index, TimeStep]
# We'll load it and organize into time_steps, num_nodes, features_per_node

df = pd.read_csv("financial_data.csv")
time_steps = len(df)  # Number of time steps
num_nodes = 4         # E.g., Interest Rate, Inflation Rate, Unemployment, Stock Index
features_per_node = 1 # One feature per node (for simplicity)

# Extract node features and reshape
node_features = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)  # Exclude the TimeStep column
node_features = node_features.view(time_steps, num_nodes, -1)

# Example adjacency matrix for 4 financial indicators (fully connected, no self-loops)
adjacency_matrix = torch.tensor([
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0]
], dtype=torch.float32)

# For demonstration, assume 'stock_index' is the last column in df
stock_index = node_features[:, -1, :]  # The stock index for each time step (node: the 4th one)

# Temporal GNN definition
class TemporalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TemporalGNN, self).__init__()
        self.gcn1 = nn.Linear(input_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, features, adj):
        # features: shape [num_nodes, input_dim]
        x = torch.matmul(adj, features)  # Aggregate neighbor features
        x = F.relu(self.gcn1(x))         # First GNN layer
        x = torch.matmul(adj, x)         # Aggregate again
        x = self.gcn2(x)                 # Second GNN layer
        return x

gnn = TemporalGNN(input_dim=1, hidden_dim=8, output_dim=1)
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Shift the stock_index to create a "future" prediction scenario
target = stock_index[1:]       # 1 step ahead
node_features = node_features[:-1]  # Align feature sequences with the target

# Training loop
for epoch in range(100):
    gnn.train()
    optimizer.zero_grad()

    predictions = []
    # Predict for each time step except the last one
    for t in range(time_steps - 1):
        # node_features[t] shape: [num_nodes, input_dim]
        pred = gnn(node_features[t], adjacency_matrix)  # Predict for all nodes
        predictions.append(pred[-1])  # The stock index is the last node
    
    predictions = torch.stack(predictions).squeeze(-1)  # [time_steps-1]
    loss = loss_fn(predictions, target.squeeze(-1))     # Compare to actual stock index
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = torch.mean(torch.abs((predictions - target.squeeze(-1)) /
                                    target.squeeze(-1))) * 100
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, MAPE: {mape.item():.2f}%")

# Print final predictions vs. actual
print("\nFinal Predictions vs Actual Target:")
for t in range(time_steps - 1):
    print(f"Time {t+1}: Predicted: {predictions[t].item():.2f}, "
          f"Actual: {target[t, 0].item():.2f}")


