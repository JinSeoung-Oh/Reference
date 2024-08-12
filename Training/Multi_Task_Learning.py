## From https://pub.towardsai.net/multi-task-learning-mtl-and-the-role-of-activation-functions-in-neural-networks-train-mlp-with-b892ffe678c8

"""
Multi-task learning is a Method in Machine Learning where Multiple related tasks are learned simultaneously, 
leveraging shared information among them to improve performance. 
Instead of training a separate model for each task, MTL trains a single model to handle multiple tasks

In MTL, some layers or parameters are shared across tasks, allowing the model to learn common features that benefit all tasks. 
The model is trained on different tasks simultaneously, and the parameters are updated based on the combined loss from all tasks.

In addition to shared layers, MTL models typically have task-specific layers that handle the unique aspects of each task.
The final output layer of the model provides the desired output for each task.
"""

## Below code is the part of MiltiTask leaning, but it is good to understand about it
class MultiTaskNet(nn.Module):
    def __init__(self):
        super(MultiTaskNet, self).__init__()
        
        # Two Shared Hidden Layer (Parameters in this layer learns general nature of the input and its relationship with the output)
        self.shared_fc1 = nn.Linear(12, 32) 
        self.shared_fc2 = nn.Linear(32, 64)
        
        self.thal_fc1 = nn.Linear(64, 32)
        self.thal_fc2 = nn.Linear(32, 3)  # 3 classes for thalassemia
        
        self.heart_fc1 = nn.Linear(64, 16)
        self.heart_fc2 = nn.Linear(16, 1)  # 1 output for heart disease
    
    def forward(self, x):
        x = F.relu(self.shared_fc1(x))
        x = F.relu(self.shared_fc2(x))
        
        thal_out = F.relu(self.thal_fc1(x))
        thal_out = self.thal_fc2(thal_out)  # Task 1: Predicting thalassemia
        
        heart_out = F.relu(self.heart_fc1(x))
        heart_out = torch.sigmoid(self.heart_fc2(heart_out)) # Task 2: Predicting heart disease
        
        return thal_out, heart_out

model = MultiTaskNet()


# Cost function
criterion_thal = nn.CrossEntropyLoss()  # Multi Class- Softmax activation
criterion_heart = nn.BCELoss()    # Binary Loss- Sigmoid activation

#Optimizers
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss_thal = 0.0
    running_loss_heart = 0.0
    
    for inputs, labels_thal, labels_heart in train_loader:
        optimizer.zero_grad()  # Making the optimizer has no slope (zero_grade)
        
        outputs_thal, outputs_heart = model(inputs)
        
        loss_thal = criterion_thal(outputs_thal, labels_thal)
        loss_heart = criterion_heart(outputs_heart.squeeze(), labels_heart)
        
        loss = loss_thal + loss_heart
        loss.backward() #Calculates the slope or gradients
        optimizer.step() # Updating gradients
        
        running_loss_thal += loss_thal.item()
        running_loss_heart += loss_heart.item()
        
    if epoch%10==0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss Thal: {running_loss_thal/len(train_loader)}, Loss Heart: {running_loss_heart/len(train_loader)}')

### Super basic, but I think, this explain is very excellent
### The Role of Activation Functions in Neural Networks
## Activations Functions Introduce Non-Linearity 😁 into the Neural Network. 
## This allows the network to learn from the errors and to capture complex patterns in the data. 
## Without the Activation function, the neural network won’t be able to learn complex relationships in the data. 
## Neural Networks can only learn linear relationships in the data without activation.



