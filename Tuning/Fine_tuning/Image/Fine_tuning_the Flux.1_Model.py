### From https://medium.com/@kram254/fine-tuning-the-flux-1-model-a-professionals-guide-to-advanced-ai-training-4a6ffc1e996a

!git clone https://github.com/ostris/ai-toolkit.git
%cd ai-toolkit
!pip3 install -r requirements.txt
!pip install peft
!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

huggingface-cli login

from datasets import load_dataset
from flux_models import FluxModel, FluxTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Load the dataset for fine-tuning (e.g., AG News)
dataset = load_dataset('ag_news')

# Preprocess the dataset for tokenization
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Load pre-trained model and tokenizer
model = FluxModel.from_pretrained('flux-model-1.0')
tokenizer = FluxTokenizer.from_pretrained('flux-model-1.0')

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the appropriate device (CPU or GPU)
model.to(device)

# Prepare data loader
train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=32, shuffle=True)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

print(f"Epoch {epoch + 1} completed. Loss: {loss.item()}")

## Evaluation Code
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Switch model to evaluation mode
model.eval()
def evaluate(model, dataset):
    predictions, true_labels = [], []
    for batch in dataset:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(inputs)
            preds = outputs.logits.argmax(dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return accuracy_score(true_labels, predictions)
accuracy = evaluate(model, tokenized_dataset['test'])
print(f"Accuracy: {accuracy * 100:.2f}%")

# Switch model to evaluation mode
model.eval()

def evaluate(model, dataset):
    predictions, true_labels = [], []
    for batch in dataset:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            outputs = model(inputs)
            preds = outputs.logits.argmax(dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(true_labels, predictions))

    return accuracy, precision, recall, f1

accuracy, precision, recall, f1 = evaluate(model, tokenized_dataset['test'])


## Advanced Optimization Techniques for Fine-Tuning
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

model.save_pretrained('./fine_tuned_flux')





