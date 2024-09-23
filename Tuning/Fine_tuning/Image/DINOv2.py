## From https://medium.com/data-science-in-your-pocket/fine-tuning-dinov2-custom-training-for-your-own-ai-projects-6e8a5a486671

## After preparing dataset

from datasets import load_dataset
from transformers import AutoModelForImageClassification
from transformers import Trainer, TrainingArguments

# Load your custom dataset
dataset = load_dataset("path_to_your_dataset")

# Load the pre-trained DINOv2 model
model = AutoModelForImageClassification.from_pretrained("facebook/dinov2-base", num_labels=YOUR_NUM_CLASSES)
model.classifier = torch.nn.Linear(model.config.hidden_size, YOUR_NUM_CLASSES)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
)

trainer.train()

predictions = trainer.predict(dataset["val"])
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(predictions.label_ids, predictions.predictions.argmax(-1))
print(f"Accuracy: {accuracy:.4f}")

####### Tips for Fine-Tuning and Optimization #######
# It is very basic thing, but basic is more important than the other
"""
Fine-tuning a large model like DINOv2 can be tricky, so here are a few tips to help you along the way:
 -1. Learning Rate: Start with a small learning rate and adjust based on performance. A typical range might be between 1e-5 and 1e-4.
 -2. Batch Size: Make sure your batch size fits within your GPU’s memory. If you run out of memory, try reducing it.
 -3. Early Stopping: Implement early stopping to avoid overfitting. If the validation loss stops improving, it might be time to stop training.
 -4. Data Augmentation: If your dataset is small, use techniques like data augmentation (rotation, cropping, etc.) to help your model generalize better.
"""
