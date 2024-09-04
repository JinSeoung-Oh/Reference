### From https://towardsdatascience.com/fine-tune-the-audio-spectrogram-transformer-with-transformers-73333c9ef717

!pip install transformers[torch] datasets[audio] audiomentations

from datasets import load_dataset
from datasets import Dataset, Audio, ClassLabel, Features
import numpy as np
from datasets import Audio, ClassLabel
from transformers import ASTFeatureExtractor
from audiomentations import Compose, AddGaussianSNR, GainTransition, Gain, ClippingDistortion, TimeStretch, PitchShift
from transformers import ASTConfig, ASTForAudioClassification
from transformers import TrainingArguments
import evaluate
import numpy as np
from transformers import Trainer

def preprocess_audio(batch):
    wavs = [audio["array"] for audio in batch["input_values"]]
    # inputs are spectrograms as torch.tensors now
    inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    
    output_batch = {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}
    return output_batch

def preprocess_audio_with_transforms(batch):
    # we apply augmentations on each waveform
    wavs = [audio_augmentations(audio["array"], sample_rate=SAMPLING_RATE) for audio in batch["input_values"]]
    inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    
    output_batch = {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}
    return output_batch

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    predictions = np.argmax(logits, axis=1)
    metrics = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
    metrics.update(precision.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    metrics.update(recall.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    metrics.update(f1.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    return metrics
  
##### Data 
esc50 = load_dataset("ashraq/esc50", split="train")
# Define class labels
class_labels = ClassLabel(names=["bang", "dog_bark"])
# Define features with audio and label columns
features = Features({
    "audio": Audio(),  # Define the audio feature
    "labels": class_labels  # Assign the class labels
})
# Construct the dataset from a dictionary
dataset = Dataset.from_dict({
    "audio": ["/audio/fold1/7061-6-0-0.wav", "/audio/fold1/7383-3-0-0.wav"],
    "labels": [0, 1],  # Corresponding labels for the audio files
}, features=features)

# get target value - class name mappings
df = esc50.select_columns(["target", "category"]).to_pandas()
class_names = df.iloc[np.unique(df["target"], return_index=True)[1]]["category"].to_list()
# cast target and audio column
esc50 = esc50.cast_column("target", ClassLabel(names=class_names))
esc50 = esc50.cast_column("audio", Audio(sampling_rate=16000))
# rename the target feature
esc50 = esc50.rename_column("target", "labels")
num_labels = len(np.unique(esc50["labels"]))

###### Load model
pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)
# we save model input name and sampling rate for later use
model_input_name = feature_extractor.model_input_names[0]  # key -> 'input_values'
SAMPLING_RATE = feature_extractor.sampling_rate

# calculate values for normalization
feature_extractor.do_normalize = False  # we set normalization to False in order to calculate the mean + std of the dataset
mean = []
std = []

# we use the transformation w/o augmentation on the training dataset to calculate the mean + std
dataset["train"].set_transform(preprocess_audio, output_all_columns=False)
for i, (audio_input, labels) in enumerate(dataset["train"]):
    cur_mean = torch.mean(dataset["train"][i][audio_input])
    cur_std = torch.std(dataset["train"][i][audio_input])
    mean.append(cur_mean)
    std.append(cur_std)
feature_extractor.mean = np.mean(mean)
feature_extractor.std = np.mean(std)
feature_extractor.do_normalize = True


# Apply the transformation to the dataset
dataset = dataset.rename_column("audio", "input_values")  # rename audio column
dataset.set_transform(preprocess_audio, output_all_columns=False)

# split training data
if "test" not in dataset:
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=0, stratify_by_column="labels")

# Setting Up Audio Augmentations
audio_augmentations = Compose([
    AddGaussianSNR(min_snr_db=10, max_snr_db=20),
    Gain(min_gain_db=-6, max_gain_db=6),
    GainTransition(min_gain_db=-6, max_gain_db=6, min_duration=0.01, max_duration=0.3, duration_unit="fraction"),
    ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=30, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2),
    PitchShift(min_semitones=-4, max_semitones=4),
], p=0.8, shuffle=True)

# Cast the audio column to the appropriate feature type and rename it
dataset = dataset.cast_column("input_values", Audio(sampling_rate=feature_extractor.sampling_rate))

# with augmentations on the training set
dataset["train"].set_transform(preprocess_audio_with_transforms, output_all_columns=False)
# w/o augmentations on the test set
dataset["test"].set_transform(preprocess_audio, output_all_columns=False)

######### Configure and Initialize the AST for Fine-Tuning
config = ASTConfig.from_pretrained(pretrained_model)
# Update configuration with the number of labels in our dataset
config.num_labels = num_labels
config.label2id = label2id
config.id2label = {v: k for k, v in label2id.items()}
# Initialize the model with the updated configuration
model = ASTForAudioClassification.from_pretrained(pretrained_model, config=config, ignore_mismatched_sizes=True)
model.init_weights()

########## Training
# Configure training run with TrainingArguments class
training_args = TrainingArguments(
    output_dir="./runs/ast_classifier",
    logging_dir="./logs/ast_classifier",
    report_to="tensorboard",
    learning_rate=5e-5,  # Learning rate
    push_to_hub=False,
    num_train_epochs=10,  # Number of epochs
    per_device_train_batch_size=8,  # Batch size per device
    eval_strategy="epoch",  # Evaluation strategy
    save_strategy="epoch",
    eval_steps=1,
    save_steps=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_strategy="steps",
    logging_steps=20,
)

accuracy = evaluate.load("accuracy")
recall = evaluate.load("recall")
precision = evaluate.load("precision")
f1 = evaluate.load("f1")
AVERAGE = "macro" if config.num_labels > 2 else "binary"

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,  # Use the metrics function from above
)

trainer.train()
