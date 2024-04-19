# From https://medium.com/@zaiinn440/llama-bitnet-training-a-1-58-bit-llm-3831e517430a

### Create the llama model with custom config. Convert it to bitnet.
model = LlamaForCausalLM(config)
convert_to_bitnet(model, copy_weights=False)
model_size = sum(t.numel() for t in model.parameters())
print(f"Model size: {model_size/1000**2:.1f}M parameters")
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

output_path = "./out"
args = TrainingArguments(
    output_dir=output_path,
    per_device_train_batch_size=BATCH_SIZE,
    logging_steps=100,
    gradient_accumulation_steps=2,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    warmup_steps=0.1,
    lr_scheduler_type="cosine",
    learning_rate=LEARNING_RATE,
    save_steps=0.25,
    fp16=True,
    report_to="wandb"
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_data["train"],
)

trainer.train()
trainer.save_model(f"{output_path}/final_model")
folder = "./out/final_model"
api = HfApi()
create_repo(
    repo_id = f"{HUGGINGFACE_ID}/{NEW_MODEL}",
    repo_type="model",
    exist_ok=True,
    token=HF_TOKEN,
)

# Upload Model files
api.upload_folder(
    folder_path=folder,
    repo_type="model",
    repo_id=f"{HUGGINGFACE_ID}/{NEW_MODEL}",
    token=HF_TOKEN,
)
