### From https://ai.gopubby.com/railway-ontology-inspired-fine-tuning-and-evaluation-of-llms-using-mlflow-for-more-effective-8068acdfc715

"""
1. Overview of the Fine-Tuning Workflow for Railway Maintenance Knowledge
   -a. Background: Large language models built on Transformer architecture are trained on vast, general-purpose datasets that enable them to perform diverse tasks. However, these models struggle with domain-specific knowledge like railway maintenance and engineering because the training data often lacks specialized technical details.
   -b. Challenges: General training data does not cover intricate technical terminology and complex maintenance processes. For example, the term “voltage transducer” might be associated with both a pantograph and an inductor, requiring the model to correctly distinguish context for reliable analysis and maintenance decisions.

2. Adoption of Railway Vehicle Ontology
   -a. Definition of Ontology: An ontology is a formal specification of concepts, relationships, and rules within a particular domain. It serves as the schema or “rule book” for knowledge graphs, organizing domain-specific information.
   -b. Hierarchical Structure of Railway Ontology:
       -1. Top-level concept: Vehicle
       -2. Hierarchy: A Vehicle contains different types of coaches, each coach includes a set of components, 
                      and these components consist of various parts or sub-parts.
       -3. This hierarchy covers maintenance activities by linking tasks to specific parts or sub-parts and 
           identifying the machinery required for inspection, maintenance, and related services.
   -c. Importance of Hierarchical Ontology:
       -1. Resolving Component Ambiguity: The hierarchical lineage helps trace components accurately, 
                                          enabling the LLM to distinguish and correctly identify specific parts.
       -2. Addressing Inconsistency in Maintenance Logs: Maintenance logs often vary because different technicians may 
                                                         use abbreviations, full terms, or omit hierarchical context. 
                                                         The ontology provides a consistent structure that reduces confusion 
                                                         and improves the accuracy of LLM interpretations.

3. Training Dataset Curation for Fine-Tuning
   -a. Need for High-Quality Data: A high-quality, domain-specific dataset is crucial for improving model performance in railway maintenance tasks.
   -b. Data Preparation Process:
       -1. Collect Historical Maintenance Logs: Gather logs that document common maintenance activities at a railway depot.
       -2. Convert Data into a Specific Conversational Format: Prepare the collected logs in a structured conversational format suitable 
           for fine-tuning the model.
   -c. Key Components in Data Formatting:
       -1. Base Model Selection: Llama-2–7b-hf is chosen as the base model for fine-tuning.
       -2. Specialized System Prompt: Include a railway maintenance-specific system prompt focusing on diagnostic and recommendation tasks.
       -3. Structured Data Using Classes: Use Conversation and Message classes to define conversations. 
           Each message includes a role (user or assistant) and content.
       -4. Special Tokens: For Llama-2–7b-hf:
           - User inputs are wrapped in [INST]…[/INST].
           - The system message is included inside the first [INST] block, wrapped with <<SYS>> tags.
           - Assistant responses do not require special tokens.

4. Example of Railway Vehicle Ontology Utilization
   -a. Structure Example: Original text illustrates a railway vehicle ontology as a hierarchical tree, 
                          starting from the top-level concept and branching into detailed sub-components.
   -b. Problem Solving: The ontology helps the model accurately identify components in varied contexts, resolves ambiguities due to 
                        multiple meanings of the same terminology, and ensures that maintenance tasks are linked correctly within the vehicle’s hierarchical structure.
"""

## data.py
@dataclass
class Message:
    role: str
    content: str

@dataclass
class Conversation:
    messages: List[Message]
    system_prompt: Optional[str] = None

class DatasetProcessor:
    def __init__(
        self,
        max_length: int = 2048,
        system_prompt: Optional[str] = None
    ):
    
        self.max_length = max_length
        self.default_system_prompt = system_prompt or (
            "You are an expert railway maintenance system. Your role is to analyze maintenance "
            "problems, recommend appropriate actions, and identify the component chain involved "
            "in the issue. Provide clear, precise recommendations based on the problem description."
            "If you do not know the answer, respond with 'I don't know.' "
            "If the question is unclear, ask for clarification."        
        )

    def format_conversation(self, conversation: Conversation) -> str:

        formatted_messages = []        
        for i, message in enumerate(conversation.messages):
            if i % 2 == 0:  # User messages
                # For the first user message, include system prompt
                if i == 0:
                    formatted_messages.append(
                        f"[INST] <<SYS>>\n{self.default_system_prompt}\n<</SYS>>\n\n"
                        f"{message.content} [/INST]"
                    )
                else:
                    formatted_messages.append(
                        f"[INST] {message.content} [/INST]"
                    )
            else:  # Assistant messages
                formatted_messages.append(message.content)
        
        return "\n".join(formatted_messages)

    def format_assistant_response(self, action: str, component_chain: str) -> str:
        
        return (
            f"Based on the analysis, here are my recommendations:\n\n"
            f"Recommended Action:\n{action}\n\n"
            f"Component Chain Analysis:\n{component_chain}"
        )

    def load_railway_maintenance_data(self, df: pd.DataFrame) -> List[Conversation]:
        
        conversations = []
        
        for _, row in df.iterrows():
            # Format user query (problem description)
            user_content = (
                f"Please analyze the following railway maintenance issue and provide "
                f"recommendations:\n\nProblem Description: {row['problem']}"
            )
            
            # Format assistant response with action and component chain
            assistant_content = self.format_assistant_response(
                row['action'],
                row['component chain']
            )
            
            messages = [
                Message(role="user", content=user_content),
                Message(role="assistant", content=assistant_content)
            ]
            
            conversations.append(Conversation(
                messages=messages,
                system_prompt=self.default_system_prompt
            ))
        
        return conversations

    def create_dataset(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8
    ) -> DatasetDict:

        # Convert DataFrame to conversations
        conversations = self.load_railway_maintenance_data(df)
        
        # Prepare conversation data
        data = [{
            "text": self.format_conversation(conv),
            "raw_data": json.dumps({
                "id": idx,
                "problem": conv.messages[0].content,
                "response": conv.messages[1].content
            })
        } for idx, conv in enumerate(conversations)]
        
        # Create DataFrame and split into train/validation
        df_processed = pd.DataFrame(data)
        train_size = int(len(df_processed) * train_ratio)
        
        train_df = df_processed[:train_size]
        val_df = df_processed[train_size:]
        
        # Convert to Dataset format
        dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df)
        })

        if add_tokenization:
                dataset = dataset.map(
                          self._tokenize_function,
                          batched=True,
                          remove_columns=dataset["train"].column_names
                )
        return dataset
      
    def _tokenize_function(
        self, 
        examples: Dict[str, List[str]]
        ) -> Dict[str, List]:

        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

## BaseModelEvaluator.py
class BaseModelEvaluator:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        max_length: int = 2048,
        device: str = "cuda"
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.max_length = max_length

    def format_prompt(self, problem: str) -> str:
        """Format the input prompt following Llama 2 chat format"""
        system_prompt = """You are an expert railway maintenance system. Your role is to analyze maintenance 
        problems, recommend appropriate actions, and identify the component chain involved in the issue. 
        Provide clear, precise recommendations based on the problem description."""
        
        prompt_template = f"""[INST] <<SYS>>{system_prompt}<</SYS>>

          Please analyze the following railway maintenance issue and provide recommendations:

          Problem Description: {problem} [/INST]"""        
        return prompt_template

    def generate_response(self, prompt: str) -> str:
        """Generate model response for a given prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the model's response part (after the prompt)
        response = response[len(prompt):].strip()
        return response

    def extract_components(self, text: str) -> Dict[str, str]:
        """Extract action and component chain from response text"""
        try:
            action = ""
            component_chain = ""
            
            # Look for Recommended Action section
            if "Recommended Action:" in text:
                action = text.split("Recommended Action:")[1]
                action = action.split("Component Chain")[0].strip()
            
            # Look for Component Chain section
            if "Component Chain" in text:
                component_chain = text.split("Component Chain")[1]
                component_chain = component_chain.split("\n\n")[0].strip()
            
            return {
                "action": action,
                "component_chain": component_chain
            }
        except Exception as e:
            print(f"Error extracting components: {e}")
            return {"action": "", "component_chain": ""}


## Config.py
@dataclass
class FineTuningConfig:
    # Model configuration
    base_model_name: str = "meta-llama/Llama-2-7b-hf"
    model_max_length: int = 2048
    
    # QLoRA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = ("q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj")  # Extended for better coverage
    
    # Memory optimization
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": False})
    
    # Training parameters
    learning_rate: float = 3e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    
    # Output configuration
    output_dir: str = "./railway_maintenance_model"

class ModelTrainer:
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.dataset_processor = DatasetProcessor(
            tokenizer_name=config.base_model_name,
            max_length=config.model_max_length
        )

    def setup_model(self):
        """Initialize and prepare the model with QLoRA configuration"""
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",  # normalized float 4
        )
        
        # Load base model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(**gradient_checkpointing_kwargs)
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA with QLoRA settings
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.config.lora_target_modules,
            # QLoRA specific settings
            inference_mode=False,
            lora_16bit_weights=False,  # Important for 4-bit quantization
        )
        
        # Get PEFT model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model

    def setup_training_args(self):
        """Configure training arguments"""
        return TrainingArguments(
            report_to="mlflow",
            run_name=f"Llama-2-7b-hf-{datetime.now().strftime('%Y-%m-%d-%H-%M-%s')}",
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_dir="./logs",
            optim="paged_adamw_8bit",
            logging_steps=100,
            load_best_model_at_end=True,
            bf16=True,
            ddp_find_unused_parameters=False
        )

    def train(self, train_df: pd.DataFrame):
        """Main training function with MLflow tracking"""
        mlflow.set_experiment("llm-finetuning")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "model_name": self.config.base_model_name,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_train_epochs,
                "batch_size": self.config.per_device_train_batch_size
            })
            
            # Prepare dataset
            dataset = self.dataset_processor.create_dataset(
                train_df,
                train_ratio=0.8,
                add_tokenization=True
            )
            
            # Setup model and training arguments
            model = self.setup_model()
            training_args = self.setup_training_args()
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                data_collator=DataCollatorForLanguageModeling(
                    self.dataset_processor.tokenizer,
                    mlm=False
                )
            )

            # use_cache=True is incompatible with gradient checkpointing.
            model.config.use_cache = False
            # Train the model
            train_result = trainer.train()
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.metrics["train_runtime"]
            })
            
            # Save the model
            trainer.save_model(self.config.output_dir)
            
            # Log the model to MLflow
            mlflow.transformers.log_model(
                transformers_model={
                  "model": trainer.model, 
                  "tokenizer": AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True)
                },
                artifact_path="model",  # This is a relative path to save model files within MLflow run
            )
            
            return model, train_result

## Eval.py
@dataclass
class EvaluationMetrics:
    bertscore: float
    bleu_score: float
    iou_score: float
    component_accuracy: float
    component_completeness: float

class ModelEvaluator:
    def __init__(
        self,
        base_model_name: str,
        peft_model_path: str,
        device: str = "cuda",
        max_length: int = 2048
    ):
        self.device = device
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load fine-tuned model
        self.model = PeftModel.from_pretrained(base_model, peft_model_path)
        self.model.eval()
        
        # Initialize BERTScore
        self.bertscore = evaluate.load("bertscore")

    def format_prompt(self, problem: str) -> str:
        """Format input following Llama 2 chat format"""
        system_prompt = """You are an expert railway maintenance system. Your role is to analyze maintenance 
        problems, recommend appropriate actions, and identify the component chain involved in the issue. 
        Provide clear, precise recommendations based on the problem description."""
        
        prompt_template = f"""[INST] <<SYS>>{system_prompt}<</SYS>>

          Please analyze the following railway maintenance issue and provide recommendations:

          Problem Description: {problem} [/INST]"""        
        return prompt_template

    def generate_response(self, prompt: str) -> str:
        """Generate model response for a given prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_return_sequences=1,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):].strip()

    def parse_component_chain(self, chain: str) -> Set[str]:
        """Parse component chain into a set of components"""
        # Clean and normalize the component chain
        chain = chain.replace("Component Chain Analysis:", "").strip()
        components = set()
        
        # Split by different possible delimiters
        for part in chain.replace('>', '/').replace(',', '/').split('/'):
            component = part.strip().lower()
            if component:
                components.add(component)
        
        return components

    def calculate_iou(self, pred_components: Set[str], true_components: Set[str]) -> float:
        """Calculate IoU score for component sets"""
        if not pred_components and not true_components:
            return 1.0
        if not pred_components or not true_components:
            return 0.0
            
        intersection = len(pred_components.intersection(true_components))
        union = len(pred_components.union(true_components))
        
        return intersection / union if union > 0 else 0.0

    def extract_action_and_chain(self, text: str) -> Dict[str, str]:
        """Extract action and component chain from response"""
        action = ""
        component_chain = ""
        
        # Extract action
        if "Recommended Action:" in text:
            action_parts = text.split("Recommended Action:")[1].split("Component Chain")
            action = action_parts[0].strip()
        
        # Extract component chain
        if "Component Chain" in text:
            chain_parts = text.split("Component Chain")[1].split("\n\n")
            component_chain = chain_parts[0].strip()
        
        return {
            "action": action,
            "component_chain": component_chain
        }

    def calculate_bleu_score(self, predicted: str, reference: str) -> float:
        """Calculate BLEU score for recommended actions"""
        # Tokenize the strings into words
        predicted_tokens = nltk.word_tokenize(predicted.lower())
        reference_tokens = nltk.word_tokenize(reference.lower())
        
        # Use smoothing function to handle cases where there are no n-gram overlaps
        smoothing = SmoothingFunction().method1
        
        # Calculate BLEU score with weights for 1-4 grams
        try:
            return sentence_bleu(
                [reference_tokens],
                predicted_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing
            )
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            return 0.0

    def evaluate_single_response(
        self,
        predicted: Dict[str, str],
        actual: Dict[str, str]
    ) -> EvaluationMetrics:
        """Evaluate a single prediction against ground truth"""
        # Calculate BERTScore for action
        bertscore_results = self.bertscore.compute(
            predictions=[predicted['action']],
            references=[actual['action']],
            lang="en"
        )
        bertscore = bertscore_results['f1'][0]
        
        # Calculate BLEU score for action
        bleu_score = self.calculate_bleu_score(
            predicted['action'],
            actual['action']
        )
        
        # Calculate IoU for component chain
        pred_components = self.parse_component_chain(predicted['component_chain'])
        true_components = self.parse_component_chain(actual['component_chain'])
        
        iou_score = self.calculate_iou(pred_components, true_components)
        
        # Calculate additional component metrics
        if true_components:
            component_accuracy = len(pred_components.intersection(true_components)) / len(pred_components) if pred_components else 0.0
            component_completeness = len(pred_components.intersection(true_components)) / len(true_components)
        else:
            component_accuracy = component_completeness = 0.0
        
        return EvaluationMetrics(
            bertscore=bertscore,
            bleu_score=bleu_score,
            iou_score=iou_score,
            component_accuracy=component_accuracy,
            component_completeness=component_completeness
        )

    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        num_samples: int = None,
        output_file: str = "evaluation_results.json"
    ) -> Dict[str, float]:
        """Evaluate model on the entire dataset"""
        if num_samples:
            df = df.sample(n=min(num_samples, len(df)))
        
        all_metrics = []
        detailed_results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            # Generate prediction
            prompt = self.format_prompt(row['problem'])
            response = self.generate_response(prompt)
            
            # Extract components
            predicted = self.extract_action_and_chain(response)
            actual = {
                'action': row['action'],
                'component_chain': row['component chain']
            }
            
            # Calculate metrics
            metrics = self.evaluate_single_response(predicted, actual)
            all_metrics.append(metrics)
            
            # Store detailed results
            detailed_results.append({
                'problem': row['problem'],
                'predicted_action': predicted['action'],
                'actual_action': actual['action'],
                'predicted_chain': predicted['component_chain'],
                'actual_chain': actual['component_chain'],
                'metrics': metrics.__dict__
            })
        
        # Calculate average metrics
        avg_metrics = {
            'avg_bertscore': np.mean([m.bertscore for m in all_metrics]),
            'avg_bleu_score': np.mean([m.bleu_score for m in all_metrics]),
            'avg_iou_score': np.mean([m.iou_score for m in all_metrics]),
            'avg_component_accuracy': np.mean([m.component_accuracy for m in all_metrics]),
            'avg_component_completeness': np.mean([m.component_completeness for m in all_metrics])
        }
        
        # Save detailed results
        with open(output_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        return avg_metrics
