## From https://medium.com/@datadrifters/starcoder-2-can-top-open-source-llm-beat-github-copilot-642735ea2fbf

"""
# Create a virtual environment
mkdir starcoder2 && cd starcoder2
python3 -m venv starcoder2-env
source starcoder2-env/bin/activate

# Install dependencies
pip3 install torch
pip3 install git+https://github.com/huggingface/transformers.git
pip3 install datasets 
pip3 install ipykernel jupyter
pip3 install --upgrade huggingface_hub
pip3 install accelerate # to run the model on a single / multi GPU
pip3 install bitsandbytes

# Loging to Huggingface Hub
huggingface-cli login

# Optionally, fire up VSCode or your favorite IDE and let's get rolling!
code .
""""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# to use 4bit use `load_in_4bit=True` instead
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

checkpoint = "bigcode/starcoder2-15b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=quantization_config)
print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")

## Test simple
inputs = tokenizer.encode("def factorial(n):", return_tensors="pt").to("cuda")
outputs = model.generate(inputs, eos_token_id=tokenizer.eos_token_id, max_length=100, num_return_sequences=1)
print(tokenizer.decode(outputs[0]))

## Test JavaScript question
inputs = tokenizer.encode("""
function filterArray(arr) {
    // Complete the function to filter out numbers greater than 10
""", return_tensors="pt").to("cuda")
outputs = model.generate(inputs, eos_token_id=tokenizer.eos_token_id, max_length=100, num_return_sequences=1)
print(tokenizer.decode(outputs[0]))


## SQL
inputs = tokenizer.encode("""
# generate a SQL query that selects all columns from a table named 'employees' where the 'salary' is greater than 50000.
SELECT * FROM employees WHERE
""", return_tensors="pt").to("cuda")
outputs = model.generate(inputs, eos_token_id=tokenizer.eos_token_id, max_length=100, num_return_sequences=1)
print(tokenizer.decode(outputs[0]))

## C++
inputs = tokenizer.encode("""
#include <iostream>
class Rectangle {
private:
    int width, height;
public:
    Rectangle(int w, int h) : width(w), height(h) {}
    // Complete the class with methods to calculate area and perimeter
    int getArea();
    int getPerimeter();
};
""", return_tensors="pt").to("cuda")

outputs = model.generate(inputs, eos_token_id=tokenizer.eos_token_id, max_length=500, num_return_sequences=1)
print(tokenizer.decode(outputs[0]))

