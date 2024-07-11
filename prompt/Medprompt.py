## https://towardsdatascience.com/understanding-and-implementing-medprompt-77bbd2777c91

def write_jsonl_file(file_path, dict_list):
    """
    Write a list of dictionaries to a JSON Lines file.

    Args:
    - file_path (str): The path to the file where the data will be written.
    - dict_list (list): A list of dictionaries to write to the file.
    """
    with open(file_path, 'w') as file:
        for dictionary in dict_list:
            json_line = json.dumps(dictionary)
            file.write(json_line + '\n')

def read_jsonl_file(file_path):
    """
    Parses a JSONL (JSON Lines) file and returns a list of dictionaries.

    Args:
        file_path (str): The path to the JSONL file to be read.

    Returns:
        list of dict: A list where each element is a dictionary representing
            a JSON object from the file.
    """
    jsonl_lines = []
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            json_object = json.loads(line)
            jsonl_lines.append(json_object)
            
    return jsonl_lines

system_prompt = """You are an expert medical professional. You are provided with a medical question with multiple answer choices.
Your goal is to think through the question carefully and explain your reasoning step by step before selecting the final answer.
Respond only with the reasoning steps and answer as specified below.
Below is the format for each question and answer:

Input:
## Question: {{question}}
{{answer_choices}}

Output:
## Answer
(model generated chain of thought explanation)
Therefore, the answer is [final model answer (e.g. A,B,C,D)]"""

def build_few_shot_prompt(system_prompt, question, examples, include_cot=True):
    """
    Builds the zero-shot prompt.

    Args:
        system_prompt (str): Task Instruction for the LLM
        content (dict): The content for which to create a query, formatted as
            required by `create_query`.

    Returns:
        list of dict: A list of messages, including a system message defining
            the task and a user message with the input question.
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    for elem in examples:
        messages.append({"role": "user", "content": create_query(elem)})
        if include_cot:
            messages.append({"role": "assistant", "content": format_answer(elem["cot"], elem["answer_idx"])})        
        else:           
            answer_string = f"""## Answer\nTherefore, the answer is {elem["answer_idx"]}"""
            messages.append({"role": "assistant", "content": answer_string})
            
    messages.append({"role": "user", "content": create_query(question)})
    return messages

def get_response(messages, model_name, temperature = 0.0, max_tokens = 10):
    """
    Obtains the responses/answers of the model through the chat-completions API.

    Args:
        messages (list of dict): The built messages provided to the API.
        model_name (str): Name of the model to access through the API
        temperature (float): A value between 0 and 1 that controls the randomness of the output.
        A temperature value of 0 ideally makes the model pick the most likely token, making the outputs deterministic.
        max_tokens (int): Maximum number of tokens that the model should generate

    Returns:
        str: The response message content from the model.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def matches_ans_option(s):
    """
    Checks if the string starts with the specific pattern 'Therefore, the answer is [A-Z]'.
    
    Args:
    s (str): The string to be checked.

    Returns:
    bool: True if the string matches the pattern, False otherwise.
    """
    return bool(re.match(r'^Therefore, the answer is [A-Z]', s))

def extract_ans_option(s):
    """
    Extracts the answer option (a single capital letter) from the start of the string.
    
    Args:
    s (str): The string containing the answer pattern.

    Returns:
    str or None: The captured answer option if the pattern is found, otherwise None.
    """
    match = re.search(r'^Therefore, the answer is ([A-Z])', s)
    if match:
        return match.group(1)  # Returns the captured alphabet
    return None 

def matches_answer_start(s):
    """
    Checks if the string starts with the markdown header '## Answer'.
    
    Args:
    s (str): The string to be checked.

    Returns:
    bool: True if the string starts with '## Answer', False otherwise.
    """
    return s.startswith("## Answer")

def validate_response(s):
    """
    Validates a multi-line string response that it starts with '## Answer' and ends with the answer pattern.
    
    Args:
    s (str): The multi-line string response to be validated.

    Returns:
    bool: True if the response is valid, False otherwise.
    """
    file_content = s.split("\n")
    
    return matches_ans_option(file_content[-1]) and matches_answer_start(s)

def parse_answer(response):
    """
    Parses a response that starts with '## Answer', extracting the reasoning and the answer choice.
    
    Args:
    response (str): The multi-line string response containing the answer and reasoning.

    Returns:
    tuple: A tuple containing the extracted CoT reasoning and the answer choice.
    """
    split_response = response.split("\n")
    assert split_response[0] == "## Answer"
    cot_reasoning = "\n".join(split_response[1:-1]).strip()
    ans_choice = extract_ans_option(split_response[-1])
    return cot_reasoning, ans_choice

#######################
train_data = read_jsonl_file("data/phrases_no_exclude_train.jsonl")

cot_responses = []
# os.mkdir("cot_responses")
existing_files = os.listdir("cot_responses/")

for idx, item in enumerate(tqdm(train_data)):
    if str(idx) + ".txt" in existing_files:
        continue
    
    prompt = build_zero_shot_prompt(system_prompt, item)
    try:
        response = get_response(prompt, model_name="gpt-4o", max_tokens=500)
        cot_responses.append(response)
        with open(os.path.join("cot_responses", str(idx) + ".txt"), "w", encoding="utf-8") as f:
            f.write(response)           
    except Exception as e :
        print(str(e))
        cot_responses.append("")


questions_dict = []
ctr = 0
for idx, question in enumerate(tqdm(train_data)):
    file =  open(os.path.join("cot_responses/", str(idx) + ".txt"), encoding="utf-8").read()
    if not validate_response(file):
        continue
    
    cot, pred_ans = parse_answer(file)
    
    dict_elem = {}
    dict_elem["idx"] = idx
    dict_elem["question"] = question["question"]
    dict_elem["answer"] = question["answer"]
    dict_elem["options"] = question["options"]
    dict_elem["cot"] = cot
    dict_elem["pred_ans"] = pred_ans
    questions_dict.append(dict_elem)        

filtered_questions_dict = []
for item in tqdm(questions_dict):
    pred_ans = item["options"][item["pred_ans"]]
    if pred_ans == item["answer"]:
        filtered_questions_dict.append(item)


def get_embedding(text, model="text-embedding-ada-002"):
    return client.embeddings.create(input = [text], model=model).data[0].embedding

for item in tqdm(filtered_questions_dict):
    item["embedding"] = get_embedding(item["question"])
    inv_options_map = {v:k for k,v in item["options"].items()}
    item["answer_idx"] = inv_options_map[item["answer"]]    

import numpy as np
from sklearn.neighbors import NearestNeighbors

embeddings = np.array([d["embedding"] for d in filtered_questions_dict])
indices = list(range(len(filtered_questions_dict)))

knn = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='cosine').fit(embeddings)

#######################
def shuffle_option_labels(answer_options):
    """
    Shuffles the options of the question.
    
    Parameters:
    answer_options (dict): A dictionary with the options.

    Returns:
    dict: A new dictionary with the shuffled options.
    """
    options = list(answer_options.values())
    random.shuffle(options)
    labels = [chr(i) for i in range(ord('A'), ord('A') + len(options))]
    shuffled_options_dict = {label: option for label, option in zip(labels, options)}
    
    return shuffled_options_dict
test_samples = read_jsonl_file("final_processed_test_set_responses_medprompt.jsonl")

for question in tqdm(test_samples, colour ="green"):
    question_variants = []
    prompt_variants = []
    cot_responses = []
    question_embedding = get_embedding(question["question"])
    distances, top_k_indices = knn.kneighbors([question_embedding], n_neighbors=5)
    top_k_dicts = [filtered_questions_dict[i] for i in top_k_indices[0]]
    question["outputs"] = []
    
    for idx in range(5):
        question_copy = question.copy()
        shuffled_options = shuffle_option_labels(question["options"])
        inv_map = {v:k for k,v in shuffled_options.items()}
        
        question_copy["options"] = shuffled_options
        question_copy["answer_idx"] = inv_map[question_copy["answer"]]
        question_variants.append(question_copy)
        prompt = build_few_shot_prompt(system_prompt,  question_copy, top_k_dicts)
        prompt_variants.append(prompt)
    
    for prompt in tqdm(prompt_variants):
        response = get_response(prompt, model_name="gpt-4o", max_tokens=500)
        cot_responses.append(response)
    
    for question_sample, answer in zip(question_variants, cot_responses):
        if validate_response(answer):
            cot, pred_ans = parse_answer(answer)
            
        else:
            cot = ""
            pred_ans = ""
                
        question["outputs"].append({"question": question_sample["question"], "options": question_sample["options"], "cot": cot, "pred_ans": question_sample["options"].get(pred_ans, "")})

def find_mode_string_list(string_list):
    """
    Finds the most frequently occurring strings.

    Parameters:
    string_list (list of str): A list of strings.
    Returns:
    list of str or None: A list containing the most frequent string(s) from the input list.
                         Returns None if the input list is empty.
    """    
    if not string_list:
        return None  

    string_counts = Counter(string_list)
    max_freq = max(string_counts.values())
    mode_strings = [string for string, count in string_counts.items() if count == max_freq]
    return mode_strings

ctr = 0 
for item in test_samples:
    pred_ans = [x["pred_ans"] for x in item["outputs"]]
    freq_ans = find_mode_string_list(pred_ans)
    
    if len(freq_ans) > 1:
        final_prediction = ""
    else:
        final_prediction = freq_ans[0]
        
    if final_prediction == item["answer"]:
        ctr +=1

print(ctr / len(test_samples))

