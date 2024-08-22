import requests
import pandas as pd
import json
import time
from tqdm import tqdm

def ask_response(instruction,prompt):
    url = "http://192.168.1.168:1980/inference_phi3"

    payload = json.dumps({
    "instruction": instruction,
    "prompt": prompt
    })
    headers = {
    'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()['response']


def save_to_csv(questions, answers, standard_answers, filename='q_not_in_system_1w.csv'):
    data = {'Question': questions, 'Generated Answer': answers, 'Standard Answer': standard_answers}
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Answer saved to {filename}")

def extract_data_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    instruction_list = [item['instruction'] for item in data[2200:2230] if 'instruction' in item]
    input_list = [item['input'] for item in data[2200:2230] if 'input' in item]
    output_list = [item['output'] for item in data[2200:2230] if 'output' in item]
    
    return instruction_list, input_list, output_list

if __name__ == "__main__":
    json_file_path = '/data2/yixu/phi3/mrc_阅读理解问答_7_29.json'
    instruction_list, input_list, output_list = extract_data_from_json(json_file_path)
    
    questions = [f"{instr}\n\n{inp}" for instr, inp in zip(instruction_list, input_list)]
    standard_answers = output_list
    generated_answers = []

    for i in tqdm(range(len(instruction_list))):
        answer = ask_response(instruction_list[i],input_list[i])
        print(answer)
        generated_answers.append(answer)
        time.sleep(1) 
    
    save_to_csv(questions, generated_answers, standard_answers)
