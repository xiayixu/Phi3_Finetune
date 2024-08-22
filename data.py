import json
import pandas as pd
import re
from tqdm import tqdm
import ast


def extract_choices_and_answer(item):
    choices_str = item['choices']
    try:
        # 使用 ast.literal_eval 来安全地解析字符串
        choices_list = ast.literal_eval(choices_str)
        answer_index = ord(item['answer']) - ord('A')
        correct_answer = choices_list[answer_index]
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing choices for item: {item}")
        raise e
    
    return correct_answer

def extract_text(text):
    # 使用正则表达式去除字母和标点符号
    text = re.sub(r'^[^\u4e00-\u9fa5]+', '', text)
    # 去除结尾的非中文字符
    text = re.sub(r'[^\u4e00-\u9fa5]+$', '', text)
    return text


file_path = '/data2/yixu/phi3/output.json'
with open(file_path, 'r') as f:
    file = json.load(f)


outputs = []
for i in tqdm(file):
    temp = {}
    question = i['query']
    instruction = f'根据篇章，生成问题的答案，仅输出答案，不要解释你的答案。'

    artical = i['content']
    answers = i['answer']
    inputs = f'篇章：{artical}\\n问题：{question} {i["choices"]}'
    choices = i['choices']

    # for v in choices:
    #     if answers in v:
    #         output = extract_text(v)
    #         break
    output = extract_choices_and_answer(i)
    temp['instruction'] = instruction
    temp['input'] = inputs
    temp['output'] = output
    outputs.append(temp)

out_path = 'mrc_阅读理解问答_7_29.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(outputs[10001:], f, ensure_ascii=False, indent=4)


# df = pd.read_json('/data2/yixu/phi3/mrc_阅读理解问答.json')

# # 显示数据框内容
# print(df.head())
