from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from peft import LoraConfig, TaskType, get_peft_model

import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBILE_DEVICES']='1'

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/inference_phi3', methods=['post'])
def inference_phi3():
    data = request.json
    print('Get request:')
    print(data)
    instruction = data.get('instruction')
    prompt = data.get('prompt')
    model_path = '/data2/yixu/LLM-Research/Phi-3-mini-4k-instruct'
    lora_path = './Phi-3_lora_729'
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",torch_dtype=torch.bfloat16,trust_remote_code=True)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True, 
        r=8, # Lora 秩
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

    messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        eos_token_id=tokenizer.encode('<|endoftext|>')[0]
    )
    outputs = generated_ids.tolist()[0][len(model_inputs[0]):]
    response = tokenizer.decode(outputs).split('<|end|>')[0]

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=False, port=1980, host='0.0.0.0')