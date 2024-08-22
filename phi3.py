import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBILE_DEVICES']='1'

df = pd.read_json('/data2/yixu/phi3/mrc_阅读理解问答_7_29_微调.json')
ds = Dataset.from_pandas(df)

def process_func(example):
    MAX_LENGTH = 500    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|user|>\n{example['instruction'] + example['input']}<|end|>\n<|assistant|>\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<|end|>\n", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenizer = AutoTokenizer.from_pretrained('/data2/yixu/LLM-Research/Phi-3-mini-4k-instruct', use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

model = AutoModelForCausalLM.from_pretrained('/data2/yixu/LLM-Research/Phi-3-mini-4k-instruct', device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True)
model.enable_input_require_grads()
print(model.dtype)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)

model = get_peft_model(model, config)


args = TrainingArguments(
    output_dir="output/Phi-3",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

lora_path='./Phi-3_lora_729'
trainer.model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)