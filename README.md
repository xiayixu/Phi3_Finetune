# Phi3_Finetune

# 使用指南

### 功能概述
`phi3.py` 文件用于对 `Phi-3` 模型进行微调。它包含了数据处理、模型配置以及训练的完整流程。

### 使用步骤
1. **数据准备**:
   - 将数据集保存为 JSON 文件格式，路径为 `/data2/yixu/phi3/mrc_阅读理解问答_7_29_微调.json`。
   - 数据集将通过 `process_func` 函数进行处理，以适应模型的输入需求。

2. **模型配置**:
   - 使用 `AutoTokenizer` 和 `AutoModelForCausalLM` 加载预训练的 `Phi-3-mini-4k-instruct` 模型。
   - 配置 `LoRA`，这是模型微调的重要部分，包括 `r`, `lora_alpha`, 和 `lora_dropout` 等参数。

3. **模型训练**:
   - 使用 `TrainingArguments` 配置训练参数，例如批量大小、学习率、保存频率等。
   - 通过 `Trainer` 对模型进行训练，并在训练完成后保存微调后的模型。

4. **保存模型**:
   - 微调后的模型和分词器将被保存到指定路径 `./Phi-3_lora_729`。

## 2. `inference.py` 文件

### 功能概述
`inference.py` 文件用于部署一个 Flask 应用，通过 RESTful API 实现对 `Phi-3` 模型的推理调用。

### 使用步骤
1. **启动服务**:
   - 运行脚本将启动一个 Flask 服务器，监听 `1980` 端口。

2. **发送请求**:
   - 客户端可以通过 POST 请求访问 `/inference_phi3` 接口，传递 `instruction` 和 `prompt`，获取模型生成的响应。

3. **模型加载**:
   - 加载的模型路径为 `/data2/yixu/LLM-Research/Phi-3-mini-4k-instruct`，并使用之前微调好的 `LoRA` 模型参数进行推理。

## 3. `test_phi3.py` 文件

### 功能概述
`test_phi3.py` 文件用于测试模型推理结果，并将结果保存为 CSV 文件。

### 使用步骤
1. **加载数据**:
   - 从指定的 JSON 文件路径加载测试数据，提取出 `instruction`，`input`，和 `output`。

2. **调用推理接口**:
   - 通过 HTTP POST 请求调用 `inference.py` 中的接口，获取模型生成的回答。

3. **保存结果**:
   - 将问题、模型生成的答案和标准答案保存为 CSV 文件，默认文件名为 `q_not_in_system_1w.csv`。
