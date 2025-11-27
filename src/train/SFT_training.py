import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from trl import SFTTrainer
from datasets import Dataset

# 1. 加载模型和分词器
model_path = "D:\\DSAA-2012-Project\\models\\Qwen2.5-Coder-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 设置pad_token（如果不存在）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 加载和预处理数据集
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def format_instruction(example):
    """将instruction, input, output格式化为模型输入"""
    if example['input'] and example['input'].strip():
        # 如果有input内容
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        # 如果input为空
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"text": text}

# 加载数据
train_data = load_dataset("sft_data.json")

# 转换为huggingface Dataset格式
formatted_data = [format_instruction(item) for item in train_data]
dataset = Dataset.from_list(formatted_data)

# 3. 数据预处理函数
def preprocess_function(examples):
    """tokenize数据，并确保只有output部分参与loss计算"""
    # tokenize文本
    model_inputs = tokenizer(
        examples["text"],
        max_length=1024,
        truncation=True,
        padding=False
    )
    
    # 创建labels副本（用于计算loss）
    labels = model_inputs["input_ids"].copy()
    
    # 关键步骤：mask掉instruction和input部分，只让output参与loss计算
    for i, text in enumerate(examples["text"]):
        # 找到"### Response:"的位置
        response_start = text.find("### Response:")
        if response_start != -1:
            # tokenize整个文本，找到response对应的token位置
            tokens = tokenizer.encode(text, add_special_tokens=False)
            response_tokens = tokenizer.encode(
                text[response_start:], 
                add_special_tokens=False
            )
            
            # 计算response的起始位置
            response_start_idx = len(tokens) - len(response_tokens)
            
            # 创建attention mask（前面部分设为0，不参与loss计算）
            for j in range(response_start_idx):
                labels[i][j] = -100
        else:
            # 如果没有找到Response，整个序列都参与训练（fallback）
            pass
            
    model_inputs["labels"] = labels
    return model_inputs

# 应用预处理
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)

# 4. 设置训练参数
training_args = TrainingArguments(
    output_dir="./sft_results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    warmup_steps=100,
    fp16=True,  # 如果GPU支持的话
    remove_unused_columns=False,
)

# 5. 创建Trainer并开始训练
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    ),
)

# 开始训练！
print("开始训练...")
trainer.train()

# 保存最终模型
trainer.save_model("./sft_final_model")
tokenizer.save_pretrained("./sft_final_model")
print("训练完成！模型已保存到 ./sft_final_model")