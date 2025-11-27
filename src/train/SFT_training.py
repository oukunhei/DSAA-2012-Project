import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

# 1. 加载模型和分词器
model_path = "D:\\DSAA-2012-Project\\models\\Qwen2.5-Coder-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto"
)

print("模型和分词器加载完成。")

# 设置pad_token（如果不存在）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 配置LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. 加载和格式化数据集
def load_and_format_dataset(file_path):
    """加载数据并格式化为训练文本"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_texts = []
    for item in data:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')
        
        # 构建训练文本格式
        if input_text.strip():
            # 如果有input内容
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}{tokenizer.eos_token}"
        else:
            # 如果input为空
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}{tokenizer.eos_token}"
        
        formatted_texts.append(text)
    
    return Dataset.from_dict({"text": formatted_texts})

# 加载数据集
dataset = load_and_format_dataset("data/train_set/SFT_set.json")

# 4. 设置训练参数
training_args = SFTConfig(
    output_dir="./sft_results",
    per_device_train_batch_size=2,  # 对于1.5B模型可以适当调整
    gradient_accumulation_steps=4,
    learning_rate=1e-4,  # LoRA通常使用稍大的学习率
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=500,
    warmup_ratio=0.1,
    fp16=True,
    remove_unused_columns=False,
    report_to=None,  # 禁用wandb等记录器
)

# 5. 创建SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)

# 开始训练！
print("开始训练...")
print(f"训练样本数: {len(dataset)}")
print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

trainer.train()

# 保存最终模型
trainer.save_model("models/sft_final_model")
print("训练完成！模型已保存到 models/sft_final_model")