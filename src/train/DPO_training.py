from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import optim
from trl import DPOConfig, DPOTrainer
from datasets import Dataset
import json

model_path = "DSAA-2012-Project/models/sft_final_model"
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

base_model_path = "DSAA-2012-Project/models/Qwen2.5-Coder-1.5B"
model_path = "DSAA-2012-Project/models/sft_final_model"

# 加载基础模型和适配器
try:
    # 首先加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # 从适配器配置中获取基础模型名称
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_config.base_model_name_or_path
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(
        base_model,
        model_path,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("模型加载成功！")
    
except Exception as e:
    print(f"加载失败: {e}")

# 加载数据集
with open ("DSAA-2012-Project/data/train_set/DPO_set.json", "r", encoding="utf-8") as f:
    data = json.load(f)
dataset = Dataset.from_list(data)

training_args = DPOConfig(
    max_length=1024,
    max_prompt_length=512,
    
    # DPO核心参数
    beta=0.1,  # KL惩罚系数[citation:1]
    loss_type="sigmoid",
    
    # 训练超参数
    learning_rate=5e-6,  # 使用较低的学习率[citation:9]
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,  # 训练轮数[citation:1]
    
    # 训练设置
    output_dir="DSAA-2012-Project/models/Qwen2.5-DPO-1.5B",
    fp16=True,  # 如果硬件支持，可以启用混合精度训练[citation:9]
    logging_steps=10,
    save_steps=500,
    remove_unused_columns=False,
)

optimizer = optim.SGD(model.parameters(), lr=training_args.learning_rate)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    optimizers=(optimizer, None),
)
trainer.train()
