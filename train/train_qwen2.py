# train_qwen_dpo.py
import os
import random
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
)
from trl import DPOTrainer, DPOConfig
import numpy as np

# ----------------------------------------------------------------------
# 配置参数
# ----------------------------------------------------------------------
MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-Coder-7B-Instruct"
DATA_FILE_PATH = "train_judge_dpo.jsonl" 
OUTPUT_DIR = "./qwen7b_judge_dpo_adapter"
NUM_TRAIN_EPOCHS = 1

# ----------------------------------------------------------------------
# 优化的数据格式化函数
# ----------------------------------------------------------------------
def format_dpo_dataset(example: dict) -> dict:
    """优化版本的数据格式化"""
    
    question = example['question']
    chosen_sql = example['chosen']['sql']
    chosen_ast = example['chosen']['ast']
    rejected_sql = example['rejected']['sql']
    rejected_ast = example['rejected']['ast']

    # 随机化顺序
    if random.random() < 0.5:
        sql_a, ast_a = chosen_sql, chosen_ast
        sql_b, ast_b = rejected_sql, rejected_ast
        chosen_response = "A"
        rejected_response = "B"
    else:
        sql_a, ast_a = rejected_sql, rejected_ast
        sql_b, ast_b = chosen_sql, chosen_ast
        chosen_response = "B"
        rejected_response = "A"
        
    # 构建prompt - 使用更简洁的模板
    user_prompt = f"""Task: Evaluate two SQL queries given a question. Choose the query (A or B) that correctly answers the question.

Question: {question}

Query [A]: {sql_a}
AST [A]: {ast_a}

Query [B]: {sql_b}  
AST [B]: {ast_b}

Which query is correct (A or B)? Respond with a single letter (A or B) only."""

    final_prompt_string = (
        f"<|im_start|>system\n"
        f"You are an expert SQL analyst. Your task is to judge the correctness of SQL queries.\n<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    return {
        "prompt": final_prompt_string,
        "chosen": chosen_response,
        "rejected": rejected_response
    }

# ----------------------------------------------------------------------
# 预格式化数据集（减少训练时的处理开销）
# ----------------------------------------------------------------------
def preprocess_dataset():
    """预处理数据集并保存到磁盘"""
    processed_data_path = "processed_dpo_dataset"
    
    if os.path.exists(processed_data_path):
        print(f"加载已预处理的数据集: {processed_data_path}")
        return load_dataset("json", data_files=processed_data_path, split="train")
    
    print("预处理数据集...")
    dataset = load_dataset("json", data_files=DATA_FILE_PATH, split="train")
    
    # 预处理所有数据
    processed_data = []
    for example in dataset:
        processed_data.append(format_dpo_dataset(example))
    
    # 保存预处理后的数据
    import json
    with open(processed_data_path, 'w') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"数据集已预处理并保存到: {processed_data_path}")
    return load_dataset("json", data_files=processed_data_path, split="train")

# ----------------------------------------------------------------------
# 主训练流程
# ----------------------------------------------------------------------
def main():
    # 检查现有检查点
    output_path = Path(OUTPUT_DIR)
    resume_from_checkpoint = False
    if output_path.exists():
        checkpoints = list(output_path.glob("checkpoint-*"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"将从中断处继续训练: {latest_checkpoint}")
            resume_from_checkpoint = str(latest_checkpoint)

    print("--- 1. 加载和预处理数据集 ---")
    dataset = preprocess_dataset()
    print(f"数据集大小: {len(dataset)}")

    # 数据集分片（多GPU训练）
    if torch.cuda.device_count() > 1:
        dataset = dataset.shard(num_shards=torch.cuda.device_count(), index=0)
        print(f"分片后数据集大小: {len(dataset)}")

    print("--- 2. 加载模型 ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.config.use_cache = False
    # 让梯度检查点正常工作
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    print("--- 3. 加载 Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    print("--- 4. 配置 LoRA ---")
    peft_config = LoraConfig(
        r=32,  # 增加rank以获得更好性能
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    print("--- 5. 配置优化的训练参数 ---")
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        # 显著增加批次大小以充分利用A800显存
        per_device_train_batch_size=16,  # 根据您的数据长度调整
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # 总批次大小 = 8 * 8 * 2(GPU) = 128
        
        # 学习率配置
        learning_rate=1e-4,  # 稍微提高学习率
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        
        # 训练配置
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=-1,
        
        # 优化器配置
        optim="adamw_torch_fused",  # 使用融合优化器
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        fp16=True,
        
        # 日志和保存
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        
        # 数据加载优化
        dataloader_pin_memory=True,
        dataloader_num_workers=4,  # 增加数据加载 workers
        dataloader_prefetch_factor=2,
        
        # DPO 特定配置
        beta=0.1,
        max_prompt_length=1024,  # 根据实际数据调整
        max_length=1152,
        
        # 梯度配置
        max_grad_norm=0.5,
        gradient_checkpointing=True,
        
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    print("--- 6. 初始化 DPOTrainer ---")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config
    )

    print("--- 7. 🚀 开始训练 ---")
    if hasattr(torch, "compile") and os.name != "nt":
        dpo_trainer.model = torch.compile(dpo_trainer.model)
    else:
        print("Skip torch.compile on Windows (MSVC cl not found).")

    if torch.cuda.is_available():
        print(f"CUDA available. devices={torch.cuda.device_count()}, name={torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Training on CPU -> 低 GPU 利用率是预期。")


    dpo_trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print("--- 8. 保存模型 ---")
    dpo_trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ 训练完成！模型保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    # 设置环境变量优化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    main()