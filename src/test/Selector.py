from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from typing import List

import os

# 加载基础模型
base_model_path = "models/Qwen2.5-Coder-1.5B"
dpo_dir = "models/Qwen2.5-DPO-1.5B/checkpoint-57"
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 加载DPO训练后的适配器
dpo_model = PeftModel.from_pretrained(base_model, dpo_dir)

# 合并适配器到基础模型
merged_model = dpo_model.merge_and_unload()

# 保存完整模型


merged_output_dir = "models/Qwen2.5-DPO-1.5B-merged"
merged_model.save_pretrained(merged_output_dir)
tokenizer.save_pretrained(merged_output_dir)
print(f"完整模型已保存到: {merged_output_dir}")

def build_nl2sql_prompt(nl_question: str, schema_info: str, candidate_sqls: List[str]) -> str:
    prompt_path = Path("src/templete/prompt.txt")
    template = prompt_path.read_text(encoding='utf-8')

    return template.format(nl_question=nl_question, schema_info=schema_info, candidate_sqls=candidate_sqls)


class Selector:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
    
    def load_models(self):
        """加载模型，如果已加载则直接返回"""
        if self._is_loaded:
            return self.model, self.tokenizer
            
        print("正在加载模型...")
        base_model_path = "models/Qwen2.5-Coder-1.5B"
        dpo_dir = "models/Qwen2.5-DPO-1.5B/checkpoint-57"
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        dpo_model = PeftModel.from_pretrained(base_model, dpo_dir)
        self.model = dpo_model.merge_and_unload()
        self.tokenizer = tokenizer
        self._is_loaded = True
        
        # 保存模型（可选）
        merged_output_dir = "models/Qwen2.5-DPO-1.5B-merged"
        self.model.save_pretrained(merged_output_dir)
        self.tokenizer.save_pretrained(merged_output_dir)
        
        return self.model, self.tokenizer
    
    def invoke(self, nl_question: str, schema_info: str, candidate_sqls: List[str]) -> str:
        model, tokenizer = self.load_models()
        prompt = build_nl2sql_prompt(nl_question, schema_info, candidate_sqls)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100)
        return tokenizer.decode(outputs[0])

if __name__ == "__main__":
    # 使用方式
    model_instance = NL2SQLModel()

    # 多次调用，模型只会加载一次
    result1 = model_instance.invoke("查询所有用户", "users表", ["SELECT * FROM users"])
    result2 = model_instance.invoke("查询订单数量", "orders表", ["SELECT COUNT(*) FROM orders"])