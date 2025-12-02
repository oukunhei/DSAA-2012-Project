import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from typing import List

import os

def build_nl2sql_prompt(nl_question: str, schema_info: str, candidate_sqls: List[str]) -> str:
    prompt_path = Path("src/templete/prompt.txt")
    template = prompt_path.read_text(encoding='utf-8')

    return template.format(nl_question=nl_question, schema_info=schema_info, candidate_sqls=candidate_sqls)


class Selector:
    def __init__(self, 
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 do_sample: bool = True,
                 **kwargs):

        self.model = None
        self.tokenizer = None
        self._is_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"模型将加载到设备: {self.device}")
        
        # 保存生成参数
        self.generation_config = {
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'do_sample': do_sample,
            **kwargs
        }
    
    def load_models(self):
        """加载模型，如果已加载则直接返回"""
        if self._is_loaded:
            return self.model, self.tokenizer
            
        print("正在加载模型...")
        
        # 检查是否已经存在合并后的模型
        merged_output_dir = "models/Qwen2.5-DPO-1.5B-merged"
        
        if os.path.exists(merged_output_dir):
            # 直接加载合并后的模型 - 添加 device_map 参数
            print(f"加载已合并的模型: {merged_output_dir}")
            self.model = AutoModelForCausalLM.from_pretrained(
                merged_output_dir,
                device_map="auto",  # 添加这一行
                torch_dtype=torch.float16  # 可选：使用半精度减少内存
            )
            self.tokenizer = AutoTokenizer.from_pretrained(merged_output_dir)
        else:
            # 加载基础模型并合并适配器
            base_model_path = "models/Qwen2.5-Coder-1.5B"
            dpo_dir = "models/Qwen2.5-DPO-1.5B/checkpoint-57"
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path, 
                dtype=torch.float16, 
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)

            dpo_model = PeftModel.from_pretrained(base_model, dpo_dir)
            self.model = dpo_model.merge_and_unload()
            self.tokenizer = tokenizer
            
            # 保存合并后的模型
            if not os.path.exists(merged_output_dir):
                os.makedirs(merged_output_dir, exist_ok=True)
                self.model.save_pretrained(merged_output_dir)
                self.tokenizer.save_pretrained(merged_output_dir)
                print(f"合并后的模型已保存到: {merged_output_dir}")
        
        self._is_loaded = True
        return self.model, self.tokenizer
    
    def invoke(self, nl_question: str, schema_info: str, candidate_sqls: List[str]) -> str:
        model, tokenizer = self.load_models()

            # 添加设备检查
        print(f"模型所在设备: {model.device}")
        print(f"是否有CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"当前GPU内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


        prompt = build_nl2sql_prompt(nl_question, schema_info, candidate_sqls)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        print(f"输入数据所在设备: {inputs['input_ids'].device}")

        input_length = inputs['input_ids'].shape[1]

        outputs = model.generate(**inputs, **self.generation_config)
        generated_only = outputs[0][input_length:]
        return tokenizer.decode(generated_only)

if __name__ == "__main__":
    # 使用方式
    model_instance =Selector(temperature=0.8, top_p=0.8)

    # 多次调用，模型只会加载一次
    result1 = model_instance.invoke("")
