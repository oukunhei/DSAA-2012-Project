import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import List

import os

def build_nl2sql_prompt(nl_question: str, schema_info: str, candidate_sqls: List[str]) -> str:
    prompt_path = Path("/ssd/yunxiou/DSAA-2012/prompt.txt")
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
        merged_output_dir = "/ssd/yunxiou/DSAA-2012/Qwen2.5-DPO-1.5B-merged"
        
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
            print("model not found. ")
        
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
        outputs = model.generate(**inputs, max_length=100)
        return tokenizer.decode(outputs[0])

if __name__ == "__main__":
    # 使用方式
    model_instance =Selector(temperature=0.8, top_p=0.8)

    # 多次调用，模型只会加载一次
    result1 = model_instance.invoke("")
