import os
from pathlib import Path
from huggingface_hub import snapshot_download
import shutil

model_name = "XGenerationLab/XiYanSQL-QwenCoder-7B-2504"
model_cache_dir = "data/yunxiou/models"
model_short_name = model_name.split('/')[-1]
model_path = Path(model_cache_dir) / model_short_name

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

def cleanup_before_download(model_path):
    """下载前清理不完整文件"""
    # 删除 .incomplete 文件
    incomplete_files = list(model_path.rglob('*.incomplete'))
    for f in incomplete_files:
        try:
            f.unlink()
            print(f"已删除不完整文件: {f}")
        except:
            pass
    
    # 删除很小的 .safetensors 文件（可能是损坏的）
    small_safetensors = [f for f in model_path.rglob('*.safetensors') if f.exists() and f.stat().st_size < 1024 * 1024]
    for f in small_safetensors:
        try:
            f.unlink()
            print(f"已删除可能损坏的文件: {f} (大小: {f.stat().st_size} bytes)")
        except:
            pass

def download_model():
    """下载模型的主函数"""
    
    # 如果目录存在但可能不完整，先清理
    if model_path.exists():
        print("检测到已存在的模型目录，检查完整性...")
        cleanup_before_download(model_path)
    
    try:
        print("开始下载模型...")
        snapshot_download(
            repo_id=model_name,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
            retries=5,
        )
        print("下载完成!")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False

# 执行下载
download_model()