# cd /root/DSAA-2012
# rm models/Qwen2.5-Coder-3B/model-00002-of-00002.safetensors
# ls -lh models/Qwen2.5-Coder-3B/

from huggingface_hub import hf_hub_download
import os

repo_id = "Qwen/Qwen2.5-Coder-3B"
local_dir = "models/Qwen2.5-Coder-3B"

# 要下载的文件名
corrupted_file = "model-00002-of-00002.safetensors"

print(f"重新下载损坏的分片: {corrupted_file}")
print("=" * 50)

# 检查文件是否已删除
local_path = os.path.join(local_dir, corrupted_file)
if os.path.exists(local_path):
    print(f"警告: 文件已存在，大小: {os.path.getsize(local_path):,} bytes")
    choice = input("是否删除并重新下载？(y/n): ")
    if choice.lower() == 'y':
        os.remove(local_path)
        print("已删除旧文件")
    else:
        print("取消下载")
        exit()

# 重新下载
try:
    print("开始下载...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=corrupted_file,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,    # 支持断点续传
        force_download=False,    # 如果文件不存在则下载
    )
    
    # 验证下载的文件
    if os.path.exists(downloaded_path):
        file_size = os.path.getsize(downloaded_path)
        print(f"✓ 下载完成!")
        print(f"  文件路径: {downloaded_path}")
        print(f"  文件大小: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        # 预期大小应该和第一个分片类似（但可能稍小）
        shard1 = os.path.join(local_dir, "model-00001-of-00002.safetensors")
        if os.path.exists(shard1):
            shard1_size = os.path.getsize(shard1)
            print(f"  第一个分片大小: {shard1_size:,} bytes")
            print(f"  比例: {file_size/shard1_size*100:.1f}%")
            
except Exception as e:
    print(f"下载失败: {e}")
    print("\n尝试强制重新下载...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=corrupted_file,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=False,   # 不续传，重新开始
            force_download=True,     # 强制下载
        )
        print("✓ 强制下载完成!")
    except Exception as e2:
        print(f"强制下载也失败: {e2}")
