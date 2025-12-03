# diagnose_safetensors.py
import os
import json
from pathlib import Path
import struct

def check_safetensors_file(filepath):
    """检查safetensors文件完整性"""
    print(f"\n检查文件: {filepath}")
    
    if not os.path.exists(filepath):
        print("  文件不存在!")
        return False
    
    file_size = os.path.getsize(filepath)
    print(f"  文件大小: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    # 尝试读取文件头
    try:
        with open(filepath, 'rb') as f:
            # 读取前8个字节（头部长度）
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                print("  错误: 文件太小，无法读取头部长度")
                return False
            
            header_len = struct.unpack('<Q', header_len_bytes)[0]
            print(f"  头部长度: {header_len} bytes")
            
            # 读取JSON头部
            header_data = f.read(header_len)
            if len(header_data) < header_len:
                print(f"  错误: 头部数据不完整 ({len(header_data)}/{header_len})")
                return False
            
            # 尝试解析JSON
            header_json = header_data.decode('utf-8')
            header = json.loads(header_json)
            print(f"  ✓ JSON头部解析成功")
            print(f"  包含张量数量: {len(header)}")
            
            # 计算预期总大小
            total_tensor_size = 0
            for tensor_info in header.values():
                if isinstance(tensor_info, dict) and 'data_offsets' in tensor_info:
                    start, end = tensor_info['data_offsets']
                    total_tensor_size = max(total_tensor_size, end)
            
            expected_size = 8 + header_len + total_tensor_size
            print(f"  预期文件大小: {expected_size:,} bytes")
            
            if file_size >= expected_size:
                print(f"  ✓ 文件大小符合预期")
                return True
            else:
                print(f"  ✗ 文件大小不完整 (缺少 {expected_size - file_size:,} bytes)")
                return False
                
    except Exception as e:
        print(f"  错误: {e}")
        return False

# 检查所有safetensors文件
model_path = Path("models/Qwen2.5-Coder-3B")
print("=== 检查Qwen2.5-Coder-3B模型文件 ===")

if model_path.exists():
    safetensor_files = list(model_path.glob("*.safetensors"))
    if safetensor_files:
        print(f"找到 {len(safetensor_files)} 个safetensors文件")
        
        for i, file in enumerate(safetensor_files, 1):
            print(f"\n[{i}/{len(safetensor_files)}] ", end="")
            if check_safetensors_file(file):
                print(f"  ✓ {file.name} 完整")
            else:
                print(f"  ✗ {file.name} 损坏或不完整")
    else:
        print("没有找到safetensors文件")
else:
    print("模型目录不存在")
