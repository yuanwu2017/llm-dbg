import os
import torch
import numpy as np
import sys
from collections.abc import Mapping, Sequence

def compare_pt_files(dir1, dir2, rtol=1e-5, atol=1e-8, output_file=None):
    """
    详细比较两个文件夹中的同名 .pt 文件差异
    
    Args:
        dir1 (str): 第一个文件夹路径
        dir2 (str): 第二个文件夹路径
        rtol (float): 相对容差
        atol (float): 绝对容差
        output_file (str, optional): 输出文件路径，不指定则输出到控制台
    """
    # 如果指定了输出文件，重定向stdout
    original_stdout = sys.stdout
    if output_file:
        try:
            sys.stdout = open(output_file, 'w', encoding='utf-8')
        except Exception as e:
            print(f"无法打开输出文件 {output_file}: {e}")
            sys.stdout = original_stdout
            return
    
    try:
        # 获取文件列表
        files1 = {f for f in os.listdir(dir1) if f.endswith('.pt')}
        files2 = {f for f in os.listdir(dir2) if f.endswith('.pt')}
        
        common_files = files1 & files2
        only_in_dir1 = files1 - files2
        only_in_dir2 = files2 - files1

        print(f"\n{'='*60}")
        print(f"文件夹比对: \n- {dir1}\n- {dir2}")
        print(f"\n共有文件: {len(common_files)} 个")
        print(f"仅在 {dir1} 中的文件: {len(only_in_dir1)} 个")
        print(f"仅在 {dir2} 中的文件: {len(only_in_dir2)} 个")
        print('='*60)

        # 比较共有文件
        for filename in sorted(common_files):
            file1 = os.path.join(dir1, filename)
            file2 = os.path.join(dir2, filename)
            
            print(f"\n🔍 正在比较文件: {filename}")
            
            try:
                data1 = torch.load(file1)
                data2 = torch.load(file2)
                
                # 比较两个对象
                compare_objects(data1, data2, rtol, atol, indent="  ")
                
            except Exception as e:
                print(f"  ❌ 文件加载失败: {str(e)}")
                continue
    finally:
        # 恢复标准输出
        if output_file and sys.stdout != original_stdout:
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"比对结果已保存到: {output_file}")

def compare_objects(obj1, obj2, rtol, atol, indent=""):
    """递归比较两个Python对象"""
    if type(obj1) != type(obj2):
        print(f"{indent}类型不同: {type(obj1)} vs {type(obj2)}")
        return
    
    if isinstance(obj1, torch.Tensor):
        compare_tensors(obj1, obj2, rtol, atol, indent)
    elif isinstance(obj1, (str, int, float, bool)) or obj1 is None:
        if obj1 != obj2:
            print(f"{indent}值不同: {obj1} != {obj2}")
    elif isinstance(obj1, Mapping):
        compare_dicts(obj1, obj2, rtol, atol, indent)
    elif isinstance(obj1, Sequence) and not isinstance(obj1, str):
        compare_sequences(obj1, obj2, rtol, atol, indent)
    else:
        print(f"{indent}⚠️ 不支持比较的类型: {type(obj1)}")

def compare_tensors(t1, t2, rtol, atol, indent):
    """比较两个Tensor的差异"""
    print(f"{indent}├── 形状: {tuple(t1.shape)} vs {tuple(t2.shape)}")
    print(f"{indent}├── 数据类型: {t1.dtype} vs {t2.dtype}")
    print(f"{indent}├── 设备: {t1.device} vs {t2.device}")
    
    if t1.shape != t2.shape:
        print(f"{indent}└── ❌ 形状不同")
        print(f"{indent}  └── reshape")
        if len(t1.shape) != len(t2.shape):
            if len(t1.shape) > len(t2.shape):
                t1 = t1.view(t1.shape[1:])
            else:
                t2 = t2.view(t2.shape[1:])
            print(f"{indent}  └── 形状: {tuple(t1.shape)} vs {tuple(t2.shape)}")
        if t1.shape[0] > t2.shape[0]:
            t1 = t1[-t2.shape[0]:]
        else:
            t2 = t2[-t1.shape[0]:]
        print(f"{indent}  └── 形状: {tuple(t1.shape)} vs {tuple(t2.shape)}")
            
        
        #return
    
    if t1.dtype != t2.dtype:
        print(f"{indent}└── ❌ 数据类型不同")
        return
    
    # 计算差异统计
    diff = torch.abs(t1 - t2)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print(f"{indent}├── 最大差异: {max_diff:.6f}")
    print(f"{indent}├── 平均差异: {mean_diff:.6f}")
    
    # 找出差异最大的位置
    if max_diff > atol:
        idx = torch.argmax(diff)
        idx_tuple = np.unravel_index(idx.cpu().numpy(), t1.shape)
        print(f"{indent}├── 最大差异位置: {idx_tuple}")
        print(f"{indent}│   ├── 值1: {t1[idx_tuple].item():.6f}")
        print(f"{indent}│   └── 值2: {t2[idx_tuple].item():.6f}")
    
    # 检查是否所有元素都接近
    if torch.allclose(t1, t2, rtol=rtol, atol=atol):
        print(f"{indent}└── ✅ 所有值在容差范围内一致 (rtol={rtol}, atol={atol})")
    else:
        print(f"{indent}└── ❌ 存在超出容差的差异")

def compare_dicts(dict1, dict2, rtol, atol, indent):
    """比较两个字典的差异"""
    keys1, keys2 = set(dict1.keys()), set(dict2.keys())
    common_keys = keys1 & keys2
    only_in_1 = keys1 - keys2
    only_in_2 = keys2 - keys1
    
    print(f"{indent}字典键比较:")
    print(f"{indent}├── 共有键: {len(common_keys)} 个")
    print(f"{indent}├── 仅在第一个对象中的键: {only_in_1}")
    print(f"{indent}└── 仅在第二个对象中的键: {only_in_2}")
    
    for key in sorted(common_keys):
        print(f"{indent}比较键: '{key}'")
        compare_objects(dict1[key], dict2[key], rtol, atol, indent + "  ")

def compare_sequences(seq1, seq2, rtol, atol, indent):
    """比较两个序列的差异"""
    len1, len2 = len(seq1), len(seq2)
    print(f"{indent}序列长度: {len1} vs {len2}")
    
    if len1 != len2:
        print(f"{indent}└── ❌ 长度不同")
        return
    
    for i, (item1, item2) in enumerate(zip(seq1, seq2)):
        print(f"{indent}比较索引 {i}:")
        compare_objects(item1, item2, rtol, atol, indent + "  ")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='详细比较两个文件夹中的 .pt 文件差异')
    parser.add_argument('dir1', type=str, help='第一个文件夹路径')
    parser.add_argument('dir2', type=str, help='第二个文件夹路径')
    parser.add_argument('--rtol', type=float, default=1e-5, help='相对容差')
    parser.add_argument('--atol', type=float, default=1e-8, help='绝对容差')
    parser.add_argument('--output', '-o', type=str, help='输出文件路径，不指定则输出到控制台')
    
    args = parser.parse_args()
    
    compare_pt_files(args.dir1, args.dir2, args.rtol, args.atol, args.output)