#Presented by KeJi
#Date : 2026-01-14

import re
import csv
from pathlib import Path

# 设备映射
DEVICE_MAP = {
    'testlaptop.log': 'Intel(R) Core(TM) Ultra 9 285H 16GB',
    'testlaptop2.log': 'Intel(R) Core? 5 220H 8GB',
    'testpi.log': 'ARM64_Raspberry_Pi_5',
    'testrk3588.log': 'RK3588 8G'
}

# 优化配置列表
CONFIGS = [
    'Python-Original', 'Python-Fixlen', 'Python-Fused', 'Python-Fused-Fixlen',
    'CPP-Original', 'CPP-Fixlen', 'CPP-Fused', 'CPP-Fused-Fixlen',
    'FullCPP-Original', 'FullCPP-Fixlen', 'FullCPP-Fused', 'FullCPP-Fused-Fixlen',
    'SIMD', 'SIMD-Fixlen', 'SIMD-Fused', 'SIMD-Fused-Fixlen'
]

# 模型分组
PARAM_MODELS = ['vim_5m', 'vim_tiny', 'vim_10m', 'vim_15m', 'vim_20m']
FLOPS_MODELS = ['vim_2gflops', 'vim_3gflops', 'vim_4gflops', 'vim_5gflops']
ALL_MODELS = PARAM_MODELS + FLOPS_MODELS

def Parse_Log_File(file_path):
    """解析单个log文件，提取各模型各配置的时间数据"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    results = {}
    current_model = None
    
    # 正则匹配模型名称
    model_pattern = re.compile(r'模型[:\s]*(\w+)')
    # 匹配unicode编码的模型名称
    model_pattern_unicode = re.compile(r'\\u6a21\\u578b[:\s]*(\w+)')
    # 正则匹配时间数据
    time_pattern = re.compile(r'^\s*([\w-]+)\s+([\d.]+)ms\s+\(min:', re.MULTILINE)
    
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 检测模型名称
        for model in ALL_MODELS:
            if f'模型: {model}' in line or f'\\u6a21\\u578b: {model}' in line or f': {model}' in line:
                current_model = model
                results[current_model] = {}
                break
        
        # 提取时间数据
        if current_model:
            match = time_pattern.match(line)
            if match:
                config_name = match.group(1)
                time_ms = float(match.group(2))
                if config_name in CONFIGS:
                    results[current_model][config_name] = time_ms
        
        i += 1
    
    return results


def Generate_Csv(output_path):
    """生成包含所有模型子表格的CSV文件"""
    devices = list(DEVICE_MAP.values())
    log_dir = Path(__file__).parent
    
    # 收集所有数据
    all_data = {}
    for log_file, device in DEVICE_MAP.items():
        file_path = log_dir / log_file
        if file_path.exists():
            all_data[device] = Parse_Log_File(file_path)
        else:
            print(f"警告: 文件 {log_file} 不存在")
            all_data[device] = {}
    
    # 写入CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 按参数量模型写入
        writer.writerow(['=== 按参数量分类的模型 ==='])
        writer.writerow([])
        
        for model in PARAM_MODELS:
            # 获取参数量信息
            param_info = Get_Model_Info(model, all_data)
            writer.writerow([f'模型: {model} ({param_info})'])
            writer.writerow(['配置'] + devices)
            
            for config in CONFIGS:
                row = [config]
                for device in devices:
                    if device in all_data and model in all_data[device]:
                        time_val = all_data[device][model].get(config, '')
                        row.append(f'{time_val:.2f}' if isinstance(time_val, float) else '')
                    else:
                        row.append('')
                writer.writerow(row)
            
            writer.writerow([])
        
        # 按FLOPs模型写入
        writer.writerow(['=== 按FLOPs分类的模型 ==='])
        writer.writerow([])
        
        for model in FLOPS_MODELS:
            param_info = Get_Model_Info(model, all_data)
            writer.writerow([f'模型: {model} ({param_info})'])
            writer.writerow(['配置'] + devices)
            
            for config in CONFIGS:
                row = [config]
                for device in devices:
                    if device in all_data and model in all_data[device]:
                        time_val = all_data[device][model].get(config, '')
                        row.append(f'{time_val:.2f}' if isinstance(time_val, float) else '')
                    else:
                        row.append('')
                writer.writerow(row)
            
            writer.writerow([])


def Get_Model_Info(model, all_data):
    """获取模型参数量信息"""
    MODEL_PARAMS = {
        'vim_5m': '5.46M',
        'vim_tiny': '7.15M',
        'vim_10m': '10.16M',
        'vim_15m': '15.04M',
        'vim_20m': '19.78M',
        'vim_2gflops': '9.46M/2G FLOPs',
        'vim_3gflops': '14.61M/3G FLOPs',
        'vim_4gflops': '17.97M/4G FLOPs',
        'vim_5gflops': '23.67M/5G FLOPs'
    }
    return MODEL_PARAMS.get(model, '')


if __name__ == '__main__':
    output_file = Path(__file__).parent / 'result_parsed.csv'
    Generate_Csv(output_file)
    print(f"结果已保存到: {output_file}")
