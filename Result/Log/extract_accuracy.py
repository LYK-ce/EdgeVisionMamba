#Presented by KeJi
#Date : 2026-01-22

import re
import json
import pandas as pd
from pathlib import Path

def Parse_Json_Log(file_path):
    """解析JSON格式的log.txt"""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if 'epoch' in data and 'test_acc1' in data:
                        records.append({
                            'epoch': data.get('epoch', 0),
                            'test_acc1': data.get('test_acc1', 0),
                            'test_acc5': data.get('test_acc5', 0),
                            'train_loss': data.get('train_loss', 0),
                            'test_loss': data.get('test_loss', 0),
                            'train_lr': data.get('train_lr', 0)
                        })
                except json.JSONDecodeError:
                    continue
    return records

def Parse_Training_Log(file_path):
    """解析标准训练日志格式（vim_tiny, vim_small等）"""
    records = []
    current_epoch = -1
    
    epoch_pattern = re.compile(r'Epoch: \[(\d+)\]')
    acc_pattern = re.compile(r'\* Acc@1 ([\d.]+) Acc@5 ([\d.]+) loss ([\d.]+)')
    network_acc_pattern = re.compile(r'Accuracy of the network on the \d+ test images: ([\d.]+)%')
    max_acc_pattern = re.compile(r'Max accuracy: ([\d.]+)%')
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        acc_match = acc_pattern.search(line)
        if acc_match:
            acc1 = float(acc_match.group(1))
            acc5 = float(acc_match.group(2))
            loss = float(acc_match.group(3))
            
            network_acc = None
            max_acc = None
            for j in range(i+1, min(i+5, len(lines))):
                net_match = network_acc_pattern.search(lines[j])
                max_match = max_acc_pattern.search(lines[j])
                if net_match:
                    network_acc = float(net_match.group(1))
                if max_match:
                    max_acc = float(max_match.group(1))
            
            records.append({
                'epoch': current_epoch,
                'test_acc1': acc1,
                'test_acc5': acc5,
                'test_loss': loss,
                'network_acc': network_acc,
                'max_acc': max_acc
            })
        
        i += 1
    
    return records

def Parse_Autovim_Log(file_path):
    """解析autovim多配置训练日志"""
    records = []
    current_epoch = -1
    
    epoch_pattern = re.compile(r'Epoch: \[(\d+)\]')
    acc_pattern = re.compile(r'\* Acc@1 ([\d.]+) Acc@5 ([\d.]+) loss ([\d.]+)')
    config_acc_pattern = re.compile(r'Accuracy of the network on the \d+ test images on config (\d+): ([\d.]+)%')
    all_acc_pattern = re.compile(r'Accuracy of the network on the \d+ test images: ([\d.]+)%')
    max_acc_pattern = re.compile(r'Max accuracy: ([\d.]+)%')
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    i = 0
    config_results = {}
    
    while i < len(lines):
        line = lines[i]
        
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        acc_match = acc_pattern.search(line)
        if acc_match:
            acc1 = float(acc_match.group(1))
            acc5 = float(acc_match.group(2))
            loss = float(acc_match.group(3))
            
            for j in range(i+1, min(i+10, len(lines))):
                config_match = config_acc_pattern.search(lines[j])
                if config_match:
                    config_id = int(config_match.group(1))
                    config_acc = float(config_match.group(2))
                    
                    key = (current_epoch, config_id)
                    config_results[key] = {
                        'epoch': current_epoch,
                        'config_id': config_id,
                        'test_acc1': acc1,
                        'test_acc5': acc5,
                        'test_loss': loss,
                        'config_acc': config_acc
                    }
                    break
                
                all_match = all_acc_pattern.search(lines[j])
                if all_match and not config_acc_pattern.search(lines[j]):
                    network_acc = float(all_match.group(1))
                    records.append({
                        'epoch': current_epoch,
                        'config_id': None,
                        'test_acc1': acc1,
                        'test_acc5': acc5,
                        'test_loss': loss,
                        'network_acc': network_acc
                    })
                    break
        
        i += 1
    
    for key in sorted(config_results.keys()):
        records.append(config_results[key])
    
    return records

def Extract_All_Logs(log_dir):
    """提取目录中所有日志文件的准确率数据"""
    log_dir = Path(log_dir)
    all_data = {}
    
    log_txt = log_dir / 'log.txt'
    if log_txt.exists():
        records = Parse_Json_Log(log_txt)
        if records:
            all_data['auto_vim_json'] = pd.DataFrame(records)
    
    autovim_files = [
        'autovim.log',
        'autovim_96.log',
        'autovim_160.log',
        'autovim_224.log'
    ]
    
    for filename in autovim_files:
        file_path = log_dir / filename
        if file_path.exists():
            records = Parse_Autovim_Log(file_path)
            if records:
                sheet_name = filename.replace('.log', '').replace('.', '_')
                all_data[sheet_name] = pd.DataFrame(records)
    
    standard_logs = [
        '2025.12.17_vimsmall_Image.log',
        'output_vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2.log'
    ]
    
    for filename in standard_logs:
        file_path = log_dir / filename
        if file_path.exists():
            records = Parse_Training_Log(file_path)
            if records:
                if 'vimsmall' in filename.lower():
                    sheet_name = 'vim_small'
                elif 'vim_tiny' in filename.lower():
                    sheet_name = 'vim_tiny'
                else:
                    sheet_name = filename[:30].replace('.', '_')
                all_data[sheet_name] = pd.DataFrame(records)
    
    random_search_file = log_dir / 'random_search_all_results.json'
    if random_search_file.exists():
        try:
            with open(random_search_file, 'r', encoding='utf-8') as f:
                search_data = json.load(f)
            if isinstance(search_data, list) and search_data:
                all_data['random_search'] = pd.DataFrame(search_data)
        except (json.JSONDecodeError, Exception):
            pass
    
    return all_data

def Save_To_Excel(data_dict, output_path):
    """保存数据到Excel文件"""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, df in data_dict.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    print(f"数据已保存到: {output_path}")

def main():
    log_dir = Path(__file__).parent
    output_path = log_dir / 'result.xlsx'
    
    print("正在提取日志文件中的准确率数据...")
    all_data = Extract_All_Logs(log_dir)
    
    if all_data:
        print(f"成功提取 {len(all_data)} 个日志文件的数据:")
        for name, df in all_data.items():
            print(f"  - {name}: {len(df)} 条记录")
        
        Save_To_Excel(all_data, output_path)
    else:
        print("未找到任何可用的日志数据")

if __name__ == '__main__':
    main()
