#Presented by KeJi
#Date : 2026-01-22

"""
提取random_search_all_results.json中的模型配置信息
写入result.xlsx的新sheet
"""

import json
import pandas as pd
from pathlib import Path

def Extract_Random_Config(json_path: str, xlsx_path: str, sheet_name: str = "RandomSearch"):
    """
    从json文件提取params, flops, accuracy，写入xlsx新sheet
    JSON文件有损坏，直接用正则提取
    """
    import re
    with open(json_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 正则提取每个模型的accuracy, params, flops
    # 模式: "accuracy": xxx, "params": xxx, "flops": xxx
    pattern = r'"accuracy":\s*([\d.]+),\s*"params":\s*(\d+),\s*"flops":\s*([\d.]+)'
    matches = re.findall(pattern, content)
    
    data = []
    for acc, params, flops in matches:
        data.append({
            'accuracy': float(acc),
            'params': int(params),
            'flops': float(flops)
        })
    
    # 提取信息
    records = []
    for i, item in enumerate(data):
        record = {
            'Model_ID': i + 1,
            'Params': item['params'],
            'FLOPs': item['flops'],
            'Accuracy': item['accuracy']
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # 格式化
    df['Params_M'] = df['Params'] / 1e6  # 参数量(M)
    df['FLOPs_G'] = df['FLOPs'] / 1e9    # FLOPs(G)
    
    # 重排列
    df = df[['Model_ID', 'Params', 'Params_M', 'FLOPs', 'FLOPs_G', 'Accuracy']]
    
    # 写入xlsx
    xlsx_file = Path(xlsx_path)
    if xlsx_file.exists():
        with pd.ExcelFile(xlsx_path) as xls:
            existing_sheets = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}
    else:
        existing_sheets = {}
    
    existing_sheets[sheet_name] = df
    
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for sheet, sheet_df in existing_sheets.items():
            sheet_df.to_excel(writer, sheet_name=sheet, index=False)
    
    print(f"提取 {len(records)} 条记录")
    print(f"已保存至 {xlsx_path} - sheet: {sheet_name}")
    print(f"\nParams范围: {df['Params_M'].min():.2f}M - {df['Params_M'].max():.2f}M")
    print(f"FLOPs范围: {df['FLOPs_G'].min():.2f}G - {df['FLOPs_G'].max():.2f}G")
    print(f"Accuracy范围: {df['Accuracy'].min():.2f}% - {df['Accuracy'].max():.2f}%")
    
    return df

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    json_path = script_dir / "random_search_all_results.json"
    xlsx_path = script_dir / "result.xlsx"
    
    df = Extract_Random_Config(str(json_path), str(xlsx_path))
    print("\n数据预览:")
    print(df.to_string(index=False))
