#Presented by KeJi
#Date: 2026-01-30

"""
Parse random_search.json and extract params/accuracy to compare.xlsx Random sheet.
"""

import json
import pandas as pd
from openpyxl import load_workbook
import argparse
import os

# 常量
DEFAULT_JSON_PATH = "random_search.json"
DEFAULT_XLSX_PATH = "compare.xlsx"
SHEET_NAME = "Random"


def Parse_Random_Search(json_path: str) -> pd.DataFrame:
    """
    解析random_search.json，提取参数量和准确率
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        DataFrame包含Params和Accuracy两列
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取params和accuracy
    records = []
    for item in data:
        records.append({
            'Params': item['params'],
            'Accuracy': item['accuracy']
        })
    
    df = pd.DataFrame(records)
    
    # 按参数量排序
    df = df.sort_values('Params', ascending=True).reset_index(drop=True)
    
    return df


def Write_To_Xlsx(df: pd.DataFrame, xlsx_path: str, sheet_name: str):
    """
    将数据写入xlsx文件的指定sheet
    
    Args:
        df: 要写入的DataFrame
        xlsx_path: xlsx文件路径
        sheet_name: sheet名称
    """
    if os.path.exists(xlsx_path):
        # 文件存在，追加sheet
        with pd.ExcelWriter(xlsx_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # 文件不存在，创建新文件
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def main():
    parser = argparse.ArgumentParser(description='Parse random_search.json to xlsx')
    parser.add_argument('-i', '--input', type=str, default=DEFAULT_JSON_PATH,
                        help=f'Input JSON file path (default: {DEFAULT_JSON_PATH})')
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_XLSX_PATH,
                        help=f'Output xlsx file path (default: {DEFAULT_XLSX_PATH})')
    parser.add_argument('-s', '--sheet', type=str, default=SHEET_NAME,
                        help=f'Sheet name (default: {SHEET_NAME})')
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建完整路径
    json_path = os.path.join(script_dir, args.input) if not os.path.isabs(args.input) else args.input
    xlsx_path = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output
    
    print(f"读取JSON: {json_path}")
    df = Parse_Random_Search(json_path)
    
    print(f"提取到 {len(df)} 条记录")
    print(f"参数量范围: {df['Params'].min():,} - {df['Params'].max():,}")
    print(f"准确率范围: {df['Accuracy'].min():.2f}% - {df['Accuracy'].max():.2f}%")
    
    print(f"\n写入到: {xlsx_path} (sheet: {args.sheet})")
    Write_To_Xlsx(df, xlsx_path, args.sheet)
    
    print("完成!")


if __name__ == "__main__":
    main()
