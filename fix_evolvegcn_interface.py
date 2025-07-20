#!/usr/bin/env python3
"""
Fix evolvegcn_go_interface.py - Remove all non-ASCII characters
This tool will replace all Chinese characters and special Unicode characters
with ASCII-compatible markers to ensure BlockEmulator Go program can call it
"""

import re

def fix_evolvegcn_interface():
    """Fix evolvegcn_go_interface.py encoding issues"""
    
    file_path = "evolvegcn_go_interface.py"
    
    print(f"[FIX] Processing {file_path}...")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Define replacement mappings
    replacements = [
        # Common Chinese characters and phrases
        ('接口', '[INTERFACE]'),
        ('程序通过此脚本调用', 'program calls through this script'),
        ('算法', 'algorithm'),
        ('输入', 'Input'),
        ('格式的节点特征和交易图', 'format node features and transaction graph'),
        ('输出', 'Output'),
        ('格式的分片结果', 'format sharding results'),
        
        # Comment markers
        ('配置路径', 'Configure paths'),
        ('导入四步骤流水线', 'Import four-step pipeline'),
        ('导入失败，使用替代方案', 'Import failed, using fallback'),
        ('解析命令行参数', 'Parse command line arguments'),
        ('读取输入数据', 'Read input data'),
        ('数据验证', 'Data validation'),
        ('节点特征', 'node_features'),
        ('交易图', 'transaction_graph'),
        
        # Process descriptions
        ('处理输入数据', 'Process input data'),
        ('运行四步骤流水线', 'Run four-step pipeline'),
        ('格式化输出结果', 'Format output results'),
        ('保存结果到文件', 'Save results to file'),
        
        # Error and status messages
        ('成功', 'SUCCESS'),
        ('失败', 'FAILED'),
        ('错误', 'ERROR'),
        ('警告', 'WARNING'),
        ('开始', 'START'),
        ('完成', 'COMPLETE'),
        ('处理中', 'PROCESSING'),
        
        # Technical terms
        ('分片', 'shard'),
        ('节点', 'node'),
        ('特征', 'feature'),
        ('结果', 'result'),
        ('数据', 'data'),
        ('文件', 'file'),
        
        # Any remaining Chinese characters
        (r'[\u4e00-\u9fff]+', '[CHINESE_CHAR]'),
        
        # Special Unicode characters that might cause issues
        ('🔄', '[LOOP]'),
        ('✅', '[SUCCESS]'),
        ('❌', '[ERROR]'),
        ('⚠️', '[WARNING]'),
        ('📊', '[DATA]'),
        ('🔧', '[CONFIG]'),
        ('🚀', '[START]'),
        ('📁', '[FILE]'),
        ('🎯', '[TARGET]'),
        ('🔗', '[LINK]'),
        ('🧠', '[AI]'),
        
        # Arrow and other symbols
        ('→', '->'),
        ('←', '<-'),
        ('↔', '<->'),
        ('⇄', '<->'),
        ('⟨', '<'),
        ('⟩', '>'),
    ]
    
    # Apply replacements
    for old, new in replacements:
        if old.startswith(r'['):  # Regex pattern
            content = re.sub(old, new, content)
        else:
            content = content.replace(old, new)
    
    # Additional cleanup for any missed non-ASCII characters
    def replace_non_ascii(match):
        char = match.group(0)
        return f'[U+{ord(char):04X}]'
    
    # Replace any remaining non-ASCII characters
    content = re.sub(r'[^\x00-\x7F]', replace_non_ascii, content)
    
    # Write back the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[SUCCESS] Fixed {file_path}")
    
    # Count lines and verify
    lines = content.split('\n')
    print(f"  Total lines: {len(lines)}")
    print(f"  File size: {len(content)} characters")
    
    # Check for any remaining problematic characters
    problematic = re.findall(r'[^\x00-\x7F]', content)
    if problematic:
        print(f"[WARNING] Found {len(problematic)} non-ASCII characters still remaining")
        unique_chars = set(problematic)
        for char in unique_chars:
            print(f"  U+{ord(char):04X}: {char}")
    else:
        print("[SUCCESS] All characters are now ASCII-compatible")

if __name__ == "__main__":
    fix_evolvegcn_interface()
