#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复特定文件中的所有特殊字符
"""

def fix_special_chars_in_integrated_pipeline():
    """修复integrated_four_step_pipeline.py中的所有特殊字符"""
    
    # 定义特殊字符替换映射
    special_replacements = {
        # 原有的emoji字符
        '🔄': '[LOOP]',
        '🎉': '[SUCCESS]',
        '📈': '[METRICS]', 
        '📁': '[FILES]',
        '🔗': '[LINK]',
        '🧠': '[AI]',
        '📄': '[FILE]',
        '📝': '[DATA]',
        '🚀': '[START]',
        '⚠️': '[WARNING]',
        '❌': '[ERROR]',
        '✅': '[SUCCESS]',
        '🔧': '[CONFIG]',
        '💾': '[SAVE]',
        '🎯': '[TARGET]',
        # 新增的特殊字符
        '⇄': '和',
        '→': '->'
    }
    
    files_to_fix = [
        'integrated_four_step_pipeline.py',
        'docker/integrated_four_step_pipeline.py'
    ]
    
    for filepath in files_to_fix:
        try:
            # 读取文件内容
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 替换特殊字符
            original_content = content
            for char, replacement in special_replacements.items():
                content = content.replace(char, replacement)
            
            # 写入修复后的内容
            if content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"[SUCCESS] Fixed {filepath}")
            else:
                print(f"[INFO] No special chars found in {filepath}")
                
        except Exception as e:
            print(f"[ERROR] Failed to fix {filepath}: {e}")

if __name__ == "__main__":
    print("[START] 修复特定文件中的所有特殊字符...")
    fix_special_chars_in_integrated_pipeline()
    print("[DONE] 修复完成")
