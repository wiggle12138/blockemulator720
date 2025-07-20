#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修复Python文件中的emoji字符以解决GBK编码问题
"""

import os
import re
import glob

# emoji替换映射表
EMOJI_REPLACEMENTS = {
    '[START]': '[START]',
    '[FIX]': '[FIX]',
    '[DEVICE]': '[DEVICE]',
    '[SPEED]': '[SPEED]',
    '[DATA]': '[DATA]',
    '[SUCCESS]': '[SUCCESS]',
    '[ERROR]': '[ERROR]',
    '[WARNING]': '[WARNING]',
    '[TARGET]': '[TARGET]',
    '[CONFIG]': '[CONFIG]',
    '[STEP1]': '[STEP1]',
    '[STEP2]': '[STEP2]',
    '[STEP3]': '[STEP3]',
    '[STEP4]': '[STEP4]',
}

def fix_emoji_in_file(filepath):
    """修复单个文件中的emoji字符"""
    try:
        # 读取文件
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含emoji
        has_emoji = any(emoji in content for emoji in EMOJI_REPLACEMENTS.keys())
        if not has_emoji:
            return False
        
        # 替换emoji字符
        original_content = content
        for emoji, replacement in EMOJI_REPLACEMENTS.items():
            content = content.replace(emoji, replacement)
        
        # 写回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"[SUCCESS] Fixed {filepath}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to fix {filepath}: {e}")
        return False

def main():
    """主函数"""
    print("[START] 开始批量修复Python文件中的emoji字符...")
    print("=" * 60)
    
    # 获取当前目录下所有Python文件
    python_files = []
    for pattern in ['*.py', '*/*.py', '*/*/*.py']:
        python_files.extend(glob.glob(pattern))
    
    fixed_count = 0
    total_count = len(python_files)
    
    for py_file in python_files:
        if fix_emoji_in_file(py_file):
            fixed_count += 1
    
    print("=" * 60)
    print(f"[SUMMARY] 处理完成: {fixed_count}/{total_count} 个文件被修复")
    print("[TARGET] 所有emoji字符已替换为ASCII兼容标记")

if __name__ == "__main__":
    main()
